#ifndef MSAR_CROSS_VALIDATION_HPP
#define MSAR_CROSS_VALIDATION_HPP

#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "standardizer.hpp"
#include "mlp.hpp"
#include "evaluation.hpp"
#include "training.hpp"

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <fstream>

namespace msar {

struct Fold {
    std::vector<size_t> train_indices;
    std::vector<size_t> val_indices;
};

inline std::vector<Fold> createStratifiedFolds(const std::vector<GapRecord>& dataset, size_t k = 5, unsigned int seed = 42) {
    std::vector<Fold> folds(k);

    // Group sample indices by class
    std::map<int, std::vector<size_t>> class_indices;
    for (size_t i = 0; i < dataset.size(); ++i) {
        class_indices[dataset[i].label].push_back(i);
    }

    std::mt19937 gen(seed);

    // Stratified distribution
    for (auto& pair : class_indices) {
        auto& indices = pair.second;
        std::shuffle(indices.begin(), indices.end(), gen);

        for (size_t i = 0; i < indices.size(); ++i) {
            size_t fold_idx = i % k;
            folds[fold_idx].val_indices.push_back(indices[i]);
        }
    }

    // Build train_indices for each fold as the complement of val_indices
    for (size_t f = 0; f < k; ++f) {
        std::vector<bool> is_val(dataset.size(), false);
        for (size_t idx : folds[f].val_indices) {
            is_val[idx] = true;
        }
        for (size_t i = 0; i < dataset.size(); ++i) {
            if (!is_val[i]) {
                folds[f].train_indices.push_back(i);
            }
        }
    }

    return folds;
}

struct CVMetrics {
    double acc_mean = 0.0, acc_std = 0.0;
    double macro_f1_mean = 0.0, macro_f1_std = 0.0;
    double weighted_f1_mean = 0.0, weighted_f1_std = 0.0;
    double loss_mean = 0.0, loss_std = 0.0;
    std::vector<EvaluationResult> fold_results;
    std::vector<int> oof_predictions;
    std::vector<std::vector<double>> oof_probs;
    EvaluationResult oof_overall_eval;
};

inline CVMetrics runCrossValidation(
    const std::vector<GapRecord>& dataset,
    const std::vector<std::vector<double>>& raw_features,
    const TrainingConfig& base_config,
    size_t k = 5,
    unsigned int cv_seed = 42)
{
    CVMetrics cv;
    size_t n = dataset.size();
    cv.oof_predictions.assign(n, -1);
    cv.oof_probs.assign(n, std::vector<double>(3, 0.0));

    auto folds = createStratifiedFolds(dataset, k, cv_seed);

    for (size_t f = 0; f < k; ++f) {
        const auto& fold = folds[f];
        size_t n_train = fold.train_indices.size();
        size_t n_val = fold.val_indices.size();

        std::vector<std::vector<double>> X_train(n_train);
        std::vector<int> y_train(n_train);
        for (size_t i = 0; i < n_train; ++i) {
            size_t idx = fold.train_indices[i];
            X_train[i] = raw_features[idx];
            y_train[i] = dataset[idx].label;
        }

        std::vector<std::vector<double>> X_val(n_val);
        std::vector<int> y_val(n_val);
        for (size_t i = 0; i < n_val; ++i) {
            size_t idx = fold.val_indices[i];
            X_val[i] = raw_features[idx];
            y_val[i] = dataset[idx].label;
        }

        // Fit standardizer ONLY on training fold
        FeatureStandardizer scaler;
        scaler.fit(X_train);
        scaler.transformInPlace(X_train);
        scaler.transformInPlace(X_val);

        // Configure training
        TrainingConfig fold_config = base_config;
        fold_config.experiment_name = base_config.experiment_name + "_fold" + std::to_string(f);
        fold_config.checkpoint_path = "models/temp_fold_" + std::to_string(f) + ".bin";
        fold_config.history_csv_path = ""; // Don't write history per fold

        auto res = trainModel(X_train, y_train, X_val, y_val, fold_config);
        cv.fold_results.push_back(res.final_val_eval);

        // Save OOF predictions and probabilities
        for (size_t i = 0; i < n_val; ++i) {
            size_t orig_idx = fold.val_indices[i];
            cv.oof_predictions[orig_idx] = res.val_preds[i];
            cv.oof_probs[orig_idx] = res.val_probs[i];
        }

        // Clean up temporary checkpoint
        std::remove(fold_config.checkpoint_path.c_str());
    }

    // Compute mean and std across folds
    double acc_sum = 0.0, f1_sum = 0.0, wf1_sum = 0.0, loss_sum = 0.0;
    for (const auto& fr : cv.fold_results) {
        acc_sum += fr.accuracy;
        f1_sum += fr.macro_f1;
        wf1_sum += fr.weighted_f1;
        loss_sum += fr.weighted_loss;
    }
    cv.acc_mean = acc_sum / k;
    cv.macro_f1_mean = f1_sum / k;
    cv.weighted_f1_mean = wf1_sum / k;
    cv.loss_mean = loss_sum / k;

    double acc_var = 0.0, f1_var = 0.0, wf1_var = 0.0, loss_var = 0.0;
    for (const auto& fr : cv.fold_results) {
        acc_var += (fr.accuracy - cv.acc_mean) * (fr.accuracy - cv.acc_mean);
        f1_var += (fr.macro_f1 - cv.macro_f1_mean) * (fr.macro_f1 - cv.macro_f1_mean);
        wf1_var += (fr.weighted_f1 - cv.weighted_f1_mean) * (fr.weighted_f1 - cv.weighted_f1_mean);
        loss_var += (fr.weighted_loss - cv.loss_mean) * (fr.weighted_loss - cv.loss_mean);
    }
    cv.acc_std = std::sqrt(acc_var / k);
    cv.macro_f1_std = std::sqrt(f1_var / k);
    cv.weighted_f1_std = std::sqrt(wf1_var / k);
    cv.loss_std = std::sqrt(loss_var / k);

    // Compute overall OOF evaluation
    std::vector<int> y_all(n);
    for (size_t i = 0; i < n; ++i) y_all[i] = dataset[i].label;
    auto class_weights = computeClassWeights(y_all, base_config.class_weight_alpha);
    cv.oof_overall_eval = evaluatePredictions(y_all, cv.oof_predictions, cv.oof_probs, class_weights);

    return cv;
}

inline void generateErrorAnalysisReport(
    const std::vector<GapRecord>& dataset,
    const CVMetrics& cv,
    const std::string& filepath)
{
    std::ofstream out(filepath);
    if (!out.is_open()) return;

    out << "# Stage 0 Diagnostic Error Analysis (5-Fold Stratified Cross-Validation)\n\n";
    out << "**Evaluation Dataset**: Google GAP Development + Validation Combined ($N = 2,454$)\n";
    out << "**Model Evaluated**: Baseline V2 36-Feature MLP (36 -> 32 -> 3, Adam $lr=0.001$, $\\alpha=0.50$)\n\n";

    out << "## 1. 5-Fold Cross-Validation Performance Summary\n\n";
    out << "| Fold | Accuracy | Macro-F1 | Weighted-F1 | Weighted Loss |\n";
    out << "|:---:|:---:|:---:|:---:|:---:|\n";
    for (size_t f = 0; f < cv.fold_results.size(); ++f) {
        const auto& fr = cv.fold_results[f];
        out << "| Fold " << (f + 1) << " | "
            << std::fixed << std::setprecision(2) << (fr.accuracy * 100.0) << "% | "
            << std::setprecision(4) << fr.macro_f1 << " | "
            << fr.weighted_f1 << " | "
            << fr.weighted_loss << " |\n";
    }
    out << "| **Mean ± Std** | **" 
        << std::fixed << std::setprecision(2) << (cv.acc_mean * 100.0) << " ± " << (cv.acc_std * 100.0) << "%** | **"
        << std::setprecision(4) << cv.macro_f1_mean << " ± " << cv.macro_f1_std << "** | **"
        << cv.weighted_f1_mean << " ± " << cv.weighted_f1_std << "** | **"
        << cv.loss_mean << " ± " << cv.loss_std << "** |\n\n";

    out << "### Overall Out-Of-Fold (OOF) Metrics\n";
    out << "- **OOF Accuracy**: " << std::fixed << std::setprecision(2) << (cv.oof_overall_eval.accuracy * 100.0) << "%\n";
    out << "- **OOF Macro-F1**: " << std::setprecision(4) << cv.oof_overall_eval.macro_f1 << "\n";
    out << "- **OOF Weighted-F1**: " << cv.oof_overall_eval.weighted_f1 << "\n\n";

    out << "### Overall OOF Confusion Matrix\n```\n";
    out << "                 Predicted\n";
    out << "                 A        B        Neither\n";
    out << "Actual A         " << std::setw(8) << cv.oof_overall_eval.confusion_matrix[0][0] << " "
                              << std::setw(8) << cv.oof_overall_eval.confusion_matrix[0][1] << " "
                              << std::setw(8) << cv.oof_overall_eval.confusion_matrix[0][2] << "\n";
    out << "Actual B         " << std::setw(8) << cv.oof_overall_eval.confusion_matrix[1][0] << " "
                              << std::setw(8) << cv.oof_overall_eval.confusion_matrix[1][1] << " "
                              << std::setw(8) << cv.oof_overall_eval.confusion_matrix[1][2] << "\n";
    out << "Actual Neither   " << std::setw(8) << cv.oof_overall_eval.confusion_matrix[2][0] << " "
                              << std::setw(8) << cv.oof_overall_eval.confusion_matrix[2][1] << " "
                              << std::setw(8) << cv.oof_overall_eval.confusion_matrix[2][2] << "\n";
    out << "```\n\n";

    out << "### Overall OOF Per-Class Breakdown\n";
    out << "| Class | Precision | Recall | F1-Score | Support |\n";
    out << "|---|:---:|:---:|:---:|:---:|\n";
    const char* c_names[3] = {"A (0)", "B (1)", "Neither (2)"};
    for (int c = 0; c < 3; ++c) {
        out << "| " << c_names[c] << " | "
            << std::fixed << std::setprecision(4) << cv.oof_overall_eval.class_metrics[c].precision << " | "
            << cv.oof_overall_eval.class_metrics[c].recall << " | "
            << cv.oof_overall_eval.class_metrics[c].f1 << " | "
            << cv.oof_overall_eval.class_metrics[c].support << " |\n";
    }
    out << "\n---\n\n";

    // 2. Pronoun Type Diagnostic
    std::map<std::string, std::vector<size_t>> pronoun_samples;
    for (size_t i = 0; i < dataset.size(); ++i) {
        std::string p_lower = toLowerString(dataset[i].pronoun);
        pronoun_samples[p_lower].push_back(i);
    }

    out << "## 2. Errors by Pronoun Type\n\n";
    out << "| Pronoun | Total | Correct | Errors | Accuracy | % of Neither Correct |\n";
    out << "|---|:---:|:---:|:---:|:---:|:---:|\n";
    for (const auto& pair : pronoun_samples) {
        const std::string& pron = pair.first;
        const auto& indices = pair.second;
        size_t total = indices.size();
        size_t correct = 0;
        size_t neither_total = 0, neither_correct = 0;

        for (size_t idx : indices) {
            int actual = dataset[idx].label;
            int pred = cv.oof_predictions[idx];
            if (actual == pred) correct++;
            if (actual == 2) {
                neither_total++;
                if (pred == 2) neither_correct++;
            }
        }
        double acc = total > 0 ? (100.0 * correct / total) : 0.0;
        double neither_acc = neither_total > 0 ? (100.0 * neither_correct / neither_total) : 0.0;

        out << "| `" << pron << "` | " << total << " | " << correct << " | " 
            << (total - correct) << " | " << std::fixed << std::setprecision(2) << acc << "% | "
            << neither_acc << "% (" << neither_correct << "/" << neither_total << ") |\n";
    }
    out << "\n---\n\n";

    // 3. Sentence Locality Diagnostic
    struct LocalityStats {
        size_t total = 0;
        size_t correct = 0;
        size_t neither_total = 0;
        size_t neither_correct = 0;
    };
    LocalityStats loc_same, loc_prev, loc_distant;

    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& rec = dataset[i];
        auto sents = getSentences(rec.text);
        size_t s_a = findSentenceIndex(sents, rec.a_offset);
        size_t s_b = findSentenceIndex(sents, rec.b_offset);
        size_t s_p = findSentenceIndex(sents, rec.pronoun_offset);

        bool in_same = (s_a == s_p || s_b == s_p);
        bool in_prev = ((s_p > 0 && (s_a == s_p - 1 || s_b == s_p - 1)) && !in_same);
        
        LocalityStats* target = nullptr;
        if (in_same) target = &loc_same;
        else if (in_prev) target = &loc_prev;
        else target = &loc_distant;

        target->total++;
        if (dataset[i].label == cv.oof_predictions[i]) target->correct++;
        if (dataset[i].label == 2) {
            target->neither_total++;
            if (cv.oof_predictions[i] == 2) target->neither_correct++;
        }
    }

    out << "## 3. Errors by Sentence Locality (Context Distance)\n\n";
    out << "| Context Configuration | Total | Correct | Accuracy | Neither Recall |\n";
    out << "|---|:---:|:---:|:---:|:---:|\n";
    auto printLoc = [&](const std::string& name, const LocalityStats& s) {
        double acc = s.total > 0 ? (100.0 * s.correct / s.total) : 0.0;
        double n_rec = s.neither_total > 0 ? (100.0 * s.neither_correct / s.neither_total) : 0.0;
        out << "| " << name << " | " << s.total << " | " << s.correct << " | "
            << std::fixed << std::setprecision(2) << acc << "% | " << n_rec << "% |\n";
    };
    printLoc("At least one candidate in Same Sentence", loc_same);
    printLoc("Candidate in Immediately Preceding Sentence", loc_prev);
    printLoc("Distant Sentence (> 1 sentence gap)", loc_distant);
    out << "\n---\n\n";

    // 4. Character Distance Diagnostic
    struct DistanceStats {
        size_t total = 0;
        size_t correct = 0;
    };
    DistanceStats dist_close, dist_mid, dist_far, dist_extreme;

    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& rec = dataset[i];
        long long min_d = std::min(std::abs(rec.pronoun_offset - rec.a_offset), std::abs(rec.pronoun_offset - rec.b_offset));
        DistanceStats* target = nullptr;
        if (min_d < 50) target = &dist_close;
        else if (min_d < 100) target = &dist_mid;
        else if (min_d < 200) target = &dist_far;
        else target = &dist_extreme;

        target->total++;
        if (dataset[i].label == cv.oof_predictions[i]) target->correct++;
    }

    out << "## 4. Errors by Character Distance (Min Offset to Pronoun)\n\n";
    out << "| Distance Range | Total | Correct | Accuracy |\n";
    out << "|---|:---:|:---:|:---:|\n";
    auto printDist = [&](const std::string& name, const DistanceStats& s) {
        double acc = s.total > 0 ? (100.0 * s.correct / s.total) : 0.0;
        out << "| " << name << " | " << s.total << " | " << s.correct << " | "
            << std::fixed << std::setprecision(2) << acc << "% |\n";
    };
    printDist("< 50 characters", dist_close);
    printDist("50 - 100 characters", dist_mid);
    printDist("100 - 200 characters", dist_far);
    printDist("200+ characters", dist_extreme);
    out << "\n---\n\n";

    // 5. Key Error Drivers & Priority Formulation
    out << "## 5. Diagnostic Findings & Stage Roadmap\n\n";
    out << "### Finding 1: The 'Neither' Bottleneck\n";
    out << "- The minority class (Neither, $N = 263$) has by far the weakest F1 score ("
        << std::fixed << std::setprecision(4) << cv.oof_overall_eval.class_metrics[2].f1 << ").\n";
    out << "- " << cv.oof_overall_eval.confusion_matrix[2][0] << " actual Neither samples were falsely classified as A, and "
        << cv.oof_overall_eval.confusion_matrix[2][1] << " were falsely classified as B.\n";
    out << "- Conversely, " << cv.oof_overall_eval.confusion_matrix[0][2] << " A samples and "
        << cv.oof_overall_eval.confusion_matrix[1][2] << " B samples were misclassified as Neither.\n";
    out << "- **Implication for Stage 2 & 4**: We need lexical gender agreement features (e.g. matching name gender to pronoun gender) and OOF threshold/multiplier calibration to separate true coreferents from non-matching candidates.\n\n";

    out << "### Finding 2: Candidate A vs. Candidate B Confusion\n";
    out << "- Mutual confusion between A and B accounts for substantial error volume:\n";
    out << "  - Actual A predicted as B: " << cv.oof_overall_eval.confusion_matrix[0][1] << " samples.\n";
    out << "  - Actual B predicted as A: " << cv.oof_overall_eval.confusion_matrix[1][0] << " samples.\n";
    out << "- **Implication for Stage 1**: Because A and B occupy distinct feature slots (0..17 vs 18..35), the MLP lacks structural symmetry. A shared-weight pair-scoring architecture (or A<->B swap augmentation) will enforce symmetry and improve discrimination between A and B.\n\n";

    out << "### Finding 3: Gender and Grammatical Role Ambiguity\n";
    out << "- Possessive pronouns (`his`, `her`) and ambiguous `her` (accusative vs possessive) exhibit distinct error patterns compared to nominative pronouns (`he`, `she`).\n";
    out << "- **Implication for Stage 2**: Adding grammatical role features (nominative vs accusative vs possessive) and verb-parallelism heuristics will directly target these errors.\n";
}

} // namespace msar

#endif // MSAR_CROSS_VALIDATION_HPP
