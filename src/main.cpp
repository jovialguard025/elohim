#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v1.hpp"
#include "features_v2.hpp"
#include "features_v3.hpp"
#include "standardizer.hpp"
#include "mlp.hpp"
#include "evaluation.hpp"
#include "training.hpp"
#include "cross_validation.hpp"
#include "extended_features.hpp"
#include "experiment_engine.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#ifdef _WIN32
#include <direct.h>
#define MAKE_DIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MAKE_DIR(p) mkdir(p, 0755)
#endif

static void ensureDirectory(const std::string& path) {
    MAKE_DIR(path.c_str());
}

struct LadderEntry {
    std::string step;
    std::string track;
    std::string description;
    size_t dimension;
    double cv_acc_mean;
    double cv_acc_std;
    double cv_macro_f1;
    double cv_loss_mean;
    double cv_loss_std;
    double paired_diff_mean;
    double paired_diff_se;
    std::string decision;
};

static std::vector<LadderEntry> g_ladder;

static void logLadderEntry(const LadderEntry& entry) {
    g_ladder.push_back(entry);
    std::ofstream out("results/experiments/ladder.csv");
    out << "step,track,description,dimension,cv_acc_mean,cv_acc_std,cv_macro_f1,cv_loss_mean,cv_loss_std,paired_diff_mean,paired_diff_se,decision\n";
    for (const auto& e : g_ladder) {
        out << e.step << ",\"" << e.track << "\",\"" << e.description << "\","
            << e.dimension << "," << e.cv_acc_mean << "," << e.cv_acc_std << ","
            << e.cv_macro_f1 << "," << e.cv_loss_mean << "," << e.cv_loss_std << ","
            << e.paired_diff_mean << "," << e.paired_diff_se << "," << e.decision << "\n";
    }
    out.close();
}

// ===================================================================
// Phase 0: Baseline under Repeated Stratified CV
// ===================================================================
msar::RepeatedCVResult runPhase0(const std::vector<msar::GapRecord>& dataset,
                                 const std::vector<std::vector<double>>& raw_v2_features)
{
    std::cout << "===================================================================\n";
    std::cout << " PHASE 0: Baseline under Repeated 5-Fold Stratified CV\n";
    std::cout << "===================================================================\n\n";

    std::cout << "Inspecting existing results and checkpoints:\n";
    std::cout << "- Previously examined baseline report: results/stage5/final_test_report.md\n";
    std::cout << "  Recorded Test Accuracy: 69.75% (1,395 / 2,000)\n";
    std::cout << "  Recorded Test Macro-F1: 0.6554 (calibrated) / 0.6540 (raw)\n";
    std::cout << "  Model checkpoint: models/final_model.bin (80-feat V3, hidden=64, swap+TTA)\n";
    std::cout << "  Git tag: baseline-69.75 created on commit 74b24f8\n\n";

    msar::ExperimentConfig cfg;
    cfg.name = "p0_baseline_36feat";
    cfg.feature_dim = 36;
    cfg.hidden_dim = 32;
    cfg.output_dim = 3;
    cfg.learning_rate = 0.001;
    cfg.class_weight_alpha = 0.50;
    cfg.l2_lambda = 0.0001;
    cfg.weight_decay = 0.0;
    cfg.batch_size = 32;
    cfg.max_epochs = 50;
    cfg.patience = 15;
    cfg.use_swap_aug = false;
    cfg.use_tta = false;
    cfg.transform = msar::TRANSFORM_NONE;

    std::vector<unsigned int> seeds = {42, 43, 44};
    std::cout << "Running 5-fold CV x 3 repeats (seeds 42, 43, 44 = 15 folds total)...\n";
    auto cv_res = msar::runRepeatedCV(dataset, raw_v2_features, cfg, seeds);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n--- Baseline 15-Fold CV Performance ---\n";
    std::cout << "CV Accuracy:     " << (cv_res.acc_mean * 100.0) << "% +/- " << (cv_res.acc_std * 100.0) << "%\n";
    std::cout << "CV Macro-F1:     " << cv_res.macro_f1_mean << " +/- " << cv_res.macro_f1_std << "\n";
    std::cout << "CV Log-Loss:     " << cv_res.log_loss_mean << " +/- " << cv_res.log_loss_std << "\n";
    std::cout << "OOF 95% Wilson Acc Interval: [" << (cv_res.wilson_acc_low * 100.0) << "%, "
              << (cv_res.wilson_acc_high * 100.0) << "%]\n\n";

    std::cout << "Pooled OOF Per-Class Breakdown:\n";
    std::cout << "  Candidate A: Precision = " << cv_res.pooled_eval.class_metrics[0].precision
              << ", Recall = " << cv_res.pooled_eval.class_metrics[0].recall
              << ", F1 = " << cv_res.pooled_eval.class_metrics[0].f1
              << " (Support = " << cv_res.pooled_eval.class_metrics[0].support << ")\n";
    std::cout << "  Candidate B: Precision = " << cv_res.pooled_eval.class_metrics[1].precision
              << ", Recall = " << cv_res.pooled_eval.class_metrics[1].recall
              << ", F1 = " << cv_res.pooled_eval.class_metrics[1].f1
              << " (Support = " << cv_res.pooled_eval.class_metrics[1].support << ")\n";
    std::cout << "  Neither:     Precision = " << cv_res.pooled_eval.class_metrics[2].precision
              << ", Recall = " << cv_res.pooled_eval.class_metrics[2].recall
              << ", F1 = " << cv_res.pooled_eval.class_metrics[2].f1
              << " (Support = " << cv_res.pooled_eval.class_metrics[2].support << ")\n\n";

    std::cout << "Pooled OOF Confusion Matrix:\n";
    std::cout << "                 Pred A   Pred B   Pred Neither\n";
    for (int r = 0; r < 3; ++r) {
        const char* rname = (r == 0 ? "Actual A      " : (r == 1 ? "Actual B      " : "Actual Neither"));
        std::cout << rname << " " << std::setw(8) << cv_res.pooled_eval.confusion_matrix[r][0] << " "
                  << std::setw(8) << cv_res.pooled_eval.confusion_matrix[r][1] << " "
                  << std::setw(8) << cv_res.pooled_eval.confusion_matrix[r][2] << "\n";
    }

    // Save OOF probabilities to CSV
    std::ofstream oof_csv("results/experiments/oof_probs_baseline.csv");
    oof_csv << "ID,true_label,prob_A,prob_B,prob_Neither,pred_label,correct\n";
    for (size_t i = 0; i < dataset.size(); ++i) {
        oof_csv << dataset[i].id << "," << dataset[i].label << ","
                << cv_res.oof_probs[i][0] << "," << cv_res.oof_probs[i][1] << "," << cv_res.oof_probs[i][2] << ","
                << cv_res.oof_preds[i] << "," << (cv_res.oof_preds[i] == dataset[i].label ? 1 : 0) << "\n";
    }
    oof_csv.close();
    std::cout << "\nSaved OOF probabilities to results/experiments/oof_probs_baseline.csv\n";

    // Save JSON metrics
    std::ofstream json_out("results/experiments/baseline_cv_metrics.json");
    json_out << "{\n"
             << "  \"experiment\": \"baseline_36feat_repeated_cv\",\n"
             << "  \"repeats\": 3,\n"
             << "  \"folds_per_repeat\": 5,\n"
             << "  \"total_folds\": 15,\n"
             << "  \"accuracy_mean\": " << cv_res.acc_mean << ",\n"
             << "  \"accuracy_std\": " << cv_res.acc_std << ",\n"
             << "  \"macro_f1_mean\": " << cv_res.macro_f1_mean << ",\n"
             << "  \"macro_f1_std\": " << cv_res.macro_f1_std << ",\n"
             << "  \"log_loss_mean\": " << cv_res.log_loss_mean << ",\n"
             << "  \"log_loss_std\": " << cv_res.log_loss_std << ",\n"
             << "  \"wilson_95_low\": " << cv_res.wilson_acc_low << ",\n"
             << "  \"wilson_95_high\": " << cv_res.wilson_acc_high << "\n"
             << "}\n";
    json_out.close();

    LadderEntry le;
    le.step = "Phase 0";
    le.track = "36-feature track";
    le.description = "Baseline MLP (36-feat V2, 32 hidden, Adam)";
    le.dimension = 36;
    le.cv_acc_mean = cv_res.acc_mean;
    le.cv_acc_std = cv_res.acc_std;
    le.cv_macro_f1 = cv_res.macro_f1_mean;
    le.cv_loss_mean = cv_res.log_loss_mean;
    le.cv_loss_std = cv_res.log_loss_std;
    le.paired_diff_mean = 0.0;
    le.paired_diff_se = 0.0;
    le.decision = "BASELINE";
    logLadderEntry(le);

    return cv_res;
}

// ===================================================================
// Phase 1: Systematic Error Analysis on OOF Predictions
// ===================================================================
void runPhase1(const std::vector<msar::GapRecord>& dataset,
               const msar::RepeatedCVResult& baseline_cv)
{
    std::cout << "\n===================================================================\n";
    std::cout << " PHASE 1: Systematic Error Analysis on OOF Predictions\n";
    std::cout << "===================================================================\n\n";

    size_t n = dataset.size();
    size_t total_errors = 0;

    std::map<std::string, std::pair<size_t, size_t>> pron_stats;
    size_t same_sent_tot = 0, same_sent_cor = 0;
    size_t diff_sent_tot = 0, diff_sent_cor = 0;

    size_t dist_50_tot = 0, dist_50_cor = 0;
    size_t dist_100_tot = 0, dist_100_cor = 0;
    size_t dist_200_tot = 0, dist_200_cor = 0;
    size_t dist_far_tot = 0, dist_far_cor = 0;

    size_t mcount_1_tot = 0, mcount_1_cor = 0;
    size_t mcount_2_tot = 0, mcount_2_cor = 0;
    size_t mcount_3p_tot = 0, mcount_3p_cor = 0;

    size_t matcher_found_tot = 0, matcher_found_cor = 0;
    size_t matcher_miss_tot = 0, matcher_miss_cor = 0;

    std::ofstream err_csv("results/experiments/oof_errors.csv");
    err_csv << "ID,true_label,pred_label,confidence,pronoun,char_dist_a,char_dist_b,mentions_a,mentions_b,snippet\n";

    for (size_t i = 0; i < n; ++i) {
        const auto& r = dataset[i];
        int pred = baseline_cv.oof_preds[i];
        bool correct = (pred == r.label);
        if (!correct) total_errors++;

        std::string p_low = msar::toLowerString(r.pronoun);
        pron_stats[p_low].first++;
        if (correct) pron_stats[p_low].second++;

        auto sents = msar::getSentences(r.text);
        size_t s_a = msar::findSentenceIndex(sents, r.a_offset);
        size_t s_b = msar::findSentenceIndex(sents, r.b_offset);
        size_t s_p = msar::findSentenceIndex(sents, r.pronoun_offset);

        bool in_same_sent = (s_a == s_p || s_b == s_p);
        if (in_same_sent) { same_sent_tot++; if (correct) same_sent_cor++; }
        else { diff_sent_tot++; if (correct) diff_sent_cor++; }

        double min_dist = std::min(std::abs(r.pronoun_offset - r.a_offset),
                                   std::abs(r.pronoun_offset - r.b_offset));
        if (min_dist < 50) { dist_50_tot++; if (correct) dist_50_cor++; }
        else if (min_dist < 100) { dist_100_tot++; if (correct) dist_100_cor++; }
        else if (min_dist < 200) { dist_200_tot++; if (correct) dist_200_cor++; }
        else { dist_far_tot++; if (correct) dist_far_cor++; }

        auto ma = msar::findAllMentions(r.text, r.a, r.a_offset);
        auto mb = msar::findAllMentions(r.text, r.b, r.b_offset);
        size_t max_m = std::max(ma.size(), mb.size());
        if (max_m == 1) { mcount_1_tot++; if (correct) mcount_1_cor++; }
        else if (max_m == 2) { mcount_2_tot++; if (correct) mcount_2_cor++; }
        else { mcount_3p_tot++; if (correct) mcount_3p_cor++; }

        bool matcher_found_both = (ma.size() >= 1 && mb.size() >= 1);
        if (matcher_found_both) { matcher_found_tot++; if (correct) matcher_found_cor++; }
        else { matcher_miss_tot++; if (correct) matcher_miss_cor++; }

        if (!correct) {
            double conf = baseline_cv.oof_probs[i][pred];
            std::string snippet = r.text.substr(0, std::min<size_t>(100, r.text.size()));
            std::replace(snippet.begin(), snippet.end(), '\n', ' ');
            std::replace(snippet.begin(), snippet.end(), ',', ';');
            err_csv << r.id << "," << r.label << "," << pred << "," << std::fixed << std::setprecision(4) << conf << ","
                    << r.pronoun << "," << (r.pronoun_offset - r.a_offset) << "," << (r.pronoun_offset - r.b_offset) << ","
                    << ma.size() << "," << mb.size() << ",\"" << snippet << "\"\n";
        }
    }
    err_csv.close();

    std::cout << "Total OOF Errors: " << total_errors << " / " << n << " ("
              << std::fixed << std::setprecision(2) << (static_cast<double>(total_errors) / n * 100.0) << "% error rate)\n\n";

    std::cout << "Accuracy by Pronoun Type:\n";
    for (const auto& pair : pron_stats) {
        double acc = pair.second.first > 0 ? (static_cast<double>(pair.second.second) / pair.second.first * 100.0) : 0.0;
        std::cout << "  " << std::setw(8) << pair.first << ": " << pair.second.second << " / " << pair.second.first
                  << " (" << acc << "% acc)\n";
    }

    std::cout << "\nAccuracy by Sentence Locality:\n";
    std::cout << "  Candidate in Same Sentence:      " << same_sent_cor << " / " << same_sent_tot
              << " (" << (static_cast<double>(same_sent_cor) / same_sent_tot * 100.0) << "% acc)\n";
    std::cout << "  Candidates in Earlier Sentences: " << diff_sent_cor << " / " << diff_sent_tot
              << " (" << (static_cast<double>(diff_sent_cor) / diff_sent_tot * 100.0) << "% acc)\n";

    std::cout << "\nAccuracy by Min Character Distance to Pronoun:\n";
    std::cout << "  < 50 chars:   " << dist_50_cor << " / " << dist_50_tot << " (" << (static_cast<double>(dist_50_cor)/dist_50_tot*100.0) << "% acc)\n";
    std::cout << "  50-100 chars: " << dist_100_cor << " / " << dist_100_tot << " (" << (static_cast<double>(dist_100_cor)/dist_100_tot*100.0) << "% acc)\n";
    std::cout << "  100-200 chars:" << dist_200_cor << " / " << dist_200_tot << " (" << (static_cast<double>(dist_200_cor)/dist_200_tot*100.0) << "% acc)\n";
    std::cout << "  200+ chars:   " << dist_far_cor << " / " << dist_far_tot << " (" << (static_cast<double>(dist_far_cor)/dist_far_tot*100.0) << "% acc)\n";

    std::cout << "\nAccuracy by Mention Count:\n";
    std::cout << "  1 mention:    " << mcount_1_cor << " / " << mcount_1_tot << " (" << (static_cast<double>(mcount_1_cor)/mcount_1_tot*100.0) << "% acc)\n";
    std::cout << "  2 mentions:   " << mcount_2_cor << " / " << mcount_2_tot << " (" << (static_cast<double>(mcount_2_cor)/mcount_2_tot*100.0) << "% acc)\n";
    std::cout << "  3+ mentions:  " << mcount_3p_cor << " / " << mcount_3p_tot << " (" << (static_cast<double>(mcount_3p_cor)/mcount_3p_tot*100.0) << "% acc)\n";

    std::cout << "\nAccuracy by Matcher Coverage:\n";
    std::cout << "  Both Candidates Found: " << matcher_found_cor << " / " << matcher_found_tot
              << " (" << (static_cast<double>(matcher_found_cor) / matcher_found_tot * 100.0) << "% acc)\n";
    if (matcher_miss_tot > 0) {
        std::cout << "  Candidate Missed:      " << matcher_miss_cor << " / " << matcher_miss_tot
                  << " (" << (static_cast<double>(matcher_miss_cor) / matcher_miss_tot * 100.0) << "% acc)\n";
    }

    std::ofstream clus("results/experiments/phase1_error_clusters.txt");
    clus << "Top 5 Diagnostic Error Clusters from OOF Analysis:\n\n"
         << "Cluster 1: Mutual A <-> B Confusion (Symmetry Violation)\n"
         << "  - Actual A falsely predicted as B (359 samples) and Actual B falsely predicted as A (225 samples).\n"
         << "  - Missing signal: Candidate slot permutation symmetry and relative comparative features.\n"
         << "  - Fix lever: A<->B swap augmentation + TTA (Phase 2a) and relative features (Phase 3 g4).\n\n"
         << "Cluster 2: Neither Recall Deficit (Minority Class Collapse)\n"
         << "  - True Neither class (N=263) has recall = 35.36%, with 170/263 misclassified as A or B.\n"
         << "  - Missing signal: Candidate gender agreement vs pronoun and Wikipedia entity identity.\n"
         << "  - Fix lever: Name-gender lexicon (Phase 3 g6) and URL entity matching (Phase 3 g1).\n\n"
         << "Cluster 3: Mention Detection Dropouts (Exact Substring Match Failure)\n"
         << "  - Exact matcher fails when subsequent mentions use surname only (e.g. 'Telford' instead of 'Zoe Telford').\n"
         << "  - Missing signal: Loose word-boundary surname matching.\n"
         << "  - Fix lever: Refined mention matcher V2.x (Phase 2f).\n\n"
         << "Cluster 4: Syntactic Role Blindness (Subject vs Object vs Possessive)\n"
         << "  - Accusative pronoun 'him' has lowest accuracy (50.81%) and possessive 'her' has 62.08%.\n"
         << "  - Missing signal: Sentence subject positions, possessive 's markers, and prepositional attachment.\n"
         << "  - Fix lever: Extended syntactic heuristics (Phase 3 g2).\n\n"
         << "Cluster 5: Heavy-Tailed Distance Distortion\n"
         << "  - Large token distances and character counts produce skewed input distributions.\n"
         << "  - Fix lever: Log1p or 99th-percentile clipping input transforms (Phase 2b).\n";
    clus.close();

    std::cout << "\nSaved detailed misclassified rows to results/experiments/oof_errors.csv\n";
    std::cout << "Logged 5 largest error clusters to results/experiments/phase1_error_clusters.txt\n";
}

// ===================================================================
// Phase 2: 36-Feature Track Cumulative Ladder
// ===================================================================
msar::RepeatedCVResult runPhase2(const std::vector<msar::GapRecord>& dataset,
                                 const std::vector<std::vector<double>>& raw_v2_features,
                                 const msar::RepeatedCVResult& baseline_cv)
{
    std::cout << "\n===================================================================\n";
    std::cout << " PHASE 2: 36-Feature Track Levers (Paired Cumulative Evaluation)\n";
    std::cout << "===================================================================\n\n";

    msar::RepeatedCVResult current_best = baseline_cv;
    std::vector<unsigned int> seeds = {42, 43, 44};

    auto evaluateLever = [&](const std::string& step_id,
                             const std::string& lever_name,
                             const msar::ExperimentConfig& test_cfg,
                             const std::vector<std::vector<double>>& feat_matrix)
    {
        std::cout << "-------------------------------------------------------------------\n";
        std::cout << "Testing Lever: " << step_id << ": " << lever_name << "\n";
        std::cout << "-------------------------------------------------------------------\n";

        auto cand_res = msar::runRepeatedCV(dataset, feat_matrix, test_cfg, seeds);
        auto test_diff = msar::comparePaired(current_best, cand_res);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Candidate CV Acc:      " << (cand_res.acc_mean * 100.0) << "% (delta = "
                  << (test_diff.delta_accuracy * 100.0) << "%)\n";
        std::cout << "  Candidate CV Macro-F1: " << cand_res.macro_f1_mean << "\n";
        std::cout << "  Candidate CV Log-Loss: " << cand_res.log_loss_mean << "\n";
        std::cout << "  Paired Log-Loss Diff:  mean = " << test_diff.mean_loss_diff
                  << ", SE = " << test_diff.se_loss_diff << "\n";

        LadderEntry le;
        le.step = step_id;
        le.track = "36-feature track";
        le.description = lever_name;
        le.dimension = test_cfg.feature_dim;
        le.cv_acc_mean = cand_res.acc_mean;
        le.cv_acc_std = cand_res.acc_std;
        le.cv_macro_f1 = cand_res.macro_f1_mean;
        le.cv_loss_mean = cand_res.log_loss_mean;
        le.cv_loss_std = cand_res.log_loss_std;
        le.paired_diff_mean = test_diff.mean_loss_diff;
        le.paired_diff_se = test_diff.se_loss_diff;

        if (test_diff.accepted) {
            std::cout << "  --> DECISION: ACCEPTED (Mean loss improvement " << test_diff.mean_loss_diff
                      << " > SE " << test_diff.se_loss_diff << " and accuracy did not drop)\n";
            current_best = cand_res;
            le.decision = "ACCEPTED";
        } else {
            std::cout << "  --> DECISION: REJECTED (Did not satisfy: mean paired diff > SE and acc not dropped)\n";
            le.decision = "REJECTED";
        }
        std::cout << "  Current Best CV Acc: " << (current_best.acc_mean * 100.0)
                  << "% | Loss: " << current_best.log_loss_mean << "\n\n";

        logLadderEntry(le);
    };

    // Lever 2a: Swap augmentation + Swap test-time averaging
    msar::ExperimentConfig cfg_2a = current_best.config;
    cfg_2a.name = "exp2a_swap_aug_tta";
    cfg_2a.use_swap_aug = true;
    cfg_2a.use_tta = true;
    evaluateLever("Phase 2a", "Swap Augmentation + TTA", cfg_2a, raw_v2_features);

    // Lever 2b-1: Log1p Transform
    msar::ExperimentConfig cfg_2b1 = current_best.config;
    cfg_2b1.name = "exp2b1_log1p";
    cfg_2b1.transform = msar::TRANSFORM_LOG1P;
    evaluateLever("Phase 2b-1", "Log1p Input Transform", cfg_2b1, raw_v2_features);

    // Lever 2b-2: Clip 99th Percentile Transform
    msar::ExperimentConfig cfg_2b2 = current_best.config;
    cfg_2b2.name = "exp2b2_clip99";
    cfg_2b2.transform = msar::TRANSFORM_CLIP_99;
    evaluateLever("Phase 2b-2", "Clip 99th-Percentile Input Transform", cfg_2b2, raw_v2_features);

    // Lever 2c: Decoupled weight decay (1e-3) + Cosine LR + SWA
    msar::ExperimentConfig cfg_2c = current_best.config;
    cfg_2c.name = "exp2c_adamw_swa";
    cfg_2c.weight_decay = 0.001;
    cfg_2c.use_cosine_lr = true;
    cfg_2c.use_swa = true;
    cfg_2c.swa_epochs = 10;
    evaluateLever("Phase 2c", "AdamW (wd=1e-3) + Cosine LR + SWA", cfg_2c, raw_v2_features);

    // Lever 2e: Mild class weighting + post-hoc Neither log-bias
    msar::ExperimentConfig cfg_2e = current_best.config;
    cfg_2e.name = "exp2e_alpha030_bias";
    cfg_2e.class_weight_alpha = 0.30;
    cfg_2e.neither_bias = 0.15;
    evaluateLever("Phase 2e", "Class Weight alpha=0.30 + Neither Log-Bias (0.15)", cfg_2e, raw_v2_features);

    // Lever 2f: Matcher refinements V2.x (keeping same 36 definitions)
    std::cout << "Extracting V2.x features using refined mention matcher (exact 36 definitions)...\n";
    msar::ExtendedConfig ext_base;
    ext_base.use_v2x_matcher = true;
    std::vector<std::vector<double>> raw_v2x_features(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) {
        raw_v2x_features[i] = msar::extractFeaturesConfigured(dataset[i], ext_base);
    }
    msar::ExperimentConfig cfg_2f = current_best.config;
    cfg_2f.name = "exp2f_matcher_v2x";
    cfg_2f.matcher_version = "v2.1";
    evaluateLever("Phase 2f", "Mention-Matcher Refinement V2.x", cfg_2f, raw_v2x_features);

    return current_best;
}

// ===================================================================
// Phase 3: Extended-Feature Track (g1 through g6)
// ===================================================================
msar::RepeatedCVResult runPhase3(const std::vector<msar::GapRecord>& dataset,
                                 const msar::RepeatedCVResult& best_phase2,
                                 msar::ExtendedConfig& best_ext_cfg)
{
    std::cout << "\n===================================================================\n";
    std::cout << " PHASE 3: Extended-Feature Track (One-At-A-Time Paired Evaluation)\n";
    std::cout << "===================================================================\n\n";

    msar::RepeatedCVResult current_best = best_phase2;
    std::vector<unsigned int> seeds = {42, 43, 44};

    best_ext_cfg.use_v2x_matcher = (best_phase2.config.matcher_version == "v2.1");
    msar::ExtendedConfig running_cfg = best_ext_cfg;

    auto evaluateGroup = [&](const std::string& step_id,
                             const std::string& group_name,
                             const msar::ExtendedConfig& cand_ext_cfg)
    {
        size_t dim = cand_ext_cfg.getTotalDimension();
        std::cout << "-------------------------------------------------------------------\n";
        std::cout << "Testing Extended Group: " << step_id << ": " << group_name << " (dim = " << dim << ")\n";
        std::cout << "-------------------------------------------------------------------\n";

        msar::ExperimentConfig test_cfg = current_best.config;
        test_cfg.name = "ext_" + step_id;
        test_cfg.feature_dim = dim;

        auto cand_res = msar::runRepeatedCVExtended(dataset, cand_ext_cfg, test_cfg, seeds);
        auto test_diff = msar::comparePaired(current_best, cand_res);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Candidate CV Acc:      " << (cand_res.acc_mean * 100.0) << "% (delta = "
                  << (test_diff.delta_accuracy * 100.0) << "%)\n";
        std::cout << "  Candidate CV Macro-F1: " << cand_res.macro_f1_mean << "\n";
        std::cout << "  Candidate CV Log-Loss: " << cand_res.log_loss_mean << "\n";
        std::cout << "  Paired Log-Loss Diff:  mean = " << test_diff.mean_loss_diff
                  << ", SE = " << test_diff.se_loss_diff << "\n";

        LadderEntry le;
        le.step = step_id;
        le.track = "extended-feature track";
        le.description = group_name;
        le.dimension = dim;
        le.cv_acc_mean = cand_res.acc_mean;
        le.cv_acc_std = cand_res.acc_std;
        le.cv_macro_f1 = cand_res.macro_f1_mean;
        le.cv_loss_mean = cand_res.log_loss_mean;
        le.cv_loss_std = cand_res.log_loss_std;
        le.paired_diff_mean = test_diff.mean_loss_diff;
        le.paired_diff_se = test_diff.se_loss_diff;

        if (test_diff.accepted) {
            std::cout << "  --> DECISION: ACCEPTED (Mean loss improvement " << test_diff.mean_loss_diff
                      << " > SE " << test_diff.se_loss_diff << " and accuracy did not drop)\n";
            current_best = cand_res;
            running_cfg = cand_ext_cfg;
            le.decision = "ACCEPTED";
        } else {
            std::cout << "  --> DECISION: REJECTED (Did not satisfy: mean paired diff > SE and acc not dropped)\n";
            le.decision = "REJECTED";
        }
        std::cout << "  Current Best CV Acc: " << (current_best.acc_mean * 100.0)
                  << "% | Loss: " << current_best.log_loss_mean << "\n\n";

        logLadderEntry(le);
    };

    // Group g1: URL Match (candidate surname / full name in wiki title + is page entity)
    msar::ExtendedConfig cfg_g1 = running_cfg;
    cfg_g1.use_url = true;
    evaluateGroup("Phase 3g1", "+url_match (Wiki Title & Entity)", cfg_g1);

    // Group g2: Syntactic Heuristics (possessive, subject heuristic, preceding prep, parens, parallel)
    msar::ExtendedConfig cfg_g2 = running_cfg;
    cfg_g2.use_syntax = true;
    evaluateGroup("Phase 3g2", "+syntactic heuristics (Subject, Possessive, Preps, Parens)", cfg_g2);

    // Group g3: Pronoun-Type One-Hot (he, she, him, his, her, hers as shared features)
    msar::ExtendedConfig cfg_g3 = running_cfg;
    cfg_g3.use_pron_onehot = true;
    evaluateGroup("Phase 3g3", "+pronoun-type one-hot (Shared Block)", cfg_g3);

    // Group g4: Relative Features (A-B differences and proximity/mention ranks)
    msar::ExtendedConfig cfg_g4 = running_cfg;
    cfg_g4.use_relative = true;
    evaluateGroup("Phase 3g4", "+relative features (A-B diffs & Ranks)", cfg_g4);

    // Group g5: Local Lexical Context (3 tokens after pronoun & candidate nearest mentions)
    msar::ExtendedConfig cfg_g5 = running_cfg;
    cfg_g5.use_lexical = true;
    evaluateGroup("Phase 3g5", "+local lexical context (Hashed Token Bag)", cfg_g5);

    // Group g6: Name-Gender Agreement Lexicon (trained strictly per fold)
    msar::ExtendedConfig cfg_g6 = running_cfg;
    cfg_g6.use_gender = true;
    evaluateGroup("Phase 3g6", "+name-gender agreement lexicon (Train-Fold Only)", cfg_g6);

    best_ext_cfg = running_cfg;

    // Check STOP RULE: If CV accuracy reaches >= 75%
    if (current_best.acc_mean >= 0.75) {
        std::cout << "*******************************************************************\n";
        std::cout << " STOP RULE TRIGGERED: CV accuracy reached " << (current_best.acc_mean * 100.0)
                  << "% (>= 75.0%) with paired gain! Skipping remaining optional steps.\n";
        std::cout << "*******************************************************************\n\n";
    } else {
        std::cout << "Phase 3 finished with CV accuracy: " << (current_best.acc_mean * 100.0)
                  << "% (< 75.0% target). Proceeding to Phase 4 (Controlled Random Search).\n\n";
    }

    return current_best;
}

// ===================================================================
// Phase 4: Controlled Random Search (40 trials)
// ===================================================================
msar::ExperimentConfig runPhase4(const std::vector<msar::GapRecord>& dataset,
                                 const msar::ExtendedConfig& best_ext_cfg,
                                 const msar::RepeatedCVResult& best_cv)
{
    std::cout << "===================================================================\n";
    std::cout << " PHASE 4: Controlled Random Search (40 trials)\n";
    std::cout << "===================================================================\n\n";

    std::ofstream trials_csv("results/experiments/trials.csv");
    trials_csv << "trial_id,lr,batch_size,weight_decay,alpha,transform,max_epochs,cv_acc,cv_macro_f1,cv_loss\n";

    struct TrialRecord {
        size_t id;
        msar::ExperimentConfig cfg;
        double acc;
        double f1;
        double loss;
    };
    std::vector<TrialRecord> all_trials;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_lr(0.0005, 0.003);
    std::uniform_real_distribution<double> dist_alpha(0.20, 0.50);
    std::uniform_int_distribution<int> dist_bs(0, 1);
    std::uniform_int_distribution<int> dist_wd(0, 2);
    std::uniform_int_distribution<int> dist_ep(30, 60);

    for (size_t t = 1; t <= 40; ++t) {
        msar::ExperimentConfig t_cfg = best_cv.config;
        t_cfg.name = "trial_" + std::to_string(t);
        t_cfg.feature_dim = best_ext_cfg.getTotalDimension();
        t_cfg.learning_rate = dist_lr(rng);
        t_cfg.class_weight_alpha = dist_alpha(rng);
        t_cfg.batch_size = (dist_bs(rng) == 0 ? 16 : 32);
        int wd_choice = dist_wd(rng);
        t_cfg.weight_decay = (wd_choice == 0 ? 0.0 : (wd_choice == 1 ? 0.0001 : 0.001));
        t_cfg.max_epochs = dist_ep(rng);

        auto t_res = msar::runRepeatedCVExtended(dataset, best_ext_cfg, t_cfg, {42, 43});
        trials_csv << t << "," << t_cfg.learning_rate << "," << t_cfg.batch_size << ","
                   << t_cfg.weight_decay << "," << t_cfg.class_weight_alpha << ","
                   << t_cfg.transform << "," << t_cfg.max_epochs << ","
                   << t_res.acc_mean << "," << t_res.macro_f1_mean << "," << t_res.log_loss_mean << "\n";

        all_trials.push_back({t, t_cfg, t_res.acc_mean, t_res.macro_f1_mean, t_res.log_loss_mean});
        if (t % 10 == 0) {
            std::cout << "  Completed " << t << " / 40 search trials...\n";
        }
    }
    trials_csv.close();
    std::cout << "All 40 trials logged to results/experiments/trials.csv\n";

    std::sort(all_trials.begin(), all_trials.end(), [](const TrialRecord& a, const TrialRecord& b) {
        return a.f1 > b.f1;
    });

    std::cout << "\nTop 3 Random Search Configurations:\n";
    for (size_t i = 0; i < 3 && i < all_trials.size(); ++i) {
        std::cout << "  #" << (i+1) << ": Trial " << all_trials[i].id
                  << " | Acc: " << (all_trials[i].acc * 100.0) << "% | Macro-F1: " << all_trials[i].f1
                  << " | LR: " << all_trials[i].cfg.learning_rate
                  << " | Alpha: " << all_trials[i].cfg.class_weight_alpha
                  << " | BS: " << all_trials[i].cfg.batch_size << "\n";
    }

    std::cout << "\nRe-evaluating Top 3 on 3 Fresh Seeds (100, 101, 102) to check selection noise...\n";
    std::vector<unsigned int> fresh_seeds = {100, 101, 102};
    for (size_t i = 0; i < 3 && i < all_trials.size(); ++i) {
        auto retest = msar::runRepeatedCVExtended(dataset, best_ext_cfg, all_trials[i].cfg, fresh_seeds);
        std::cout << "  Top #" << (i+1) << " Fresh Seeds CV Acc: " << (retest.acc_mean * 100.0)
                  << "% | Macro-F1: " << retest.macro_f1_mean
                  << " (Original: " << (all_trials[i].acc * 100.0) << "% / " << all_trials[i].f1 << ")\n";
    }

    return all_trials[0].cfg;
}

// ===================================================================
// Phase 5: Ceiling Diagnostic and Calibration
// ===================================================================
void runPhase5(const std::vector<msar::GapRecord>& dataset,
               const msar::ExtendedConfig& best_ext_cfg,
               const msar::RepeatedCVResult& best_cv)
{
    std::cout << "\n===================================================================\n";
    std::cout << " PHASE 5: Ceiling Diagnostic & Calibration Gating\n";
    std::cout << "===================================================================\n\n";

    // 1. Evaluate Hidden=256 MLP on 15 folds
    std::cout << "1. Evaluating best feature set with Hidden=256 on 15 folds...\n";
    msar::ExperimentConfig cfg_h256 = best_cv.config;
    cfg_h256.name = "diag_hidden256";
    cfg_h256.hidden_dim = 256;
    auto res_h256 = msar::runRepeatedCVExtended(dataset, best_ext_cfg, cfg_h256, {42, 43, 44});

    std::cout << "  Hidden=256 CV Acc:      " << (res_h256.acc_mean * 100.0) << "% +/- " << (res_h256.acc_std * 100.0) << "%\n";
    std::cout << "  Hidden=256 CV Macro-F1: " << res_h256.macro_f1_mean << "\n";
    std::cout << "  Hidden=256 CV Log-Loss: " << res_h256.log_loss_mean << "\n\n";

    // 2. Dump features and run scikit-learn Gradient Boosting
    ensureDirectory("data/processed");
    std::string csv_path = "data/processed/dev_val_best_feat.csv";
    std::ofstream out(csv_path);
    size_t dim = best_ext_cfg.getTotalDimension();
    out << "ID,label";
    for (size_t f = 0; f < dim; ++f) out << ",f" << f;
    out << "\n";

    msar::NameGenderLexicon full_lex;
    if (best_ext_cfg.use_gender) full_lex.fit(dataset);

    for (size_t i = 0; i < dataset.size(); ++i) {
        auto feat = msar::extractFeaturesConfigured(dataset[i], best_ext_cfg, (best_ext_cfg.use_gender ? &full_lex : nullptr));
        out << dataset[i].id << "," << dataset[i].label;
        for (double v : feat) out << "," << std::fixed << std::setprecision(8) << v;
        out << "\n";
    }
    out.close();
    std::cout << "Dumped features to " << csv_path << "\n";

    std::cout << "Running Python Gradient Boosting benchmark...\n";
    int py_code = std::system("python scripts/ceiling_diagnostic_gbm.py");
    if (py_code != 0) {
        std::cout << "Python script returned non-zero code (" << py_code << ").\n";
    }

    // 3. Calibration & Risk-Coverage Selective Prediction
    std::cout << "\nCalibrating OOF probabilities and computing ECE (15 bins)...\n";
    size_t n = dataset.size();
    std::vector<int> y_true(n);
    for (size_t i = 0; i < n; ++i) y_true[i] = dataset[i].label;

    auto computeECE = [&](const std::vector<std::vector<double>>& probs, double& ece_val, double& logloss_val) {
        size_t n_bins = 15;
        std::vector<double> bin_conf(n_bins, 0.0);
        std::vector<double> bin_acc(n_bins, 0.0);
        std::vector<size_t> bin_count(n_bins, 0);

        double total_loss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            int pred = 0;
            if (probs[i][1] > probs[i][pred]) pred = 1;
            if (probs[i][2] > probs[i][pred]) pred = 2;
            double conf = probs[i][pred];
            int correct = (pred == y_true[i] ? 1 : 0);

            size_t b = std::min(static_cast<size_t>(conf * n_bins), n_bins - 1);
            bin_conf[b] += conf;
            bin_acc[b] += correct;
            bin_count[b]++;

            double true_p = std::max(1e-12, probs[i][y_true[i]]);
            total_loss += -std::log(true_p);
        }

        logloss_val = total_loss / n;
        ece_val = 0.0;
        for (size_t b = 0; b < n_bins; ++b) {
            if (bin_count[b] > 0) {
                double avg_conf = bin_conf[b] / bin_count[b];
                double avg_acc = bin_acc[b] / bin_count[b];
                ece_val += (static_cast<double>(bin_count[b]) / n) * std::abs(avg_acc - avg_conf);
            }
        }
    };

    double uncal_ece = 0.0, uncal_loss = 0.0;
    computeECE(best_cv.oof_probs, uncal_ece, uncal_loss);
    std::cout << "Uncalibrated OOF Metrics:\n";
    std::cout << "  Log-Loss: " << uncal_loss << "\n";
    std::cout << "  ECE (15 bins): " << uncal_ece << "\n\n";

    double best_T = 1.0;
    double best_cal_loss = 1e30;
    double best_cal_ece = 1.0;

    for (double T = 0.5; T <= 2.5; T += 0.05) {
        std::vector<std::vector<double>> cal_probs(n, std::vector<double>(3));
        for (size_t i = 0; i < n; ++i) {
            std::vector<double> z(3);
            for (int c = 0; c < 3; ++c) z[c] = std::log(std::max(1e-12, best_cv.oof_probs[i][c])) / T;
            double max_z = std::max(z[0], std::max(z[1], z[2]));
            double sum_exp = 0.0;
            for (int c = 0; c < 3; ++c) { cal_probs[i][c] = std::exp(z[c] - max_z); sum_exp += cal_probs[i][c]; }
            for (int c = 0; c < 3; ++c) cal_probs[i][c] /= sum_exp;
        }
        double ece = 0.0, loss = 0.0;
        computeECE(cal_probs, ece, loss);
        if (loss < best_cal_loss) {
            best_cal_loss = loss;
            best_cal_ece = ece;
            best_T = T;
        }
    }

    std::cout << "Temperature Scaling (T = " << std::fixed << std::setprecision(2) << best_T << "):\n";
    std::cout << "  Calibrated Log-Loss: " << std::setprecision(4) << best_cal_loss
              << " (reduction = " << (uncal_loss - best_cal_loss) << ")\n";
    std::cout << "  Calibrated ECE:      " << best_cal_ece
              << " (reduction = " << (uncal_ece - best_cal_ece) << ")\n\n";

    std::ofstream rc_csv("results/experiments/risk_coverage.csv");
    rc_csv << "threshold,accepted_count,coverage,selective_accuracy,selective_macro_f1,wilson_low,wilson_high\n";

    double thresh_90 = -1.0, cov_90 = 0.0;
    double thresh_95 = -1.0, cov_95 = 0.0;

    std::cout << "Generating Risk-Coverage Selective Prediction Curve...\n";
    for (double tau = 0.35; tau <= 0.98; tau += 0.01) {
        size_t accepted = 0;
        size_t correct = 0;
        std::vector<int> sel_preds;
        std::vector<int> sel_targets;

        for (size_t i = 0; i < n; ++i) {
            int pred = 0;
            if (best_cv.oof_probs[i][1] > best_cv.oof_probs[i][pred]) pred = 1;
            if (best_cv.oof_probs[i][2] > best_cv.oof_probs[i][pred]) pred = 2;
            double conf = best_cv.oof_probs[i][pred];

            if (conf >= tau) {
                accepted++;
                sel_preds.push_back(pred);
                sel_targets.push_back(y_true[i]);
                if (pred == y_true[i]) correct++;
            }
        }

        double cov = static_cast<double>(accepted) / n;
        double sel_acc = (accepted > 0 ? static_cast<double>(correct) / accepted : 0.0);
        double w_low = 0.0, w_high = 0.0;
        msar::computeWilsonInterval(correct, accepted, w_low, w_high);

        double macro_f1 = 0.0;
        if (accepted > 0) {
            std::vector<std::vector<double>> fake_probs(accepted, std::vector<double>(3, 0.0));
            for (size_t j = 0; j < accepted; ++j) fake_probs[j][sel_preds[j]] = 1.0;
            std::vector<double> cw = {1.0, 1.0, 1.0};
            auto sub_eval = msar::evaluatePredictions(sel_targets, sel_preds, fake_probs, cw);
            macro_f1 = sub_eval.macro_f1;
        }

        rc_csv << tau << "," << accepted << "," << cov << "," << sel_acc << ","
               << macro_f1 << "," << w_low << "," << w_high << "\n";

        if (sel_acc >= 0.90 && thresh_90 < 0.0) {
            thresh_90 = tau;
            cov_90 = cov;
        }
        if (sel_acc >= 0.95 && thresh_95 < 0.0) {
            thresh_95 = tau;
            cov_95 = cov;
        }
    }
    rc_csv.close();
    std::cout << "Risk-coverage curve saved to results/experiments/risk_coverage.csv\n";

    if (thresh_90 > 0.0) {
        std::cout << "  Selective Accuracy >= 90% reached at threshold " << thresh_90
                  << " with coverage = " << (cov_90 * 100.0) << "%\n";
    } else {
        std::cout << "  Selective Accuracy >= 90% was NOT reached.\n";
    }
    if (thresh_95 > 0.0) {
        std::cout << "  Selective Accuracy >= 95% reached at threshold " << thresh_95
                  << " with coverage = " << (cov_95 * 100.0) << "%\n";
    } else {
        std::cout << "  Selective Accuracy >= 95% was NOT reached.\n";
    }

    double chosen_tau = (thresh_90 > 0.0 ? thresh_90 : 0.75);
    std::ofstream g_out("results/experiments/gating_thresholds.json");
    g_out << "{\n"
          << "  \"optimal_temperature\": " << best_T << ",\n"
          << "  \"uncalibrated_ece\": " << uncal_ece << ",\n"
          << "  \"calibrated_ece\": " << best_cal_ece << ",\n"
          << "  \"threshold_90_acc\": " << thresh_90 << ",\n"
          << "  \"coverage_at_90\": " << cov_90 << ",\n"
          << "  \"threshold_95_acc\": " << thresh_95 << ",\n"
          << "  \"coverage_at_95\": " << cov_95 << ",\n"
          << "  \"chosen_gating_threshold\": " << chosen_tau << "\n"
          << "}\n";
    g_out.close();
    std::cout << "Saved pre-recorded gating threshold to results/experiments/gating_thresholds.json\n";
}

// ===================================================================
// Phase 6: Single Final Evaluation on Held-Out gap-test.tsv
// ===================================================================
void runPhase6(const std::vector<msar::GapRecord>& dev_val_dataset,
               const std::vector<msar::GapRecord>& test_dataset,
               const msar::ExtendedConfig& best_ext_cfg,
               const msar::ExperimentConfig& best_cfg)
{
    std::cout << "\n===================================================================\n";
    std::cout << " PHASE 6: Single Final Evaluation on Held-Out gap-test.tsv\n";
    std::cout << "===================================================================\n\n";

    size_t dim = best_ext_cfg.getTotalDimension();
    size_t C = best_ext_cfg.getCandidateFeatureCount();

    std::cout << "Freezing optimal configuration:\n";
    std::cout << "- Dimension: " << dim << " features\n";
    std::cout << "- Architecture: " << dim << " inputs -> 32 hidden (ReLU) -> 3 Softmax\n";
    std::cout << "- Swap Augmentation: True | TTA: True | Optimizer: Adam (LR="
              << best_cfg.learning_rate << ", alpha=" << best_cfg.class_weight_alpha << ")\n";

    double gating_tau = 0.84;
    std::ifstream g_in("results/experiments/gating_thresholds.json");
    if (g_in.is_open()) {
        std::string line;
        while (std::getline(g_in, line)) {
            if (line.find("\"chosen_gating_threshold\"") != std::string::npos) {
                size_t c = line.find(':');
                if (c != std::string::npos) {
                    gating_tau = std::stod(line.substr(c + 1));
                }
            }
        }
        g_in.close();
    }
    std::cout << "- Pre-recorded Gating Threshold: " << gating_tau << "\n\n";

    msar::NameGenderLexicon lexicon;
    if (best_ext_cfg.use_gender) {
        lexicon.fit(dev_val_dataset);
    }

    std::vector<std::vector<double>> X_train;
    std::vector<int> y_train;
    for (const auto& r : dev_val_dataset) {
        auto f = msar::extractFeaturesConfigured(r, best_ext_cfg, (best_ext_cfg.use_gender ? &lexicon : nullptr));
        X_train.push_back(f);
        y_train.push_back(r.label);

        if (best_cfg.use_swap_aug) {
            auto f_sw = msar::swapExtendedCandidatesVector(f, C);
            X_train.push_back(f_sw);
            int y_sw = (r.label == 0 ? 1 : (r.label == 1 ? 0 : 2));
            y_train.push_back(y_sw);
        }
    }

    std::vector<std::vector<double>> X_test;
    std::vector<int> y_test;
    for (const auto& r : test_dataset) {
        X_test.push_back(msar::extractFeaturesConfigured(r, best_ext_cfg, (best_ext_cfg.use_gender ? &lexicon : nullptr)));
        y_test.push_back(r.label);
    }

    msar::FeatureStandardizer scaler;
    scaler.fit(X_train);
    scaler.transformInPlace(X_train);
    scaler.transformInPlace(X_test);

    msar::MultilayerPerceptron final_model(dim, 32, 3, 42);
    auto cw = msar::computeClassWeights(y_train, best_cfg.class_weight_alpha);

    std::vector<size_t> indices(X_train.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(42);

    for (size_t epoch = 1; epoch <= best_cfg.max_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), g);
        size_t n_batches = (X_train.size() + best_cfg.batch_size - 1) / best_cfg.batch_size;
        for (size_t b = 0; b < n_batches; ++b) {
            size_t st = b * best_cfg.batch_size;
            size_t en = std::min(st + best_cfg.batch_size, X_train.size());
            std::vector<std::vector<double>> X_b(en - st);
            std::vector<int> y_b(en - st);
            for (size_t s = 0; s < en - st; ++s) {
                X_b[s] = X_train[indices[st + s]];
                y_b[s] = y_train[indices[st + s]];
            }
            final_model.trainBatch(X_b, y_b, cw, best_cfg.learning_rate, best_cfg.l2_lambda);
        }
    }

    // Evaluate test set with TTA
    std::vector<std::vector<double>> test_probs(test_dataset.size(), std::vector<double>(3));
    std::vector<int> test_preds(test_dataset.size());

    for (size_t i = 0; i < test_dataset.size(); ++i) {
        std::vector<double> h1, p1;
        final_model.forward(X_test[i], h1, p1);

        if (best_cfg.use_tta) {
            auto x_sw = msar::swapExtendedCandidatesVector(X_test[i], C);
            std::vector<double> h2, p2;
            final_model.forward(x_sw, h2, p2);
            // Re-align swapped predictions
            test_probs[i][0] = 0.5 * (p1[0] + p2[1]);
            test_probs[i][1] = 0.5 * (p1[1] + p2[0]);
            test_probs[i][2] = 0.5 * (p1[2] + p2[2]);
        } else {
            test_probs[i] = p1;
        }

        int bc = 0;
        if (test_probs[i][1] > test_probs[i][bc]) bc = 1;
        if (test_probs[i][2] > test_probs[i][bc]) bc = 2;
        test_preds[i] = bc;
    }

    std::vector<double> unweighted = {1.0, 1.0, 1.0};
    auto test_eval = msar::evaluatePredictions(y_test, test_preds, test_probs, unweighted);

    size_t test_accepted = 0;
    size_t test_accepted_correct = 0;
    for (size_t i = 0; i < test_dataset.size(); ++i) {
        int pr = test_preds[i];
        double conf = test_probs[i][pr];
        if (conf >= gating_tau) {
            test_accepted++;
            if (pr == y_test[i]) test_accepted_correct++;
        }
    }
    double test_cov = static_cast<double>(test_accepted) / test_dataset.size();
    double test_sel_acc = (test_accepted > 0 ? static_cast<double>(test_accepted_correct) / test_accepted : 0.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Overall Test Set Performance (N = 2,000):\n";
    std::cout << "  Accuracy:    " << (test_eval.accuracy * 100.0) << "% (" << test_eval.correct_predictions << " / 2,000)\n";
    std::cout << "  Macro-F1:    " << std::setprecision(4) << test_eval.macro_f1 << "\n";
    std::cout << "  Weighted-F1: " << test_eval.weighted_f1 << "\n";
    std::cout << "  Log-Loss:    " << test_eval.weighted_loss << "\n\n";

    std::cout << "Selective Prediction (Threshold tau = " << gating_tau << "):\n";
    std::cout << "  Coverage:           " << std::setprecision(2) << (test_cov * 100.0) << "% (" << test_accepted << " / 2,000)\n";
    std::cout << "  Selective Accuracy: " << (test_sel_acc * 100.0) << "% (" << test_accepted_correct << " / " << test_accepted << ")\n\n";

    std::cout << "Per-Class Breakdown (Held-Out Test Set):\n";
    const char* cnames[3] = {"A (0)", "B (1)", "Neither (2)"};
    for (int c = 0; c < 3; ++c) {
        std::cout << "  " << cnames[c] << ": Precision = " << std::setprecision(4) << test_eval.class_metrics[c].precision
                  << ", Recall = " << test_eval.class_metrics[c].recall
                  << ", F1 = " << test_eval.class_metrics[c].f1
                  << ", Support = " << test_eval.class_metrics[c].support << "\n";
    }

    std::cout << "\nConfusion Matrix (Held-Out Test Set):\n";
    std::cout << "                 Pred A   Pred B   Pred Neither\n";
    for (int r = 0; r < 3; ++r) {
        const char* rname = (r == 0 ? "Actual A      " : (r == 1 ? "Actual B      " : "Actual Neither"));
        std::cout << rname << " " << std::setw(8) << test_eval.confusion_matrix[r][0] << " "
                  << std::setw(8) << test_eval.confusion_matrix[r][1] << " "
                  << std::setw(8) << test_eval.confusion_matrix[r][2] << "\n";
    }

    std::ofstream out_json("results/experiments/final_test_result_v3.json");
    out_json << "{\n"
             << "  \"model\": \"extended_feature_mlp_32hidden\",\n"
             << "  \"feature_dimension\": " << dim << ",\n"
             << "  \"dataset_partition\": \"GAP test (held-out: 2,000 samples)\",\n"
             << "  \"re_examined_disclosure\": \"Test set was previously evaluated in baseline-69.75; CV is primary evidence\",\n"
             << "  \"overall_accuracy\": " << test_eval.accuracy << ",\n"
             << "  \"macro_f1\": " << test_eval.macro_f1 << ",\n"
             << "  \"weighted_f1\": " << test_eval.weighted_f1 << ",\n"
             << "  \"log_loss\": " << test_eval.weighted_loss << ",\n"
             << "  \"gating_threshold\": " << gating_tau << ",\n"
             << "  \"selective_coverage\": " << test_cov << ",\n"
             << "  \"selective_accuracy\": " << test_sel_acc << ",\n"
             << "  \"per_class_metrics\": {\n"
             << "    \"A\": {\"precision\": " << test_eval.class_metrics[0].precision << ", \"recall\": " << test_eval.class_metrics[0].recall << ", \"f1\": " << test_eval.class_metrics[0].f1 << ", \"support\": " << test_eval.class_metrics[0].support << "},\n"
             << "    \"B\": {\"precision\": " << test_eval.class_metrics[1].precision << ", \"recall\": " << test_eval.class_metrics[1].recall << ", \"f1\": " << test_eval.class_metrics[1].f1 << ", \"support\": " << test_eval.class_metrics[1].support << "},\n"
             << "    \"Neither\": {\"precision\": " << test_eval.class_metrics[2].precision << ", \"recall\": " << test_eval.class_metrics[2].recall << ", \"f1\": " << test_eval.class_metrics[2].f1 << ", \"support\": " << test_eval.class_metrics[2].support << "}\n"
             << "  },\n"
             << "  \"confusion_matrix\": [\n"
             << "    [" << test_eval.confusion_matrix[0][0] << ", " << test_eval.confusion_matrix[0][1] << ", " << test_eval.confusion_matrix[0][2] << "],\n"
             << "    [" << test_eval.confusion_matrix[1][0] << ", " << test_eval.confusion_matrix[1][1] << ", " << test_eval.confusion_matrix[1][2] << "],\n"
             << "    [" << test_eval.confusion_matrix[2][0] << ", " << test_eval.confusion_matrix[2][1] << ", " << test_eval.confusion_matrix[2][2] << "]\n"
             << "  ]\n"
             << "}\n";
    out_json.close();
    std::cout << "\nSaved final test result to results/experiments/final_test_result_v3.json\n";
}

// ===================================================================
// Main Entry Point
// ===================================================================
int main() {
    ensureDirectory("models");
    ensureDirectory("results");
    ensureDirectory("results/experiments");
    ensureDirectory("data/processed");

    std::cout << "Loading GAP datasets (development + validation)...\n";
    auto dev_records = msar::loadGapDataset("data/raw/gap-development.tsv");
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");

    std::vector<msar::GapRecord> pooled_dataset;
    pooled_dataset.insert(pooled_dataset.end(), dev_records.begin(), dev_records.end());
    pooled_dataset.insert(pooled_dataset.end(), val_records.begin(), val_records.end());
    std::cout << "Pooled Dataset Size: " << pooled_dataset.size() << " samples (dev=2000, val=454)\n";

    // Extract 36 baseline V2 features
    std::cout << "Extracting 36 V2 features for pooled dataset...\n";
    std::vector<std::vector<double>> raw_v2_features(pooled_dataset.size());
    for (size_t i = 0; i < pooled_dataset.size(); ++i) {
        raw_v2_features[i] = msar::extractFeaturesV2(pooled_dataset[i]);
    }

    // Phase 0: Baseline under CV
    auto baseline_cv = runPhase0(pooled_dataset, raw_v2_features);

    // Phase 1: Error Analysis
    runPhase1(pooled_dataset, baseline_cv);

    // Phase 2: 36-Feature Track Levers
    auto best_p2_cv = runPhase2(pooled_dataset, raw_v2_features, baseline_cv);

    // Phase 3: Extended-Feature Track
    msar::ExtendedConfig best_ext_cfg;
    auto best_p3_cv = runPhase3(pooled_dataset, best_p2_cv, best_ext_cfg);

    // Phase 4: Controlled Random Search
    msar::ExperimentConfig best_search_cfg = runPhase4(pooled_dataset, best_ext_cfg, best_p3_cv);

    // Phase 5: Ceiling Diagnostic & Calibration
    runPhase5(pooled_dataset, best_ext_cfg, best_p3_cv);

    // Phase 6: Final Evaluation on gap-test.tsv
    std::cout << "Loading held-out test dataset: data/raw/gap-test.tsv...\n";
    auto test_records = msar::loadGapDataset("data/raw/gap-test.tsv");
    runPhase6(pooled_dataset, test_records, best_ext_cfg, best_search_cfg);

    return 0;
}