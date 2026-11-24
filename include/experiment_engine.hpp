#ifndef MSAR_EXPERIMENT_ENGINE_HPP
#define MSAR_EXPERIMENT_ENGINE_HPP

#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v1.hpp"
#include "features_v2.hpp"
#include "standardizer.hpp"
#include "mlp.hpp"
#include "evaluation.hpp"
#include "training.hpp"
#include "cross_validation.hpp"
#include "extended_features.hpp"

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
#include <numeric>

namespace msar {

enum TransformType {
    TRANSFORM_NONE = 0,
    TRANSFORM_LOG1P = 1,
    TRANSFORM_CLIP_99 = 2
};

struct ExperimentConfig {
    std::string name = "exp";
    size_t feature_dim = 36;
    size_t hidden_dim = 32;
    size_t output_dim = 3;
    double learning_rate = 0.001;
    double class_weight_alpha = 0.50;
    double l2_lambda = 0.0001;
    double adam_beta1 = 0.9;
    double adam_beta2 = 0.999;
    double label_smoothing = 0.0;
    double focal_gamma = 0.0;
    double dropout_rate = 0.0;
    double weight_decay = 0.0; // Decoupled weight decay (AdamW)
    size_t batch_size = 32;
    size_t max_epochs = 50;
    size_t patience = 15;
    bool use_swap_aug = false;
    bool use_tta = false;
    size_t candidate_feature_count = 0; // 0 means two equal candidate blocks and no shared features
    size_t ensemble_members = 1; // Independent MLPs trained on each fold's same training partition
    double ensemble_sample_fraction = 1.0; // Optional row-group subsampling per member (keeps swaps paired)
    bool ensemble_bootstrap = false; // Record-level bootstrap sampling per member
    bool ensemble_geometric_mean = false; // Log-pool member probabilities instead of arithmetic averaging
    bool use_early_stopping = true; // Disabled by repeated CV so outer folds remain score-only
    TransformType transform = TRANSFORM_NONE;
    bool use_cosine_lr = false;
    bool use_swa = false;
    size_t swa_epochs = 10;
    double neither_bias = 0.0; // Post-hoc log-bias for Neither class
    double temperature = 1.0;
    std::string matcher_version = "v2.0"; // "v2.0" (exact) or "v2.1" (loose/robust)
};

struct FoldEvaluation {
    size_t fold_idx = 0;
    unsigned int seed = 0;
    double accuracy = 0.0;
    double macro_f1 = 0.0;
    double weighted_f1 = 0.0;
    double log_loss = 0.0;
    EvaluationResult eval_res;
    std::vector<int> fold_labels;
    std::vector<std::vector<double>> fold_probabilities;
};

struct RepeatedCVResult {
    ExperimentConfig config;
    std::vector<FoldEvaluation> folds; // 15 folds
    double acc_mean = 0.0, acc_std = 0.0;
    double macro_f1_mean = 0.0, macro_f1_std = 0.0;
    double log_loss_mean = 0.0, log_loss_std = 0.0;
    double wilson_acc_low = 0.0, wilson_acc_high = 0.0;
    EvaluationResult pooled_eval;
    std::vector<std::vector<double>> oof_probs; // averaged across 3 repeats: N x 3
    std::vector<int> oof_preds;                // N
};

inline void computeWilsonInterval(size_t correct, size_t total, double& low, double& high, double z = 1.95996) {
    if (total == 0) { low = 0.0; high = 0.0; return; }
    double p = static_cast<double>(correct) / static_cast<double>(total);
    double n = static_cast<double>(total);
    double z2 = z * z;
    double denom = 1.0 + z2 / n;
    double center = (p + z2 / (2.0 * n)) / denom;
    double margin = z * std::sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom;
    low = std::max(0.0, center - margin);
    high = std::min(1.0, center + margin);
}

// Transform functions
inline void applyTransformTrain(std::vector<std::vector<double>>& X_train,
                                std::vector<std::vector<double>>& X_val,
                                TransformType t)
{
    if (t == TRANSFORM_NONE || X_train.empty()) return;

    if (t == TRANSFORM_LOG1P) {
        for (auto& row : X_train) {
            for (auto& val : row) {
                double s = (val >= 0.0 ? 1.0 : -1.0);
                val = s * std::log1p(std::abs(val));
            }
        }
        for (auto& row : X_val) {
            for (auto& val : row) {
                double s = (val >= 0.0 ? 1.0 : -1.0);
                val = s * std::log1p(std::abs(val));
            }
        }
    } else if (t == TRANSFORM_CLIP_99) {
        size_t d = X_train[0].size();
        size_t n = X_train.size();
        for (size_t col = 0; col < d; ++col) {
            std::vector<double> vals(n);
            for (size_t r = 0; r < n; ++r) vals[r] = X_train[r][col];
            std::sort(vals.begin(), vals.end());
            size_t idx99 = static_cast<size_t>(0.99 * n);
            if (idx99 >= n) idx99 = n - 1;
            double cap = vals[idx99];
            size_t idx01 = static_cast<size_t>(0.01 * n);
            double floor_val = vals[idx01];

            for (size_t r = 0; r < n; ++r) {
                if (X_train[r][col] > cap) X_train[r][col] = cap;
                if (X_train[r][col] < floor_val) X_train[r][col] = floor_val;
            }
            for (size_t r = 0; r < X_val.size(); ++r) {
                if (X_val[r][col] > cap) X_val[r][col] = cap;
                if (X_val[r][col] < floor_val) X_val[r][col] = floor_val;
            }
        }
    }
}

// Extended MLP training with AdamW, Cosine LR, and SWA
inline TrainingResult trainModelExtended(
    std::vector<std::vector<double>>& X_train,
    const std::vector<int>& y_train,
    std::vector<std::vector<double>>& X_val,
    const std::vector<int>& y_val,
    const ExperimentConfig& cfg,
    unsigned int fold_seed)
{
    TrainingResult result;
    result.config.experiment_name = cfg.name;
    result.config.feature_dim = cfg.feature_dim;
    result.config.hidden_dim = cfg.hidden_dim;
    result.config.output_dim = cfg.output_dim;
    result.config.learning_rate = cfg.learning_rate;
    result.config.class_weight_alpha = cfg.class_weight_alpha;
    result.config.l2_lambda = cfg.l2_lambda;
    result.config.batch_size = cfg.batch_size;
    result.config.max_epochs = cfg.max_epochs;
    result.config.patience = cfg.patience;
    result.config.use_tta = cfg.use_tta;

    result.class_weights = computeClassWeights(y_train, cfg.class_weight_alpha);

    MultilayerPerceptron model(cfg.feature_dim, cfg.hidden_dim, cfg.output_dim, fold_seed);

    size_t n_train = X_train.size();
    size_t num_batches = (n_train + cfg.batch_size - 1) / cfg.batch_size;

    std::vector<size_t> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 shuffle_gen(fold_seed + 101);

    double best_macro_f1 = -1.0;
    double best_loss = 1e30;
    size_t best_epoch = 0;
    size_t epochs_no_improve = 0;

    std::string temp_ckpt = "models/temp_ext_" + std::to_string(fold_seed) + ".bin";

    // For SWA (Stochastic Weight Averaging)
    std::vector<double> swa_W1(model.W1.size(), 0.0);
    std::vector<double> swa_b1(model.b1.size(), 0.0);
    std::vector<double> swa_W2(model.W2.size(), 0.0);
    std::vector<double> swa_b2(model.b2.size(), 0.0);
    size_t swa_count = 0;

    for (size_t epoch = 1; epoch <= cfg.max_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), shuffle_gen);

        // Cosine learning rate
        double current_lr = cfg.learning_rate;
        if (cfg.use_cosine_lr) {
            double progress = static_cast<double>(epoch - 1) / static_cast<double>(cfg.max_epochs);
            double lr_min = 0.05 * cfg.learning_rate;
            current_lr = lr_min + 0.5 * (cfg.learning_rate - lr_min) * (1.0 + std::cos(3.141592653589793 * progress));
        }

        double total_epoch_loss = 0.0;
        size_t total_samples_epoch = 0;

        for (size_t b = 0; b < num_batches; ++b) {
            size_t start_idx = b * cfg.batch_size;
            size_t end_idx = std::min(start_idx + cfg.batch_size, n_train);
            size_t current_bs = end_idx - start_idx;

            std::vector<std::vector<double>> X_batch(current_bs);
            std::vector<int> y_batch(current_bs);
            for (size_t s = 0; s < current_bs; ++s) {
                size_t idx = indices[start_idx + s];
                X_batch[s] = X_train[idx];
                y_batch[s] = y_train[idx];
            }

            model.trainBatch(X_batch, y_batch, result.class_weights, current_lr, cfg.l2_lambda,
                             cfg.adam_beta1, cfg.adam_beta2, 1e-8, cfg.label_smoothing,
                             cfg.focal_gamma, cfg.dropout_rate);

            // Decoupled weight decay (AdamW)
            if (cfg.weight_decay > 0.0) {
                double wd_step = current_lr * cfg.weight_decay;
                for (auto& w : model.W1) w -= wd_step * w;
                for (auto& w : model.W2) w -= wd_step * w;
            }

            // Accumulate training loss
            for (size_t s = 0; s < current_bs; ++s) {
                std::vector<double> h, p;
                model.forward(X_batch[s], h, p);
                int tgt = y_batch[s];
                double prob = std::max(1e-12, std::min(1.0, p[tgt]));
                total_epoch_loss += -std::log(prob) * result.class_weights[tgt];
                total_samples_epoch++;
            }
        }

        // SWA accumulation
        if (cfg.use_swa && epoch > (cfg.max_epochs - cfg.swa_epochs)) {
            swa_count++;
            for (size_t k = 0; k < model.W1.size(); ++k) swa_W1[k] += model.W1[k];
            for (size_t k = 0; k < model.b1.size(); ++k) swa_b1[k] += model.b1[k];
            for (size_t k = 0; k < model.W2.size(); ++k) swa_W2[k] += model.W2[k];
            for (size_t k = 0; k < model.b2.size(); ++k) swa_b2[k] += model.b2[k];
        }

        if (cfg.use_early_stopping) {
            // This is suitable for a dedicated validation split. Repeated CV
            // disables it so the outer scoring fold cannot select an epoch.
            std::vector<std::vector<double>> val_probs;
            std::vector<int> val_preds = predictModel(model, X_val, val_probs, cfg.use_tta,
                                                      cfg.candidate_feature_count);
            auto val_eval = evaluatePredictions(y_val, val_preds, val_probs, result.class_weights);

            EpochRecord ep;
            ep.epoch = epoch;
            ep.train_loss = total_samples_epoch > 0 ? (total_epoch_loss / total_samples_epoch) : 0.0;
            ep.val_loss = val_eval.weighted_loss;
            ep.val_acc = val_eval.accuracy;
            ep.val_macro_f1 = val_eval.macro_f1;
            result.history.push_back(ep);

            bool is_best = false;
            if (val_eval.macro_f1 > best_macro_f1 + 1e-5) {
                is_best = true;
            } else if (std::abs(val_eval.macro_f1 - best_macro_f1) <= 1e-5 && val_eval.weighted_loss < best_loss) {
                is_best = true;
            }

            if (is_best) {
                best_macro_f1 = val_eval.macro_f1;
                best_loss = val_eval.weighted_loss;
                best_epoch = epoch;
                epochs_no_improve = 0;
                model.saveCheckpoint(temp_ckpt);
            } else {
                epochs_no_improve++;
                if (epochs_no_improve >= cfg.patience) {
                    break;
                }
            }
        }
    }

    // If SWA is enabled and collected, use SWA weights
    if (cfg.use_swa && swa_count > 0) {
        for (size_t k = 0; k < model.W1.size(); ++k) model.W1[k] = swa_W1[k] / static_cast<double>(swa_count);
        for (size_t k = 0; k < model.b1.size(); ++k) model.b1[k] = swa_b1[k] / static_cast<double>(swa_count);
        for (size_t k = 0; k < model.W2.size(); ++k) model.W2[k] = swa_W2[k] / static_cast<double>(swa_count);
        for (size_t k = 0; k < model.b2.size(); ++k) model.b2[k] = swa_b2[k] / static_cast<double>(swa_count);
        result.best_epoch = cfg.max_epochs;
    } else if (cfg.use_early_stopping) {
        // Restore best checkpoint
        try {
            model.loadCheckpoint(temp_ckpt);
        } catch (...) {}
        result.best_epoch = best_epoch;
    } else {
        // Fixed-epoch fit: validation data has not influenced model selection.
        result.best_epoch = cfg.max_epochs;
    }
    std::remove(temp_ckpt.c_str());

    std::vector<std::vector<double>> best_val_probs;
    std::vector<int> best_val_preds = predictModel(model, X_val, best_val_probs, cfg.use_tta,
                                                   cfg.candidate_feature_count);

    // Apply post-hoc Neither log-bias and/or temperature if configured
    if (cfg.neither_bias != 0.0 || cfg.temperature != 1.0) {
        double T = std::max(1e-4, cfg.temperature);
        for (size_t i = 0; i < best_val_probs.size(); ++i) {
            std::vector<double> z(3);
            for (int c = 0; c < 3; ++c) {
                z[c] = std::log(std::max(1e-12, best_val_probs[i][c])) / T;
            }
            z[2] += cfg.neither_bias; // add bias to Neither class
            double max_z = std::max(z[0], std::max(z[1], z[2]));
            double sum_exp = 0.0;
            for (int c = 0; c < 3; ++c) {
                best_val_probs[i][c] = std::exp(z[c] - max_z);
                sum_exp += best_val_probs[i][c];
            }
            for (int c = 0; c < 3; ++c) best_val_probs[i][c] /= sum_exp;
            int bc = 0;
            if (best_val_probs[i][1] > best_val_probs[i][bc]) bc = 1;
            if (best_val_probs[i][2] > best_val_probs[i][bc]) bc = 2;
            best_val_preds[i] = bc;
        }
    }

    result.val_preds = best_val_preds;
    result.val_probs = best_val_probs;
    std::vector<double> unweighted = {1.0, 1.0, 1.0};
    result.final_val_eval = evaluatePredictions(y_val, best_val_preds, best_val_probs, unweighted);
    result.best_val_acc = result.final_val_eval.accuracy;
    result.best_val_macro_f1 = result.final_val_eval.macro_f1;
    result.best_val_loss = result.final_val_eval.weighted_loss;

    return result;
}

// 3-Repeated 5-Fold Stratified Cross-Validation
inline RepeatedCVResult runRepeatedCV(
    const std::vector<GapRecord>& dataset,
    const std::vector<std::vector<double>>& raw_features,
    const ExperimentConfig& cfg,
    const std::vector<unsigned int>& seeds = {42, 43, 44})
{
    RepeatedCVResult res;
    res.config = cfg;
    size_t n = dataset.size();
    res.oof_probs.assign(n, std::vector<double>(3, 0.0));
    res.oof_preds.assign(n, -1);

    size_t k = 5;
    size_t total_folds = seeds.size() * k;
    res.folds.reserve(total_folds);

    std::vector<int> y_all(n);
    for (size_t i = 0; i < n; ++i) y_all[i] = dataset[i].label;

    size_t fold_counter = 0;
    for (unsigned int seed : seeds) {
        auto folds = createStratifiedFolds(dataset, k, seed);

        for (size_t f = 0; f < k; ++f) {
            const auto& fold = folds[f];
            size_t n_train = fold.train_indices.size();
            size_t n_val = fold.val_indices.size();

            std::vector<std::vector<double>> X_train;
            std::vector<int> y_train;
            X_train.reserve(cfg.use_swap_aug ? 2 * n_train : n_train);
            y_train.reserve(cfg.use_swap_aug ? 2 * n_train : n_train);

            for (size_t i = 0; i < n_train; ++i) {
                size_t idx = fold.train_indices[i];
                X_train.push_back(raw_features[idx]);
                y_train.push_back(dataset[idx].label);

                if (cfg.use_swap_aug) {
                    X_train.push_back(swapCandidatesVector(raw_features[idx]));
                    int y_orig = dataset[idx].label;
                    int y_swapped = (y_orig == 0 ? 1 : (y_orig == 1 ? 0 : 2));
                    y_train.push_back(y_swapped);
                }
            }

            std::vector<std::vector<double>> X_val(n_val);
            std::vector<int> y_val(n_val);
            for (size_t i = 0; i < n_val; ++i) {
                size_t idx = fold.val_indices[i];
                X_val[i] = raw_features[idx];
                y_val[i] = dataset[idx].label;
            }

            // Apply feature transform
            applyTransformTrain(X_train, X_val, cfg.transform);

            // Fit standardizer ONLY on training fold
            FeatureStandardizer scaler;
            scaler.fit(X_train);
            scaler.transformInPlace(X_train);
            scaler.transformInPlace(X_val);

            // Train model
            unsigned int fold_run_seed = seed * 1000 + static_cast<unsigned int>(f);
            ExperimentConfig fold_cfg = cfg;
            fold_cfg.use_early_stopping = false;
            auto train_res = trainModelExtended(X_train, y_train, X_val, y_val, fold_cfg, fold_run_seed);

            FoldEvaluation fe;
            fe.fold_idx = fold_counter++;
            fe.seed = seed;
            fe.accuracy = train_res.final_val_eval.accuracy;
            fe.macro_f1 = train_res.final_val_eval.macro_f1;
            fe.weighted_f1 = train_res.final_val_eval.weighted_f1;
            fe.log_loss = train_res.final_val_eval.weighted_loss;
            fe.eval_res = train_res.final_val_eval;
            fe.fold_labels = y_val;
            fe.fold_probabilities = train_res.val_probs;
            res.folds.push_back(fe);

            // Accumulate OOF probabilities (will divide by seeds.size() at end)
            for (size_t i = 0; i < n_val; ++i) {
                size_t orig_idx = fold.val_indices[i];
                for (int c = 0; c < 3; ++c) {
                    res.oof_probs[orig_idx][c] += train_res.val_probs[i][c] / static_cast<double>(seeds.size());
                }
            }
        }
    }

    // Compute mean and std over the 15 folds
    double acc_sum = 0.0, f1_sum = 0.0, loss_sum = 0.0;
    for (const auto& fe : res.folds) {
        acc_sum += fe.accuracy;
        f1_sum += fe.macro_f1;
        loss_sum += fe.log_loss;
    }
    res.acc_mean = acc_sum / total_folds;
    res.macro_f1_mean = f1_sum / total_folds;
    res.log_loss_mean = loss_sum / total_folds;

    double acc_var = 0.0, f1_var = 0.0, loss_var = 0.0;
    for (const auto& fe : res.folds) {
        acc_var += (fe.accuracy - res.acc_mean) * (fe.accuracy - res.acc_mean);
        f1_var += (fe.macro_f1 - res.macro_f1_mean) * (fe.macro_f1 - res.macro_f1_mean);
        loss_var += (fe.log_loss - res.log_loss_mean) * (fe.log_loss - res.log_loss_mean);
    }
    res.acc_std = std::sqrt(acc_var / total_folds);
    res.macro_f1_std = std::sqrt(f1_var / total_folds);
    res.log_loss_std = std::sqrt(loss_var / total_folds);

    // Compute ensemble predictions from averaged OOF probabilities
    size_t correct_oof = 0;
    for (size_t i = 0; i < n; ++i) {
        int bc = 0;
        if (res.oof_probs[i][1] > res.oof_probs[i][bc]) bc = 1;
        if (res.oof_probs[i][2] > res.oof_probs[i][bc]) bc = 2;
        res.oof_preds[i] = bc;
        if (bc == y_all[i]) correct_oof++;
    }

    computeWilsonInterval(correct_oof, n, res.wilson_acc_low, res.wilson_acc_high);

    std::vector<double> unweighted = {1.0, 1.0, 1.0};
    res.pooled_eval = evaluatePredictions(y_all, res.oof_preds, res.oof_probs, unweighted);

    return res;
}

// Paired Statistical Comparison on identical 15 folds
struct PairedTestResult {
    double mean_loss_diff; // baseline_loss - candidate_loss (positive means candidate is better)
    double se_loss_diff;
    double delta_accuracy; // candidate_acc - baseline_acc
    bool loss_improved;    // mean_loss_diff > se_loss_diff
    bool acc_not_dropped;  // delta_accuracy >= -1e-5
    bool accepted;
};

inline PairedTestResult comparePaired(const RepeatedCVResult& baseline, const RepeatedCVResult& candidate) {
    PairedTestResult r;
    size_t M = baseline.folds.size();
    std::vector<double> diffs(M);

    double sum_diff = 0.0;
    for (size_t i = 0; i < M; ++i) {
        // Positive means candidate loss is smaller (improvement)
        diffs[i] = baseline.folds[i].log_loss - candidate.folds[i].log_loss;
        sum_diff += diffs[i];
    }
    r.mean_loss_diff = sum_diff / static_cast<double>(M);

    double var_diff = 0.0;
    for (size_t i = 0; i < M; ++i) {
        double dev = diffs[i] - r.mean_loss_diff;
        var_diff += dev * dev;
    }
    if (M > 1) {
        r.se_loss_diff = std::sqrt((var_diff / static_cast<double>(M - 1)) / static_cast<double>(M));
    } else {
        r.se_loss_diff = 0.0;
    }

    r.delta_accuracy = candidate.acc_mean - baseline.acc_mean;
    r.loss_improved = (r.mean_loss_diff > r.se_loss_diff);
    r.acc_not_dropped = (r.delta_accuracy >= -1e-5);
    r.accepted = r.loss_improved && r.acc_not_dropped;

    return r;
}

inline RepeatedCVResult runRepeatedCVExtended(
    const std::vector<GapRecord>& dataset,
    const ExtendedConfig& ext_cfg,
    const ExperimentConfig& cfg,
    const std::vector<unsigned int>& seeds = {42, 43, 44})
{
    RepeatedCVResult res;
    res.config = cfg;
    size_t n = dataset.size();
    res.oof_probs.assign(n, std::vector<double>(3, 0.0));
    res.oof_preds.assign(n, -1);

    size_t k = 5;
    size_t total_folds = seeds.size() * k;
    res.folds.reserve(total_folds);

    std::vector<int> y_all(n);
    for (size_t i = 0; i < n; ++i) y_all[i] = dataset[i].label;

    size_t C = ext_cfg.getCandidateFeatureCount();
    size_t dim = ext_cfg.getTotalDimension();

    size_t fold_counter = 0;
    for (unsigned int seed : seeds) {
        auto folds = createStratifiedFolds(dataset, k, seed);

        for (size_t f = 0; f < k; ++f) {
            const auto& fold = folds[f];
            size_t n_train = fold.train_indices.size();
            size_t n_val = fold.val_indices.size();

            // Fold-derived resources are fitted only on the training partition.
            NameGenderLexicon lexicon;
            ContextPathLexicon path_lexicon;
            FoldIdfLexicon idf_lexicon;
            if (ext_cfg.use_gender || ext_cfg.use_path_lexicon || ext_cfg.use_tfidf_overlap ||
                ext_cfg.use_local_tfidf_overlap) {
                std::vector<GapRecord> train_records;
                train_records.reserve(n_train);
                for (size_t idx : fold.train_indices) train_records.push_back(dataset[idx]);
                if (ext_cfg.use_gender) lexicon.fit(train_records);
                if (ext_cfg.use_path_lexicon) path_lexicon.fit(train_records);
                if (ext_cfg.use_tfidf_overlap || ext_cfg.use_local_tfidf_overlap) idf_lexicon.fit(train_records);
            }

            std::vector<std::vector<double>> X_train;
            std::vector<int> y_train;
            X_train.reserve(cfg.use_swap_aug ? 2 * n_train : n_train);
            y_train.reserve(cfg.use_swap_aug ? 2 * n_train : n_train);

            for (size_t i = 0; i < n_train; ++i) {
                size_t idx = fold.train_indices[i];
                auto f_vec = extractFeaturesConfigured(dataset[idx], ext_cfg,
                    ext_cfg.use_gender ? &lexicon : nullptr,
                    ext_cfg.use_path_lexicon ? &path_lexicon : nullptr,
                    (ext_cfg.use_tfidf_overlap || ext_cfg.use_local_tfidf_overlap) ? &idf_lexicon : nullptr);
                X_train.push_back(f_vec);
                y_train.push_back(dataset[idx].label);

                if (cfg.use_swap_aug) {
                    auto f_swap = swapExtendedCandidatesVector(f_vec, C);
                    X_train.push_back(f_swap);
                    int y_orig = dataset[idx].label;
                    int y_swapped = (y_orig == 0 ? 1 : (y_orig == 1 ? 0 : 2));
                    y_train.push_back(y_swapped);
                }
            }

            std::vector<std::vector<double>> X_val(n_val);
            std::vector<int> y_val(n_val);
            for (size_t i = 0; i < n_val; ++i) {
                size_t idx = fold.val_indices[i];
                X_val[i] = extractFeaturesConfigured(dataset[idx], ext_cfg,
                    ext_cfg.use_gender ? &lexicon : nullptr,
                    ext_cfg.use_path_lexicon ? &path_lexicon : nullptr,
                    (ext_cfg.use_tfidf_overlap || ext_cfg.use_local_tfidf_overlap) ? &idf_lexicon : nullptr);
                y_val[i] = dataset[idx].label;
            }

            // Apply feature transform
            applyTransformTrain(X_train, X_val, cfg.transform);

            // Fit standardizer ONLY on training fold
            FeatureStandardizer scaler;
            scaler.fit(X_train);
            scaler.transformInPlace(X_train);
            scaler.transformInPlace(X_val);

            // Train model
            ExperimentConfig fold_cfg = cfg;
            fold_cfg.feature_dim = dim;
            fold_cfg.candidate_feature_count = C;
            fold_cfg.use_early_stopping = false;
            const size_t member_count = std::max<size_t>(1, cfg.ensemble_members);
            std::vector<std::vector<double>> fold_val_probs(n_val, std::vector<double>(3, 0.0));
            for (size_t member = 0; member < member_count; ++member) {
                const unsigned int fold_run_seed = seed * 1000 + static_cast<unsigned int>(f) +
                    static_cast<unsigned int>(member * 100000);
                std::vector<std::vector<double>>* member_x = &X_train;
                std::vector<int>* member_y = &y_train;
                std::vector<std::vector<double>> sampled_x;
                std::vector<int> sampled_y;
                const double sample_fraction = std::max(0.05, std::min(1.0, cfg.ensemble_sample_fraction));
                if (sample_fraction < 1.0 || cfg.ensemble_bootstrap) {
                    const size_t augmentation_factor = cfg.use_swap_aug ? 2 : 1;
                    const size_t group_count = X_train.size() / augmentation_factor;
                    std::vector<size_t> groups(group_count);
                    std::iota(groups.begin(), groups.end(), 0);
                    std::mt19937 sample_rng(fold_run_seed + 0x9e3779b9U);
                    const size_t selected = std::max<size_t>(1, static_cast<size_t>(
                        std::round(sample_fraction * static_cast<double>(group_count))));
                    if (cfg.ensemble_bootstrap) {
                        std::uniform_int_distribution<size_t> draw(0, group_count - 1);
                        for (size_t g = 0; g < selected; ++g) groups[g] = draw(sample_rng);
                    } else {
                        std::shuffle(groups.begin(), groups.end(), sample_rng);
                    }
                    sampled_x.reserve(selected * augmentation_factor);
                    sampled_y.reserve(selected * augmentation_factor);
                    for (size_t g = 0; g < selected; ++g) {
                        const size_t start = groups[g] * augmentation_factor;
                        for (size_t offset = 0; offset < augmentation_factor; ++offset) {
                            sampled_x.push_back(X_train[start + offset]);
                            sampled_y.push_back(y_train[start + offset]);
                        }
                    }
                    member_x = &sampled_x;
                    member_y = &sampled_y;
                }
                const auto train_res = trainModelExtended(*member_x, *member_y, X_val, y_val,
                                                          fold_cfg, fold_run_seed);
                for (size_t i = 0; i < n_val; ++i) {
                    for (int c = 0; c < 3; ++c) {
                        const double p = std::max(1e-12, train_res.val_probs[i][c]);
                        fold_val_probs[i][c] += (cfg.ensemble_geometric_mean ? std::log(p) : p) /
                                                static_cast<double>(member_count);
                    }
                }
            }

            if (cfg.ensemble_geometric_mean) {
                for (size_t i = 0; i < n_val; ++i) {
                    const double max_logp = *std::max_element(fold_val_probs[i].begin(), fold_val_probs[i].end());
                    double sum_exp = 0.0;
                    for (int c = 0; c < 3; ++c) {
                        fold_val_probs[i][c] = std::exp(fold_val_probs[i][c] - max_logp);
                        sum_exp += fold_val_probs[i][c];
                    }
                    for (int c = 0; c < 3; ++c) fold_val_probs[i][c] /= sum_exp;
                }
            }

            std::vector<int> fold_val_preds(n_val, 0);
            for (size_t i = 0; i < n_val; ++i) {
                if (fold_val_probs[i][1] > fold_val_probs[i][fold_val_preds[i]]) fold_val_preds[i] = 1;
                if (fold_val_probs[i][2] > fold_val_probs[i][fold_val_preds[i]]) fold_val_preds[i] = 2;
            }
            const std::vector<double> unweighted = {1.0, 1.0, 1.0};
            const auto fold_eval = evaluatePredictions(y_val, fold_val_preds, fold_val_probs, unweighted);

            FoldEvaluation fe;
            fe.fold_idx = fold_counter++;
            fe.seed = seed;
            fe.accuracy = fold_eval.accuracy;
            fe.macro_f1 = fold_eval.macro_f1;
            fe.weighted_f1 = fold_eval.weighted_f1;
            fe.log_loss = fold_eval.weighted_loss;
            fe.eval_res = fold_eval;
            fe.fold_labels = y_val;
            fe.fold_probabilities = fold_val_probs;
            res.folds.push_back(fe);

            for (size_t i = 0; i < n_val; ++i) {
                size_t orig_idx = fold.val_indices[i];
                for (int c = 0; c < 3; ++c) {
                    res.oof_probs[orig_idx][c] += fold_val_probs[i][c] / static_cast<double>(seeds.size());
                }
            }
        }
    }

    // Compute mean and std over the 15 folds
    double acc_sum = 0.0, f1_sum = 0.0, loss_sum = 0.0;
    for (const auto& fe : res.folds) {
        acc_sum += fe.accuracy;
        f1_sum += fe.macro_f1;
        loss_sum += fe.log_loss;
    }
    res.acc_mean = acc_sum / total_folds;
    res.macro_f1_mean = f1_sum / total_folds;
    res.log_loss_mean = loss_sum / total_folds;

    double acc_var = 0.0, f1_var = 0.0, loss_var = 0.0;
    for (const auto& fe : res.folds) {
        acc_var += (fe.accuracy - res.acc_mean) * (fe.accuracy - res.acc_mean);
        f1_var += (fe.macro_f1 - res.macro_f1_mean) * (fe.macro_f1 - res.macro_f1_mean);
        loss_var += (fe.log_loss - res.log_loss_mean) * (fe.log_loss - res.log_loss_mean);
    }
    res.acc_std = std::sqrt(acc_var / total_folds);
    res.macro_f1_std = std::sqrt(f1_var / total_folds);
    res.log_loss_std = std::sqrt(loss_var / total_folds);

    size_t correct_oof = 0;
    for (size_t i = 0; i < n; ++i) {
        int bc = 0;
        if (res.oof_probs[i][1] > res.oof_probs[i][bc]) bc = 1;
        if (res.oof_probs[i][2] > res.oof_probs[i][bc]) bc = 2;
        res.oof_preds[i] = bc;
        if (bc == y_all[i]) correct_oof++;
    }

    computeWilsonInterval(correct_oof, n, res.wilson_acc_low, res.wilson_acc_high);

    std::vector<double> unweighted = {1.0, 1.0, 1.0};
    res.pooled_eval = evaluatePredictions(y_all, res.oof_preds, res.oof_probs, unweighted);

    return res;
}

} // namespace msar

#endif // MSAR_EXPERIMENT_ENGINE_HPP
