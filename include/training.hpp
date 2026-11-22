#ifndef MSAR_TRAINING_HPP
#define MSAR_TRAINING_HPP

#include "mlp.hpp"
#include "evaluation.hpp"
#include "standardizer.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace msar {

struct TrainingConfig {
    std::string experiment_name = "exp";
    size_t feature_dim = 36;
    size_t hidden_dim = 32;
    size_t output_dim = 3;
    double learning_rate = 0.001;
    double class_weight_alpha = 0.50;
    double l2_lambda = 0.0;
    size_t batch_size = 32;
    size_t max_epochs = 500;
    size_t patience = 50;
    unsigned int seed = 42;
    std::string checkpoint_path = "";
    std::string history_csv_path = "";
};

struct EpochRecord {
    size_t epoch;
    double train_loss;
    double val_loss;
    double val_acc;
    double val_macro_f1;
};

struct TrainingResult {
    TrainingConfig config;
    size_t best_epoch;
    double best_val_acc;
    double best_val_macro_f1;
    double best_val_loss;
    EvaluationResult final_val_eval;
    std::vector<EpochRecord> history;
    std::vector<double> class_weights;
    std::vector<int> val_preds;
    std::vector<std::vector<double>> val_probs;
};

inline std::vector<double> computeClassWeights(const std::vector<int>& y_train, double alpha) {
    std::vector<double> counts(3, 0.0);
    for (int y : y_train) {
        if (y >= 0 && y < 3) counts[y] += 1.0;
    }

    std::vector<double> raw_w(3);
    double sum_raw = 0.0;
    for (int c = 0; c < 3; ++c) {
        if (counts[c] > 0.0) {
            raw_w[c] = std::pow(1.0 / counts[c], alpha);
        } else {
            raw_w[c] = 1.0;
        }
        sum_raw += raw_w[c];
    }

    // Normalize so mean weight is 1.0 (sum is 3.0)
    std::vector<double> norm_w(3);
    for (int c = 0; c < 3; ++c) {
        norm_w[c] = raw_w[c] / (sum_raw / 3.0);
    }
    return norm_w;
}

inline double computeEpochLoss(
    const MultilayerPerceptron& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    const std::vector<double>& class_weights)
{
    double total_loss = 0.0;
    double weight_sum = 0.0;
    std::vector<double> h, p;

    for (size_t i = 0; i < X.size(); ++i) {
        model.forward(X[i], h, p);
        int target = y[i];
        double prob = p[target];
        if (prob < 1e-12) prob = 1e-12;
        if (prob > 1.0) prob = 1.0;
        double w = class_weights[target];
        total_loss += w * (-std::log(prob));
        weight_sum += w;
    }
    return weight_sum > 0.0 ? (total_loss / weight_sum) : 0.0;
}

inline TrainingResult trainModel(
    const std::vector<std::vector<double>>& X_train,
    const std::vector<int>& y_train,
    const std::vector<std::vector<double>>& X_val,
    const std::vector<int>& y_val,
    const TrainingConfig& config)
{
    TrainingResult result;
    result.config = config;
    result.class_weights = computeClassWeights(y_train, config.class_weight_alpha);

    MultilayerPerceptron model(config.feature_dim, config.hidden_dim, config.output_dim, config.seed);

    size_t n_train = X_train.size();
    std::vector<size_t> indices(n_train);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 shuffle_gen(config.seed);

    double best_macro_f1 = -1.0;
    double best_acc = 0.0;
    double best_loss = 1e30;
    size_t best_epoch = 0;
    size_t epochs_no_improve = 0;

    std::string temp_ckpt = config.checkpoint_path.empty() ? "temp_best.bin" : config.checkpoint_path;

    for (size_t epoch = 1; epoch <= config.max_epochs; ++epoch) {
        // Shuffle training samples
        std::shuffle(indices.begin(), indices.end(), shuffle_gen);

        // Mini-batch updates
        for (size_t b_start = 0; b_start < n_train; b_start += config.batch_size) {
            size_t b_end = std::min(n_train, b_start + config.batch_size);
            std::vector<std::vector<double>> X_batch;
            std::vector<int> y_batch;
            X_batch.reserve(b_end - b_start);
            y_batch.reserve(b_end - b_start);

            for (size_t idx = b_start; idx < b_end; ++idx) {
                size_t orig_idx = indices[idx];
                X_batch.push_back(X_train[orig_idx]);
                y_batch.push_back(y_train[orig_idx]);
            }

            model.trainBatch(
                X_batch, y_batch, result.class_weights,
                config.learning_rate, config.l2_lambda
            );
        }

        // Evaluate on training loss and validation set
        double train_loss = computeEpochLoss(model, X_train, y_train, result.class_weights);

        std::vector<std::vector<double>> val_probs;
        std::vector<int> val_preds = model.predict(X_val, val_probs);
        EvaluationResult val_eval = evaluatePredictions(y_val, val_preds, val_probs, result.class_weights);

        EpochRecord ep_rec = {epoch, train_loss, val_eval.weighted_loss, val_eval.accuracy, val_eval.macro_f1};
        result.history.push_back(ep_rec);

        // Checkpoint selection:
        // Highest validation macro-F1; tie-breaker: lower weighted validation loss
        bool is_best = false;
        if (val_eval.macro_f1 > best_macro_f1 + 1e-6) {
            is_best = true;
        } else if (std::abs(val_eval.macro_f1 - best_macro_f1) <= 1e-6 && val_eval.weighted_loss < best_loss) {
            is_best = true;
        }

        if (is_best) {
            best_macro_f1 = val_eval.macro_f1;
            best_acc = val_eval.accuracy;
            best_loss = val_eval.weighted_loss;
            best_epoch = epoch;
            epochs_no_improve = 0;
            model.saveCheckpoint(temp_ckpt);
        } else {
            epochs_no_improve++;
            if (epochs_no_improve >= config.patience) {
                break;
            }
        }
    }

    // Restore best checkpoint
    model.loadCheckpoint(temp_ckpt);

    // Final evaluation on validation set using restored best checkpoint
    std::vector<std::vector<double>> best_val_probs;
    std::vector<int> best_val_preds = model.predict(X_val, best_val_probs);
    result.val_preds = best_val_preds;
    result.val_probs = best_val_probs;
    result.final_val_eval = evaluatePredictions(y_val, best_val_preds, best_val_probs, result.class_weights);
    result.best_epoch = best_epoch;
    result.best_val_acc = result.final_val_eval.accuracy;
    result.best_val_macro_f1 = result.final_val_eval.macro_f1;
    result.best_val_loss = result.final_val_eval.weighted_loss;

    // Save history CSV if path provided
    if (!config.history_csv_path.empty()) {
        std::ofstream h_out(config.history_csv_path);
        if (h_out.is_open()) {
            h_out << "epoch,train_loss,val_loss,val_acc,val_macro_f1\n";
            for (const auto& h : result.history) {
                h_out << h.epoch << "," << h.train_loss << "," << h.val_loss << ","
                      << h.val_acc << "," << h.val_macro_f1 << "\n";
            }
        }
    }

    return result;
}

} // namespace msar

#endif // MSAR_TRAINING_HPP
