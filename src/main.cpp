#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v1.hpp"
#include "features_v2.hpp"
#include "features_v3.hpp"
#include "standardizer.hpp"
#include "mlp.hpp"
#include "evaluation.hpp"
#include "training.hpp"
#include "correlation.hpp"
#include "cross_validation.hpp"
#include "mention_ranker.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MAKE_DIR(p) _mkdir(p)
#else
#define MAKE_DIR(p) mkdir(p, 0755)
#endif

void ensureDir(const std::string& path) {
    MAKE_DIR(path.c_str());
}

// ===================================================================
//  Stage 0: Baseline 5-Fold Stratified Cross-Validation
// ===================================================================
int runStage0() {
    std::cout << "===================================================================\n";
    std::cout << " Stage 0: 5-Fold Stratified Cross-Validation & Diagnostic Error Analysis\n";
    std::cout << "===================================================================\n\n";

    ensureDir("results");
    ensureDir("models");

    std::string dev_path = "data/raw/gap-development.tsv";
    std::string val_path = "data/raw/gap-validation.tsv";

    std::cout << "[Step 1] Loading GAP development and validation partitions...\n";
    auto dev_records = msar::loadGapDataset(dev_path);
    auto val_records = msar::loadGapDataset(val_path);

    std::vector<msar::GapRecord> combined;
    combined.insert(combined.end(), dev_records.begin(), dev_records.end());
    combined.insert(combined.end(), val_records.begin(), val_records.end());

    auto stats = msar::computePartitionStats(combined);
    std::cout << "  Combined Dataset: " << stats.total_samples << " samples (A=" << stats.count_a
              << ", B=" << stats.count_b << ", Neither=" << stats.count_neither << ")\n";

    std::cout << "\n[Step 2] Extracting 36-feature V2 representations for all samples...\n";
    std::vector<std::vector<double>> raw_features;
    raw_features.reserve(combined.size());
    for (const auto& r : combined) {
        raw_features.push_back(msar::extractFeaturesV2(r));
    }

    std::cout << "\n[Step 3] Running 5-Fold Stratified Cross-Validation (Seed = 42)...\n";
    msar::TrainingConfig base_cfg;
    base_cfg.experiment_name = "stage0_cv_baseline";
    base_cfg.feature_dim = 36;
    base_cfg.hidden_dim = 32;
    base_cfg.output_dim = 3;
    base_cfg.learning_rate = 0.001;
    base_cfg.class_weight_alpha = 0.50;
    base_cfg.batch_size = 32;
    base_cfg.max_epochs = 500;
    base_cfg.patience = 50;
    base_cfg.seed = 42;

    auto cv = msar::runCrossValidation(combined, raw_features, base_cfg, 5, 42);

    std::cout << "\n===================================================================\n";
    std::cout << " STAGE 0 CROSS-VALIDATION RESULTS (N = 2,454, 5 FOLDS)\n";
    std::cout << "===================================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    for (size_t f = 0; f < cv.fold_results.size(); ++f) {
        std::cout << "  Fold " << (f + 1) << ": Acc = " << (cv.fold_results[f].accuracy * 100.0) 
                  << "% | Macro-F1 = " << std::setprecision(4) << cv.fold_results[f].macro_f1
                  << " | Weighted-F1 = " << cv.fold_results[f].weighted_f1 << "\n";
    }
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Mean Accuracy:    " << std::setprecision(2) << (cv.acc_mean * 100.0) << " +/- " << (cv.acc_std * 100.0) << "%\n";
    std::cout << "  Mean Macro-F1:    " << std::setprecision(4) << cv.macro_f1_mean << " +/- " << cv.macro_f1_std << "\n";
    std::cout << "  Mean Weighted-F1: " << cv.weighted_f1_mean << " +/- " << cv.weighted_f1_std << "\n";
    std::cout << "  Mean Loss:        " << cv.loss_mean << " +/- " << cv.loss_std << "\n";
    std::cout << "-------------------------------------------------------------------\n\n";

    std::cout << "Overall Out-Of-Fold (OOF) Metrics:\n";
    std::cout << cv.oof_overall_eval.toString() << "\n";

    std::cout << "\n[Step 4] Generating detailed error analysis in results/error_analysis.md...\n";
    msar::generateErrorAnalysisReport(combined, cv, "results/error_analysis.md");
    std::cout << "  Saved results/error_analysis.md\n";

    std::ofstream cv_csv("results/cv_summary.csv");
    if (cv_csv.is_open()) {
        cv_csv << "Stage,Description,Accuracy_Mean,Accuracy_Std,MacroF1_Mean,MacroF1_Std,WeightedF1_Mean,WeightedF1_Std,Loss_Mean,Loss_Std\n";
        cv_csv << "Stage 0,Baseline V2 36-feat MLP,"
               << std::fixed << std::setprecision(4)
               << cv.acc_mean << "," << cv.acc_std << ","
               << cv.macro_f1_mean << "," << cv.macro_f1_std << ","
               << cv.weighted_f1_mean << "," << cv.weighted_f1_std << ","
               << cv.loss_mean << "," << cv.loss_std << "\n";
        std::cout << "  Saved results/cv_summary.csv\n";
    }

    return 0;
}

// ===================================================================
//  Stage 1: Symmetry & Data Augmentation (A<->B Swap + TTA)
// ===================================================================
int runStage1() {
    std::cout << "===================================================================\n";
    std::cout << " Stage 1: Symmetry & Data Augmentation (A<->B Swap + TTA)\n";
    std::cout << "===================================================================\n\n";

    std::string dev_path = "data/raw/gap-development.tsv";
    std::string val_path = "data/raw/gap-validation.tsv";

    auto dev_records = msar::loadGapDataset(dev_path);
    auto val_records = msar::loadGapDataset(val_path);
    std::vector<msar::GapRecord> combined;
    combined.insert(combined.end(), dev_records.begin(), dev_records.end());
    combined.insert(combined.end(), val_records.begin(), val_records.end());

    std::cout << "[Step 1] Extracting 36-feature V2 representations...\n";
    std::vector<std::vector<double>> raw_features;
    raw_features.reserve(combined.size());
    for (const auto& r : combined) {
        raw_features.push_back(msar::extractFeaturesV2(r));
    }

    std::cout << "\n[Step 2] Running 5-Fold Stratified CV with A<->B Swap Augmentation & TTA...\n";
    msar::TrainingConfig cfg;
    cfg.experiment_name = "stage1_swap_tta";
    cfg.feature_dim = 36;
    cfg.hidden_dim = 32;
    cfg.output_dim = 3;
    cfg.learning_rate = 0.001;
    cfg.class_weight_alpha = 0.50;
    cfg.batch_size = 32;
    cfg.max_epochs = 500;
    cfg.patience = 50;
    cfg.seed = 42;

    auto cv = msar::runCrossValidation(combined, raw_features, cfg, 5, 42, true, true);

    std::cout << "\n===================================================================\n";
    std::cout << " STAGE 1 CROSS-VALIDATION RESULTS (N = 2,454, 5 FOLDS)\n";
    std::cout << "===================================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    for (size_t f = 0; f < cv.fold_results.size(); ++f) {
        std::cout << "  Fold " << (f + 1) << ": Acc = " << (cv.fold_results[f].accuracy * 100.0) 
                  << "% | Macro-F1 = " << std::setprecision(4) << cv.fold_results[f].macro_f1
                  << " | Weighted-F1 = " << cv.fold_results[f].weighted_f1 << "\n";
    }
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Mean Accuracy:    " << std::setprecision(2) << (cv.acc_mean * 100.0) << " +/- " << (cv.acc_std * 100.0) << "%\n";
    std::cout << "  Mean Macro-F1:    " << std::setprecision(4) << cv.macro_f1_mean << " +/- " << cv.macro_f1_std << "\n";
    std::cout << "  Mean Weighted-F1: " << cv.weighted_f1_mean << " +/- " << cv.weighted_f1_std << "\n";
    std::cout << "  Mean Loss:        " << cv.loss_mean << " +/- " << cv.loss_std << "\n";
    std::cout << "-------------------------------------------------------------------\n\n";

    std::cout << "Overall Out-Of-Fold (OOF) Metrics:\n";
    std::cout << cv.oof_overall_eval.toString() << "\n";

    std::ofstream cv_csv("results/cv_summary.csv", std::ios::app);
    if (cv_csv.is_open()) {
        cv_csv << "Stage 1,A<->B Swap Augmentation + TTA (V2 36-feat),"
               << std::fixed << std::setprecision(4)
               << cv.acc_mean << "," << cv.acc_std << ","
               << cv.macro_f1_mean << "," << cv.macro_f1_std << ","
               << cv.weighted_f1_mean << "," << cv.weighted_f1_std << ","
               << cv.loss_mean << "," << cv.loss_std << "\n";
    }

    return 0;
}

// ===================================================================
//  Stage 2: Coreference Signal Features (V3 80-Features + Swap & TTA)
// ===================================================================
int runStage2() {
    std::cout << "===================================================================\n";
    std::cout << " Stage 2: Coreference Signal Features (V3 80-Features + Swap Aug & TTA)\n";
    std::cout << "===================================================================\n\n";

    std::string dev_path = "data/raw/gap-development.tsv";
    std::string val_path = "data/raw/gap-validation.tsv";

    auto dev_records = msar::loadGapDataset(dev_path);
    auto val_records = msar::loadGapDataset(val_path);
    std::vector<msar::GapRecord> combined;
    combined.insert(combined.end(), dev_records.begin(), dev_records.end());
    combined.insert(combined.end(), val_records.begin(), val_records.end());

    std::cout << "[Step 1] Loaded " << combined.size() << " samples.\n";
    std::cout << "[Step 2] Running 5-Fold Stratified CV with 80-dim V3 Features...\n";

    msar::TrainingConfig cfg;
    cfg.experiment_name = "stage2_features_v3";
    cfg.feature_dim = 80;
    cfg.hidden_dim = 64;
    cfg.output_dim = 3;
    cfg.learning_rate = 0.001;
    cfg.class_weight_alpha = 0.50;
    cfg.l2_lambda = 0.0001;
    cfg.batch_size = 32;
    cfg.max_epochs = 500;
    cfg.patience = 50;
    cfg.seed = 42;

    auto cv = msar::runCrossValidationV3(combined, cfg, 5, 42, true, true);

    std::cout << "\n===================================================================\n";
    std::cout << " STAGE 2 CROSS-VALIDATION RESULTS (N = 2,454, 5 FOLDS)\n";
    std::cout << "===================================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    for (size_t f = 0; f < cv.fold_results.size(); ++f) {
        std::cout << "  Fold " << (f + 1) << ": Acc = " << (cv.fold_results[f].accuracy * 100.0) 
                  << "% | Macro-F1 = " << std::setprecision(4) << cv.fold_results[f].macro_f1
                  << " | Weighted-F1 = " << cv.fold_results[f].weighted_f1 << "\n";
    }
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "  Mean Accuracy:    " << std::setprecision(2) << (cv.acc_mean * 100.0) << " +/- " << (cv.acc_std * 100.0) << "%\n";
    std::cout << "  Mean Macro-F1:    " << std::setprecision(4) << cv.macro_f1_mean << " +/- " << cv.macro_f1_std << "\n";
    std::cout << "  Mean Weighted-F1: " << cv.weighted_f1_mean << " +/- " << cv.weighted_f1_std << "\n";
    std::cout << "  Mean Loss:        " << cv.loss_mean << " +/- " << cv.loss_std << "\n";
    std::cout << "-------------------------------------------------------------------\n\n";

    std::cout << "Overall Out-Of-Fold (OOF) Metrics:\n";
    std::cout << cv.oof_overall_eval.toString() << "\n";

    std::ofstream cv_csv("results/cv_summary.csv", std::ios::app);
    if (cv_csv.is_open()) {
        cv_csv << "Stage 2,V3 Clean Features (80-feat) + Page Subject + Syntactic Heuristics,"
               << std::fixed << std::setprecision(4)
               << cv.acc_mean << "," << cv.acc_std << ","
               << cv.macro_f1_mean << "," << cv.macro_f1_std << ","
               << cv.weighted_f1_mean << "," << cv.weighted_f1_std << ","
               << cv.loss_mean << "," << cv.loss_std << "\n";
    }

    return 0;
}

// ===================================================================
//  Stage 3: Model Capacity & Optimisation Grid Search
// ===================================================================
int runStage3() {
    std::cout << "===================================================================\n";
    std::cout << " Stage 3: Model Capacity & Optimisation Grid Search (V3 80-Features)\n";
    std::cout << "===================================================================\n\n";

    ensureDir("results");
    ensureDir("results/stage3");
    ensureDir("models");

    std::string dev_path = "data/raw/gap-development.tsv";
    std::string val_path = "data/raw/gap-validation.tsv";

    auto dev_records = msar::loadGapDataset(dev_path);
    auto val_records = msar::loadGapDataset(val_path);
    std::vector<msar::GapRecord> combined;
    combined.insert(combined.end(), dev_records.begin(), dev_records.end());
    combined.insert(combined.end(), val_records.begin(), val_records.end());

    auto stats = msar::computePartitionStats(combined);
    std::cout << "[Step 1] Loaded " << stats.total_samples << " samples (A=" << stats.count_a
              << ", B=" << stats.count_b << ", Neither=" << stats.count_neither << ")\n\n";

    struct GridConfig {
        std::string name;
        size_t hidden_dim;
        double lr;
        double l2_lambda;
        double alpha;
    };

    std::vector<GridConfig> grid = {
        {"s3_h64_lr001_l2_0001",    64,  0.001,  0.0001, 0.50},
        {"s3_h64_lr001_l2_0005",    64,  0.001,  0.0005, 0.50},
        {"s3_h64_lr0005_l2_0001",   64,  0.0005, 0.0001, 0.50},
        {"s3_h128_lr001_l2_0001",   128, 0.001,  0.0001, 0.50},
        {"s3_h128_lr0005_l2_0001",  128, 0.0005, 0.0001, 0.50},
    };

    msar::CVMetrics best_cv;
    std::string best_name;
    double best_macro_f1 = -1.0;
    GridConfig best_grid_cfg = grid[0];

    std::ofstream grid_csv("results/stage3/grid_search_results.csv");
    grid_csv << "Config,Hidden,LR,L2,Alpha,Accuracy_Mean,Accuracy_Std,MacroF1_Mean,MacroF1_Std,WeightedF1_Mean,WeightedF1_Std,Loss_Mean,Loss_Std\n";

    for (size_t gi = 0; gi < grid.size(); ++gi) {
        const auto& gc = grid[gi];
        std::cout << "-------------------------------------------------------------------\n";
        std::cout << "[Config " << (gi + 1) << "/" << grid.size() << "] " << gc.name 
                  << " (Hidden=" << gc.hidden_dim << ", LR=" << gc.lr 
                  << ", L2=" << gc.l2_lambda << ", Alpha=" << gc.alpha << ")\n";
        std::cout << "-------------------------------------------------------------------\n";

        msar::TrainingConfig cfg;
        cfg.experiment_name = gc.name;
        cfg.feature_dim = 80;
        cfg.hidden_dim = gc.hidden_dim;
        cfg.output_dim = 3;
        cfg.learning_rate = gc.lr;
        cfg.class_weight_alpha = gc.alpha;
        cfg.l2_lambda = gc.l2_lambda;
        cfg.batch_size = 32;
        cfg.max_epochs = 500;
        cfg.patience = 50;
        cfg.seed = 42;

        auto cv = msar::runCrossValidationV3(combined, cfg, 5, 42, true, true);

        std::cout << "  Mean Accuracy:    " << std::fixed << std::setprecision(2) << (cv.acc_mean * 100.0) << " +/- " << (cv.acc_std * 100.0) << "%\n";
        std::cout << "  Mean Macro-F1:    " << std::setprecision(4) << cv.macro_f1_mean << " +/- " << cv.macro_f1_std << "\n";
        std::cout << "  Mean Weighted-F1: " << cv.weighted_f1_mean << " +/- " << cv.weighted_f1_std << "\n";
        std::cout << "  Mean Loss:        " << cv.loss_mean << " +/- " << cv.loss_std << "\n\n";

        grid_csv << gc.name << "," << gc.hidden_dim << "," << gc.lr << "," << gc.l2_lambda << "," << gc.alpha << ","
                 << std::fixed << std::setprecision(4)
                 << cv.acc_mean << "," << cv.acc_std << ","
                 << cv.macro_f1_mean << "," << cv.macro_f1_std << ","
                 << cv.weighted_f1_mean << "," << cv.weighted_f1_std << ","
                 << cv.loss_mean << "," << cv.loss_std << "\n";
        grid_csv.flush();

        if (cv.macro_f1_mean > best_macro_f1) {
            best_macro_f1 = cv.macro_f1_mean;
            best_cv = cv;
            best_name = gc.name;
            best_grid_cfg = gc;
        }
    }

    std::cout << "===================================================================\n";
    std::cout << " STAGE 3 BEST CONFIGURATION: " << best_name << "\n";
    std::cout << "===================================================================\n";
    std::cout << "  Hidden: " << best_grid_cfg.hidden_dim << " | LR: " << best_grid_cfg.lr
              << " | L2: " << best_grid_cfg.l2_lambda << " | Alpha: " << best_grid_cfg.alpha << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Mean Accuracy:    " << (best_cv.acc_mean * 100.0) << " +/- " << (best_cv.acc_std * 100.0) << "%\n";
    std::cout << std::setprecision(4);
    std::cout << "  Mean Macro-F1:    " << best_cv.macro_f1_mean << " +/- " << best_cv.macro_f1_std << "\n";
    std::cout << "  Mean Weighted-F1: " << best_cv.weighted_f1_mean << " +/- " << best_cv.weighted_f1_std << "\n";
    std::cout << "  Mean Loss:        " << best_cv.loss_mean << " +/- " << best_cv.loss_std << "\n\n";
    std::cout << "Overall OOF Metrics:\n";
    std::cout << best_cv.oof_overall_eval.toString() << "\n";

    std::ofstream best_cfg_file("results/stage3/best_config.txt");
    best_cfg_file << best_name << "\n"
                  << best_grid_cfg.hidden_dim << "\n"
                  << best_grid_cfg.lr << "\n"
                  << best_grid_cfg.l2_lambda << "\n"
                  << best_grid_cfg.alpha << "\n";

    std::ofstream cv_csv("results/cv_summary.csv", std::ios::app);
    if (cv_csv.is_open()) {
        cv_csv << "Stage 3," << best_name << " (best of grid search),"
               << std::fixed << std::setprecision(4)
               << best_cv.acc_mean << "," << best_cv.acc_std << ","
               << best_cv.macro_f1_mean << "," << best_cv.macro_f1_std << ","
               << best_cv.weighted_f1_mean << "," << best_cv.weighted_f1_std << ","
               << best_cv.loss_mean << "," << best_cv.loss_std << "\n";
    }

    return 0;
}

// ===================================================================
//  Stage 4: Decision-Rule Tuning (OOF Probability Calibration)
// ===================================================================
int runStage4() {
    std::cout << "===================================================================\n";
    std::cout << " Stage 4: Decision-Rule Tuning (Per-Class Probability Scaling)\n";
    std::cout << "===================================================================\n\n";

    ensureDir("results");
    ensureDir("results/stage4");
    ensureDir("models");

    size_t best_hidden = 64;
    double best_lr = 0.001;
    double best_l2 = 0.0001;
    double best_alpha = 0.50;
    {
        std::ifstream cfg_file("results/stage3/best_config.txt");
        if (cfg_file.is_open()) {
            std::string name;
            std::getline(cfg_file, name);
            cfg_file >> best_hidden >> best_lr >> best_l2 >> best_alpha;
            std::cout << "[Info] Loaded Stage 3 best config: " << name 
                      << " (H=" << best_hidden << ", LR=" << best_lr 
                      << ", L2=" << best_l2 << ", Alpha=" << best_alpha << ")\n\n";
        } else {
            std::cout << "[Info] Using default optimal config (H=64, LR=0.001, L2=0.0001, Alpha=0.50)\n\n";
        }
    }

    std::string dev_path = "data/raw/gap-development.tsv";
    std::string val_path = "data/raw/gap-validation.tsv";
    auto dev_records = msar::loadGapDataset(dev_path);
    auto val_records = msar::loadGapDataset(val_path);
    std::vector<msar::GapRecord> combined;
    combined.insert(combined.end(), dev_records.begin(), dev_records.end());
    combined.insert(combined.end(), val_records.begin(), val_records.end());

    std::cout << "[Step 1] Running 5-fold CV to obtain calibrated OOF probability distributions...\n";

    msar::TrainingConfig cfg;
    cfg.experiment_name = "stage4_oof";
    cfg.feature_dim = 80;
    cfg.hidden_dim = best_hidden;
    cfg.output_dim = 3;
    cfg.learning_rate = best_lr;
    cfg.class_weight_alpha = best_alpha;
    cfg.l2_lambda = best_l2;
    cfg.batch_size = 32;
    cfg.max_epochs = 500;
    cfg.patience = 50;
    cfg.seed = 42;

    auto cv = msar::runCrossValidationV3(combined, cfg, 5, 42, true, true);

    std::cout << "  OOF Accuracy (before calibration): " << std::fixed << std::setprecision(2) 
              << (cv.oof_overall_eval.accuracy * 100.0) << "%\n";
    std::cout << "  OOF Macro-F1 (before calibration): " << std::setprecision(4) 
              << cv.oof_overall_eval.macro_f1 << "\n\n";

    std::cout << "[Step 2] Grid search over per-class scaling factors...\n";

    std::vector<int> y_all(combined.size());
    for (size_t i = 0; i < combined.size(); ++i) y_all[i] = combined[i].label;
    auto class_weights = msar::computeClassWeights(y_all, best_alpha);

    double best_scale_a = 1.0, best_scale_b = 1.0, best_scale_n = 1.0;
    double best_cal_macro_f1 = cv.oof_overall_eval.macro_f1;

    std::vector<double> scale_values = {0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30};
    size_t total_combos = scale_values.size() * scale_values.size() * scale_values.size();

    for (double sa : scale_values) {
        for (double sb : scale_values) {
            for (double sn : scale_values) {
                std::vector<int> scaled_preds(combined.size());
                std::vector<std::vector<double>> scaled_probs(combined.size(), std::vector<double>(3));

                for (size_t i = 0; i < combined.size(); ++i) {
                    scaled_probs[i][0] = cv.oof_probs[i][0] * sa;
                    scaled_probs[i][1] = cv.oof_probs[i][1] * sb;
                    scaled_probs[i][2] = cv.oof_probs[i][2] * sn;

                    double sum = scaled_probs[i][0] + scaled_probs[i][1] + scaled_probs[i][2];
                    if (sum > 0) {
                        scaled_probs[i][0] /= sum;
                        scaled_probs[i][1] /= sum;
                        scaled_probs[i][2] /= sum;
                    }

                    int best_c = 0;
                    if (scaled_probs[i][1] > scaled_probs[i][best_c]) best_c = 1;
                    if (scaled_probs[i][2] > scaled_probs[i][best_c]) best_c = 2;
                    scaled_preds[i] = best_c;
                }

                auto eval = msar::evaluatePredictions(y_all, scaled_preds, scaled_probs, class_weights);
                if (eval.macro_f1 > best_cal_macro_f1) {
                    best_cal_macro_f1 = eval.macro_f1;
                    best_scale_a = sa;
                    best_scale_b = sb;
                    best_scale_n = sn;
                }
            }
        }
    }

    std::cout << "  Tested " << total_combos << " scaling factor combinations.\n\n";

    std::vector<int> final_preds(combined.size());
    std::vector<std::vector<double>> final_probs(combined.size(), std::vector<double>(3));

    for (size_t i = 0; i < combined.size(); ++i) {
        final_probs[i][0] = cv.oof_probs[i][0] * best_scale_a;
        final_probs[i][1] = cv.oof_probs[i][1] * best_scale_b;
        final_probs[i][2] = cv.oof_probs[i][2] * best_scale_n;
        double sum = final_probs[i][0] + final_probs[i][1] + final_probs[i][2];
        if (sum > 0) {
            final_probs[i][0] /= sum;
            final_probs[i][1] /= sum;
            final_probs[i][2] /= sum;
        }
        int best_c = 0;
        if (final_probs[i][1] > final_probs[i][best_c]) best_c = 1;
        if (final_probs[i][2] > final_probs[i][best_c]) best_c = 2;
        final_preds[i] = best_c;
    }

    auto cal_eval = msar::evaluatePredictions(y_all, final_preds, final_probs, class_weights);

    std::cout << "===================================================================\n";
    std::cout << " STAGE 4 RESULTS: Decision-Rule Calibration\n";
    std::cout << "===================================================================\n";
    std::cout << "  Best Scale Factors: A=" << best_scale_a << ", B=" << best_scale_b << ", N=" << best_scale_n << "\n";
    std::cout << "  Before calibration: Accuracy=" << std::fixed << std::setprecision(2) << (cv.oof_overall_eval.accuracy * 100.0) << "%, Macro-F1=" << std::setprecision(4) << cv.oof_overall_eval.macro_f1 << "\n";
    std::cout << "  After  calibration: Accuracy=" << std::setprecision(2) << (cal_eval.accuracy * 100.0) << "%, Macro-F1=" << std::setprecision(4) << cal_eval.macro_f1 << "\n";
    std::cout << "  Delta Accuracy: " << std::setprecision(2) << ((cal_eval.accuracy - cv.oof_overall_eval.accuracy) * 100.0) << " pp\n";
    std::cout << "  Delta Macro-F1: " << std::setprecision(4) << (cal_eval.macro_f1 - cv.oof_overall_eval.macro_f1) << "\n\n";

    std::cout << "Calibrated OOF Metrics:\n";
    std::cout << cal_eval.toString() << "\n";

    std::ofstream sf("results/stage4/scaling_factors.json");
    sf << "{\n"
       << "  \"scale_A\": " << best_scale_a << ",\n"
       << "  \"scale_B\": " << best_scale_b << ",\n"
       << "  \"scale_Neither\": " << best_scale_n << ",\n"
       << "  \"oof_macro_f1_before\": " << cv.oof_overall_eval.macro_f1 << ",\n"
       << "  \"oof_macro_f1_after\": " << cal_eval.macro_f1 << ",\n"
       << "  \"oof_accuracy_before\": " << cv.oof_overall_eval.accuracy << ",\n"
       << "  \"oof_accuracy_after\": " << cal_eval.accuracy << "\n"
       << "}\n";

    std::ofstream s4_cfg("results/stage4/best_config.txt");
    s4_cfg << best_hidden << "\n" << best_lr << "\n" << best_l2 << "\n" << best_alpha << "\n"
           << best_scale_a << "\n" << best_scale_b << "\n" << best_scale_n << "\n";

    std::ofstream cv_csv("results/cv_summary.csv", std::ios::app);
    if (cv_csv.is_open()) {
        cv_csv << "Stage 4,Decision-Rule Calibration (A=" << best_scale_a << " B=" << best_scale_b << " N=" << best_scale_n << "),"
               << std::fixed << std::setprecision(4)
               << cal_eval.accuracy << "," << 0.0 << ","
               << cal_eval.macro_f1 << "," << 0.0 << ","
               << cal_eval.weighted_f1 << "," << 0.0 << ","
               << cal_eval.weighted_loss << "," << 0.0 << "\n";
    }

    return 0;
}

// ===================================================================
//  Stage 5: Final Test Evaluation (GAP Test Set N = 2,000)
// ===================================================================
int runStage5() {
    std::cout << "===================================================================\n";
    std::cout << " Stage 5: Final Test Evaluation (Train on Dev+Val, Eval on Test)\n";
    std::cout << "===================================================================\n\n";

    ensureDir("results");
    ensureDir("results/stage5");
    ensureDir("models");

    size_t best_hidden = 64;
    double best_lr = 0.001;
    double best_l2 = 0.0001;
    double best_alpha = 0.50;
    double scale_a = 1.0, scale_b = 1.0, scale_n = 1.0;
    {
        std::ifstream cfg_file("results/stage4/best_config.txt");
        if (cfg_file.is_open()) {
            cfg_file >> best_hidden >> best_lr >> best_l2 >> best_alpha;
            cfg_file >> scale_a >> scale_b >> scale_n;
            std::cout << "[Info] Loaded config: H=" << best_hidden << ", LR=" << best_lr 
                      << ", L2=" << best_l2 << ", Alpha=" << best_alpha 
                      << ", Scale A=" << scale_a << ", B=" << scale_b << ", N=" << scale_n << "\n\n";
        } else {
            std::cout << "[Info] Using standard optimal config.\n\n";
        }
    }

    std::string dev_path = "data/raw/gap-development.tsv";
    std::string val_path = "data/raw/gap-validation.tsv";
    std::string test_path = "data/raw/gap-test.tsv";

    auto dev_records = msar::loadGapDataset(dev_path);
    auto val_records = msar::loadGapDataset(val_path);
    auto test_records = msar::loadGapDataset(test_path);

    std::vector<msar::GapRecord> train_set;
    train_set.insert(train_set.end(), dev_records.begin(), dev_records.end());
    train_set.insert(train_set.end(), val_records.begin(), val_records.end());

    auto train_stats = msar::computePartitionStats(train_set);
    auto test_stats = msar::computePartitionStats(test_records);
    std::cout << "[Step 1] Training set: " << train_stats.total_samples << " samples (A=" << train_stats.count_a
              << ", B=" << train_stats.count_b << ", Neither=" << train_stats.count_neither << ")\n";
    std::cout << "         Test set:     " << test_stats.total_samples << " samples (A=" << test_stats.count_a
              << ", B=" << test_stats.count_b << ", Neither=" << test_stats.count_neither << ")\n\n";

    std::cout << "[Step 2] Extracting 80-dim V3 representations...\n";
    std::vector<std::vector<double>> X_train_raw, X_test_raw;
    std::vector<int> y_train, y_test;

    X_train_raw.reserve(train_set.size());
    y_train.reserve(train_set.size());
    for (const auto& r : train_set) {
        X_train_raw.push_back(msar::extractFeaturesV3(r));
        y_train.push_back(r.label);
    }

    X_test_raw.reserve(test_records.size());
    y_test.reserve(test_records.size());
    for (const auto& r : test_records) {
        X_test_raw.push_back(msar::extractFeaturesV3(r));
        y_test.push_back(r.label);
    }

    std::cout << "[Step 3] Applying A<->B swap augmentation on training data...\n";
    std::vector<std::vector<double>> X_train_aug;
    std::vector<int> y_train_aug;
    X_train_aug.reserve(2 * X_train_raw.size());
    y_train_aug.reserve(2 * X_train_raw.size());

    for (size_t i = 0; i < X_train_raw.size(); ++i) {
        X_train_aug.push_back(X_train_raw[i]);
        y_train_aug.push_back(y_train[i]);

        X_train_aug.push_back(msar::swapCandidatesVector(X_train_raw[i]));
        int y_swapped = (y_train[i] == 0 ? 1 : (y_train[i] == 1 ? 0 : 2));
        y_train_aug.push_back(y_swapped);
    }
    std::cout << "  Augmented training set: " << X_train_aug.size() << " samples\n\n";

    std::cout << "[Step 4] Fitting feature standardizer on training data...\n";
    msar::FeatureStandardizer scaler;
    scaler.fit(X_train_aug);
    scaler.transformInPlace(X_train_aug);

    std::vector<std::vector<double>> X_test_std(X_test_raw.size());
    for (size_t i = 0; i < X_test_raw.size(); ++i) {
        X_test_std[i] = scaler.transformSingle(X_test_raw[i]);
    }
    scaler.save("models/standardizer_v3_final.txt");

    // Partition training data into train (dev 2000) and early-stopping validation (val 454)
    size_t n_val = val_records.size(); // 454
    size_t n_dev = dev_records.size(); // 2000

    std::vector<std::vector<double>> X_es_train, X_es_val;
    std::vector<int> y_es_train, y_es_val;

    for (size_t i = 0; i < n_dev; ++i) {
        X_es_train.push_back(X_train_aug[2 * i]);
        y_es_train.push_back(y_train_aug[2 * i]);
        X_es_train.push_back(X_train_aug[2 * i + 1]);
        y_es_train.push_back(y_train_aug[2 * i + 1]);
    }
    for (size_t i = n_dev; i < train_set.size(); ++i) {
        X_es_val.push_back(X_train_aug[2 * i]);
        y_es_val.push_back(y_train_aug[2 * i]);
    }

    std::cout << "[Step 5] Training final model (ES-Train=" << X_es_train.size() 
              << ", ES-Val=" << X_es_val.size() << ")...\n";

    msar::TrainingConfig train_cfg;
    train_cfg.experiment_name = "stage5_final";
    train_cfg.feature_dim = 80;
    train_cfg.hidden_dim = best_hidden;
    train_cfg.output_dim = 3;
    train_cfg.learning_rate = best_lr;
    train_cfg.class_weight_alpha = best_alpha;
    train_cfg.l2_lambda = best_l2;
    train_cfg.batch_size = 32;
    train_cfg.max_epochs = 500;
    train_cfg.patience = 50;
    train_cfg.seed = 42;
    train_cfg.use_tta = true;
    train_cfg.checkpoint_path = "models/final_model.bin";
    train_cfg.history_csv_path = "results/stage5/training_history.csv";

    auto train_res = msar::trainModel(X_es_train, y_es_train, X_es_val, y_es_val, train_cfg);

    std::cout << "  Best Epoch:      " << train_res.best_epoch << "\n";
    std::cout << "  ES-Val Accuracy: " << std::fixed << std::setprecision(2) << (train_res.best_val_acc * 100.0) << "%\n";
    std::cout << "  ES-Val Macro-F1: " << std::setprecision(4) << train_res.best_val_macro_f1 << "\n\n";

    std::cout << "[Step 6] Evaluating on untouched GAP test set (N = " << test_records.size() << ")...\n\n";

    msar::MultilayerPerceptron model(80, best_hidden, 3);
    model.loadCheckpoint("models/final_model.bin");

    std::vector<std::vector<double>> test_probs_raw;
    auto test_preds_raw = msar::predictModel(model, X_test_std, test_probs_raw, true);
    auto test_class_weights = msar::computeClassWeights(y_test, best_alpha);
    auto test_eval_raw = msar::evaluatePredictions(y_test, test_preds_raw, test_probs_raw, test_class_weights);

    std::vector<std::vector<double>> test_probs_cal = test_probs_raw;
    std::vector<int> test_preds_cal(test_probs_cal.size());
    for (size_t i = 0; i < test_probs_cal.size(); ++i) {
        test_probs_cal[i][0] *= scale_a;
        test_probs_cal[i][1] *= scale_b;
        test_probs_cal[i][2] *= scale_n;
        double sum = test_probs_cal[i][0] + test_probs_cal[i][1] + test_probs_cal[i][2];
        if (sum > 0) {
            test_probs_cal[i][0] /= sum;
            test_probs_cal[i][1] /= sum;
            test_probs_cal[i][2] /= sum;
        }
        int best_c = 0;
        if (test_probs_cal[i][1] > test_probs_cal[i][best_c]) best_c = 1;
        if (test_probs_cal[i][2] > test_probs_cal[i][best_c]) best_c = 2;
        test_preds_cal[i] = best_c;
    }
    auto test_eval_cal = msar::evaluatePredictions(y_test, test_preds_cal, test_probs_cal, test_class_weights);

    std::cout << "===================================================================\n";
    std::cout << " STAGE 5: FINAL TEST SET RESULTS (N = 2,000)\n";
    std::cout << "===================================================================\n\n";
    std::cout << "--- Raw Inference (Without Calibration) ---\n";
    std::cout << test_eval_raw.toString() << "\n";
    std::cout << "--- Calibrated Inference (Scale A=" << scale_a << ", B=" << scale_b << ", N=" << scale_n << ") ---\n";
    std::cout << test_eval_cal.toString() << "\n";

    std::ofstream rf("results/stage5/final_test_report.md");
    rf << "# Final Test Evaluation Report\n\n";
    rf << "## Configuration\n";
    rf << "- **Model**: MLP (80 -> " << best_hidden << " -> 3) with ReLU + Softmax\n";
    rf << "- **Features**: Clean V3 (80-dim: 40 per candidate x 2)\n";
    rf << "- **Training Data**: GAP Development (2000) + GAP Validation (454) = 2454 samples\n";
    rf << "- **Augmentation**: A<->B Swap (2x training data) + TTA at inference\n";
    rf << "- **Optimizer**: Adam (LR=" << best_lr << ", L2=" << best_l2 << ")\n";
    rf << "- **Class Weighting**: Inverse-frequency^" << best_alpha << "\n";
    rf << "- **Calibration Multipliers**: Scale A=" << scale_a << ", B=" << scale_b << ", Neither=" << scale_n << "\n";
    rf << "- **Best Epoch**: " << train_res.best_epoch << "\n\n";

    rf << "## Test Set Performance (N = 2,000)\n\n";
    rf << "| Metric | Raw Inference | Calibrated Inference |\n";
    rf << "|---|:---:|:---:|\n";
    rf << "| **Accuracy** | **" << std::fixed << std::setprecision(2) << (test_eval_raw.accuracy * 100.0) << "%** | **"
       << (test_eval_cal.accuracy * 100.0) << "%** |\n";
    rf << "| **Macro-F1** | " << std::setprecision(4) << test_eval_raw.macro_f1 << " | " << test_eval_cal.macro_f1 << " |\n";
    rf << "| **Weighted-F1** | " << test_eval_raw.weighted_f1 << " | " << test_eval_cal.weighted_f1 << " |\n";
    rf << "| **Weighted Loss** | " << test_eval_raw.weighted_loss << " | " << test_eval_cal.weighted_loss << " |\n\n";

    rf << "### Per-Class Breakdown (Raw Inference)\n";
    rf << "| Class | Precision | Recall | F1-Score | Support |\n";
    rf << "|---|---|---|---|---|\n";
    const char* cnames[3] = {"A (0)", "B (1)", "Neither (2)"};
    for (int c = 0; c < 3; ++c) {
        rf << "| " << cnames[c] << " | " << std::fixed << std::setprecision(4)
           << test_eval_raw.class_metrics[c].precision << " | "
           << test_eval_raw.class_metrics[c].recall << " | "
           << test_eval_raw.class_metrics[c].f1 << " | "
           << test_eval_raw.class_metrics[c].support << " |\n";
    }

    rf << "\n### Confusion Matrix (Raw Inference)\n```\n";
    rf << "                 Predicted\n";
    rf << "                 A        B        Neither\n";
    rf << "Actual A         " << std::setw(8) << test_eval_raw.confusion_matrix[0][0] << " "
                              << std::setw(8) << test_eval_raw.confusion_matrix[0][1] << " "
                              << std::setw(8) << test_eval_raw.confusion_matrix[0][2] << "\n";
    rf << "Actual B         " << std::setw(8) << test_eval_raw.confusion_matrix[1][0] << " "
                              << std::setw(8) << test_eval_raw.confusion_matrix[1][1] << " "
                              << std::setw(8) << test_eval_raw.confusion_matrix[1][2] << "\n";
    rf << "Actual Neither   " << std::setw(8) << test_eval_raw.confusion_matrix[2][0] << " "
                              << std::setw(8) << test_eval_raw.confusion_matrix[2][1] << " "
                              << std::setw(8) << test_eval_raw.confusion_matrix[2][2] << "\n";
    rf << "```\n";
    rf.close();

    std::ofstream cv_csv("results/cv_summary.csv", std::ios::app);
    if (cv_csv.is_open()) {
        cv_csv << "Stage 5 (Test - Raw),Final Test Set Evaluation (Raw),"
               << std::fixed << std::setprecision(4)
               << test_eval_raw.accuracy << ",0,"
               << test_eval_raw.macro_f1 << ",0,"
               << test_eval_raw.weighted_f1 << ",0,"
               << test_eval_raw.weighted_loss << ",0\n";
        cv_csv << "Stage 5 (Test - Calibrated),Final Test Set Evaluation (Calibrated),"
               << test_eval_cal.accuracy << ",0,"
               << test_eval_cal.macro_f1 << ",0,"
               << test_eval_cal.weighted_f1 << ",0,"
               << test_eval_cal.weighted_loss << ",0\n";
    }

    std::cout << "Final test report written to results/stage5/final_test_report.md\n";
    std::cout << "Model checkpoint saved to models/final_model.bin\n";
    std::cout << "Standardizer saved to models/standardizer_v3_final.txt\n";

    return 0;
}

// ===================================================================
//  Main Entry Point
// ===================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: msar_reproduce.exe --stage0|--stage1|--stage2|--stage3|--stage4|--stage5|--all\n";
        return 1;
    }

    std::string flag = argv[1];

    if (flag == "--stage0") return runStage0();
    if (flag == "--stage1") return runStage1();
    if (flag == "--stage2") return runStage2();
    if (flag == "--stage3") return runStage3();
    if (flag == "--stage4") return runStage4();
    if (flag == "--stage5") return runStage5();
    if (flag == "--all") {
        int r;
        r = runStage0(); if (r != 0) return r;
        r = runStage1(); if (r != 0) return r;
        r = runStage2(); if (r != 0) return r;
        r = runStage3(); if (r != 0) return r;
        r = runStage4(); if (r != 0) return r;
        r = runStage5(); if (r != 0) return r;
        return 0;
    }

    std::cerr << "Unknown flag: " << flag << "\n";
    std::cerr << "Use --stage0, --stage1, --stage2, --stage3, --stage4, --stage5, or --all\n";
    return 1;
}
