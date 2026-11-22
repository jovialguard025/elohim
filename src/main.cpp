#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v1.hpp"
#include "features_v2.hpp"
#include "standardizer.hpp"
#include "mlp.hpp"
#include "evaluation.hpp"
#include "training.hpp"
#include "correlation.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
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

void saveExperimentJson(const std::string& filepath, const msar::TrainingResult& res, const std::string& feat_name, const std::string& arch_str) {
    std::ofstream out(filepath);
    if (!out.is_open()) return;

    out << "{\n";
    out << "  \"experiment_name\": \"" << res.config.experiment_name << "\",\n";
    out << "  \"dataset_partition\": \"GAP development (train: 2000), GAP validation (val: 454)\",\n";
    out << "  \"feature_representation\": \"" << feat_name << "\",\n";
    out << "  \"network_architecture\": \"" << arch_str << "\",\n";
    out << "  \"optimizer\": \"Adam\",\n";
    out << "  \"learning_rate\": " << res.config.learning_rate << ",\n";
    out << "  \"batch_size\": " << res.config.batch_size << ",\n";
    out << "  \"max_epochs\": " << res.config.max_epochs << ",\n";
    out << "  \"early_stopping_patience\": " << res.config.patience << ",\n";
    out << "  \"class_weight_exponent\": " << res.config.class_weight_alpha << ",\n";
    out << "  \"l2_lambda\": " << res.config.l2_lambda << ",\n";
    out << "  \"random_seed\": " << res.config.seed << ",\n";
    out << "  \"best_epoch\": " << res.best_epoch << ",\n";
    out << "  \"final_training_loss\": " << (res.history.empty() ? 0.0 : res.history[res.best_epoch > 0 ? res.best_epoch - 1 : 0].train_loss) << ",\n";
    out << "  \"validation_loss\": " << res.best_val_loss << ",\n";
    out << "  \"validation_accuracy\": " << res.best_val_acc << ",\n";
    out << "  \"macro_f1\": " << res.best_val_macro_f1 << ",\n";
    out << "  \"per_class_metrics\": {\n";
    out << "    \"A\": {\"precision\": " << res.final_val_eval.class_metrics[0].precision << ", \"recall\": " << res.final_val_eval.class_metrics[0].recall << ", \"f1\": " << res.final_val_eval.class_metrics[0].f1 << ", \"support\": " << res.final_val_eval.class_metrics[0].support << "},\n";
    out << "    \"B\": {\"precision\": " << res.final_val_eval.class_metrics[1].precision << ", \"recall\": " << res.final_val_eval.class_metrics[1].recall << ", \"f1\": " << res.final_val_eval.class_metrics[1].f1 << ", \"support\": " << res.final_val_eval.class_metrics[1].support << "},\n";
    out << "    \"Neither\": {\"precision\": " << res.final_val_eval.class_metrics[2].precision << ", \"recall\": " << res.final_val_eval.class_metrics[2].recall << ", \"f1\": " << res.final_val_eval.class_metrics[2].f1 << ", \"support\": " << res.final_val_eval.class_metrics[2].support << "}\n";
    out << "  },\n";
    out << "  \"confusion_matrix\": [\n";
    for (int r = 0; r < 3; ++r) {
        out << "    [" << res.final_val_eval.confusion_matrix[r][0] << ", "
                       << res.final_val_eval.confusion_matrix[r][1] << ", "
                       << res.final_val_eval.confusion_matrix[r][2] << "]"
                       << (r < 2 ? ",\n" : "\n");
    }
    out << "  ],\n";
    out << "  \"checkpoint_file\": \"" << res.config.checkpoint_path << "\"\n";
    out << "}\n";
}

int runAllExperiments();

int runInference(int argc, char* argv[]) {
    if (argc < 8) {
        std::cout << "Usage: msar_reproduce.exe --predict \"<text>\" \"<pronoun>\" <pron_offset> \"<cand_A>\" <a_offset> \"<cand_B>\" <b_offset>\n";
        return 1;
    }
    std::string text = argv[2];
    std::string pronoun = argv[3];
    long long pron_offset = std::stoll(argv[4]);
    std::string cand_a = argv[5];
    long long a_offset = std::stoll(argv[6]);
    std::string cand_b = argv[7];
    long long b_offset = std::stoll(argv[8]);

    msar::GapRecord rec;
    rec.id = "inference-sample";
    rec.text = text;
    rec.pronoun = pronoun;
    rec.pronoun_offset = pron_offset;
    rec.a = cand_a;
    rec.a_offset = a_offset;
    rec.b = cand_b;
    rec.b_offset = b_offset;

    // Load standardizer
    msar::FeatureStandardizer scaler;
    try {
        scaler.load("models/standardizer_v2.txt");
    } catch (const std::exception& e) {
        std::cerr << "Error loading standardizer: " << e.what() << "\nPlease run experiments first.\n";
        return 1;
    }

    // Load best model
    msar::MultilayerPerceptron model(36, 32, 3);
    try {
        model.loadCheckpoint("models/best_model.bin");
    } catch (const std::exception& e) {
        std::cerr << "Error loading model checkpoint: " << e.what() << "\nPlease run experiments first.\n";
        return 1;
    }

    auto raw_features = msar::extractFeaturesV2(rec);
    auto std_features = scaler.transformSingle(raw_features);

    std::vector<double> h, probs;
    model.forward(std_features, h, probs);

    int best_c = 0;
    if (probs[1] > probs[best_c]) best_c = 1;
    if (probs[2] > probs[best_c]) best_c = 2;

    const char* class_names[3] = {"Candidate A", "Candidate B", "Neither"};

    std::cout << "\n--- Antecedent Prediction Result ---\n";
    std::cout << "Text: \"" << text << "\"\n";
    std::cout << "Pronoun: \"" << pronoun << "\" at offset " << pron_offset << "\n";
    std::cout << "Candidate A: \"" << cand_a << "\" at offset " << a_offset << "\n";
    std::cout << "Candidate B: \"" << cand_b << "\" at offset " << b_offset << "\n\n";
    std::cout << "Predicted Class: " << class_names[best_c] << " (Class " << best_c << ")\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Class Probabilities:\n";
    std::cout << "  P(Candidate A: " << cand_a << ") = " << (probs[0] * 100.0) << "%\n";
    std::cout << "  P(Candidate B: " << cand_b << ") = " << (probs[1] * 100.0) << "%\n";
    std::cout << "  P(Neither)                     = " << (probs[2] * 100.0) << "%\n";

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--predict") {
        return runInference(argc, argv);
    }
    return runAllExperiments();
}

int runAllExperiments() {
    std::cout << "===================================================================\n";
    std::cout << " MAR / MSAR Antecedent Classifier: Research Reproduction Framework\n";
    std::cout << "===================================================================\n\n";

    std::string data_raw_dir = "data/raw";
    std::string train_path = data_raw_dir + "/gap-development.tsv";
    std::string val_path = data_raw_dir + "/gap-validation.tsv";
    std::string test_path = data_raw_dir + "/gap-test.tsv";

    // Create necessary directories
    ensureDir("data");
    ensureDir("data/raw");
    ensureDir("data/processed");
    ensureDir("models");
    ensureDir("results");
    ensureDir("results/baseline");
    ensureDir("results/experiments");
    ensureDir("results/comparisons");
    ensureDir("docs");

    // Load datasets
    std::cout << "[Step 1] Loading and verifying GAP coreference datasets...\n";
    std::vector<msar::GapRecord> dev_records;
    std::vector<msar::GapRecord> val_records;
    std::vector<msar::GapRecord> test_records;

    try {
        dev_records = msar::loadGapDataset(train_path);
        val_records = msar::loadGapDataset(val_path);
        test_records = msar::loadGapDataset(test_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << "\n";
        return 1;
    }

    auto dev_stats = msar::computePartitionStats(dev_records);
    auto val_stats = msar::computePartitionStats(val_records);
    auto test_stats = msar::computePartitionStats(test_records);

    std::cout << "  Development: " << dev_stats.total_samples << " samples (A=" << dev_stats.count_a 
              << ", B=" << dev_stats.count_b << ", Neither=" << dev_stats.count_neither << ")\n";
    std::cout << "  Validation:  " << val_stats.total_samples << " samples (A=" << val_stats.count_a 
              << ", B=" << val_stats.count_b << ", Neither=" << val_stats.count_neither << ")\n";
    std::cout << "  Test:        " << test_stats.total_samples << " samples (A=" << test_stats.count_a 
              << ", B=" << test_stats.count_b << ", Neither=" << test_stats.count_neither << ")\n";

    // Extract V1 (12 features)
    std::cout << "\n[Step 2] Extracting 12-feature representations (Baseline V1)...\n";
    std::vector<std::vector<double>> X_dev_v1, X_val_v1;
    std::vector<int> y_dev, y_val;

    for (const auto& r : dev_records) {
        X_dev_v1.push_back(msar::extractFeaturesV1(r));
        y_dev.push_back(r.label);
    }
    for (const auto& r : val_records) {
        X_val_v1.push_back(msar::extractFeaturesV1(r));
        y_val.push_back(r.label);
    }

    // Extract V2 (36 features)
    std::cout << "[Step 3] Extracting 36-feature representations (Expanded V2)...\n";
    std::vector<std::vector<double>> X_dev_v2, X_val_v2;

    for (const auto& r : dev_records) {
        X_dev_v2.push_back(msar::extractFeaturesV2(r));
    }
    for (const auto& r : val_records) {
        X_val_v2.push_back(msar::extractFeaturesV2(r));
    }

    // Standardizers (fitted on development / training set only)
    std::cout << "[Step 4] Fitting feature standardizers on development partition only...\n";
    msar::FeatureStandardizer scaler_v1;
    scaler_v1.fit(X_dev_v1);
    scaler_v1.transformInPlace(X_dev_v1);
    scaler_v1.transformInPlace(X_val_v1);
    scaler_v1.save("models/standardizer_v1.txt");

    msar::FeatureStandardizer scaler_v2;
    scaler_v2.fit(X_dev_v2);
    scaler_v2.transformInPlace(X_dev_v2);
    scaler_v2.transformInPlace(X_val_v2);
    scaler_v2.save("models/standardizer_v2.txt");

    // Feature correlation analysis on V2 dev set
    std::cout << "[Step 5] Analyzing feature correlations on development data (Experiment 6 prep)...\n";
    auto corr_analysis = msar::analyzeFeatureCorrelations(X_dev_v2);
    std::cout << "  Pairs with |r| >= 0.90: " << corr_analysis.pairs_ge_90.size() << "\n";
    std::cout << "  Pairs with |r| >= 0.95: " << corr_analysis.pairs_ge_95.size() << "\n";
    std::cout << "  Pairs with |r| >= 0.99: " << corr_analysis.pairs_ge_99.size() << "\n";

    // Save correlation report
    {
        std::ofstream corr_out("results/experiments/correlation_report.txt");
        corr_out << "Feature Correlation Audit (GAP Development Set):\n";
        corr_out << "Total Features: 36\n";
        corr_out << "Pairs with |r| >= 0.90: " << corr_analysis.pairs_ge_90.size() << "\n";
        corr_out << "Pairs with |r| >= 0.95: " << corr_analysis.pairs_ge_95.size() << "\n";
        corr_out << "Pairs with |r| >= 0.99: " << corr_analysis.pairs_ge_99.size() << "\n\n";
        corr_out << "Top Correlated Feature Pairs (|r| >= 0.90):\n";
        corr_out << "---------------------------------------------------------\n";
        for (const auto& cp : corr_analysis.pairs_ge_90) {
            corr_out << "Feature " << std::setw(2) << cp.feat_i << " <-> Feature " 
                     << std::setw(2) << cp.feat_j << " : r = " << std::fixed << std::setprecision(4) << cp.r << "\n";
        }
    }

    // 30-feature reduced dataset for Exp 6
    auto X_dev_v2_reduced = msar::applyFeatureReduction(X_dev_v2, corr_analysis.retained_features);
    auto X_val_v2_reduced = msar::applyFeatureReduction(X_val_v2, corr_analysis.retained_features);

    std::vector<msar::TrainingResult> all_results;

    // Experiment 1: 12-feature baseline
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "Running Experiment 1: Original 12-feature Baseline\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        msar::TrainingConfig cfg;
        cfg.experiment_name = "exp1_baseline_12feat";
        cfg.feature_dim = 12;
        cfg.hidden_dim = 32;
        cfg.output_dim = 3;
        cfg.learning_rate = 0.001;
        cfg.class_weight_alpha = 0.50;
        cfg.checkpoint_path = "models/model_exp1_baseline.bin";
        cfg.history_csv_path = "results/baseline/exp1_history.csv";

        auto res = msar::trainModel(X_dev_v1, y_dev, X_val_v1, y_val, cfg);
        all_results.push_back(res);
        saveExperimentJson("results/baseline/exp1_result.json", res, "12 features (V1)", "12 -> 32 -> 3 MLP");
        std::cout << "  Best Epoch: " << res.best_epoch << " | Val Acc: " << (res.best_val_acc * 100.0) 
                  << "% | Macro-F1: " << res.best_val_macro_f1 << " | Val Loss: " << res.best_val_loss << "\n";
    }

    // Experiment 2: V2, Inverse-frequency class weighting (alpha = 1.0)
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "Running Experiment 2: 36-feature V2, Inverse-Frequency Weighting (alpha=1.0)\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        msar::TrainingConfig cfg;
        cfg.experiment_name = "exp2_v2_inv_freq_alpha100";
        cfg.feature_dim = 36;
        cfg.hidden_dim = 32;
        cfg.output_dim = 3;
        cfg.learning_rate = 0.001;
        cfg.class_weight_alpha = 1.00;
        cfg.checkpoint_path = "models/model_exp2_inv_freq.bin";
        cfg.history_csv_path = "results/experiments/exp2_history.csv";

        auto res = msar::trainModel(X_dev_v2, y_dev, X_val_v2, y_val, cfg);
        all_results.push_back(res);
        saveExperimentJson("results/experiments/exp2_result.json", res, "36 features (V2)", "36 -> 32 -> 3 MLP");
        std::cout << "  Best Epoch: " << res.best_epoch << " | Val Acc: " << (res.best_val_acc * 100.0) 
                  << "% | Macro-F1: " << res.best_val_macro_f1 << " | Val Loss: " << res.best_val_loss << "\n";
    }

    // Experiment 3: Reduced learning rate (lr = 0.0003)
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "Running Experiment 3: Reduced Learning Rate (lr=0.0003, alpha=1.0)\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        msar::TrainingConfig cfg;
        cfg.experiment_name = "exp3_reduced_lr_0003";
        cfg.feature_dim = 36;
        cfg.hidden_dim = 32;
        cfg.output_dim = 3;
        cfg.learning_rate = 0.0003;
        cfg.class_weight_alpha = 1.00;
        cfg.checkpoint_path = "models/model_exp3_reduced_lr.bin";
        cfg.history_csv_path = "results/experiments/exp3_history.csv";

        auto res = msar::trainModel(X_dev_v2, y_dev, X_val_v2, y_val, cfg);
        all_results.push_back(res);
        saveExperimentJson("results/experiments/exp3_result.json", res, "36 features (V2)", "36 -> 32 -> 3 MLP");
        std::cout << "  Best Epoch: " << res.best_epoch << " | Val Acc: " << (res.best_val_acc * 100.0) 
                  << "% | Macro-F1: " << res.best_val_macro_f1 << " | Val Loss: " << res.best_val_loss << "\n";
    }

    // Experiment 4: Square-root class weighting (alpha = 0.50, lr = 0.001) - BEST MODEL
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "Running Experiment 4: Square-Root Class Weighting (alpha=0.50, lr=0.001) [Target Best]\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        msar::TrainingConfig cfg;
        cfg.experiment_name = "exp4_best_sqrt_alpha050";
        cfg.feature_dim = 36;
        cfg.hidden_dim = 32;
        cfg.output_dim = 3;
        cfg.learning_rate = 0.001;
        cfg.class_weight_alpha = 0.50;
        cfg.checkpoint_path = "models/model_exp4_best.bin";
        cfg.history_csv_path = "results/experiments/exp4_history.csv";

        auto res = msar::trainModel(X_dev_v2, y_dev, X_val_v2, y_val, cfg);
        all_results.push_back(res);
        saveExperimentJson("results/experiments/exp4_result.json", res, "36 features (V2)", "36 -> 32 -> 3 MLP");
        std::cout << "  Best Epoch: " << res.best_epoch << " | Val Acc: " << (res.best_val_acc * 100.0) 
                  << "% | Macro-F1: " << res.best_val_macro_f1 << " | Val Loss: " << res.best_val_loss << "\n";
    }

    // Experiment 5: Class-weight exponent 0.65
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "Running Experiment 5: Class-Weight Exponent 0.65\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        msar::TrainingConfig cfg;
        cfg.experiment_name = "exp5_class_weight_alpha065";
        cfg.feature_dim = 36;
        cfg.hidden_dim = 32;
        cfg.output_dim = 3;
        cfg.learning_rate = 0.001;
        cfg.class_weight_alpha = 0.65;
        cfg.checkpoint_path = "models/model_exp5_alpha065.bin";
        cfg.history_csv_path = "results/experiments/exp5_history.csv";

        auto res = msar::trainModel(X_dev_v2, y_dev, X_val_v2, y_val, cfg);
        all_results.push_back(res);
        saveExperimentJson("results/experiments/exp5_result.json", res, "36 features (V2)", "36 -> 32 -> 3 MLP");
        std::cout << "  Best Epoch: " << res.best_epoch << " | Val Acc: " << (res.best_val_acc * 100.0) 
                  << "% | Macro-F1: " << res.best_val_macro_f1 << " | Val Loss: " << res.best_val_loss << "\n";
    }

    // Experiment 6: Feature correlation reduced model (30 features)
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "Running Experiment 6: Feature Correlation Ablation (30 features)\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        msar::TrainingConfig cfg;
        cfg.experiment_name = "exp6_reduced_30feat";
        cfg.feature_dim = 30;
        cfg.hidden_dim = 32;
        cfg.output_dim = 3;
        cfg.learning_rate = 0.001;
        cfg.class_weight_alpha = 0.50;
        cfg.checkpoint_path = "models/model_exp6_reduced.bin";
        cfg.history_csv_path = "results/experiments/exp6_history.csv";

        auto res = msar::trainModel(X_dev_v2_reduced, y_dev, X_val_v2_reduced, y_val, cfg);
        all_results.push_back(res);
        saveExperimentJson("results/experiments/exp6_result.json", res, "30 features (Ablated V2)", "30 -> 32 -> 3 MLP");
        std::cout << "  Best Epoch: " << res.best_epoch << " | Val Acc: " << (res.best_val_acc * 100.0) 
                  << "% | Macro-F1: " << res.best_val_macro_f1 << " | Val Loss: " << res.best_val_loss << "\n";
    }

    // Experiment 7: Larger Hidden Layer (hidden = 64)
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "Running Experiment 7: Larger Hidden Layer (hidden=64)\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        msar::TrainingConfig cfg;
        cfg.experiment_name = "exp7_hidden64";
        cfg.feature_dim = 36;
        cfg.hidden_dim = 64;
        cfg.output_dim = 3;
        cfg.learning_rate = 0.001;
        cfg.class_weight_alpha = 0.50;
        cfg.checkpoint_path = "models/model_exp7_hidden64.bin";
        cfg.history_csv_path = "results/experiments/exp7_history.csv";

        auto res = msar::trainModel(X_dev_v2, y_dev, X_val_v2, y_val, cfg);
        all_results.push_back(res);
        saveExperimentJson("results/experiments/exp7_result.json", res, "36 features (V2)", "36 -> 64 -> 3 MLP");
        std::cout << "  Best Epoch: " << res.best_epoch << " | Val Acc: " << (res.best_val_acc * 100.0) 
                  << "% | Macro-F1: " << res.best_val_macro_f1 << " | Val Loss: " << res.best_val_loss << "\n";
    }

    // Experiment 8: L2 Regularization (lambda = 0.0001)
    std::cout << "\n-------------------------------------------------------------------\n";
    std::cout << "Running Experiment 8: L2 Regularization (lambda=0.0001)\n";
    std::cout << "-------------------------------------------------------------------\n";
    {
        msar::TrainingConfig cfg;
        cfg.experiment_name = "exp8_l2_reg_00001";
        cfg.feature_dim = 36;
        cfg.hidden_dim = 32;
        cfg.output_dim = 3;
        cfg.learning_rate = 0.001;
        cfg.class_weight_alpha = 0.50;
        cfg.l2_lambda = 0.0001;
        cfg.checkpoint_path = "models/model_exp8_l2.bin";
        cfg.history_csv_path = "results/experiments/exp8_history.csv";

        auto res = msar::trainModel(X_dev_v2, y_dev, X_val_v2, y_val, cfg);
        all_results.push_back(res);
        saveExperimentJson("results/experiments/exp8_result.json", res, "36 features (V2)", "36 -> 32 -> 3 MLP (L2=1e-4)");
        std::cout << "  Best Epoch: " << res.best_epoch << " | Val Acc: " << (res.best_val_acc * 100.0) 
                  << "% | Macro-F1: " << res.best_val_macro_f1 << " | Val Loss: " << res.best_val_loss << "\n";
    }

    // Identify and save Best Model overall (by highest macro-F1, tie breaker: lower val loss)
    size_t best_idx = 0;
    for (size_t i = 1; i < all_results.size(); ++i) {
        if (all_results[i].best_val_macro_f1 > all_results[best_idx].best_val_macro_f1 + 1e-6) {
            best_idx = i;
        } else if (std::abs(all_results[i].best_val_macro_f1 - all_results[best_idx].best_val_macro_f1) <= 1e-6 &&
                   all_results[i].best_val_loss < all_results[best_idx].best_val_loss) {
            best_idx = i;
        }
    }

    std::cout << "\n===================================================================\n";
    std::cout << " MODEL SELECTION & COMPARISON SUMMARY\n";
    std::cout << "===================================================================\n";
    std::cout << "Selected Best Experiment: " << all_results[best_idx].config.experiment_name << "\n";
    std::cout << "Validation Accuracy: " << (all_results[best_idx].best_val_acc * 100.0) << "%\n";
    std::cout << "Validation Macro-F1: " << all_results[best_idx].best_val_macro_f1 << "\n";
    std::cout << "Weighted Val Loss:   " << all_results[best_idx].best_val_loss << "\n";
    std::cout << "Best Epoch:          " << all_results[best_idx].best_epoch << "\n\n";

    std::cout << all_results[best_idx].final_val_eval.toString() << "\n";

    // Save final best model checkpoint & standardizer
    {
        std::ifstream src(all_results[best_idx].config.checkpoint_path, std::ios::binary);
        std::ofstream dst("models/best_model.bin", std::ios::binary);
        if (src && dst) dst << src.rdbuf();
    }

    // Historical comparison reference metrics
    struct RefMetric {
        std::string name;
        double ref_acc;
        double ref_f1;
    };
    std::vector<RefMetric> ref_metrics = {
        {"Exp 1 (12-feat Baseline)", 54.85, 0.4900},
        {"Exp 2 (V2, alpha=1.00)",   56.17, 0.5226},
        {"Exp 3 (Reduced LR)",       55.95, 0.5143},
        {"Exp 4 (Sqrt Weighting)",   63.00, 0.5604},
        {"Exp 5 (Alpha=0.65)",       62.56, 0.5606},
        {"Exp 6 (Correlation Red.)", 59.91, 0.5187},
        {"Exp 7 (Hidden=64)",        60.13, 0.5315},
        {"Exp 8 (L2 Reg)",           63.00, 0.5604}
    };

    // Save comparison table CSV
    std::ofstream csv_cmp("results/comparisons/experiment_comparison.csv");
    csv_cmp << "Experiment,Features,Hidden,LR,Alpha,L2,Best_Epoch,Reprod_Acc,Ref_Acc,Reprod_MacroF1,Ref_MacroF1,Reprod_ValLoss\n";
    for (size_t i = 0; i < all_results.size(); ++i) {
        const auto& r = all_results[i];
        csv_cmp << r.config.experiment_name << ","
                << r.config.feature_dim << ","
                << r.config.hidden_dim << ","
                << r.config.learning_rate << ","
                << r.config.class_weight_alpha << ","
                << r.config.l2_lambda << ","
                << r.best_epoch << ","
                << std::fixed << std::setprecision(2) << (r.best_val_acc * 100.0) << "%,"
                << ref_metrics[i].ref_acc << "%,"
                << std::setprecision(4) << r.best_val_macro_f1 << ","
                << ref_metrics[i].ref_f1 << ","
                << r.best_val_loss << "\n";
    }

    // Save comparison Markdown table
    std::ofstream md_cmp("results/comparisons/experiment_comparison.md");
    md_cmp << "# Experiment Comparison: Historical Reference vs. Reproduced Baseline\n\n";
    md_cmp << "| Experiment | Features | Hidden | LR | Alpha | L2 | Best Epoch | Reprod. Acc | Ref. Acc | Reprod. Macro-F1 | Ref. Macro-F1 | Reprod. Loss |\n";
    md_cmp << "|---|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (size_t i = 0; i < all_results.size(); ++i) {
        const auto& r = all_results[i];
        md_cmp << "| " << r.config.experiment_name << " | "
               << r.config.feature_dim << " | "
               << r.config.hidden_dim << " | "
               << r.config.learning_rate << " | "
               << r.config.class_weight_alpha << " | "
               << r.config.l2_lambda << " | "
               << r.best_epoch << " | "
               << std::fixed << std::setprecision(2) << (r.best_val_acc * 100.0) << "% | "
               << ref_metrics[i].ref_acc << "% | "
               << std::setprecision(4) << r.best_val_macro_f1 << " | "
               << ref_metrics[i].ref_f1 << " | "
               << r.best_val_loss << " |\n";
    }

    std::cout << "\nAll experiments completed successfully!\n";
    std::cout << "Reports saved to results/comparisons/experiment_comparison.csv and .md\n";
    return 0;
}
