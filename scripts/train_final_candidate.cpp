#include "data_loader.hpp"
#include "extended_features.hpp"
#include "mlp.hpp"
#include "standardizer.hpp"
#include "training.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    const auto dev = msar::loadGapDataset("data/raw/gap-development.tsv");
    const auto val = msar::loadGapDataset("data/raw/gap-validation.tsv");
    std::vector<msar::GapRecord> data = dev;
    data.insert(data.end(), val.begin(), val.end());

    msar::ExtendedConfig features;
    features.use_url = true;
    features.use_syntax = true;
    features.use_v2x_matcher = true;
    msar::ExtendedConfig local_features = features;
    local_features.use_local_tfidf_overlap = true;
    const size_t base_dim = features.getTotalDimension();
    const size_t base_candidate_dim = features.getCandidateFeatureCount();
    const size_t local_dim = local_features.getTotalDimension();
    const size_t local_candidate_dim = local_features.getCandidateFeatureCount();

    msar::FoldIdfLexicon idf_lexicon;
    idf_lexicon.fit(data);
    idf_lexicon.save("models/final_candidate_local_tfidf_idf.txt");

    std::vector<std::vector<double>> X_base, X_local;
    std::vector<int> y;
    X_base.reserve(data.size() * 2);
    X_local.reserve(data.size() * 2);
    y.reserve(data.size() * 2);
    for (const auto& rec : data) {
        auto base_row = msar::extractFeaturesConfigured(rec, features);
        auto local_row = msar::extractFeaturesConfigured(rec, local_features, nullptr, nullptr, &idf_lexicon);
        X_base.push_back(base_row);
        X_local.push_back(local_row);
        y.push_back(rec.label);
        X_base.push_back(msar::swapExtendedCandidatesVector(base_row, base_candidate_dim));
        X_local.push_back(msar::swapExtendedCandidatesVector(local_row, local_candidate_dim));
        y.push_back(rec.label == 0 ? 1 : (rec.label == 1 ? 0 : 2));
    }

    msar::FeatureStandardizer base_scaler, local_scaler;
    base_scaler.fit(X_base);
    base_scaler.transformInPlace(X_base);
    base_scaler.save("models/final_candidate_scaler.txt");
    local_scaler.fit(X_local);
    local_scaler.transformInPlace(X_local);
    local_scaler.save("models/final_candidate_local_tfidf_scaler.txt");

    constexpr size_t kMembersPerSchedule = 10;
    constexpr size_t kHidden = 32;
    constexpr size_t kBatch = 32;
    constexpr size_t kEpochs = 20;
    constexpr size_t kSwaEpochs = 5;
    constexpr double kLearningRate = 0.0012;
    constexpr double kAlpha = 0.24;
    constexpr double kL2 = 0.0001;
    const auto class_weights = msar::computeClassWeights(y, kAlpha);

    auto trainVariant = [&](std::vector<std::vector<double>>& X, size_t dim,
                            const std::string& artifact_prefix, unsigned int seed_offset) {
      std::vector<size_t> indices(X.size());
      std::iota(indices.begin(), indices.end(), 0);
      for (int schedule = 0; schedule < 2; ++schedule) {
        const bool cosine = schedule == 1;
        const std::string schedule_name = cosine ? "cosine" : "constant";
        for (size_t member = 0; member < kMembersPerSchedule; ++member) {
            const unsigned int seed = seed_offset + 4242U + static_cast<unsigned int>(member * 100003U);
            msar::MultilayerPerceptron model(dim, kHidden, 3, seed);
            std::mt19937 shuffle_rng(seed + 101U);
            std::vector<double> swa_W1(model.W1.size(), 0.0);
            std::vector<double> swa_b1(model.b1.size(), 0.0);
            std::vector<double> swa_W2(model.W2.size(), 0.0);
            std::vector<double> swa_b2(model.b2.size(), 0.0);
            size_t swa_count = 0;
            for (size_t epoch = 0; epoch < kEpochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), shuffle_rng);
                const double progress = static_cast<double>(epoch) / static_cast<double>(kEpochs);
                const double lr_min = 0.05 * kLearningRate;
                const double current_lr = cosine
                    ? lr_min + 0.5 * (kLearningRate - lr_min) *
                        (1.0 + std::cos(3.141592653589793 * progress))
                    : kLearningRate;
                for (size_t start = 0; start < indices.size(); start += kBatch) {
                    const size_t end = std::min(start + kBatch, indices.size());
                    std::vector<std::vector<double>> batch_x(end - start);
                    std::vector<int> batch_y(end - start);
                    for (size_t i = start; i < end; ++i) {
                        batch_x[i - start] = X[indices[i]];
                        batch_y[i - start] = y[indices[i]];
                    }
                    model.trainBatch(batch_x, batch_y, class_weights, current_lr, kL2);
                }
                if (epoch >= kEpochs - kSwaEpochs) {
                    ++swa_count;
                    for (size_t i = 0; i < model.W1.size(); ++i) swa_W1[i] += model.W1[i];
                    for (size_t i = 0; i < model.b1.size(); ++i) swa_b1[i] += model.b1[i];
                    for (size_t i = 0; i < model.W2.size(); ++i) swa_W2[i] += model.W2[i];
                    for (size_t i = 0; i < model.b2.size(); ++i) swa_b2[i] += model.b2[i];
                }
            }
            for (size_t i = 0; i < model.W1.size(); ++i) model.W1[i] = swa_W1[i] / static_cast<double>(swa_count);
            for (size_t i = 0; i < model.b1.size(); ++i) model.b1[i] = swa_b1[i] / static_cast<double>(swa_count);
            for (size_t i = 0; i < model.W2.size(); ++i) model.W2[i] = swa_W2[i] / static_cast<double>(swa_count);
            for (size_t i = 0; i < model.b2.size(); ++i) model.b2[i] = swa_b2[i] / static_cast<double>(swa_count);
            const std::string path = "models/final_candidate_" + artifact_prefix + schedule_name +
                                     "_member_" + std::to_string(member) + ".bin";
            model.saveCheckpoint(path);
            std::cout << "saved " << path << "\n";
        }
      }
    };
    trainVariant(X_base, base_dim, "", 0U);
    trainVariant(X_local, local_dim, "local_tfidf_", 700000U);

    std::ofstream config("models/final_candidate_config.json");
    if (!config) throw std::runtime_error("Unable to write final candidate config.");
    config << std::fixed << std::setprecision(8)
           << "{\n"
           << "  \"dataset\": \"GAP development + validation\",\n"
           << "  \"test_loaded\": false,\n"
           << "  \"feature_dimension\": " << base_dim << ",\n"
           << "  \"candidate_block_dimension\": " << base_candidate_dim << ",\n"
           << "  \"local_tfidf_feature_dimension\": " << local_dim << ",\n"
           << "  \"local_tfidf_candidate_block_dimension\": " << local_candidate_dim << ",\n"
           << "  \"hidden_dimension\": " << kHidden << ",\n"
           << "  \"ensemble_members_per_schedule\": " << kMembersPerSchedule << ",\n"
           << "  \"probability_blend\": {\"constant_schedule\": 0.5, \"cosine_schedule\": 0.5},\n"
           << "  \"feature_probability_blend\": {\"base\": 1.0, \"local_tfidf\": 0.0},\n"
           << "  \"neither_logit_bias\": -0.10,\n"
           << "  \"him_nearest_candidate_logit_bias\": -0.10,\n"
           << "  \"cosine_min_learning_rate_ratio\": 0.05,\n"
           << "  \"epochs\": " << kEpochs << ",\n"
           << "  \"stochastic_weight_averaging\": true,\n"
           << "  \"swa_epochs\": " << kSwaEpochs << ",\n"
           << "  \"batch_size\": " << kBatch << ",\n"
           << "  \"learning_rate\": " << kLearningRate << ",\n"
           << "  \"class_weight_alpha\": " << kAlpha << ",\n"
           << "  \"l2\": " << kL2 << ",\n"
           << "  \"swap_augmentation\": true,\n"
           << "  \"test_time_candidate_swap\": true\n"
           << "}\n";
    std::cout << "trained " << (4 * kMembersPerSchedule) << " models on " << data.size()
              << " records (with swap augmentation); test data was not loaded\n";
    return 0;
}
