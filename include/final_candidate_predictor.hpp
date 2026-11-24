#ifndef MSAR_FINAL_CANDIDATE_PREDICTOR_HPP
#define MSAR_FINAL_CANDIDATE_PREDICTOR_HPP

#include "data_loader.hpp"
#include "extended_features.hpp"
#include "mlp.hpp"
#include "standardizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace msar {

// Inference wrapper for scripts/train_final_candidate.cpp artifacts.
// Probability order is [Candidate A, Candidate B, Neither].
class FinalCandidatePredictor {
public:
    explicit FinalCandidatePredictor(const std::string& models_dir = "models",
                                     size_t members_per_schedule = 10,
                                     double neither_logit_bias = -0.10,
                                     double him_nearest_candidate_logit_bias = -0.10,
                                     double local_tfidf_weight = 0.0)
        : members_per_schedule_(members_per_schedule),
          neither_logit_bias_(neither_logit_bias),
          him_nearest_candidate_logit_bias_(him_nearest_candidate_logit_bias),
          local_tfidf_weight_(local_tfidf_weight) {
        if (local_tfidf_weight_ < 0.0 || local_tfidf_weight_ > 1.0) {
            throw std::runtime_error("Local TF-IDF blend weight must be between 0 and 1.");
        }
        base_features_.use_url = true;
        base_features_.use_syntax = true;
        base_features_.use_v2x_matcher = true;
        local_features_ = base_features_;
        local_features_.use_local_tfidf_overlap = true;
        base_candidate_dim_ = base_features_.getCandidateFeatureCount();
        local_candidate_dim_ = local_features_.getCandidateFeatureCount();
        base_feature_dim_ = base_features_.getTotalDimension();
        local_feature_dim_ = local_features_.getTotalDimension();
        base_scaler_.load(models_dir + "/final_candidate_scaler.txt");
        if (local_tfidf_weight_ > 0.0) {
            local_scaler_.load(models_dir + "/final_candidate_local_tfidf_scaler.txt");
            idf_lexicon_.load(models_dir + "/final_candidate_local_tfidf_idf.txt");
        }
        if (base_scaler_.feature_dim != base_feature_dim_ ||
            (local_tfidf_weight_ > 0.0 && local_scaler_.feature_dim != local_feature_dim_) ||
            members_per_schedule_ == 0) {
            throw std::runtime_error("Final candidate scaler or ensemble dimensions do not match.");
        }
        loadSchedule(models_dir, "constant", base_feature_dim_, base_constant_models_);
        loadSchedule(models_dir, "cosine", base_feature_dim_, base_cosine_models_);
        if (local_tfidf_weight_ > 0.0) {
            loadSchedule(models_dir, "local_tfidf_constant", local_feature_dim_, local_constant_models_);
            loadSchedule(models_dir, "local_tfidf_cosine", local_feature_dim_, local_cosine_models_);
        }
    }

    std::vector<double> predictProbabilities(const GapRecord& record) const {
        const auto base = predictVariant(record, base_features_, base_scaler_, base_candidate_dim_,
                                         base_feature_dim_, base_constant_models_, base_cosine_models_, nullptr);
        std::vector<double> probs = base;
        if (local_tfidf_weight_ > 0.0) {
            const auto local = predictVariant(record, local_features_, local_scaler_, local_candidate_dim_,
                                              local_feature_dim_, local_constant_models_, local_cosine_models_,
                                              &idf_lexicon_);
            for (int c = 0; c < 3; ++c)
                probs[c] = (1.0 - local_tfidf_weight_) * base[c] + local_tfidf_weight_ * local[c];
        }
        applyNeitherBias(probs, neither_logit_bias_);
        if (toLowerString(record.pronoun) == "him" && him_nearest_candidate_logit_bias_ != 0.0) {
            const long long distance_a = std::llabs(record.pronoun_offset - record.a_offset);
            const long long distance_b = std::llabs(record.pronoun_offset - record.b_offset);
            if (distance_a < distance_b) applyClassBias(probs, 0, him_nearest_candidate_logit_bias_);
            else if (distance_b < distance_a) applyClassBias(probs, 1, him_nearest_candidate_logit_bias_);
        }
        return probs;
    }

    int predict(const GapRecord& record) const {
        const auto probs = predictProbabilities(record);
        int best = 0;
        if (probs[1] > probs[best]) best = 1;
        if (probs[2] > probs[best]) best = 2;
        return best;
    }

private:
    ExtendedConfig base_features_;
    ExtendedConfig local_features_;
    FeatureStandardizer base_scaler_;
    FeatureStandardizer local_scaler_;
    FoldIdfLexicon idf_lexicon_;
    std::vector<MultilayerPerceptron> base_constant_models_;
    std::vector<MultilayerPerceptron> base_cosine_models_;
    std::vector<MultilayerPerceptron> local_constant_models_;
    std::vector<MultilayerPerceptron> local_cosine_models_;
    size_t members_per_schedule_;
    size_t base_candidate_dim_ = 0;
    size_t local_candidate_dim_ = 0;
    size_t base_feature_dim_ = 0;
    size_t local_feature_dim_ = 0;
    double neither_logit_bias_;
    double him_nearest_candidate_logit_bias_;
    double local_tfidf_weight_;

    void loadSchedule(const std::string& models_dir,
                      const std::string& schedule,
                      size_t feature_dim,
                      std::vector<MultilayerPerceptron>& models) {
        models.reserve(members_per_schedule_);
        for (size_t i = 0; i < members_per_schedule_; ++i) {
            models.emplace_back(feature_dim, 32, 3, static_cast<unsigned int>(i + 1));
            const std::string path = models_dir + "/final_candidate_" + schedule +
                                     "_member_" + std::to_string(i) + ".bin";
            models.back().loadCheckpoint(path);
            if (models.back().input_dim != feature_dim || models.back().hidden_dim != 32 ||
                models.back().output_dim != 3) {
                throw std::runtime_error("Final candidate checkpoint has incompatible architecture: " + path);
            }
        }
    }

    std::vector<double> predictVariant(const GapRecord& record, const ExtendedConfig& features,
                                       const FeatureStandardizer& scaler, size_t candidate_dim,
                                       size_t feature_dim,
                                       const std::vector<MultilayerPerceptron>& constant_models,
                                       const std::vector<MultilayerPerceptron>& cosine_models,
                                       const FoldIdfLexicon* idf) const {
        const auto raw = extractFeaturesConfigured(record, features, nullptr, nullptr, idf);
        if (raw.size() != feature_dim) throw std::runtime_error("Unexpected final candidate feature dimension.");
        const auto swapped_raw = swapExtendedCandidatesVector(raw, candidate_dim);
        const auto normal = scaler.transformSingle(raw);
        const auto swapped = scaler.transformSingle(swapped_raw);
        const auto constant = predictSchedule(constant_models, normal, swapped);
        const auto cosine = predictSchedule(cosine_models, normal, swapped);
        std::vector<double> probs(3);
        for (int c = 0; c < 3; ++c) probs[c] = 0.5 * (constant[c] + cosine[c]);
        return probs;
    }

    static std::vector<double> predictSchedule(const std::vector<MultilayerPerceptron>& models,
                                               const std::vector<double>& normal,
                                               const std::vector<double>& swapped) {
        std::vector<double> result(3, 0.0);
        for (const auto& model : models) {
            std::vector<double> h, p_normal, p_swapped;
            model.forward(normal, h, p_normal);
            model.forward(swapped, h, p_swapped);
            result[0] += 0.5 * (p_normal[0] + p_swapped[1]);
            result[1] += 0.5 * (p_normal[1] + p_swapped[0]);
            result[2] += 0.5 * (p_normal[2] + p_swapped[2]);
        }
        for (double& p : result) p /= static_cast<double>(models.size());
        return result;
    }

    static void applyNeitherBias(std::vector<double>& probs, double bias) {
        applyClassBias(probs, 2, bias);
    }

    static void applyClassBias(std::vector<double>& probs, int target_class, double bias) {
        double logits[3];
        double max_logit = -1e300;
        for (int c = 0; c < 3; ++c) {
            logits[c] = std::log(std::max(1e-12, probs[c]));
            if (c == target_class) logits[c] += bias;
            max_logit = std::max(max_logit, logits[c]);
        }
        double total = 0.0;
        for (int c = 0; c < 3; ++c) {
            probs[c] = std::exp(logits[c] - max_logit);
            total += probs[c];
        }
        for (double& p : probs) p /= total;
    }
};

} // namespace msar

#endif // MSAR_FINAL_CANDIDATE_PREDICTOR_HPP
