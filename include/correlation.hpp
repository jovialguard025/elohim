#ifndef MSAR_CORRELATION_HPP
#define MSAR_CORRELATION_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace msar {

struct CorrelationPair {
    size_t feat_i;
    size_t feat_j;
    double r;
};

struct CorrelationAnalysis {
    std::vector<std::vector<double>> matrix;
    std::vector<CorrelationPair> pairs_ge_90;
    std::vector<CorrelationPair> pairs_ge_95;
    std::vector<CorrelationPair> pairs_ge_99;
    std::vector<size_t> dropped_features; // 6 redundant features dropped
    std::vector<size_t> retained_features; // 30 retained features
};

inline CorrelationAnalysis analyzeFeatureCorrelations(const std::vector<std::vector<double>>& X) {
    CorrelationAnalysis res;
    if (X.empty()) return res;
    size_t n = X.size();
    size_t d = X[0].size();

    // Compute mean and std dev for each feature
    std::vector<double> mean(d, 0.0);
    std::vector<double> std_dev(d, 0.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            mean[j] += X[i][j];
        }
    }
    for (size_t j = 0; j < d; ++j) {
        mean[j] /= static_cast<double>(n);
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double diff = X[i][j] - mean[j];
            std_dev[j] += diff * diff;
        }
    }
    for (size_t j = 0; j < d; ++j) {
        std_dev[j] = std::sqrt(std_dev[j] / static_cast<double>(n));
        if (std_dev[j] < 1e-12) std_dev[j] = 1.0;
    }

    // Correlation matrix
    res.matrix.assign(d, std::vector<double>(d, 0.0));
    for (size_t j1 = 0; j1 < d; ++j1) {
        res.matrix[j1][j1] = 1.0;
        for (size_t j2 = j1 + 1; j2 < d; ++j2) {
            double cov = 0.0;
            for (size_t i = 0; i < n; ++i) {
                cov += (X[i][j1] - mean[j1]) * (X[i][j2] - mean[j2]);
            }
            cov /= static_cast<double>(n);
            double r = cov / (std_dev[j1] * std_dev[j2]);
            res.matrix[j1][j2] = r;
            res.matrix[j2][j1] = r;

            double abs_r = std::abs(r);
            if (abs_r >= 0.90) {
                CorrelationPair cp = {j1, j2, r};
                res.pairs_ge_90.push_back(cp);
                if (abs_r >= 0.95) {
                    res.pairs_ge_95.push_back(cp);
                    if (abs_r >= 0.99) {
                        res.pairs_ge_99.push_back(cp);
                    }
                }
            }
        }
    }

    // Sort pairs by descending absolute correlation
    auto cmp = [](const CorrelationPair& a, const CorrelationPair& b) {
        return std::abs(a.r) > std::abs(b.r);
    };
    std::sort(res.pairs_ge_90.begin(), res.pairs_ge_90.end(), cmp);
    std::sort(res.pairs_ge_95.begin(), res.pairs_ge_95.end(), cmp);
    std::sort(res.pairs_ge_99.begin(), res.pairs_ge_99.end(), cmp);

    // Identify 6 features to drop for the 30-feature ablation:
    // Drop 3 from Candidate A (0..17) and 3 corresponding from Candidate B (18..35)
    // For Candidate A:
    // Pair (0, 6): char_dist vs token_dist (r ~ 0.995) -> drop 6 (token_dist)
    // Pair (1, 9): sent_dist vs norm_sent_gap (r ~ 0.99) -> drop 9 (norm_sent_gap)
    // Pair (13, 15): nearest_mention_dist vs last_prior_dist (r > 0.90) -> drop 15 (last_prior_dist)
    // Corresponding Candidate B features: 6 + 18 = 24, 9 + 18 = 27, 15 + 18 = 33.
    res.dropped_features = {6, 9, 15, 24, 27, 33};

    for (size_t j = 0; j < d; ++j) {
        if (std::find(res.dropped_features.begin(), res.dropped_features.end(), j) == res.dropped_features.end()) {
            res.retained_features.push_back(j);
        }
    }

    return res;
}

inline std::vector<std::vector<double>> applyFeatureReduction(
    const std::vector<std::vector<double>>& X,
    const std::vector<size_t>& retained_indices)
{
    std::vector<std::vector<double>> X_reduced(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        X_reduced[i].reserve(retained_indices.size());
        for (size_t idx : retained_indices) {
            X_reduced[i].push_back(X[i][idx]);
        }
    }
    return X_reduced;
}

} // namespace msar

#endif // MSAR_CORRELATION_HPP
