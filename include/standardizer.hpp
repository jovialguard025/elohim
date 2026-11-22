#ifndef MSAR_STANDARDIZER_HPP
#define MSAR_STANDARDIZER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace msar {

class FeatureStandardizer {
public:
    std::vector<double> mean;
    std::vector<double> std_dev;
    size_t feature_dim = 0;
    bool is_fitted = false;

    void fit(const std::vector<std::vector<double>>& X) {
        if (X.empty()) {
            throw std::runtime_error("Cannot fit standardizer on empty dataset.");
        }
        feature_dim = X[0].size();
        size_t n = X.size();

        mean.assign(feature_dim, 0.0);
        std_dev.assign(feature_dim, 0.0);

        // Compute mean
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < feature_dim; ++d) {
                mean[d] += X[i][d];
            }
        }
        for (size_t d = 0; d < feature_dim; ++d) {
            mean[d] /= static_cast<double>(n);
        }

        // Compute variance and std dev
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < feature_dim; ++d) {
                double diff = X[i][d] - mean[d];
                std_dev[d] += diff * diff;
            }
        }
        for (size_t d = 0; d < feature_dim; ++d) {
            std_dev[d] = std::sqrt(std_dev[d] / static_cast<double>(n));
            // Safe handling for zero or near-zero std dev
            if (std_dev[d] < 1e-12) {
                std_dev[d] = 1.0;
            }
        }

        is_fitted = true;
    }

    std::vector<double> transformSingle(const std::vector<double>& x) const {
        if (!is_fitted) {
            throw std::runtime_error("Standardizer is not fitted.");
        }
        std::vector<double> result(feature_dim);
        for (size_t d = 0; d < feature_dim; ++d) {
            result[d] = (x[d] - mean[d]) / std_dev[d];
        }
        return result;
    }

    void transformInPlace(std::vector<std::vector<double>>& X) const {
        if (!is_fitted) {
            throw std::runtime_error("Standardizer is not fitted.");
        }
        for (auto& row : X) {
            for (size_t d = 0; d < feature_dim; ++d) {
                row[d] = (row[d] - mean[d]) / std_dev[d];
            }
        }
    }

    void save(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open standardizer file for writing: " + filepath);
        }
        out << feature_dim << "\n";
        for (size_t d = 0; d < feature_dim; ++d) {
            out << mean[d] << (d + 1 == feature_dim ? "\n" : " ");
        }
        for (size_t d = 0; d < feature_dim; ++d) {
            out << std_dev[d] << (d + 1 == feature_dim ? "\n" : " ");
        }
    }

    void load(const std::string& filepath) {
        std::ifstream in(filepath);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open standardizer file for reading: " + filepath);
        }
        in >> feature_dim;
        mean.resize(feature_dim);
        std_dev.resize(feature_dim);
        for (size_t d = 0; d < feature_dim; ++d) {
            in >> mean[d];
        }
        for (size_t d = 0; d < feature_dim; ++d) {
            in >> std_dev[d];
        }
        is_fitted = true;
    }
};

} // namespace msar

#endif // MSAR_STANDARDIZER_HPP
