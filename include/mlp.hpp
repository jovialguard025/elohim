#ifndef MSAR_MLP_HPP
#define MSAR_MLP_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace msar {

class MultilayerPerceptron {
public:
    size_t input_dim;
    size_t hidden_dim;
    size_t output_dim;

    // Weights and biases:
    // W1: input_dim x hidden_dim, row-major (input_dim rows, hidden_dim cols)
    std::vector<double> W1;
    std::vector<double> b1; // hidden_dim
    // W2: hidden_dim x output_dim, row-major
    std::vector<double> W2;
    std::vector<double> b2; // output_dim

    // Adam optimizer first and second moments:
    std::vector<double> mW1, vW1;
    std::vector<double> mb1, vb1;
    std::vector<double> mW2, vW2;
    std::vector<double> mb2, vb2;
    size_t time_step = 0;
    std::mt19937 dropout_rng;

    MultilayerPerceptron(size_t in_dim = 36, size_t hid_dim = 32, size_t out_dim = 3, unsigned int seed = 42)
        : input_dim(in_dim), hidden_dim(hid_dim), output_dim(out_dim) {
        initWeights(seed);
    }

    void initWeights(unsigned int seed) {
        std::mt19937 gen(seed);
        // He / Kaiming normal initialization: std = sqrt(2 / fan_in)
        double std_w1 = std::sqrt(2.0 / static_cast<double>(input_dim));
        double std_w2 = std::sqrt(2.0 / static_cast<double>(hidden_dim));
        std::normal_distribution<double> dist_w1(0.0, std_w1);
        std::normal_distribution<double> dist_w2(0.0, std_w2);

        W1.resize(input_dim * hidden_dim);
        b1.assign(hidden_dim, 0.0);
        W2.resize(hidden_dim * output_dim);
        b2.assign(output_dim, 0.0);

        for (size_t i = 0; i < W1.size(); ++i) W1[i] = dist_w1(gen);
        for (size_t i = 0; i < W2.size(); ++i) W2[i] = dist_w2(gen);

        mW1.assign(W1.size(), 0.0);
        vW1.assign(W1.size(), 0.0);
        mb1.assign(b1.size(), 0.0);
        vb1.assign(b1.size(), 0.0);

        mW2.assign(W2.size(), 0.0);
        vW2.assign(W2.size(), 0.0);
        mb2.assign(b2.size(), 0.0);
        vb2.assign(b2.size(), 0.0);

        time_step = 0;
        dropout_rng.seed(seed ^ 0x85ebca6bU);
    }

    // Forward pass for a single input vector
    void forward(const std::vector<double>& x, std::vector<double>& h, std::vector<double>& p) const {
        h.resize(hidden_dim);
        for (size_t j = 0; j < hidden_dim; ++j) {
            double sum = b1[j];
            for (size_t i = 0; i < input_dim; ++i) {
                sum += x[i] * W1[i * hidden_dim + j];
            }
            h[j] = sum > 0.0 ? sum : 0.0; // ReLU
        }

        // Output logits
        std::vector<double> z(output_dim);
        double max_z = -1e30;
        for (size_t k = 0; k < output_dim; ++k) {
            double sum = b2[k];
            for (size_t j = 0; j < hidden_dim; ++j) {
                sum += h[j] * W2[j * output_dim + k];
            }
            z[k] = sum;
            if (sum > max_z) max_z = sum;
        }

        // Numerically stable Softmax
        p.resize(output_dim);
        double sum_exp = 0.0;
        for (size_t k = 0; k < output_dim; ++k) {
            p[k] = std::exp(z[k] - max_z);
            sum_exp += p[k];
        }
        for (size_t k = 0; k < output_dim; ++k) {
            p[k] /= sum_exp;
        }
    }

    // Predict batch
    std::vector<int> predict(const std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& probs) const {
        std::vector<int> preds(X.size());
        probs.resize(X.size());
        std::vector<double> h;
        for (size_t i = 0; i < X.size(); ++i) {
            forward(X[i], h, probs[i]);
            int best_c = 0;
            double best_p = probs[i][0];
            for (size_t c = 1; c < output_dim; ++c) {
                if (probs[i][c] > best_p) {
                    best_p = probs[i][c];
                    best_c = static_cast<int>(c);
                }
            }
            preds[i] = best_c;
        }
        return preds;
    }

    // Train on a mini-batch using Adam
    void trainBatch(
        const std::vector<std::vector<double>>& X_batch,
        const std::vector<int>& y_batch,
        const std::vector<double>& class_weights,
        double lr,
        double l2_lambda,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double eps = 1e-8,
        double label_smoothing = 0.0,
        double focal_gamma = 0.0,
        double dropout_rate = 0.0)
    {
        size_t batch_sz = X_batch.size();
        if (batch_sz == 0) return;

        time_step++;

        std::vector<std::vector<double>> H_batch(batch_sz, std::vector<double>(hidden_dim));
        std::vector<std::vector<double>> P_batch(batch_sz, std::vector<double>(output_dim));
        std::vector<std::vector<double>> dropout_masks(batch_sz, std::vector<double>(hidden_dim, 1.0));
        const double dropout = std::max(0.0, std::min(0.8, dropout_rate));
        const double dropout_scale = dropout < 1.0 ? 1.0 / (1.0 - dropout) : 1.0;
        std::bernoulli_distribution keep_unit(1.0 - dropout);

        double batch_weight_sum = 0.0;
        for (size_t s = 0; s < batch_sz; ++s) {
            forward(X_batch[s], H_batch[s], P_batch[s]);
            if (dropout > 0.0) {
                for (size_t j = 0; j < hidden_dim; ++j) {
                    if (keep_unit(dropout_rng)) {
                        dropout_masks[s][j] = dropout_scale;
                        H_batch[s][j] *= dropout_scale;
                    } else {
                        dropout_masks[s][j] = 0.0;
                        H_batch[s][j] = 0.0;
                    }
                }
                double max_z = -1e30;
                std::vector<double> z(output_dim);
                for (size_t k = 0; k < output_dim; ++k) {
                    double sum = b2[k];
                    for (size_t j = 0; j < hidden_dim; ++j) sum += H_batch[s][j] * W2[j * output_dim + k];
                    z[k] = sum;
                    if (sum > max_z) max_z = sum;
                }
                double sum_exp = 0.0;
                for (size_t k = 0; k < output_dim; ++k) {
                    P_batch[s][k] = std::exp(z[k] - max_z);
                    sum_exp += P_batch[s][k];
                }
                for (size_t k = 0; k < output_dim; ++k) P_batch[s][k] /= sum_exp;
            }
            int target = y_batch[s];
            batch_weight_sum += class_weights[target];
        }
        if (batch_weight_sum <= 0.0) batch_weight_sum = 1.0;

        // Gradients accumulation
        std::vector<double> grad_W1(W1.size(), 0.0);
        std::vector<double> grad_b1(b1.size(), 0.0);
        std::vector<double> grad_W2(W2.size(), 0.0);
        std::vector<double> grad_b2(b2.size(), 0.0);

        for (size_t s = 0; s < batch_sz; ++s) {
            int target = y_batch[s];
            double sample_weight = class_weights[target] / batch_weight_sum;
            double focal_gradient_scale = 1.0;
            if (focal_gamma > 0.0 && label_smoothing == 0.0) {
                const double pt = std::max(1e-12, std::min(1.0, P_batch[s][target]));
                const double one_minus_pt = std::max(0.0, 1.0 - pt);
                focal_gradient_scale = std::pow(one_minus_pt, focal_gamma);
                if (one_minus_pt > 0.0) {
                    focal_gradient_scale += focal_gamma * pt *
                        std::pow(one_minus_pt, focal_gamma - 1.0) * std::log(pt);
                }
            }

            // Output gradient dL/dz_k = sample_weight * (p_k - y_k)
            std::vector<double> dL_dz(output_dim);
            for (size_t k = 0; k < output_dim; ++k) {
                const double target_prob = label_smoothing / static_cast<double>(output_dim) +
                    (static_cast<int>(k) == target ? 1.0 - label_smoothing : 0.0);
                double diff = P_batch[s][k] - target_prob;
                dL_dz[k] = sample_weight * focal_gradient_scale * diff;
            }

            // Gradients for W2 and b2
            for (size_t j = 0; j < hidden_dim; ++j) {
                for (size_t k = 0; k < output_dim; ++k) {
                    grad_W2[j * output_dim + k] += H_batch[s][j] * dL_dz[k];
                }
            }
            for (size_t k = 0; k < output_dim; ++k) {
                grad_b2[k] += dL_dz[k];
            }

            // Backprop into hidden layer
            std::vector<double> dL_dh(hidden_dim, 0.0);
            for (size_t j = 0; j < hidden_dim; ++j) {
                for (size_t k = 0; k < output_dim; ++k) {
                    dL_dh[j] += dL_dz[k] * W2[j * output_dim + k];
                }
                // ReLU derivative
                if (H_batch[s][j] <= 0.0 || dropout_masks[s][j] == 0.0) {
                    dL_dh[j] = 0.0;
                } else {
                    dL_dh[j] *= dropout_masks[s][j];
                }
            }

            // Gradients for W1 and b1
            for (size_t i = 0; i < input_dim; ++i) {
                for (size_t j = 0; j < hidden_dim; ++j) {
                    grad_W1[i * hidden_dim + j] += X_batch[s][i] * dL_dh[j];
                }
            }
            for (size_t j = 0; j < hidden_dim; ++j) {
                grad_b1[j] += dL_dh[j];
            }
        }

        // L2 Regularization gradient on weights
        if (l2_lambda > 0.0) {
            for (size_t idx = 0; idx < W1.size(); ++idx) {
                grad_W1[idx] += l2_lambda * W1[idx];
            }
            for (size_t idx = 0; idx < W2.size(); ++idx) {
                grad_W2[idx] += l2_lambda * W2[idx];
            }
        }

        // Adam parameter update
        double bias_corr1 = 1.0 - std::pow(beta1, static_cast<double>(time_step));
        double bias_corr2 = 1.0 - std::pow(beta2, static_cast<double>(time_step));

        auto updateParam = [&](std::vector<double>& theta, const std::vector<double>& g, std::vector<double>& m, std::vector<double>& v) {
            for (size_t idx = 0; idx < theta.size(); ++idx) {
                m[idx] = beta1 * m[idx] + (1.0 - beta1) * g[idx];
                v[idx] = beta2 * v[idx] + (1.0 - beta2) * g[idx] * g[idx];
                double m_hat = m[idx] / bias_corr1;
                double v_hat = v[idx] / bias_corr2;
                theta[idx] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        };

        updateParam(W1, grad_W1, mW1, vW1);
        updateParam(b1, grad_b1, mb1, vb1);
        updateParam(W2, grad_W2, mW2, vW2);
        updateParam(b2, grad_b2, mb2, vb2);
    }

    void saveCheckpoint(const std::string& filepath) const {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Cannot open checkpoint file for writing: " + filepath);
        }
        out.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
        out.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(hidden_dim));
        out.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));

        out.write(reinterpret_cast<const char*>(W1.data()), W1.size() * sizeof(double));
        out.write(reinterpret_cast<const char*>(b1.data()), b1.size() * sizeof(double));
        out.write(reinterpret_cast<const char*>(W2.data()), W2.size() * sizeof(double));
        out.write(reinterpret_cast<const char*>(b2.data()), b2.size() * sizeof(double));
    }

    void loadCheckpoint(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open checkpoint file for reading: " + filepath);
        }
        in.read(reinterpret_cast<char*>(&input_dim), sizeof(input_dim));
        in.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
        in.read(reinterpret_cast<char*>(&output_dim), sizeof(output_dim));

        W1.resize(input_dim * hidden_dim);
        b1.resize(hidden_dim);
        W2.resize(hidden_dim * output_dim);
        b2.resize(output_dim);

        in.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(double));
        in.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(double));
        in.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(double));
        in.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(double));
    }
};

} // namespace msar

#endif // MSAR_MLP_HPP
