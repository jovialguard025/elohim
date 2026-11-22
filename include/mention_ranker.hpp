#ifndef MSAR_MENTION_RANKER_HPP
#define MSAR_MENTION_RANKER_HPP

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace msar {

// Shared-weight mention ranking network:
// Given candidate feature vector f_A (dim D_cand) and f_B (dim D_cand),
// computes s_A = score(f_A), s_B = score(f_B), and s_N = b_N (learned Neither threshold).
// Logits z = [s_A, s_B, s_N]
// Softmax(z) produces exact permutation-equivariant probabilities for A and B!
class MentionRankingMLP {
public:
    size_t cand_dim;
    size_t hidden_dim;
    size_t output_dim = 3;

    // Weights:
    // W1: cand_dim x hidden_dim
    // b1: hidden_dim
    // W2: hidden_dim (scores candidate representation into a single scalar)
    // b_cand: scalar bias for candidate scores
    // b_neither: scalar learned score for "Neither"
    std::vector<double> W1;
    std::vector<double> b1;
    std::vector<double> W2;
    double b_cand = 0.0;
    double b_neither = 0.0;

    // Adam optimizer moments:
    std::vector<double> mW1, vW1;
    std::vector<double> mb1, vb1;
    std::vector<double> mW2, vW2;
    double mb_cand = 0.0, vb_cand = 0.0;
    double mb_neither = 0.0, vb_neither = 0.0;
    size_t time_step = 0;

    MentionRankingMLP(size_t c_dim = 18, size_t hid_dim = 32, unsigned int seed = 42)
        : cand_dim(c_dim), hidden_dim(hid_dim) {
        initWeights(seed);
    }

    void initWeights(unsigned int seed) {
        std::mt19937 gen(seed);
        double std_w1 = std::sqrt(2.0 / static_cast<double>(cand_dim));
        double std_w2 = std::sqrt(2.0 / static_cast<double>(hidden_dim));
        std::normal_distribution<double> dist_w1(0.0, std_w1);
        std::normal_distribution<double> dist_w2(0.0, std_w2);

        W1.resize(cand_dim * hidden_dim);
        b1.assign(hidden_dim, 0.0);
        W2.resize(hidden_dim);
        b_cand = 0.0;
        b_neither = 0.0;

        for (size_t i = 0; i < W1.size(); ++i) W1[i] = dist_w1(gen);
        for (size_t i = 0; i < W2.size(); ++i) W2[i] = dist_w2(gen);

        mW1.assign(W1.size(), 0.0);
        vW1.assign(W1.size(), 0.0);
        mb1.assign(b1.size(), 0.0);
        vb1.assign(b1.size(), 0.0);

        mW2.assign(W2.size(), 0.0);
        vW2.assign(W2.size(), 0.0);
        mb_cand = 0.0; vb_cand = 0.0;
        mb_neither = 0.0; vb_neither = 0.0;

        time_step = 0;
    }

    // Forward pass for single sample with 2*cand_dim features
    void forward(
        const std::vector<double>& x,
        std::vector<double>& h_a,
        std::vector<double>& h_b,
        std::vector<double>& probs) const
    {
        h_a.resize(hidden_dim);
        h_b.resize(hidden_dim);

        // Compute h_A
        for (size_t j = 0; j < hidden_dim; ++j) {
            double sum = b1[j];
            for (size_t i = 0; i < cand_dim; ++i) {
                sum += x[i] * W1[i * hidden_dim + j];
            }
            h_a[j] = sum > 0.0 ? sum : 0.0;
        }

        // Compute h_B
        for (size_t j = 0; j < hidden_dim; ++j) {
            double sum = b1[j];
            for (size_t i = 0; i < cand_dim; ++i) {
                sum += x[cand_dim + i] * W1[i * hidden_dim + j];
            }
            h_b[j] = sum > 0.0 ? sum : 0.0;
        }

        // Compute candidate scores s_A and s_B
        double s_a = b_cand;
        double s_b = b_cand;
        for (size_t j = 0; j < hidden_dim; ++j) {
            s_a += h_a[j] * W2[j];
            s_b += h_b[j] * W2[j];
        }
        double s_neither = b_neither;

        // Stable Softmax
        double max_z = std::max({s_a, s_b, s_neither});
        double exp_a = std::exp(s_a - max_z);
        double exp_b = std::exp(s_b - max_z);
        double exp_n = std::exp(s_neither - max_z);
        double sum_exp = exp_a + exp_b + exp_n;

        probs.resize(3);
        probs[0] = exp_a / sum_exp;
        probs[1] = exp_b / sum_exp;
        probs[2] = exp_n / sum_exp;
    }

    std::vector<int> predict(
        const std::vector<std::vector<double>>& X,
        std::vector<std::vector<double>>& probs) const
    {
        std::vector<int> preds(X.size());
        probs.resize(X.size());
        std::vector<double> ha, hb;
        for (size_t i = 0; i < X.size(); ++i) {
            forward(X[i], ha, hb, probs[i]);
            int best_c = 0;
            double best_p = probs[i][0];
            if (probs[i][1] > best_p) {
                best_p = probs[i][1];
                best_c = 1;
            }
            if (probs[i][2] > best_p) {
                best_c = 2;
            }
            preds[i] = best_c;
        }
        return preds;
    }

    void trainBatch(
        const std::vector<std::vector<double>>& X_batch,
        const std::vector<int>& y_batch,
        const std::vector<double>& class_weights,
        double lr,
        double l2_lambda,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double eps = 1e-8)
    {
        size_t batch_sz = X_batch.size();
        if (batch_sz == 0) return;

        time_step++;

        std::vector<std::vector<double>> Ha(batch_sz, std::vector<double>(hidden_dim));
        std::vector<std::vector<double>> Hb(batch_sz, std::vector<double>(hidden_dim));
        std::vector<std::vector<double>> P(batch_sz, std::vector<double>(3));

        double batch_weight_sum = 0.0;
        for (size_t s = 0; s < batch_sz; ++s) {
            forward(X_batch[s], Ha[s], Hb[s], P[s]);
            batch_weight_sum += class_weights[y_batch[s]];
        }
        if (batch_weight_sum <= 0.0) batch_weight_sum = 1.0;

        std::vector<double> grad_W1(W1.size(), 0.0);
        std::vector<double> grad_b1(b1.size(), 0.0);
        std::vector<double> grad_W2(W2.size(), 0.0);
        double grad_b_cand = 0.0;
        double grad_b_neither = 0.0;

        for (size_t s = 0; s < batch_sz; ++s) {
            int target = y_batch[s];
            double sample_w = class_weights[target] / batch_weight_sum;

            // Output gradients: dL/dz_k
            double g0 = sample_w * (P[s][0] - (target == 0 ? 1.0 : 0.0));
            double g1 = sample_w * (P[s][1] - (target == 1 ? 1.0 : 0.0));
            double g2 = sample_w * (P[s][2] - (target == 2 ? 1.0 : 0.0));

            // Gradients for candidate scoring head
            for (size_t j = 0; j < hidden_dim; ++j) {
                grad_W2[j] += g0 * Ha[s][j] + g1 * Hb[s][j];
            }
            grad_b_cand += (g0 + g1);
            grad_b_neither += g2;

            // Backprop into shared representations Ha and Hb
            for (size_t j = 0; j < hidden_dim; ++j) {
                double dL_dha = g0 * W2[j];
                double dL_dhb = g1 * W2[j];

                double delta_a = (Ha[s][j] > 0.0) ? dL_dha : 0.0;
                double delta_b = (Hb[s][j] > 0.0) ? dL_dhb : 0.0;

                grad_b1[j] += delta_a + delta_b;

                for (size_t i = 0; i < cand_dim; ++i) {
                    grad_W1[i * hidden_dim + j] += X_batch[s][i] * delta_a 
                                                 + X_batch[s][cand_dim + i] * delta_b;
                }
            }
        }

        // L2 Regularization
        if (l2_lambda > 0.0) {
            for (size_t idx = 0; idx < W1.size(); ++idx) grad_W1[idx] += l2_lambda * W1[idx];
            for (size_t idx = 0; idx < W2.size(); ++idx) grad_W2[idx] += l2_lambda * W2[idx];
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

        auto updateScalar = [&](double& theta, double g, double& m, double& v) {
            m = beta1 * m + (1.0 - beta1) * g;
            v = beta2 * v + (1.0 - beta2) * g * g;
            double m_hat = m / bias_corr1;
            double v_hat = v / bias_corr2;
            theta -= lr * m_hat / (std::sqrt(v_hat) + eps);
        };

        updateParam(W1, grad_W1, mW1, vW1);
        updateParam(b1, grad_b1, mb1, vb1);
        updateParam(W2, grad_W2, mW2, vW2);
        updateScalar(b_cand, grad_b_cand, mb_cand, vb_cand);
        updateScalar(b_neither, grad_b_neither, mb_neither, vb_neither);
    }
};

} // namespace msar

#endif // MSAR_MENTION_RANKER_HPP
