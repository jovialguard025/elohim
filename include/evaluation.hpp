#ifndef MSAR_EVALUATION_HPP
#define MSAR_EVALUATION_HPP

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>

namespace msar {

struct ClassMetrics {
    double precision;
    double recall;
    double f1;
    size_t support;
};

struct EvaluationResult {
    double accuracy;
    double macro_f1;
    double weighted_loss;
    std::vector<ClassMetrics> class_metrics; // size 3: 0 (A), 1 (B), 2 (Neither)
    int confusion_matrix[3][3];              // [actual][predicted]
    size_t total_samples;
    size_t correct_predictions;

    std::string toString() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "Accuracy: " << (accuracy * 100.0) << "% (" << correct_predictions << " / " << total_samples << ")\n";
        ss << "Macro-F1: " << macro_f1 << "\n";
        ss << "Weighted Loss: " << weighted_loss << "\n\n";

        ss << "Per-Class Breakdown:\n";
        ss << "---------------------------------------------------------\n";
        ss << std::left << std::setw(12) << "Class" 
           << std::setw(12) << "Precision" 
           << std::setw(12) << "Recall" 
           << std::setw(12) << "F1-Score" 
           << std::setw(10) << "Support" << "\n";
        ss << "---------------------------------------------------------\n";
        const char* class_names[3] = {"A (0)", "B (1)", "Neither (2)"};
        for (int c = 0; c < 3; ++c) {
            ss << std::left << std::setw(12) << class_names[c]
               << std::setw(12) << class_metrics[c].precision
               << std::setw(12) << class_metrics[c].recall
               << std::setw(12) << class_metrics[c].f1
               << std::setw(10) << class_metrics[c].support << "\n";
        }
        ss << "---------------------------------------------------------\n\n";

        ss << "Confusion Matrix:\n";
        ss << "                 Predicted\n";
        ss << "                 A        B        Neither\n";
        ss << "Actual A         " << std::setw(8) << confusion_matrix[0][0] << " "
                                  << std::setw(8) << confusion_matrix[0][1] << " "
                                  << std::setw(8) << confusion_matrix[0][2] << "\n";
        ss << "Actual B         " << std::setw(8) << confusion_matrix[1][0] << " "
                                  << std::setw(8) << confusion_matrix[1][1] << " "
                                  << std::setw(8) << confusion_matrix[1][2] << "\n";
        ss << "Actual Neither   " << std::setw(8) << confusion_matrix[2][0] << " "
                                  << std::setw(8) << confusion_matrix[2][1] << " "
                                  << std::setw(8) << confusion_matrix[2][2] << "\n";
        return ss.str();
    }
};

inline EvaluationResult evaluatePredictions(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    const std::vector<std::vector<double>>& probs,
    const std::vector<double>& class_weights)
{
    EvaluationResult res;
    res.total_samples = y_true.size();
    res.correct_predictions = 0;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            res.confusion_matrix[r][c] = 0;
        }
    }

    double total_weighted_loss = 0.0;
    double weight_sum = 0.0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        int act = y_true[i];
        int pr = y_pred[i];
        if (act >= 0 && act < 3 && pr >= 0 && pr < 3) {
            res.confusion_matrix[act][pr]++;
        }
        if (act == pr) {
            res.correct_predictions++;
        }

        double w = class_weights[act];
        double p = probs[i][act];
        if (p < 1e-12) p = 1e-12;
        if (p > 1.0) p = 1.0;
        total_weighted_loss += w * (-std::log(p));
        weight_sum += w;
    }

    res.accuracy = res.total_samples > 0 ? static_cast<double>(res.correct_predictions) / static_cast<double>(res.total_samples) : 0.0;
    res.weighted_loss = weight_sum > 0 ? (total_weighted_loss / weight_sum) : 0.0;

    res.class_metrics.resize(3);
    double sum_f1 = 0.0;

    for (int c = 0; c < 3; ++c) {
        int tp = res.confusion_matrix[c][c];
        int fp = 0;
        int fn = 0;
        for (int r = 0; r < 3; ++r) {
            if (r != c) fp += res.confusion_matrix[r][c];
        }
        for (int col = 0; col < 3; ++col) {
            if (col != c) fn += res.confusion_matrix[c][col];
        }

        double prec = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double rec = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1 = (prec + rec) > 0 ? (2.0 * prec * rec) / (prec + rec) : 0.0;
        size_t support = tp + fn;

        res.class_metrics[c] = {prec, rec, f1, support};
        sum_f1 += f1;
    }

    res.macro_f1 = sum_f1 / 3.0;
    return res;
}

} // namespace msar

#endif // MSAR_EVALUATION_HPP
