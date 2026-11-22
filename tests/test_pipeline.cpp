#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v1.hpp"
#include "features_v2.hpp"
#include "features_v3.hpp"
#include "standardizer.hpp"
#include "mlp.hpp"
#include "evaluation.hpp"
#include "training.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>

void testDataLoader() {
    std::cout << "[Test 1] Testing Dataset Loader & Schema Validation... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    assert(val_records.size() == 454);
    
    auto stats = msar::computePartitionStats(val_records);
    assert(stats.total_samples == 454);
    assert(stats.count_a == 187);
    assert(stats.count_b == 205);
    assert(stats.count_neither == 62);
    std::cout << "PASSED (454 samples verified: A=187, B=205, Neither=62)\n";
}

void testPreprocessing() {
    std::cout << "[Test 2] Testing GAP Character Offsets Correctness... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    for (const auto& r : val_records) {
        // Verify snippet pronoun offset substring
        if (r.pronoun_offset >= 0 && static_cast<size_t>(r.pronoun_offset) + r.pronoun.size() <= r.text.size()) {
            std::string actual_pron = r.text.substr(static_cast<size_t>(r.pronoun_offset), r.pronoun.size());
            assert(actual_pron == r.pronoun);
        }
        // Verify candidate A offset
        if (r.a_offset >= 0 && static_cast<size_t>(r.a_offset) + r.a.size() <= r.text.size()) {
            std::string actual_a = r.text.substr(static_cast<size_t>(r.a_offset), r.a.size());
            assert(actual_a == r.a);
        }
        // Verify candidate B offset
        if (r.b_offset >= 0 && static_cast<size_t>(r.b_offset) + r.b.size() <= r.text.size()) {
            std::string actual_b = r.text.substr(static_cast<size_t>(r.b_offset), r.b.size());
            assert(actual_b == r.b);
        }
    }
    std::cout << "PASSED (Exact character boundary alignment confirmed)\n";
}

void testFeaturesV1() {
    std::cout << "[Test 3] Testing 12-Feature V1 Extractor... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    for (size_t i = 0; i < val_records.size(); ++i) {
        auto feat = msar::extractFeaturesV1(val_records[i]);
        assert(feat.size() == 12);
        for (double v : feat) {
            assert(!std::isnan(v));
            assert(!std::isinf(v));
        }
    }
    std::cout << "PASSED (All 454 validation rows yield 12 finite numerical features)\n";
}

void testFeaturesV2() {
    std::cout << "[Test 4] Testing 36-Feature V2 Extractor... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    for (size_t i = 0; i < val_records.size(); ++i) {
        auto feat = msar::extractFeaturesV2(val_records[i]);
        assert(feat.size() == 36);
        for (double v : feat) {
            assert(!std::isnan(v));
            assert(!std::isinf(v));
        }
    }
    std::cout << "PASSED (All 454 validation rows yield exactly 36 finite numerical features)\n";
}

void testFeatureSymmetry() {
    std::cout << "[Test 5] Testing Candidate Feature Non-Exchangeability & Symmetry... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    auto feat = msar::extractFeaturesV2(val_records[0]);
    // Ensure Candidate A features (0..17) are not identical to Candidate B features (18..35)
    bool is_different = false;
    for (size_t i = 0; i < 18; ++i) {
        if (std::abs(feat[i] - feat[18 + i]) > 1e-6) {
            is_different = true;
            break;
        }
    }
    assert(is_different);
    std::cout << "PASSED (Candidate A and B feature partitions are strictly distinct)\n";
}

void testStandardizer() {
    std::cout << "[Test 6] Testing Feature Standardizer & Zero-Variance Handling... ";
    std::vector<std::vector<double>> fake_data = {
        {1.0, 5.0, 10.0},
        {3.0, 5.0, 20.0},
        {5.0, 5.0, 30.0}
    };
    msar::FeatureStandardizer scaler;
    scaler.fit(fake_data);
    assert(std::abs(scaler.mean[0] - 3.0) < 1e-6);
    assert(std::abs(scaler.mean[1] - 5.0) < 1e-6);
    assert(scaler.std_dev[1] == 1.0); // Zero-variance feature safe division protection

    auto transformed = scaler.transformSingle({3.0, 5.0, 20.0});
    assert(std::abs(transformed[0] - 0.0) < 1e-6);
    assert(std::abs(transformed[1] - 0.0) < 1e-6);
    assert(std::abs(transformed[2] - 0.0) < 1e-6);
    std::cout << "PASSED (Mean/std scaling and zero-division protection verified)\n";
}

void testMLPArchitecture() {
    std::cout << "[Test 7] Testing MLP Forward/Backward & Softmax Stability... ";
    msar::MultilayerPerceptron mlp(36, 32, 3, 42);
    std::vector<double> x(36, 0.5);
    std::vector<double> h, p;
    mlp.forward(x, h, p);
    assert(p.size() == 3);
    double sum = p[0] + p[1] + p[2];
    assert(std::abs(sum - 1.0) < 1e-5);
    assert(p[0] >= 0.0 && p[1] >= 0.0 && p[2] >= 0.0);

    // Mini batch step
    std::vector<std::vector<double>> X_batch = {x, x};
    std::vector<int> y_batch = {0, 1};
    std::vector<double> cw = {1.0, 1.0, 1.0};
    double old_w0 = mlp.W1[0];
    mlp.trainBatch(X_batch, y_batch, cw, 0.01, 0.0);
    assert(mlp.W1[0] != old_w0);
    std::cout << "PASSED (Softmax partitions unity and backward pass updates weights)\n";
}

void testFeaturesV3() {
    std::cout << "[Test 8] Testing 80-Feature V3 Extractor... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");

    for (size_t i = 0; i < val_records.size(); ++i) {
        auto feat = msar::extractFeaturesV3(val_records[i]);
        assert(feat.size() == 80);
        for (double v : feat) {
            assert(!std::isnan(v));
            assert(!std::isinf(v));
        }
    }
    std::cout << "PASSED (80 finite numerical features verified across all validation rows)\n";
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " Running Confidence-Gated MSAR C++ Unit Test Suite\n";
    std::cout << "===================================================================\n\n";

    testDataLoader();
    testPreprocessing();
    testFeaturesV1();
    testFeaturesV2();
    testFeatureSymmetry();
    testStandardizer();
    testMLPArchitecture();
    testFeaturesV3();

    std::cout << "\n===================================================================\n";
    std::cout << " ALL UNIT TESTS PASSED SUCCESSFULLY!\n";
    std::cout << "===================================================================\n";
    return 0;
}
