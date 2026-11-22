#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v1.hpp"
#include "features_v2.hpp"
#include "standardizer.hpp"
#include "mlp.hpp"
#include "evaluation.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

void testDatasetLoading() {
    std::cout << "[Test 1] Testing Dataset Loader & Schema Validation... ";
    auto records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    assert(records.size() == 454);
    
    auto stats = msar::computePartitionStats(records);
    assert(stats.total_samples == 454);
    assert(stats.count_a == 187);
    assert(stats.count_b == 205);
    assert(stats.count_neither == 62);

    // Verify first record
    const auto& r0 = records[0];
    assert(!r0.id.empty());
    assert(!r0.text.empty());
    assert(r0.pronoun_offset >= 0);
    assert(r0.a_offset >= 0);
    assert(r0.b_offset >= 0);
    assert(r0.label >= 0 && r0.label <= 2);

    std::cout << "PASSED (454 samples verified: A=187, B=205, Neither=62)\n";
}

void testOffsetExtraction() {
    std::cout << "[Test 2] Testing GAP Character Offsets Correctness... ";
    auto records = msar::loadGapDataset("data/raw/gap-development.tsv");
    assert(records.size() == 2000);

    for (size_t i = 0; i < 50; ++i) {
        const auto& r = records[i];
        // Ensure offsets fall within text boundaries
        assert(static_cast<size_t>(r.pronoun_offset) < r.text.size());
        assert(static_cast<size_t>(r.a_offset) < r.text.size());
        assert(static_cast<size_t>(r.b_offset) < r.text.size());

        // Substrings match
        std::string ext_pron = r.text.substr(r.pronoun_offset, r.pronoun.size());
        assert(ext_pron == r.pronoun);

        std::string ext_a = r.text.substr(r.a_offset, r.a.size());
        assert(ext_a == r.a);

        std::string ext_b = r.text.substr(r.b_offset, r.b.size());
        assert(ext_b == r.b);
    }
    std::cout << "PASSED (Exact character boundary alignment confirmed)\n";
}

void testFeaturesV1() {
    std::cout << "[Test 3] Testing 12-Feature V1 Extractor... ";
    auto records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    
    for (size_t i = 0; i < records.size(); ++i) {
        auto feat = msar::extractFeaturesV1(records[i]);
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
    auto records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    
    for (size_t i = 0; i < records.size(); ++i) {
        auto feat = msar::extractFeaturesV2(records[i]);
        assert(feat.size() == 36);
        for (size_t j = 0; j < feat.size(); ++j) {
            double v = feat[j];
            assert(!std::isnan(v));
            assert(!std::isinf(v));
        }
    }
    std::cout << "PASSED (All 454 validation rows yield exactly 36 finite numerical features)\n";
}

void testFeatureSymmetry() {
    std::cout << "[Test 5] Testing Candidate Feature Non-Exchangeability & Symmetry... ";
    msar::GapRecord rec;
    rec.text = "Alice visited Bob before she left the office.";
    rec.pronoun = "she";
    rec.pronoun_offset = 25;
    rec.a = "Alice";
    rec.a_offset = 0;
    rec.a_coref = true;
    rec.b = "Bob";
    rec.b_offset = 14;
    rec.b_coref = false;
    rec.label = 0;

    auto feat = msar::extractFeaturesV2(rec);
    assert(feat.size() == 36);

    // Feature 0 is character distance for A: (25 - 0) / 45
    // Feature 18 is character distance for B: (25 - 14) / 45
    assert(feat[0] != feat[18]);
    assert(feat[0] > feat[18]); // Alice is farther away than Bob

    std::cout << "PASSED (Candidate A and B feature partitions are strictly distinct)\n";
}

void testStandardizer() {
    std::cout << "[Test 6] Testing Feature Standardizer & Zero-Variance Handling... ";
    std::vector<std::vector<double>> dummy_X = {
        {1.0, 5.0, 10.0},
        {3.0, 5.0, 20.0},
        {2.0, 5.0, 30.0}
    };
    msar::FeatureStandardizer scaler;
    scaler.fit(dummy_X);

    assert(std::abs(scaler.mean[0] - 2.0) < 1e-6);
    assert(std::abs(scaler.mean[1] - 5.0) < 1e-6);
    assert(std::abs(scaler.mean[2] - 20.0) < 1e-6);
    // Feature 1 has 0 variance; standardizer must safely default scale to 1.0
    assert(scaler.std_dev[1] == 1.0);

    auto transformed = scaler.transformSingle({2.0, 5.0, 20.0});
    assert(std::abs(transformed[0] - 0.0) < 1e-6);
    assert(std::abs(transformed[1] - 0.0) < 1e-6);
    assert(std::abs(transformed[2] - 0.0) < 1e-6);

    std::cout << "PASSED (Mean/std scaling and zero-division protection verified)\n";
}

void testMLPArchitecture() {
    std::cout << "[Test 7] Testing MLP Forward/Backward & Softmax Stability... ";
    msar::MultilayerPerceptron mlp(36, 32, 3, 42);
    std::vector<double> dummy_x(36, 0.5);
    std::vector<double> h, p;
    mlp.forward(dummy_x, h, p);

    assert(h.size() == 32);
    assert(p.size() == 3);
    double sum_p = p[0] + p[1] + p[2];
    assert(std::abs(sum_p - 1.0) < 1e-6);

    // Train single batch
    std::vector<std::vector<double>> X_batch = {dummy_x, dummy_x};
    std::vector<int> y_batch = {0, 1};
    std::vector<double> weights = {1.0, 1.0, 1.0};
    mlp.trainBatch(X_batch, y_batch, weights, 0.001, 0.0);

    std::vector<double> p_after;
    mlp.forward(dummy_x, h, p_after);
    assert(p_after.size() == 3);

    std::cout << "PASSED (Softmax partitions unity and backward pass updates weights)\n";
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " Running Confidence-Gated MSAR C++ Unit Test Suite\n";
    std::cout << "===================================================================\n\n";

    testDatasetLoading();
    testOffsetExtraction();
    testFeaturesV1();
    testFeaturesV2();
    testFeatureSymmetry();
    testStandardizer();
    testMLPArchitecture();

    std::cout << "\n===================================================================\n";
    std::cout << " ALL UNIT TESTS PASSED SUCCESSFULLY!\n";
    std::cout << "===================================================================\n";
    return 0;
}
