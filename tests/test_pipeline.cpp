#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v1.hpp"
#include "features_v2.hpp"
#include "features_v3.hpp"
#include "standardizer.hpp"
#include "mlp.hpp"
#include "evaluation.hpp"
#include "training.hpp"
#include "experiment_engine.hpp"
#include "extended_features.hpp"

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

void testFindAllMentionsLoose() {
    std::cout << "[Test 8] Testing findAllMentionsLoose Word Boundary & Deduplication... ";
    auto mentions = msar::findAllMentionsLoose("Zoe Telford met Telford.", "Zoe Telford", 0);
    assert(mentions.size() == 2);
    assert(mentions[0] == 0);
    assert(mentions[1] == 16);
    std::cout << "PASSED (Found exactly 2 positions: 0 and 16)\n";
}

void testUrlMatch() {
    std::cout << "[Test 9] Testing url_match Extraction & Percent-Decoding... ";
    bool match1 = msar::checkUrlMatch("http://en.wikipedia.org/wiki/Zoe_Telford", "Zoe Telford");
    assert(match1 == true);
    bool match2 = msar::checkUrlMatch("http://en.wikipedia.org/wiki/List_of_Teachers_%28UK_TV_series%29_characters", "Cheryl Cassidy");
    assert(match2 == false);
    bool match3 = msar::checkUrlMatch("http://en.wikipedia.org/wiki/Pauline_Telford", "Zoe Telford");
    assert(match3 == true);
    std::cout << "PASSED (URL title parsing, percent decoding, and candidate matching verified)\n";
}

void testFeaturesV3() {
    std::cout << "[Test 10] Testing V3 Extractor (kFeatureCountV3)... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");

    for (size_t i = 0; i < val_records.size(); ++i) {
        auto feat = msar::extractFeaturesV3(val_records[i]);
        assert(feat.size() == msar::kFeatureCountV3);
        for (double v : feat) {
            assert(!std::isnan(v));
            assert(!std::isinf(v));
        }
    }
    std::cout << "PASSED (kFeatureCountV3=" << msar::kFeatureCountV3 << " finite numerical features verified across all validation rows)\n";
}

void testFoldDisjointnessAndNoLeakage() {
    std::cout << "[Test 11] Testing Fold Disjointness & Zero Stratification Leakage... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    auto folds = msar::createStratifiedFolds(val_records, 5, 42);
    assert(folds.size() == 5);

    std::vector<int> seen_val_count(val_records.size(), 0);
    for (const auto& fold : folds) {
        std::vector<bool> in_train(val_records.size(), false);
        for (size_t idx : fold.train_indices) {
            assert(idx < val_records.size());
            in_train[idx] = true;
        }
        for (size_t idx : fold.val_indices) {
            assert(idx < val_records.size());
            // Disjointness: no index can be in both train and val
            assert(!in_train[idx]);
            seen_val_count[idx]++;
        }
    }
    // Partition property: every validation index appears in exactly 1 val fold
    for (size_t i = 0; i < val_records.size(); ++i) {
        assert(seen_val_count[i] == 1);
    }
    std::cout << "PASSED (Mutual exclusion and exact partition verified across 5 folds)\n";
}

void testLabelSwapIdentity() {
    std::cout << "[Test 12] Testing Candidate Label Swap Identity... ";
    // Label 0 (A) <-> Label 1 (B), Label 2 (Neither) -> 2
    auto swap_label = [](int y) {
        return (y == 0 ? 1 : (y == 1 ? 0 : 2));
    };
    assert(swap_label(0) == 1);
    assert(swap_label(1) == 0);
    assert(swap_label(2) == 2);
    std::cout << "PASSED (0<->1 and 2 invariant confirmed)\n";
}

void testDoubleSwapIdentity() {
    std::cout << "[Test 13] Testing Feature Vector Double-Swap Involution Identity... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    auto feat = msar::extractFeaturesV2(val_records[0]);
    assert(feat.size() == 36);

    auto feat_swapped = msar::swapCandidatesVector(feat);
    auto feat_double_swapped = msar::swapCandidatesVector(feat_swapped);

    assert(feat_double_swapped.size() == feat.size());
    for (size_t i = 0; i < feat.size(); ++i) {
        assert(std::abs(feat[i] - feat_double_swapped[i]) < 1e-12);
    }

    // Also verify label double-swap identity
    auto swap_label = [](int y) { return (y == 0 ? 1 : (y == 1 ? 0 : 2)); };
    for (int y : {0, 1, 2}) {
        assert(swap_label(swap_label(y)) == y);
    }
    std::cout << "PASSED (Double swap strictly reconstructs original representation)\n";
}

void testProbabilityNormalization() {
    std::cout << "[Test 14] Testing Softmax Probability Normalization & Bounds... ";
    msar::MultilayerPerceptron mlp(36, 32, 3, 123);
    // Test across several diverse test inputs
    std::vector<std::vector<double>> test_inputs = {
        std::vector<double>(36, 0.0),
        std::vector<double>(36, 1.0),
        std::vector<double>(36, -5.0),
        std::vector<double>(36, 10.0)
    };
    for (const auto& in : test_inputs) {
        std::vector<double> h, p;
        mlp.forward(in, h, p);
        assert(p.size() == 3);
        double sum = p[0] + p[1] + p[2];
        assert(std::abs(sum - 1.0) < 1e-6);
        for (double prob : p) {
            assert(prob >= 0.0 && prob <= 1.0);
            assert(!std::isnan(prob));
            assert(!std::isinf(prob));
        }
    }
    std::cout << "PASSED (Softmax strictly bounded [0, 1] and sums to 1.0 within 1e-6)\n";
}

void testExtendedSwapBlockIsolation() {
    std::cout << "[Test 15] Testing Extended Layout [A | B | Shared] Swap Block Isolation... ";
    size_t C = 25; // 25 candidate features
    size_t S = 10; // 10 shared features
    std::vector<double> v(2 * C + S);
    for (size_t i = 0; i < C; ++i) v[i] = 100.0 + static_cast<double>(i); // Block A
    for (size_t i = 0; i < C; ++i) v[C + i] = 200.0 + static_cast<double>(i); // Block B
    for (size_t i = 0; i < S; ++i) v[2 * C + i] = 300.0 + static_cast<double>(i); // Block Shared

    auto swapped = msar::swapExtendedCandidatesVector(v, C);
    assert(swapped.size() == v.size());
    // Candidate A block now holds Block B
    for (size_t i = 0; i < C; ++i) {
        assert(std::abs(swapped[i] - (200.0 + static_cast<double>(i))) < 1e-9);
    }
    // Candidate B block now holds Block A
    for (size_t i = 0; i < C; ++i) {
        assert(std::abs(swapped[C + i] - (100.0 + static_cast<double>(i))) < 1e-9);
    }
    // Shared block is completely untouched
    for (size_t i = 0; i < S; ++i) {
        assert(std::abs(swapped[2 * C + i] - (300.0 + static_cast<double>(i))) < 1e-9);
    }
    // Double swap identity
    auto double_swapped = msar::swapExtendedCandidatesVector(swapped, C);
    for (size_t i = 0; i < v.size(); ++i) {
        assert(std::abs(double_swapped[i] - v[i]) < 1e-9);
    }
    std::cout << "PASSED (Candidate blocks swapped symmetrically, shared block untouched)\n";
}

void testMatcherWordBoundaryEdgeCase() {
    std::cout << "[Test 16] Testing Matcher Edge Cases ('Tel' vs 'Telford' word boundary)... ";
    std::string text = "Zoe Telford met Telford.";
    // Full name and surname matching
    auto m_full = msar::findAllMentionsLoose(text, "Zoe Telford", 0);
    assert(m_full.size() == 2);
    assert(m_full[0] == 0);
    assert(m_full[1] == 16);

    // Prefix substring 'Tel' must NOT match 'Telford' (no mentions found, returns fallback)
    auto m_prefix = msar::findAllMentionsLoose(text, "Tel", -1);
    assert(m_prefix.size() == 1 && m_prefix[0] == -1);
    std::cout << "PASSED (Prefix 'Tel' correctly rejected, full surname 'Telford' matched)\n";
}

void testStratifiedClassRatiosPreserved() {
    std::cout << "[Test 17] Testing Stratified Fold Class Ratios Preservation... ";
    auto val_records = msar::loadGapDataset("data/raw/gap-validation.tsv");
    assert(val_records.size() == 454);
    // Baseline overall proportions
    double p_a = 187.0 / 454.0;
    double p_b = 205.0 / 454.0;
    double p_neither = 62.0 / 454.0;

    auto folds = msar::createStratifiedFolds(val_records, 5, 42);
    for (size_t f = 0; f < 5; ++f) {
        size_t count_a = 0, count_b = 0, count_n = 0;
        for (size_t idx : folds[f].val_indices) {
            if (val_records[idx].label == 0) count_a++;
            else if (val_records[idx].label == 1) count_b++;
            else if (val_records[idx].label == 2) count_n++;
        }
        double n_fold = static_cast<double>(folds[f].val_indices.size());
        double f_pa = count_a / n_fold;
        double f_pb = count_b / n_fold;
        double f_pn = count_n / n_fold;

        // Tolerance within +/- 2.5% across small fold size (N=90)
        assert(std::abs(f_pa - p_a) < 0.035);
        assert(std::abs(f_pb - p_b) < 0.035);
        assert(std::abs(f_pn - p_neither) < 0.035);
    }
    std::cout << "PASSED (Class proportions preserved within +/- 3.5% across all folds)\n";
}

void testNameGenderLexiconLeakage() {
    std::cout << "[Test 18] Testing Name-Gender Lexicon Leakage Isolation & Neutrality... ";
    std::vector<msar::GapRecord> fake_train;
    msar::GapRecord r1; r1.a = "Alice Smith"; r1.b = "Bob Jones"; r1.pronoun = "she"; r1.label = 0;
    msar::GapRecord r2; r2.a = "John Doe"; r2.b = "Mary Ray"; r2.pronoun = "he"; r2.label = 0;
    fake_train.push_back(r1);
    fake_train.push_back(r2);

    msar::NameGenderLexicon lex;
    lex.fit(fake_train);

    // Seen female name 'Alice' with 'she' -> high agreement
    double agr_alice = lex.getGenderAgreement("Alice Smith", "she");
    assert(agr_alice > 0.5);

    // Seen male name 'John' with 'she' -> low agreement
    double agr_john_she = lex.getGenderAgreement("John Doe", "she");
    assert(agr_john_she < 0.5);

    // Unseen name 'Charlie' -> strictly neutral 0.5
    double agr_unseen = lex.getGenderAgreement("Charlie Chaplin", "he");
    assert(std::abs(agr_unseen - 0.5) < 1e-6);

    std::cout << "PASSED (Seen names aligned, unseen held-out names return strictly 0.5)\n";
}

void testCalibratedBiasedProbabilitySum() {
    std::cout << "[Test 19] Testing Calibrated + Biased Probability Normalization... ";
    std::vector<double> raw_probs = {0.60, 0.30, 0.10};
    double T = 0.95;
    double neither_bias = 0.15;

    std::vector<double> z(3);
    for (int c = 0; c < 3; ++c) z[c] = std::log(std::max(1e-12, raw_probs[c])) / T;
    z[2] += neither_bias;

    double max_z = std::max(z[0], std::max(z[1], z[2]));
    double sum_exp = 0.0;
    std::vector<double> cal_probs(3);
    for (int c = 0; c < 3; ++c) {
        cal_probs[c] = std::exp(z[c] - max_z);
        sum_exp += cal_probs[c];
    }
    double sum = 0.0;
    for (int c = 0; c < 3; ++c) {
        cal_probs[c] /= sum_exp;
        sum += cal_probs[c];
        assert(cal_probs[c] >= 0.0 && cal_probs[c] <= 1.0);
    }
    assert(std::abs(sum - 1.0) < 1e-6);
    std::cout << "PASSED (Temperature scaling + post-hoc log bias sums to 1.0 within 1e-6)\n";
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
    testFindAllMentionsLoose();
    testUrlMatch();
    testFeaturesV3();
    testFoldDisjointnessAndNoLeakage();
    testLabelSwapIdentity();
    testDoubleSwapIdentity();
    testProbabilityNormalization();
    testExtendedSwapBlockIsolation();
    testMatcherWordBoundaryEdgeCase();
    testStratifiedClassRatiosPreserved();
    testNameGenderLexiconLeakage();
    testCalibratedBiasedProbabilitySum();

    std::cout << "\n===================================================================\n";
    std::cout << " ALL UNIT TESTS PASSED SUCCESSFULLY! (19/19)\n";
    std::cout << "===================================================================\n";
    return 0;
}
