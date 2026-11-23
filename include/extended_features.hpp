#ifndef MSAR_EXTENDED_FEATURES_HPP
#define MSAR_EXTENDED_FEATURES_HPP

#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v2.hpp"
#include "features_v3.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <sstream>

namespace msar {

// ===================================================================
// URL Extraction & Parsing (Wiki Title, Surname, Entity Match)
// ===================================================================
inline void checkUrlFeatures(const std::string& url, const std::string& cand_name,
                             double& has_mention, double& is_page_entity)
{
    has_mention = 0.0;
    is_page_entity = 0.0;
    std::string title = extractWikiTitle(url);
    if (title.empty() || cand_name.empty()) return;

    std::string cand_lower = toLowerString(cand_name);
    auto cand_toks = tokenize(cand_lower);
    if (cand_toks.empty()) return;
    std::string surname = cand_toks.back();

    if (title.find(cand_lower) != std::string::npos) {
        has_mention = 1.0;
    } else if (title.find(surname) != std::string::npos) {
        has_mention = 1.0;
    }

    if (title == cand_lower || title == surname) {
        is_page_entity = 1.0;
    } else if (title.find(cand_lower) == 0) {
        is_page_entity = 1.0;
    }
}

// ===================================================================
// Name-Gender Agreement Lexicon (Fitted on Training Folds Only)
// ===================================================================
struct NameGenderLexicon {
    std::unordered_map<std::string, double> male_counts;
    std::unordered_map<std::string, double> female_counts;

    void fit(const std::vector<GapRecord>& train_records) {
        male_counts.clear();
        female_counts.clear();

        for (const auto& r : train_records) {
            std::string p_lower = toLowerString(r.pronoun);
            bool is_male = (p_lower == "he" || p_lower == "him" || p_lower == "his");
            bool is_fem  = (p_lower == "she" || p_lower == "her" || p_lower == "hers");

            if (!is_male && !is_fem) continue;

            if (r.label == 0) {
                auto toks = tokenize(r.a);
                if (!toks.empty()) {
                    std::string fname = toLowerString(toks.front());
                    if (is_male) male_counts[fname] += 1.0;
                    else female_counts[fname] += 1.0;
                }
            } else if (r.label == 1) {
                auto toks = tokenize(r.b);
                if (!toks.empty()) {
                    std::string fname = toLowerString(toks.front());
                    if (is_male) male_counts[fname] += 1.0;
                    else female_counts[fname] += 1.0;
                }
            }
        }
    }

    double getGenderAgreement(const std::string& name, const std::string& pronoun) const {
        auto toks = tokenize(name);
        if (toks.empty()) return 0.5;
        std::string fname = toLowerString(toks.front());

        double m = 0.0;
        double f = 0.0;
        auto it_m = male_counts.find(fname);
        if (it_m != male_counts.end()) m = it_m->second;
        auto it_f = female_counts.find(fname);
        if (it_f != female_counts.end()) f = it_f->second;

        // Laplace smoothed probability
        double p_male = (m + 1.0) / (m + f + 2.0);

        std::string p_lower = toLowerString(pronoun);
        bool is_male = (p_lower == "he" || p_lower == "him" || p_lower == "his");
        bool is_fem  = (p_lower == "she" || p_lower == "her" || p_lower == "hers");

        if (is_male) return p_male;
        if (is_fem) return 1.0 - p_male;
        return 0.5;
    }
};

// ===================================================================
// Local Lexical Context Hash Bag (16 buckets)
// ===================================================================
inline std::vector<double> extractHashedBag(const std::vector<std::string>& tokens, size_t num_buckets = 16) {
    std::vector<double> bag(num_buckets, 0.0);
    for (const auto& tok : tokens) {
        std::string low = toLowerString(tok);
        unsigned long h = 5381;
        for (char c : low) h = ((h << 5) + h) + static_cast<unsigned char>(c);
        bag[h % num_buckets] += 1.0;
    }
    return bag;
}

// ===================================================================
// Extended Feature Configuration Specification
// Layout: [ Candidate A block (C) | Candidate B block (C) | Shared block (S) ]
// ===================================================================
struct ExtendedConfig {
    bool use_url = false;         // g1: +2 cand
    bool use_syntax = false;      // g2: +6 cand
    bool use_pron_onehot = false; // g3: +6 shared
    bool use_relative = false;    // g4: +6 cand
    bool use_lexical = false;     // g5: +16 cand, +16 shared
    bool use_gender = false;      // g6: +1 cand
    bool use_v2x_matcher = true;

    size_t getCandidateFeatureCount() const {
        size_t c = 18; // Base V2
        if (use_url) c += 2;
        if (use_syntax) c += 6;
        if (use_relative) c += 6;
        if (use_lexical) c += 16;
        if (use_gender) c += 1;
        return c;
    }

    size_t getSharedFeatureCount() const {
        size_t s = 0;
        if (use_pron_onehot) s += 6;
        if (use_lexical) s += 16;
        return s;
    }

    size_t getTotalDimension() const {
        return 2 * getCandidateFeatureCount() + getSharedFeatureCount();
    }
};

// Candidate V2 / V2.x extraction helper
inline std::vector<double> extractCandidateV2Base(
    const std::string& cand_name,
    long long cand_offset,
    const GapRecord& rec,
    const std::vector<TextSpan>& sents,
    const std::vector<std::string>& all_tokens,
    bool use_v2x_matcher)
{
    if (!use_v2x_matcher) {
        return extractFeaturesV2Candidate(cand_name, cand_offset, rec.pronoun, rec.pronoun_offset, rec.text, sents, all_tokens);
    }

    std::vector<double> feat;
    feat.reserve(18);
    double text_len = static_cast<double>(std::max<size_t>(rec.text.size(), 1));
    double total_sents = static_cast<double>(std::max<size_t>(sents.size(), 1));
    double total_tokens = static_cast<double>(std::max<size_t>(all_tokens.size(), 1));

    size_t s_cand = findSentenceIndex(sents, cand_offset);
    size_t s_pron = findSentenceIndex(sents, rec.pronoun_offset);
    long long sent_diff = static_cast<long long>(s_pron) - static_cast<long long>(s_cand);

    auto mentions = findAllMentionsLoose(rec.text, cand_name, cand_offset);

    // 1. Character distance
    feat.push_back(static_cast<double>(rec.pronoun_offset - cand_offset) / text_len);
    // 2. Sentence distance
    feat.push_back(static_cast<double>(sent_diff));
    // 3. Candidate mention frequency
    feat.push_back(static_cast<double>(mentions.size()));
    // 4. Sentence Jaccard overlap
    std::string cand_sent_str = rec.text.substr(sents[s_cand].start, sents[s_cand].end - sents[s_cand].start);
    std::string pron_sent_str = rec.text.substr(sents[s_pron].start, sents[s_pron].end - sents[s_pron].start);
    feat.push_back(computeJaccard(tokenize(cand_sent_str), tokenize(pron_sent_str)));
    // 5. Candidate-name token length
    feat.push_back(static_cast<double>(countWords(cand_name)));
    // 6. Precedes-pronoun indicator
    feat.push_back(cand_offset < rec.pronoun_offset ? 1.0 : 0.0);
    // 7. Normalized word/token distance
    std::string text_before_cand = (cand_offset > 0) ? rec.text.substr(0, static_cast<size_t>(cand_offset)) : "";
    std::string text_before_pron = (rec.pronoun_offset > 0) ? rec.text.substr(0, static_cast<size_t>(rec.pronoun_offset)) : "";
    double tokens_before_cand = static_cast<double>(countWords(text_before_cand));
    double tokens_before_pron = static_cast<double>(countWords(text_before_pron));
    feat.push_back((tokens_before_pron - tokens_before_cand) / total_tokens);
    // 8. Same-sentence indicator
    feat.push_back(s_cand == s_pron ? 1.0 : 0.0);
    // 9. Immediately-preceding-sentence indicator
    feat.push_back(sent_diff == 1 ? 1.0 : 0.0);
    // 10. Normalized sentence gap
    feat.push_back(std::abs(static_cast<double>(sent_diff)) / total_sents);

    // 11-15. Mention statistics
    double mentions_before = 0.0;
    double mentions_after = 0.0;
    double min_dist_val = text_len;
    std::vector<long long> prior_mentions;

    for (long long m : mentions) {
        if (m < rec.pronoun_offset) {
            mentions_before += 1.0;
            prior_mentions.push_back(m);
        } else if (m > rec.pronoun_offset) {
            mentions_after += 1.0;
        }
        double dist = std::abs(static_cast<double>(m - rec.pronoun_offset));
        if (dist < min_dist_val) min_dist_val = dist;
    }
    feat.push_back(mentions_before);
    feat.push_back(mentions_after);
    feat.push_back(min_dist_val / text_len);
    feat.push_back(prior_mentions.empty() ? -1.0 : (static_cast<double>(prior_mentions.front()) / text_len));
    feat.push_back(prior_mentions.empty() ? -1.0 : (static_cast<double>(rec.pronoun_offset - prior_mentions.back()) / text_len));

    // 16. Local-context Jaccard overlap
    size_t cand_start_win = (cand_offset >= 60) ? static_cast<size_t>(cand_offset - 60) : 0;
    size_t cand_end_win = std::min<size_t>(rec.text.size(), static_cast<size_t>(cand_offset) + cand_name.size() + 60);
    size_t pron_start_win = (rec.pronoun_offset >= 60) ? static_cast<size_t>(rec.pronoun_offset - 60) : 0;
    size_t pron_end_win = std::min<size_t>(rec.text.size(), static_cast<size_t>(rec.pronoun_offset) + rec.pronoun.size() + 60);
    feat.push_back(computeJaccard(tokenize(rec.text.substr(cand_start_win, cand_end_win - cand_start_win)),
                                  tokenize(rec.text.substr(pron_start_win, pron_end_win - pron_start_win))));

    // 17. Candidate mention in the pronoun sentence
    double in_pron_sent = 0.0;
    for (long long m : mentions) {
        if (static_cast<size_t>(m) >= sents[s_pron].start && static_cast<size_t>(m) < sents[s_pron].end) {
            in_pron_sent = 1.0;
            break;
        }
    }
    feat.push_back(in_pron_sent);

    // 18. Mentions strictly between candidate and pronoun
    long long lo = std::min(cand_offset, rec.pronoun_offset);
    long long hi = std::max(cand_offset, rec.pronoun_offset);
    double mentions_between = 0.0;
    for (long long m : mentions) {
        if (m > lo && m < hi) mentions_between += 1.0;
    }
    feat.push_back(mentions_between);

    return feat;
}

// Master Feature Extraction by Configuration
inline std::vector<double> extractFeaturesConfigured(
    const GapRecord& rec,
    const ExtendedConfig& cfg,
    const NameGenderLexicon* lexicon = nullptr)
{
    auto sents = getSentences(rec.text);
    auto all_tokens = tokenize(rec.text);
    double text_len = static_cast<double>(std::max<size_t>(rec.text.size(), 1));
    double total_tokens = static_cast<double>(std::max<size_t>(all_tokens.size(), 1));

    // Base 18 candidate features
    std::vector<double> feat_a = extractCandidateV2Base(rec.a, rec.a_offset, rec, sents, all_tokens, cfg.use_v2x_matcher);
    std::vector<double> feat_b = extractCandidateV2Base(rec.b, rec.b_offset, rec, sents, all_tokens, cfg.use_v2x_matcher);

    // g1: URL Match (2 features per candidate: has_mention, is_page_entity)
    if (cfg.use_url) {
        double m_a = 0.0, e_a = 0.0;
        double m_b = 0.0, e_b = 0.0;
        checkUrlFeatures(rec.url, rec.a, m_a, e_a);
        checkUrlFeatures(rec.url, rec.b, m_b, e_b);
        feat_a.push_back(m_a);
        feat_a.push_back(e_a);
        feat_b.push_back(m_b);
        feat_b.push_back(e_b);
    }

    // g2: Syntactic Heuristics (6 features per candidate)
    if (cfg.use_syntax) {
        auto add_syntax = [&](const std::string& cand_name, long long cand_offset, std::vector<double>& feat) {
            size_t s_cand = findSentenceIndex(sents, cand_offset);
            size_t s_pron = findSentenceIndex(sents, rec.pronoun_offset);

            // 1. Possessive candidate
            size_t cand_end = static_cast<size_t>(cand_offset) + cand_name.size();
            double is_poss = 0.0;
            if (cand_end + 2 <= rec.text.size() && rec.text.substr(cand_end, 2) == "'s") {
                is_poss = 1.0;
            } else if (cand_end + 4 <= rec.text.size() && rec.text.substr(cand_end, 4) == "\xe2\x80\x99s") {
                is_poss = 1.0;
            } else if (cand_end + 3 <= rec.text.size() && rec.text.substr(cand_end, 3) == "’s") {
                is_poss = 1.0;
            }
            feat.push_back(is_poss);

            // 2. Sentence-first-capitalized-name (Subject heuristic)
            double is_sent_first = 0.0;
            if (cand_offset >= static_cast<long long>(sents[s_cand].start)) {
                size_t sent_st = sents[s_cand].start;
                bool prior_cap_word = false;
                bool in_w = false;
                size_t w_start = 0;
                for (size_t k = sent_st; k < static_cast<size_t>(cand_offset); ++k) {
                    if (std::isalpha(static_cast<unsigned char>(rec.text[k]))) {
                        if (!in_w) { in_w = true; w_start = k; }
                    } else {
                        if (in_w) {
                            if (std::isupper(static_cast<unsigned char>(rec.text[w_start]))) {
                                std::string word = rec.text.substr(w_start, k - w_start);
                                std::string word_l = toLowerString(word);
                                static const std::unordered_set<std::string> sent_starters = {
                                    "the", "a", "an", "in", "on", "at", "after", "before", "during",
                                    "however", "meanwhile", "later", "then", "although", "while",
                                    "when", "since", "because", "he", "she", "it", "they"
                                };
                                if (sent_starters.count(word_l) == 0) {
                                    prior_cap_word = true;
                                    break;
                                }
                            }
                            in_w = false;
                        }
                    }
                }
                if (!prior_cap_word) is_sent_first = 1.0;
            }
            feat.push_back(is_sent_first);

            // 3. Preceding-preposition indicator
            static const std::unordered_set<std::string> preps = {
                "of", "with", "to", "by", "for", "from", "about", "against", "at", "on", "in"
            };
            std::string text_before = (cand_offset > 0) ? rec.text.substr(0, static_cast<size_t>(cand_offset)) : "";
            auto toks_before = tokenize(text_before);
            double after_prep = (!toks_before.empty() && preps.count(toks_before.back()) > 0) ? 1.0 : 0.0;
            feat.push_back(after_prep);

            // 4. Inside parentheses
            double inside_parens = 0.0;
            if (cand_offset >= static_cast<long long>(sents[s_cand].start) &&
                cand_end <= sents[s_cand].end)
            {
                size_t open_count = 0;
                for (size_t k = sents[s_cand].start; k < static_cast<size_t>(cand_offset); ++k) {
                    if (rec.text[k] == '(') open_count++;
                    else if (rec.text[k] == ')') { if (open_count > 0) open_count--; }
                }
                if (open_count > 0) inside_parens = 1.0;
            }
            feat.push_back(inside_parens);

            // 5. Parallel_possessive
            std::string pron_lower = toLowerString(rec.pronoun);
            bool pron_poss = (pron_lower == "his" || pron_lower == "her" || pron_lower == "hers");
            feat.push_back((is_poss > 0.5 && pron_poss) ? 1.0 : 0.0);

            // 6. Parallel_subject
            bool cand_initial = (cand_offset - static_cast<long long>(sents[s_cand].start) <= 5) || (is_sent_first > 0.5);
            bool pron_initial = (rec.pronoun_offset - static_cast<long long>(sents[s_pron].start) <= 5);
            bool pron_subj = (pron_lower == "he" || pron_lower == "she");
            feat.push_back((cand_initial && (pron_initial || pron_subj)) ? 1.0 : 0.0);
        };
        add_syntax(rec.a, rec.a_offset, feat_a);
        add_syntax(rec.b, rec.b_offset, feat_b);
    }

    // g4: Relative Candidate Features (6 features per candidate)
    if (cfg.use_relative) {
        double dist_a = std::abs(static_cast<double>(rec.pronoun_offset - rec.a_offset));
        double dist_b = std::abs(static_cast<double>(rec.pronoun_offset - rec.b_offset));

        // Diff in char dist (normalized)
        feat_a.push_back((dist_b - dist_a) / text_len);
        feat_b.push_back((dist_a - dist_b) / text_len);

        // Diff in token distance
        std::string tb_a = (rec.a_offset > 0) ? rec.text.substr(0, static_cast<size_t>(rec.a_offset)) : "";
        std::string tb_b = (rec.b_offset > 0) ? rec.text.substr(0, static_cast<size_t>(rec.b_offset)) : "";
        std::string tb_p = (rec.pronoun_offset > 0) ? rec.text.substr(0, static_cast<size_t>(rec.pronoun_offset)) : "";
        double tdist_a = std::abs(static_cast<double>(countWords(tb_p) - countWords(tb_a)));
        double tdist_b = std::abs(static_cast<double>(countWords(tb_p) - countWords(tb_b)));
        feat_a.push_back((tdist_b - tdist_a) / total_tokens);
        feat_b.push_back((tdist_a - tdist_b) / total_tokens);

        // Diff in sentence distance
        size_t s_a = findSentenceIndex(sents, rec.a_offset);
        size_t s_b = findSentenceIndex(sents, rec.b_offset);
        size_t s_p = findSentenceIndex(sents, rec.pronoun_offset);
        double sdiff_a = std::abs(static_cast<double>(s_p) - static_cast<double>(s_a));
        double sdiff_b = std::abs(static_cast<double>(s_p) - static_cast<double>(s_b));
        feat_a.push_back(sdiff_b - sdiff_a);
        feat_b.push_back(sdiff_a - sdiff_b);

        // Mention count difference
        auto mentions_a = findAllMentionsLoose(rec.text, rec.a, rec.a_offset);
        auto mentions_b = findAllMentionsLoose(rec.text, rec.b, rec.b_offset);
        double cnt_a = static_cast<double>(mentions_a.size());
        double cnt_b = static_cast<double>(mentions_b.size());
        feat_a.push_back(cnt_a - cnt_b);
        feat_b.push_back(cnt_b - cnt_a);

        // Rank by proximity (1.0 if closer, 0.0 otherwise)
        feat_a.push_back(dist_a < dist_b ? 1.0 : (dist_a == dist_b ? 0.5 : 0.0));
        feat_b.push_back(dist_b < dist_a ? 1.0 : (dist_b == dist_a ? 0.5 : 0.0));

        // Rank by mention count (1.0 if more mentions, 0.0 otherwise)
        feat_a.push_back(cnt_a > cnt_b ? 1.0 : (cnt_a == cnt_b ? 0.5 : 0.0));
        feat_b.push_back(cnt_b > cnt_a ? 1.0 : (cnt_b == cnt_a ? 0.5 : 0.0));
    }

    // g5: Local Lexical Context (16 buckets per candidate for 3 tokens after nearest mention)
    std::vector<double> shared_lexical;
    if (cfg.use_lexical) {
        auto get_after_tokens = [&](long long offset, size_t name_len) {
            size_t end_pos = static_cast<size_t>(offset) + name_len;
            if (end_pos >= rec.text.size()) return std::vector<std::string>();
            std::string after_str = rec.text.substr(end_pos, std::min<size_t>(80, rec.text.size() - end_pos));
            auto toks = tokenize(after_str);
            if (toks.size() > 3) toks.resize(3);
            return toks;
        };

        // Candidate A nearest mention tokens
        auto mentions_a = findAllMentionsLoose(rec.text, rec.a, rec.a_offset);
        long long nearest_a = rec.a_offset;
        double min_d_a = text_len;
        for (long long m : mentions_a) {
            double d = std::abs(static_cast<double>(m - rec.pronoun_offset));
            if (d < min_d_a) { min_d_a = d; nearest_a = m; }
        }
        auto bag_a = extractHashedBag(get_after_tokens(nearest_a, rec.a.size()), 16);
        feat_a.insert(feat_a.end(), bag_a.begin(), bag_a.end());

        // Candidate B nearest mention tokens
        auto mentions_b = findAllMentionsLoose(rec.text, rec.b, rec.b_offset);
        long long nearest_b = rec.b_offset;
        double min_d_b = text_len;
        for (long long m : mentions_b) {
            double d = std::abs(static_cast<double>(m - rec.pronoun_offset));
            if (d < min_d_b) { min_d_b = d; nearest_b = m; }
        }
        auto bag_b = extractHashedBag(get_after_tokens(nearest_b, rec.b.size()), 16);
        feat_b.insert(feat_b.end(), bag_b.begin(), bag_b.end());

        // Pronoun 3 tokens after (for shared block)
        auto pron_after = get_after_tokens(rec.pronoun_offset, rec.pronoun.size());
        shared_lexical = extractHashedBag(pron_after, 16);
    }

    // g6: Name-Gender Agreement Lexicon (1 feature per candidate)
    if (cfg.use_gender) {
        double g_a = (lexicon != nullptr) ? lexicon->getGenderAgreement(rec.a, rec.pronoun) : 0.5;
        double g_b = (lexicon != nullptr) ? lexicon->getGenderAgreement(rec.b, rec.pronoun) : 0.5;
        feat_a.push_back(g_a);
        feat_b.push_back(g_b);
    }

    // Assemble final vector: [ Candidate A (C) | Candidate B (C) | Shared (S) ]
    std::vector<double> combined;
    combined.reserve(feat_a.size() + feat_b.size() + 30);
    combined.insert(combined.end(), feat_a.begin(), feat_a.end());
    combined.insert(combined.end(), feat_b.begin(), feat_b.end());

    // g3: Pronoun-Type One-Hot (6 shared features)
    if (cfg.use_pron_onehot) {
        std::string p = toLowerString(rec.pronoun);
        combined.push_back(p == "he"   ? 1.0 : 0.0);
        combined.push_back(p == "she"  ? 1.0 : 0.0);
        combined.push_back(p == "him"  ? 1.0 : 0.0);
        combined.push_back(p == "his"  ? 1.0 : 0.0);
        combined.push_back(p == "her"  ? 1.0 : 0.0);
        combined.push_back(p == "hers" ? 1.0 : 0.0);
    }

    // g5: Local Lexical Shared Bag (16 shared features)
    if (cfg.use_lexical) {
        combined.insert(combined.end(), shared_lexical.begin(), shared_lexical.end());
    }

    return combined;
}

// Specialized swap function for layout [Candidate A (C) | Candidate B (C) | Shared (S)]
inline std::vector<double> swapExtendedCandidatesVector(const std::vector<double>& x, size_t cand_feat_count) {
    size_t total = x.size();
    size_t shared_count = (total > 2 * cand_feat_count) ? (total - 2 * cand_feat_count) : 0;
    std::vector<double> swapped(total);

    // Swap A block and B block
    for (size_t i = 0; i < cand_feat_count; ++i) {
        swapped[i] = x[cand_feat_count + i];
        swapped[cand_feat_count + i] = x[i];
    }
    // Copy shared block untouched
    for (size_t i = 0; i < shared_count; ++i) {
        swapped[2 * cand_feat_count + i] = x[2 * cand_feat_count + i];
    }
    return swapped;
}

} // namespace msar

#endif // MSAR_EXTENDED_FEATURES_HPP