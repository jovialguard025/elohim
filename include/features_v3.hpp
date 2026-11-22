#ifndef MSAR_FEATURES_V3_HPP
#define MSAR_FEATURES_V3_HPP

#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v2.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cctype>

namespace msar {

// URL page title normalization
inline std::string extractWikiTitle(const std::string& url) {
    if (url.empty()) return "";
    size_t last_slash = url.find_last_of('/');
    if (last_slash == std::string::npos || last_slash + 1 >= url.size()) return "";
    std::string title = url.substr(last_slash + 1);

    // Replace underscores with spaces
    for (char& c : title) {
        if (c == '_') c = ' ';
    }

    // Strip disambiguation parentheses e.g. " (UK TV series)"
    size_t paren = title.find('(');
    if (paren != std::string::npos) {
        title = trim(title.substr(0, paren));
    }
    return toLowerString(title);
}

inline bool isCommonVerbToken(const std::string& token) {
    static const std::unordered_set<std::string> verbs = {
        "is", "was", "are", "were", "been", "has", "have", "had", "became", "said",
        "told", "asked", "went", "joined", "began", "started", "played", "served",
        "wrote", "married", "died", "born", "graduated", "worked", "moved", "attended",
        "returned", "called", "received", "won", "published", "left", "entered", "met"
    };
    if (verbs.count(token)) return true;
    if (token.size() >= 3 && token.substr(token.size() - 2) == "ed") return true;
    return false;
}

inline bool isPrepositionToken(const std::string& token) {
    static const std::unordered_set<std::string> preps = {
        "of", "to", "with", "by", "for", "from", "in", "at", "on", "into", "through",
        "against", "during", "before", "after", "about", "between", "under", "without"
    };
    return preps.count(token) > 0;
}

inline std::string getSurname(const std::string& name) {
    auto toks = tokenize(name);
    return toks.empty() ? "" : toks.back();
}

// Extract 40 features for a single candidate (80 features total for candidate pair)
inline std::vector<double> extractFeaturesV3Candidate(
    const std::string& cand_name,
    long long cand_offset,
    const std::string& other_cand_name,
    long long other_cand_offset,
    const std::string& pron_str,
    long long pron_offset,
    const std::string& text,
    const std::string& wiki_url,
    const std::vector<TextSpan>& sents,
    const std::vector<std::string>& all_tokens)
{
    // First 18 features from V2
    auto feat = extractFeaturesV2Candidate(cand_name, cand_offset, pron_str, pron_offset, text, sents, all_tokens);
    feat.reserve(40);

    std::string pron_lower = toLowerString(pron_str);
    std::string cand_lower = toLowerString(cand_name);
    std::string other_lower = toLowerString(other_cand_name);

    size_t s_cand = findSentenceIndex(sents, cand_offset);
    size_t s_pron = findSentenceIndex(sents, pron_offset);

    // 19. Pronoun gender features (2 features)
    bool pron_is_female = (pron_lower == "she" || pron_lower == "her" || pron_lower == "hers" || pron_lower == "herself");
    bool pron_is_male = (pron_lower == "he" || pron_lower == "his" || pron_lower == "him" || pron_lower == "himself");
    feat.push_back(pron_is_female ? 1.0 : 0.0);
    feat.push_back(pron_is_male ? 1.0 : 0.0);

    // 20. Pronoun grammatical case features (4 features)
    bool pron_is_nom = (pron_lower == "he" || pron_lower == "she");
    bool pron_is_acc = (pron_lower == "him");
    bool pron_is_poss = (pron_lower == "his" || pron_lower == "hers");
    bool pron_is_her = (pron_lower == "her");
    feat.push_back(pron_is_nom ? 1.0 : 0.0);
    feat.push_back(pron_is_acc ? 1.0 : 0.0);
    feat.push_back(pron_is_poss ? 1.0 : 0.0);
    feat.push_back(pron_is_her ? 1.0 : 0.0); // Ambiguous her indicator

    // 21. Honorific Gender Mismatch (1 feature)
    bool cand_has_male_hon = (cand_lower.find("mr.") != std::string::npos || cand_lower.find("mr ") != std::string::npos ||
                              cand_lower.find("sir ") != std::string::npos || cand_lower.find("lord ") != std::string::npos);
    bool cand_has_fem_hon  = (cand_lower.find("mrs.") != std::string::npos || cand_lower.find("mrs ") != std::string::npos ||
                              cand_lower.find("ms.") != std::string::npos || cand_lower.find("ms ") != std::string::npos ||
                              cand_lower.find("miss ") != std::string::npos || cand_lower.find("lady ") != std::string::npos);
    double hon_mismatch = ((cand_has_male_hon && pron_is_female) || (cand_has_fem_hon && pron_is_male)) ? 1.0 : 0.0;
    feat.push_back(hon_mismatch);

    // 22. Wikipedia Page-Subject Alignment (from URL metadata) (3 features)
    std::string wiki_title = extractWikiTitle(wiki_url);
    double page_exact_match = 0.0;
    double page_surname_match = 0.0;
    double page_substring_match = 0.0;
    if (!wiki_title.empty()) {
        if (wiki_title == cand_lower) page_exact_match = 1.0;
        std::string cand_sur = getSurname(cand_lower);
        std::string page_sur = getSurname(wiki_title);
        if (!cand_sur.empty() && cand_sur == page_sur) page_surname_match = 1.0;
        if (wiki_title.find(cand_lower) != std::string::npos || cand_lower.find(wiki_title) != std::string::npos) {
            page_substring_match = 1.0;
        }
    }
    feat.push_back(page_exact_match);
    feat.push_back(page_surname_match);
    feat.push_back(page_substring_match);

    // 23. Syntactic-Position Heuristics (6 features)
    size_t cand_sent_start = sents[s_cand].start;
    double is_sent_initial = (cand_offset - static_cast<long long>(cand_sent_start) <= 5) ? 1.0 : 0.0;
    feat.push_back(is_sent_initial);

    std::string text_before = (cand_offset > 0) ? text.substr(0, static_cast<size_t>(cand_offset)) : "";
    auto toks_before = tokenize(text_before);
    double is_prep_obj = (!toks_before.empty() && isPrepositionToken(toks_before.back())) ? 1.0 : 0.0;
    feat.push_back(is_prep_obj);

    size_t cand_end_off = static_cast<size_t>(cand_offset) + cand_name.size();
    std::string text_after = (cand_end_off < text.size()) ? text.substr(cand_end_off) : "";
    auto toks_after = tokenize(text_after);
    double is_post_verb = (!toks_after.empty() && isCommonVerbToken(toks_after.front())) ? 1.0 : 0.0;
    feat.push_back(is_post_verb);

    double in_parens_or_appositive = 0.0;
    if (cand_offset >= 2 && cand_end_off + 2 <= text.size()) {
        char before_c = text[cand_offset - 1];
        char after_c = text[cand_end_off];
        if ((before_c == '(' && after_c == ')') || (before_c == ',' && after_c == ',')) {
            in_parens_or_appositive = 1.0;
        }
    }
    feat.push_back(in_parens_or_appositive);

    double has_possessive_s = 0.0;
    if (cand_end_off + 2 <= text.size()) {
        std::string s_check = text.substr(cand_end_off, 2);
        if (s_check == "'s" || s_check == "’s") has_possessive_s = 1.0;
    }
    feat.push_back(has_possessive_s);

    size_t pron_sent_start = sents[s_pron].start;
    double pron_is_initial = (pron_offset - static_cast<long long>(pron_sent_start) <= 5) ? 1.0 : 0.0;
    double parallelism = (is_sent_initial > 0.5 && pron_is_initial > 0.5) ? 1.0 : 0.0;
    feat.push_back(parallelism);

    // 24. Mention Structure (3 features)
    double is_sub_of_other = (other_lower.find(cand_lower) != std::string::npos || cand_lower.find(other_lower) != std::string::npos) ? 1.0 : 0.0;
    feat.push_back(is_sub_of_other);

    double shared_surname = (getSurname(cand_lower) == getSurname(other_lower) && !getSurname(cand_lower).empty()) ? 1.0 : 0.0;
    feat.push_back(shared_surname);

    long long lo = std::min(cand_offset, pron_offset);
    long long hi = std::max(cand_offset, pron_offset);
    double competing_entities = 0.0;
    if (hi > lo + static_cast<long long>(cand_name.size())) {
        std::string between_text = text.substr(static_cast<size_t>(lo), static_cast<size_t>(hi - lo));
        bool in_word = false;
        bool is_cap = false;
        for (char c : between_text) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                if (!in_word) {
                    in_word = true;
                    is_cap = std::isupper(static_cast<unsigned char>(c));
                }
            } else {
                if (in_word && is_cap) competing_entities += 1.0;
                in_word = false;
                is_cap = false;
            }
        }
        if (in_word && is_cap) competing_entities += 1.0;
    }
    feat.push_back(competing_entities / 10.0);

    // 25. Discourse & Salience (3 features)
    double is_first_mention_in_doc = (cand_offset < 100) ? 1.0 : 0.0;
    feat.push_back(is_first_mention_in_doc);

    double closer_than_other = (std::abs(cand_offset - pron_offset) < std::abs(other_cand_offset - pron_offset)) ? 1.0 : 0.0;
    feat.push_back(closer_than_other);

    auto cand_mentions = findAllMentions(text, cand_name, cand_offset);
    auto other_mentions = findAllMentions(text, other_cand_name, other_cand_offset);
    double more_frequent_than_other = (cand_mentions.size() > other_mentions.size()) ? 1.0 : 0.0;
    feat.push_back(more_frequent_than_other);

    return feat;
}

inline std::vector<double> extractFeaturesV3(const GapRecord& rec) {
    auto sents = getSentences(rec.text);
    auto all_tokens = tokenize(rec.text);

    auto feat_a = extractFeaturesV3Candidate(
        rec.a, rec.a_offset, rec.b, rec.b_offset,
        rec.pronoun, rec.pronoun_offset, rec.text, rec.url,
        sents, all_tokens
    );

    auto feat_b = extractFeaturesV3Candidate(
        rec.b, rec.b_offset, rec.a, rec.a_offset,
        rec.pronoun, rec.pronoun_offset, rec.text, rec.url,
        sents, all_tokens
    );

    std::vector<double> combined;
    combined.reserve(feat_a.size() + feat_b.size());
    combined.insert(combined.end(), feat_a.begin(), feat_a.end());
    combined.insert(combined.end(), feat_b.begin(), feat_b.end());
    return combined;
}

} // namespace msar

#endif // MSAR_FEATURES_V3_HPP
