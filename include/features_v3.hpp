#ifndef MSAR_FEATURES_V3_HPP
#define MSAR_FEATURES_V3_HPP

#include "data_loader.hpp"
#include "preprocessing.hpp"
#include "features_v2.hpp"

#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <sstream>
#include <iomanip>

namespace msar {

constexpr size_t kFeatureCountV3 = 56;

inline std::string percentDecode(const std::string& str) {
    std::string res;
    res.reserve(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '%' && i + 2 < str.size() &&
            std::isxdigit(static_cast<unsigned char>(str[i+1])) &&
            std::isxdigit(static_cast<unsigned char>(str[i+2]))) {
            int value = 0;
            std::istringstream iss(str.substr(i + 1, 2));
            iss >> std::hex >> value;
            res.push_back(static_cast<char>(value));
            i += 2;
        } else {
            res.push_back(str[i]);
        }
    }
    return res;
}

inline std::string extractWikiTitle(const std::string& url) {
    if (url.empty()) return "";
    size_t wiki_pos = url.find("/wiki/");
    std::string raw_title;
    if (wiki_pos != std::string::npos) {
        raw_title = url.substr(wiki_pos + 6);
    } else {
        size_t last_slash = url.find_last_of('/');
        if (last_slash != std::string::npos && last_slash + 1 < url.size()) {
            raw_title = url.substr(last_slash + 1);
        } else {
            raw_title = url;
        }
    }
    for (char& c : raw_title) {
        if (c == '_') c = ' ';
    }
    std::string decoded = percentDecode(raw_title);
    return toLowerString(decoded);
}

inline std::string getSurname(const std::string& name) {
    auto toks = tokenize(name);
    return toks.empty() ? "" : toks.back();
}

inline bool checkUrlMatch(const std::string& wiki_url, const std::string& cand_name) {
    std::string title = extractWikiTitle(wiki_url);
    if (title.empty() || cand_name.empty()) return false;
    std::string cand_lower = toLowerString(cand_name);
    std::string sur_lower = toLowerString(getSurname(cand_name));
    if (title.find(cand_lower) != std::string::npos) return true;
    if (!sur_lower.empty() && sur_lower.size() >= 3 && title.find(sur_lower) != std::string::npos) return true;
    return false;
}

inline std::vector<double> extractFeaturesV3Candidate(
    const std::string& cand_name,
    long long cand_offset,
    const std::string& pron_str,
    long long pron_offset,
    const std::string& text,
    const std::string& wiki_url,
    const std::vector<TextSpan>& sents,
    const std::vector<std::string>& all_tokens)
{
    std::vector<double> feat;
    feat.reserve(25);

    double text_len = static_cast<double>(std::max<size_t>(text.size(), 1));
    double total_sents = static_cast<double>(std::max<size_t>(sents.size(), 1));
    double total_tokens = static_cast<double>(std::max<size_t>(all_tokens.size(), 1));

    size_t s_cand = findSentenceIndex(sents, cand_offset);
    size_t s_pron = findSentenceIndex(sents, pron_offset);
    long long sent_diff = static_cast<long long>(s_pron) - static_cast<long long>(s_cand);

    // V3 uses findAllMentionsLoose
    auto mentions = findAllMentionsLoose(text, cand_name, cand_offset);

    // 1. Character distance
    double char_dist = static_cast<double>(pron_offset - cand_offset) / text_len;
    feat.push_back(char_dist);

    // 2. Sentence distance
    double sent_dist = static_cast<double>(sent_diff);
    feat.push_back(sent_dist);

    // 3. Candidate mention frequency (with loose mentions)
    double mention_freq = static_cast<double>(mentions.size());
    feat.push_back(mention_freq);

    // 4. Candidate/pronoun sentence Jaccard overlap
    std::string cand_sent_str = text.substr(sents[s_cand].start, sents[s_cand].end - sents[s_cand].start);
    std::string pron_sent_str = text.substr(sents[s_pron].start, sents[s_pron].end - sents[s_pron].start);
    auto toks_cand_sent = tokenize(cand_sent_str);
    auto toks_pron_sent = tokenize(pron_sent_str);
    double jaccard_sent = computeJaccard(toks_cand_sent, toks_pron_sent);
    feat.push_back(jaccard_sent);

    // 5. Candidate-name token length
    double name_tok_len = static_cast<double>(countWords(cand_name));
    feat.push_back(name_tok_len);

    // 6. Precedes-pronoun indicator
    double precedes = (cand_offset < pron_offset) ? 1.0 : 0.0;
    feat.push_back(precedes);

    // 7. Normalized word/token distance
    std::string text_before_cand = (cand_offset > 0) ? text.substr(0, static_cast<size_t>(cand_offset)) : "";
    std::string text_before_pron = (pron_offset > 0) ? text.substr(0, static_cast<size_t>(pron_offset)) : "";
    double tokens_before_cand = static_cast<double>(countWords(text_before_cand));
    double tokens_before_pron = static_cast<double>(countWords(text_before_pron));
    double token_dist = (tokens_before_pron - tokens_before_cand) / total_tokens;
    feat.push_back(token_dist);

    // 8. Same-sentence indicator
    double same_sent = (s_cand == s_pron) ? 1.0 : 0.0;
    feat.push_back(same_sent);

    // 9. Immediately-preceding-sentence indicator
    double immed_prev = (sent_diff == 1) ? 1.0 : 0.0;
    feat.push_back(immed_prev);

    // 10. Normalized sentence gap
    double norm_sent_gap = std::abs(static_cast<double>(sent_diff)) / total_sents;
    feat.push_back(norm_sent_gap);

    // 11. Mentions before the pronoun
    double mentions_before = 0.0;
    // 12. Mentions after the pronoun
    double mentions_after = 0.0;
    double min_dist_val = text_len;
    std::vector<long long> prior_mentions;

    for (long long m : mentions) {
        if (m < pron_offset) {
            mentions_before += 1.0;
            prior_mentions.push_back(m);
        } else if (m > pron_offset) {
            mentions_after += 1.0;
        }
        double dist = std::abs(static_cast<double>(m - pron_offset));
        if (dist < min_dist_val) {
            min_dist_val = dist;
        }
    }
    feat.push_back(mentions_before);
    feat.push_back(mentions_after);

    // 13. Distance to the nearest mention
    double nearest_mention_dist = min_dist_val / text_len;
    feat.push_back(nearest_mention_dist);

    // 14. Position of the first mention before the pronoun
    double first_mention_pos = prior_mentions.empty() ? -1.0 : (static_cast<double>(prior_mentions.front()) / text_len);
    feat.push_back(first_mention_pos);

    // 15. Distance from the last prior mention
    double last_prior_dist = prior_mentions.empty() ? -1.0 : (static_cast<double>(pron_offset - prior_mentions.back()) / text_len);
    feat.push_back(last_prior_dist);

    // 16. Local-context Jaccard overlap (window of 60 chars)
    size_t cand_start_win = (cand_offset >= 60) ? static_cast<size_t>(cand_offset - 60) : 0;
    size_t cand_end_win = std::min<size_t>(text.size(), static_cast<size_t>(cand_offset) + cand_name.size() + 60);
    std::string cand_win_str = text.substr(cand_start_win, cand_end_win - cand_start_win);

    size_t pron_start_win = (pron_offset >= 60) ? static_cast<size_t>(pron_offset - 60) : 0;
    size_t pron_end_win = std::min<size_t>(text.size(), static_cast<size_t>(pron_offset) + pron_str.size() + 60);
    std::string pron_win_str = text.substr(pron_start_win, pron_end_win - pron_start_win);

    auto toks_cand_win = tokenize(cand_win_str);
    auto toks_pron_win = tokenize(pron_win_str);
    double local_jaccard = computeJaccard(toks_cand_win, toks_pron_win);
    feat.push_back(local_jaccard);

    // 17. Candidate mention in the pronoun sentence
    double in_pron_sent = 0.0;
    size_t p_sent_start = sents[s_pron].start;
    size_t p_sent_end = sents[s_pron].end;
    for (long long m : mentions) {
        if (static_cast<size_t>(m) >= p_sent_start && static_cast<size_t>(m) < p_sent_end) {
            in_pron_sent = 1.0;
            break;
        }
    }
    feat.push_back(in_pron_sent);

    // 18. Mentions between candidate and pronoun
    long long lo = std::min(cand_offset, pron_offset);
    long long hi = std::max(cand_offset, pron_offset);
    double mentions_between = 0.0;
    for (long long m : mentions) {
        if (m > lo && m < hi) {
            mentions_between += 1.0;
        }
    }
    feat.push_back(mentions_between);

    // --- 7 New Candidate Features ---

    // 19 (a). url_match
    bool url_hit = checkUrlMatch(wiki_url, cand_name);
    feat.push_back(url_hit ? 1.0 : 0.0);

    // 20 (b). is_sentence_first_mention (subject heuristic)
    double is_sent_first = 0.0;
    if (cand_offset >= static_cast<long long>(sents[s_cand].start)) {
        size_t sent_st = sents[s_cand].start;
        bool prior_cap_word = false;
        bool in_w = false;
        size_t w_start = 0;
        for (size_t k = sent_st; k < static_cast<size_t>(cand_offset); ++k) {
            if (std::isalpha(static_cast<unsigned char>(text[k]))) {
                if (!in_w) {
                    in_w = true;
                    w_start = k;
                }
            } else {
                if (in_w) {
                    if (std::isupper(static_cast<unsigned char>(text[w_start]))) {
                        std::string word = text.substr(w_start, k - w_start);
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
        if (!prior_cap_word) {
            is_sent_first = 1.0;
        }
    }
    feat.push_back(is_sent_first);

    // 21 (c). is_possessive ("'s" or "’s")
    size_t cand_end = static_cast<size_t>(cand_offset) + cand_name.size();
    double is_poss = 0.0;
    if (cand_end + 2 <= text.size() && text.substr(cand_end, 2) == "'s") {
        is_poss = 1.0;
    } else if (cand_end + 4 <= text.size() && text.substr(cand_end, 4) == "\xe2\x80\x99s") {
        is_poss = 1.0;
    } else if (cand_end + 3 <= text.size() && text.substr(cand_end, 3) == "’s") {
        is_poss = 1.0;
    }
    feat.push_back(is_poss);

    // 22 (d). after_preposition
    static const std::unordered_set<std::string> preps = {
        "of", "with", "to", "by", "for", "from", "about", "against", "at", "on", "in"
    };
    std::string text_before = (cand_offset > 0) ? text.substr(0, static_cast<size_t>(cand_offset)) : "";
    auto toks_before = tokenize(text_before);
    double after_prep = (!toks_before.empty() && preps.count(toks_before.back()) > 0) ? 1.0 : 0.0;
    feat.push_back(after_prep);

    // 23 (e). parallel_possessive
    std::string pron_lower = toLowerString(pron_str);
    bool pron_poss = (pron_lower == "his" || pron_lower == "her" || pron_lower == "hers");
    double parallel_poss = (is_poss > 0.5 && pron_poss) ? 1.0 : 0.0;
    feat.push_back(parallel_poss);

    // 24 (f). parallel_subject
    bool cand_initial = (cand_offset - static_cast<long long>(sents[s_cand].start) <= 5) || (is_sent_first > 0.5);
    bool pron_initial = (pron_offset - static_cast<long long>(sents[s_pron].start) <= 5);
    bool pron_subj = (pron_lower == "he" || pron_lower == "she");
    double parallel_subj = (cand_initial && (pron_initial || pron_subj)) ? 1.0 : 0.0;
    feat.push_back(parallel_subj);

    // 25 (g). candidate_in_parenthesis
    double in_paren = 0.0;
    size_t s_st = sents[s_cand].start;
    size_t s_en = sents[s_cand].end;
    bool has_open_paren = false;
    for (long long p = cand_offset - 1; p >= static_cast<long long>(s_st); --p) {
        if (text[p] == ')') break;
        if (text[p] == '(') {
            has_open_paren = true;
            break;
        }
    }
    if (has_open_paren) {
        for (size_t p = cand_end; p < s_en; ++p) {
            if (text[p] == '(') break;
            if (text[p] == ')') {
                in_paren = 1.0;
                break;
            }
        }
    }
    feat.push_back(in_paren);

    return feat;
}

inline std::vector<double> extractFeaturesV3(const GapRecord& rec) {
    auto sents = getSentences(rec.text);
    auto all_tokens = tokenize(rec.text);

    auto feat_a = extractFeaturesV3Candidate(
        rec.a, rec.a_offset, rec.pronoun, rec.pronoun_offset,
        rec.text, rec.url, sents, all_tokens
    );

    auto feat_b = extractFeaturesV3Candidate(
        rec.b, rec.b_offset, rec.pronoun, rec.pronoun_offset,
        rec.text, rec.url, sents, all_tokens
    );

    std::vector<double> combined;
    combined.reserve(kFeatureCountV3);
    combined.insert(combined.end(), feat_a.begin(), feat_a.end());
    combined.insert(combined.end(), feat_b.begin(), feat_b.end());

    // Plus shared features: pronoun-type one-hot (he, she, him, his, her, hers)
    std::string p = toLowerString(rec.pronoun);
    combined.push_back(p == "he"   ? 1.0 : 0.0);
    combined.push_back(p == "she"  ? 1.0 : 0.0);
    combined.push_back(p == "him"  ? 1.0 : 0.0);
    combined.push_back(p == "his"  ? 1.0 : 0.0);
    combined.push_back(p == "her"  ? 1.0 : 0.0);
    combined.push_back(p == "hers" ? 1.0 : 0.0);

    return combined;
}

} // namespace msar

#endif // MSAR_FEATURES_V3_HPP
