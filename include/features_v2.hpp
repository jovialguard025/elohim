#ifndef MSAR_FEATURES_V2_HPP
#define MSAR_FEATURES_V2_HPP

#include "data_loader.hpp"
#include "preprocessing.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace msar {

inline std::vector<double> extractFeaturesV2Candidate(
    const std::string& cand_name,
    long long cand_offset,
    const std::string& pron_str,
    long long pron_offset,
    const std::string& text,
    const std::vector<TextSpan>& sents,
    const std::vector<std::string>& all_tokens)
{
    std::vector<double> feat;
    feat.reserve(18);

    double text_len = static_cast<double>(std::max<size_t>(text.size(), 1));
    double total_sents = static_cast<double>(std::max<size_t>(sents.size(), 1));
    double total_tokens = static_cast<double>(std::max<size_t>(all_tokens.size(), 1));

    size_t s_cand = findSentenceIndex(sents, cand_offset);
    size_t s_pron = findSentenceIndex(sents, pron_offset);

    long long sent_diff = static_cast<long long>(s_pron) - static_cast<long long>(s_cand);

    // Mentions
    auto mentions = findAllMentions(text, cand_name, cand_offset);

    // 1. Character distance
    double char_dist = static_cast<double>(pron_offset - cand_offset) / text_len;
    feat.push_back(char_dist);

    // 2. Sentence distance
    double sent_dist = static_cast<double>(sent_diff);
    feat.push_back(sent_dist);

    // 3. Candidate mention frequency
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

    // 16. Local-context Jaccard overlap (window of 60 chars around candidate and pronoun)
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

    // 18. Mentions between the selected occurrence and the pronoun
    long long lo = std::min(cand_offset, pron_offset);
    long long hi = std::max(cand_offset, pron_offset);
    double mentions_between = 0.0;
    for (long long m : mentions) {
        if (m > lo && m < hi) {
            mentions_between += 1.0;
        }
    }
    feat.push_back(mentions_between);

    return feat;
}

inline std::vector<double> extractFeaturesV2(const GapRecord& rec) {
    auto sents = getSentences(rec.text);
    auto all_tokens = tokenize(rec.text);

    auto feat_a = extractFeaturesV2Candidate(rec.a, rec.a_offset, rec.pronoun, rec.pronoun_offset, rec.text, sents, all_tokens);
    auto feat_b = extractFeaturesV2Candidate(rec.b, rec.b_offset, rec.pronoun, rec.pronoun_offset, rec.text, sents, all_tokens);

    std::vector<double> combined;
    combined.reserve(36);
    combined.insert(combined.end(), feat_a.begin(), feat_a.end());
    combined.insert(combined.end(), feat_b.begin(), feat_b.end());
    return combined;
}

} // namespace msar

#endif // MSAR_FEATURES_V2_HPP
