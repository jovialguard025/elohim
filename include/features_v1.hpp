#ifndef MSAR_FEATURES_V1_HPP
#define MSAR_FEATURES_V1_HPP

#include "data_loader.hpp"
#include "preprocessing.hpp"
#include <vector>
#include <cmath>

namespace msar {

inline std::vector<double> extractFeaturesV1Candidate(
    const std::string& cand_name,
    long long cand_offset,
    const std::string& pron_str,
    long long pron_offset,
    const std::string& text,
    const std::vector<TextSpan>& sents)
{
    std::vector<double> feat;
    feat.reserve(6);
    (void)pron_str;

    double text_len = static_cast<double>(std::max<size_t>(text.size(), 1));
    double total_sents = static_cast<double>(std::max<size_t>(sents.size(), 1));

    size_t s_cand = findSentenceIndex(sents, cand_offset);
    size_t s_pron = findSentenceIndex(sents, pron_offset);

    // 1. Normalized character distance between candidate and pronoun
    double char_dist = static_cast<double>(pron_offset - cand_offset) / text_len;
    feat.push_back(char_dist);

    // 2. Normalized sentence-boundary distance
    double sent_dist = static_cast<double>(static_cast<long long>(s_pron) - static_cast<long long>(s_cand)) / total_sents;
    feat.push_back(sent_dist);

    // 3. Candidate mention frequency (exact phrase matching)
    auto mentions = findAllMentions(text, cand_name, cand_offset);
    feat.push_back(static_cast<double>(mentions.size()));

    // 4. Candidate/pronoun sentence Jaccard similarity
    std::string cand_sent_str = text.substr(sents[s_cand].start, sents[s_cand].end - sents[s_cand].start);
    std::string pron_sent_str = text.substr(sents[s_pron].start, sents[s_pron].end - sents[s_pron].start);
    auto toks_cand_sent = tokenize(cand_sent_str);
    auto toks_pron_sent = tokenize(pron_sent_str);
    double jaccard = computeJaccard(toks_cand_sent, toks_pron_sent);
    feat.push_back(jaccard);

    // 5. Candidate-name word count
    double name_word_count = static_cast<double>(countWords(cand_name));
    feat.push_back(name_word_count);

    // 6. Whether candidate appears before the pronoun (indicator)
    double precedes = (cand_offset < pron_offset) ? 1.0 : 0.0;
    feat.push_back(precedes);

    return feat;
}

inline std::vector<double> extractFeaturesV1(const GapRecord& rec) {
    auto sents = getSentences(rec.text);
    auto feat_a = extractFeaturesV1Candidate(rec.a, rec.a_offset, rec.pronoun, rec.pronoun_offset, rec.text, sents);
    auto feat_b = extractFeaturesV1Candidate(rec.b, rec.b_offset, rec.pronoun, rec.pronoun_offset, rec.text, sents);

    std::vector<double> combined;
    combined.reserve(12);
    combined.insert(combined.end(), feat_a.begin(), feat_a.end());
    combined.insert(combined.end(), feat_b.begin(), feat_b.end());
    return combined;
}

} // namespace msar

#endif // MSAR_FEATURES_V1_HPP
