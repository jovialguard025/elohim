#ifndef MSAR_PREPROCESSING_HPP
#define MSAR_PREPROCESSING_HPP

#include <string>
#include <vector>
#include <unordered_set>
#include <cctype>
#include <algorithm>
#include <cmath>

namespace msar {

struct TextSpan {
    size_t start;
    size_t end;
};

inline bool isSentenceEndChar(char c) {
    return c == '.' || c == '?' || c == '!';
}

inline std::vector<TextSpan> getSentences(const std::string& text) {
    std::vector<TextSpan> sents;
    size_t start = 0;
    size_t n = text.size();

    for (size_t i = 0; i < n; ++i) {
        if (isSentenceEndChar(text[i])) {
            // Check if boundary is at EOF or followed by whitespace/quote
            if (i + 1 == n || std::isspace(static_cast<unsigned char>(text[i + 1])) || text[i + 1] == '\"' || text[i + 1] == '\'') {
                sents.push_back({start, i + 1});
                size_t next_start = i + 1;
                while (next_start < n && (std::isspace(static_cast<unsigned char>(text[next_start])) || text[next_start] == '\"' || text[next_start] == '\'')) {
                    next_start++;
                }
                start = next_start;
                i = (next_start > 0 ? next_start - 1 : 0);
            }
        }
    }
    if (start < n) {
        sents.push_back({start, n});
    }
    if (sents.empty()) {
        sents.push_back({0, n});
    }
    return sents;
}

inline size_t findSentenceIndex(const std::vector<TextSpan>& sents, long long offset) {
    if (offset < 0 || sents.empty()) return 0;
    size_t off = static_cast<size_t>(offset);
    for (size_t i = 0; i < sents.size(); ++i) {
        if (off >= sents[i].start && off < sents[i].end) {
            return i;
        }
    }
    // If offset is at or beyond last sentence end, assign to last
    if (off >= sents.back().start) {
        return sents.size() - 1;
    }
    return 0;
}

inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current.push_back(std::tolower(static_cast<unsigned char>(c)));
        } else {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

inline double computeJaccard(const std::vector<std::string>& toks1, const std::vector<std::string>& toks2) {
    if (toks1.empty() && toks2.empty()) return 0.0;
    std::unordered_set<std::string> s1(toks1.begin(), toks1.end());
    std::unordered_set<std::string> s2(toks2.begin(), toks2.end());
    
    size_t intersection_sz = 0;
    for (const auto& w : s1) {
        if (s2.count(w)) intersection_sz++;
    }
    size_t union_sz = s1.size() + s2.size() - intersection_sz;
    return union_sz > 0 ? static_cast<double>(intersection_sz) / union_sz : 0.0;
}

inline std::string toLowerString(const std::string& s) {
    std::string res = s;
    std::transform(res.begin(), res.end(), res.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return res;
}

inline std::vector<long long> findAllMentions(const std::string& text, const std::string& phrase, long long fallback_offset) {
    std::vector<long long> mentions;
    if (phrase.empty() || text.empty()) {
        mentions.push_back(fallback_offset);
        return mentions;
    }

    std::string text_lower = toLowerString(text);
    std::string phrase_lower = toLowerString(phrase);

    size_t pos = text_lower.find(phrase_lower, 0);
    while (pos != std::string::npos) {
        mentions.push_back(static_cast<long long>(pos));
        pos = text_lower.find(phrase_lower, pos + 1);
    }

    if (mentions.empty()) {
        mentions.push_back(fallback_offset);
    }
    return mentions;
}

inline size_t countWords(const std::string& text) {
    auto tokens = tokenize(text);
    return tokens.size();
}

} // namespace msar

#endif // MSAR_PREPROCESSING_HPP
