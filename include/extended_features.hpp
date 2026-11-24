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
#include <utility>
#include <cctype>
#include <numeric>
#include <fstream>
#include <stdexcept>

namespace msar {

// ===================================================================
// URL Extraction & Parsing (Wiki Title, Surname, Entity Match)
// ===================================================================
inline bool containsTokenSequence(const std::string& text, const std::string& phrase) {
    const auto text_tokens = tokenize(text);
    const auto phrase_tokens = tokenize(phrase);
    if (phrase_tokens.empty() || phrase_tokens.size() > text_tokens.size()) return false;
    for (size_t start = 0; start + phrase_tokens.size() <= text_tokens.size(); ++start) {
        bool match = true;
        for (size_t offset = 0; offset < phrase_tokens.size(); ++offset) {
            if (text_tokens[start + offset] != phrase_tokens[offset]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

inline void checkUrlFeatures(const std::string& url, const std::string& cand_name,
                             double& has_mention, double& is_page_entity,
                             bool strict_boundaries = false)
{
    has_mention = 0.0;
    is_page_entity = 0.0;
    std::string title = extractWikiTitle(url);
    if (title.empty() || cand_name.empty()) return;

    std::string cand_lower = toLowerString(cand_name);
    auto cand_toks = tokenize(cand_lower);
    if (cand_toks.empty()) return;
    std::string surname = cand_toks.back();

    const bool full_name_match = strict_boundaries
        ? containsTokenSequence(title, cand_lower)
        : title.find(cand_lower) != std::string::npos;
    const bool surname_match = strict_boundaries
        ? containsTokenSequence(title, surname)
        : title.find(surname) != std::string::npos;
    if (full_name_match || surname_match) {
        has_mention = 1.0;
    }

    const auto title_toks = tokenize(title);
    const bool title_is_name = title_toks == cand_toks;
    const bool title_is_surname = title_toks.size() == 1 && title_toks.front() == surname;
    const bool title_starts_with_name = !cand_toks.empty() &&
        title_toks.size() >= cand_toks.size() &&
        std::equal(cand_toks.begin(), cand_toks.end(), title_toks.begin());
    if (strict_boundaries ? (title_is_name || title_is_surname || title_starts_with_name)
                          : (title == cand_lower || title == surname || title.find(cand_lower) == 0)) {
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

// Fold-fitted unigram association scores for the lexical path between a
// candidate mention and its pronoun. Training-row features use leave-one-ID
// out counts so the row's own label cannot define its feature value.
struct ContextPathLexicon {
    struct RecordCounts {
        std::unordered_map<std::string, size_t> positive_tokens;
        std::unordered_map<std::string, size_t> negative_tokens;
        size_t positive_examples = 0;
        size_t negative_examples = 0;
    };

    std::unordered_map<std::string, size_t> positive_counts;
    std::unordered_map<std::string, size_t> negative_counts;
    std::unordered_map<std::string, RecordCounts> per_record;
    size_t positive_token_total = 0;
    size_t negative_token_total = 0;
    size_t positive_examples = 0;
    size_t negative_examples = 0;

    static std::vector<std::string> pathTokens(const GapRecord& rec,
                                               const std::string& name,
                                               long long name_offset) {
        const long long p = rec.pronoun_offset;
        const long long n = name_offset;
        long long lo = 0;
        long long hi = 0;
        if (n <= p) {
            lo = n + static_cast<long long>(name.size());
            hi = p;
        } else {
            lo = p + static_cast<long long>(rec.pronoun.size());
            hi = n;
        }
        lo = std::max<long long>(0, std::min<long long>(lo, static_cast<long long>(rec.text.size())));
        hi = std::max<long long>(lo, std::min<long long>(hi, static_cast<long long>(rec.text.size())));
        auto tokens = tokenize(rec.text.substr(static_cast<size_t>(lo), static_cast<size_t>(hi - lo)));
        static const std::unordered_set<std::string> stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "than", "as", "at", "by", "for",
            "from", "in", "into", "of", "on", "onto", "to", "with", "without", "about", "after",
            "before", "between", "during", "through", "while", "although", "because", "that", "which",
            "who", "whom", "whose", "what", "when", "where", "why", "how", "he", "she", "him", "her",
            "his", "hers", "it", "they", "them", "their", "this", "these", "those", "was", "were", "is",
            "are", "be", "been", "being", "has", "have", "had", "do", "does", "did", "will", "would",
            "could", "should", "can", "may", "might", "must"
        };
        const auto name_tokens = tokenize(name);
        const auto other_name_tokens = tokenize(name == rec.a ? rec.b : rec.a);
        std::unordered_set<std::string> excluded_name_tokens(name_tokens.begin(), name_tokens.end());
        excluded_name_tokens.insert(other_name_tokens.begin(), other_name_tokens.end());
        std::vector<std::string> filtered;
        std::unordered_set<std::string> seen;
        for (const auto& token : tokens) {
            if (token.size() < 3 || stopwords.count(token) || excluded_name_tokens.count(token)) continue;
            if (seen.insert(token).second) filtered.push_back(token);
        }
        return filtered;
    }

    void fit(const std::vector<GapRecord>& records) {
        positive_counts.clear();
        negative_counts.clear();
        per_record.clear();
        positive_token_total = negative_token_total = 0;
        positive_examples = negative_examples = 0;

        for (const auto& rec : records) {
            auto& record_counts = per_record[rec.id];
            const std::string names[2] = {rec.a, rec.b};
            const long long offsets[2] = {rec.a_offset, rec.b_offset};
            for (int candidate = 0; candidate < 2; ++candidate) {
                const bool positive = rec.label == candidate;
                const auto tokens = pathTokens(rec, names[candidate], offsets[candidate]);
                if (positive) {
                    positive_examples++;
                    record_counts.positive_examples++;
                    for (const auto& token : tokens) {
                        positive_counts[token]++;
                        record_counts.positive_tokens[token]++;
                        positive_token_total++;
                    }
                } else {
                    negative_examples++;
                    record_counts.negative_examples++;
                    for (const auto& token : tokens) {
                        negative_counts[token]++;
                        record_counts.negative_tokens[token]++;
                        negative_token_total++;
                    }
                }
            }
        }
    }

    double score(const GapRecord& rec, const std::string& name, long long name_offset) const {
        const auto tokens = pathTokens(rec, name, name_offset);
        if (tokens.empty()) return 0.0;
        const size_t vocabulary = std::max<size_t>(1, positive_counts.size() + negative_counts.size());
        size_t pos_total = positive_token_total;
        size_t neg_total = negative_token_total;
        size_t pos_examples = positive_examples;
        size_t neg_examples = negative_examples;
        const RecordCounts* excluded = nullptr;
        auto record_it = per_record.find(rec.id);
        if (record_it != per_record.end()) {
            excluded = &record_it->second;
            pos_total -= std::min(pos_total, static_cast<size_t>(std::accumulate(
                excluded->positive_tokens.begin(), excluded->positive_tokens.end(), size_t(0),
                [](size_t sum, const auto& item) { return sum + item.second; })));
            neg_total -= std::min(neg_total, static_cast<size_t>(std::accumulate(
                excluded->negative_tokens.begin(), excluded->negative_tokens.end(), size_t(0),
                [](size_t sum, const auto& item) { return sum + item.second; })));
            pos_examples -= std::min(pos_examples, excluded->positive_examples);
            neg_examples -= std::min(neg_examples, excluded->negative_examples);
        }
        (void)pos_examples;
        (void)neg_examples;

        double total = 0.0;
        for (const auto& token : tokens) {
            size_t pos = 0;
            size_t neg = 0;
            auto p_it = positive_counts.find(token);
            if (p_it != positive_counts.end()) pos = p_it->second;
            auto n_it = negative_counts.find(token);
            if (n_it != negative_counts.end()) neg = n_it->second;
            if (excluded) {
                auto ep = excluded->positive_tokens.find(token);
                if (ep != excluded->positive_tokens.end()) pos -= std::min(pos, ep->second);
                auto en = excluded->negative_tokens.find(token);
                if (en != excluded->negative_tokens.end()) neg -= std::min(neg, en->second);
            }
            const double p_pos = (static_cast<double>(pos) + 1.0) /
                                 (static_cast<double>(pos_total) + static_cast<double>(vocabulary));
            const double p_neg = (static_cast<double>(neg) + 1.0) /
                                 (static_cast<double>(neg_total) + static_cast<double>(vocabulary));
            total += std::log(p_pos) - std::log(p_neg);
        }
        return total / std::sqrt(static_cast<double>(tokens.size()));
    }
};

struct FoldIdfLexicon {
    std::unordered_map<std::string, size_t> document_frequency;
    size_t document_count = 0;

    void fit(const std::vector<GapRecord>& records) {
        document_frequency.clear();
        document_count = records.size();
        for (const auto& rec : records) {
            const auto tokens = tokenize(rec.text);
            const std::unordered_set<std::string> unique(tokens.begin(), tokens.end());
            for (const auto& token : unique) document_frequency[token]++;
        }
    }

    void save(const std::string& path) const {
        std::ofstream out(path);
        if (!out) throw std::runtime_error("Unable to save fold IDF lexicon: " + path);
        std::vector<std::string> tokens;
        tokens.reserve(document_frequency.size());
        for (const auto& item : document_frequency) tokens.push_back(item.first);
        std::sort(tokens.begin(), tokens.end());
        out << document_count << ' ' << tokens.size() << '\n';
        for (const auto& token : tokens) out << token << ' ' << document_frequency.at(token) << '\n';
    }

    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("Unable to load fold IDF lexicon: " + path);
        size_t count = 0;
        in >> document_count >> count;
        document_frequency.clear();
        for (size_t i = 0; i < count; ++i) {
            std::string token;
            size_t frequency = 0;
            in >> token >> frequency;
            if (!in) throw std::runtime_error("Malformed fold IDF lexicon: " + path);
            document_frequency[token] = frequency;
        }
    }

    double idf(const std::string& token) const {
        const auto it = document_frequency.find(token);
        const double df = it == document_frequency.end() ? 0.0 : static_cast<double>(it->second);
        return std::log((static_cast<double>(document_count) + 1.0) / (df + 1.0)) + 1.0;
    }

    double sentenceCosine(const GapRecord& rec, long long candidate_offset) const {
        const auto sentences = getSentences(rec.text);
        const size_t cand_sentence = findSentenceIndex(sentences, candidate_offset);
        const size_t pron_sentence = findSentenceIndex(sentences, rec.pronoun_offset);
        const auto cand_tokens = tokenize(rec.text.substr(sentences[cand_sentence].start,
            sentences[cand_sentence].end - sentences[cand_sentence].start));
        const auto pron_tokens = tokenize(rec.text.substr(sentences[pron_sentence].start,
            sentences[pron_sentence].end - sentences[pron_sentence].start));
        std::unordered_map<std::string, double> cand_weights;
        std::unordered_map<std::string, double> pron_weights;
        for (const auto& token : cand_tokens) cand_weights[token] += idf(token);
        for (const auto& token : pron_tokens) pron_weights[token] += idf(token);
        double dot = 0.0;
        double cand_norm = 0.0;
        double pron_norm = 0.0;
        for (const auto& item : cand_weights) {
            cand_norm += item.second * item.second;
            const auto it = pron_weights.find(item.first);
            if (it != pron_weights.end()) dot += item.second * it->second;
        }
        for (const auto& item : pron_weights) pron_norm += item.second * item.second;
        const double denom = std::sqrt(cand_norm * pron_norm);
        return denom > 0.0 ? dot / denom : 0.0;
    }

    double localContextCosine(const GapRecord& rec, const std::string& candidate_name,
                              long long candidate_offset) const {
        const auto mentions = findAllMentionsLoose(rec.text, candidate_name, candidate_offset);
        long long nearest_prior = candidate_offset;
        for (long long mention : mentions) {
            if (mention < rec.pronoun_offset && mention > nearest_prior) nearest_prior = mention;
        }
        constexpr size_t window = 90;
        const size_t cand_start = nearest_prior > static_cast<long long>(window)
            ? static_cast<size_t>(nearest_prior) - window : 0;
        const size_t cand_end = std::min(rec.text.size(),
            static_cast<size_t>(std::max<long long>(0, nearest_prior)) + candidate_name.size() + window);
        const size_t pron_start = rec.pronoun_offset > static_cast<long long>(window)
            ? static_cast<size_t>(rec.pronoun_offset) - window : 0;
        const size_t pron_end = std::min(rec.text.size(),
            static_cast<size_t>(std::max<long long>(0, rec.pronoun_offset)) + rec.pronoun.size() + window);
        const auto cand_tokens = tokenize(rec.text.substr(cand_start, cand_end - cand_start));
        const auto pron_tokens = tokenize(rec.text.substr(pron_start, pron_end - pron_start));
        std::unordered_set<std::string> excluded;
        for (const auto& token : tokenize(rec.a)) excluded.insert(token);
        for (const auto& token : tokenize(rec.b)) excluded.insert(token);
        static const std::unordered_set<std::string> function_words = {
            "a","an","the","and","or","but","if","of","to","in","on","at","by","for","from",
            "with","as","is","are","was","were","be","been","being","has","have","had","do","does",
            "did","he","him","his","she","her","hers","it","they","them","their","this","that"
        };
        auto make_weights = [&](const std::vector<std::string>& tokens) {
            std::unordered_map<std::string, double> out;
            for (const auto& token : tokens) {
                if (token.size() < 3 || excluded.count(token) || function_words.count(token)) continue;
                out[token] += idf(token);
            }
            return out;
        };
        const auto cand_weights = make_weights(cand_tokens);
        const auto pron_weights = make_weights(pron_tokens);
        double dot = 0.0, cand_norm = 0.0, pron_norm = 0.0;
        for (const auto& item : cand_weights) {
            cand_norm += item.second * item.second;
            const auto it = pron_weights.find(item.first);
            if (it != pron_weights.end()) dot += item.second * it->second;
        }
        for (const auto& item : pron_weights) pron_norm += item.second * item.second;
        const double denom = std::sqrt(cand_norm * pron_norm);
        return denom > 0.0 ? dot / denom : 0.0;
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
    bool strict_url_boundaries = false;
    bool use_recent_syntax = false; // +6 features per candidate at the nearest prior mention
    bool use_role_features = false; // +6 shallow grammatical-role cues per candidate
    bool use_path_lexicon = false; // +1 fold-fitted lexical-path association per candidate
    bool use_tfidf_overlap = false; // +1 fold-fitted IDF-weighted sentence overlap per candidate
    bool use_local_tfidf_overlap = false; // +1 fold-fitted IDF cosine of local mention/pronoun windows
    bool use_title_gender = false; // +1 explicit title/pronoun gender agreement cue per candidate
    bool use_him_distance_interaction = false; // +1 candidate distance interaction for accusative "him"
    bool use_subject_for_object_pronoun = false; // +2 subject-role cues when pronoun is an object form
    bool use_prior_pronoun_links = false; // +2 candidate cues for prior gender-compatible pronoun chains
    bool use_nearby_prior_pronoun_chain = false; // +2 recency/count cues around prior compatible pronouns

    size_t getCandidateFeatureCount() const {
        size_t c = 18; // Base V2
        if (use_url) c += 2;
        if (use_syntax) c += 6;
        if (use_relative) c += 6;
        if (use_lexical) c += 16;
        if (use_gender) c += 1;
        if (use_recent_syntax) c += 6;
        if (use_role_features) c += 6;
        if (use_path_lexicon) c += 1;
        if (use_tfidf_overlap) c += 1;
        if (use_local_tfidf_overlap) c += 1;
        if (use_title_gender) c += 1;
        if (use_him_distance_interaction) c += 1;
        if (use_subject_for_object_pronoun) c += 2;
        if (use_prior_pronoun_links) c += 2;
        if (use_nearby_prior_pronoun_chain) c += 2;
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

inline std::string roleWordBefore(const std::string& text, size_t offset, size_t lower_bound) {
    size_t end = std::min(offset, text.size());
    while (end > lower_bound && !std::isalnum(static_cast<unsigned char>(text[end - 1]))) --end;
    size_t start = end;
    while (start > lower_bound && std::isalnum(static_cast<unsigned char>(text[start - 1]))) --start;
    if (start == end) return "";
    return toLowerString(text.substr(start, end - start));
}

inline std::string roleWordAfter(const std::string& text, size_t offset, size_t upper_bound) {
    size_t start = std::min(offset, text.size());
    upper_bound = std::min(upper_bound, text.size());
    while (start < upper_bound && !std::isalnum(static_cast<unsigned char>(text[start]))) ++start;
    size_t end = start;
    while (end < upper_bound && std::isalnum(static_cast<unsigned char>(text[end]))) ++end;
    if (start == end) return "";
    return toLowerString(text.substr(start, end - start));
}

inline bool isCommonFiniteVerb(const std::string& word) {
    static const std::unordered_set<std::string> verbs = {
        "am", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
        "do", "does", "did", "says", "said", "say", "tells", "told", "tell", "asks", "asked",
        "ask", "calls", "called", "call", "finds", "found", "find", "meets", "met", "meet",
        "marries", "married", "marry", "wins", "won", "win", "loses", "lost", "lose", "plays",
        "played", "play", "works", "worked", "work", "serves", "served", "serve", "uses", "used",
        "use", "creates", "created", "create", "produces", "produced", "produce", "visits", "visited",
        "visit", "leads", "led", "lead", "joins", "joined", "join", "helps", "helped", "help",
        "announces", "announced", "announce", "reports", "reported", "report", "believes", "believed",
        "believe", "thinks", "thought", "think", "knows", "knew", "know", "takes", "took", "take",
        "gives", "gave", "give", "writes", "wrote", "write", "dies", "died", "die", "becomes", "became",
        "become", "moves", "moved", "move", "begins", "began", "begin", "continues", "continued", "continue",
        "appears", "appeared", "appear", "returns", "returned", "return", "remains", "remained", "remain",
        "seems", "seemed", "seem", "lives", "lived", "live", "leaves", "left", "leave", "runs", "ran", "run"
    };
    return verbs.count(word) > 0;
}

inline bool isCommonPreposition(const std::string& word) {
    static const std::unordered_set<std::string> preps = {
        "of", "with", "to", "by", "for", "from", "about", "against", "at", "on", "in", "into",
        "over", "under", "after", "before", "through", "during", "between", "among", "without", "within"
    };
    return preps.count(word) > 0;
}

inline std::string wordImmediatelyBefore(const std::string& text, long long offset) {
    if (offset <= 0) return "";
    size_t end = std::min(static_cast<size_t>(offset), text.size());
    while (end > 0 && !std::isalnum(static_cast<unsigned char>(text[end - 1]))) --end;
    size_t start = end;
    while (start > 0 && std::isalpha(static_cast<unsigned char>(text[start - 1]))) --start;
    return toLowerString(text.substr(start, end - start));
}

inline double explicitTitlePronounAgreement(const GapRecord& rec, long long candidate_offset) {
    const std::string title = wordImmediatelyBefore(rec.text, candidate_offset);
    static const std::unordered_set<std::string> male_titles = {
        "mr", "mister", "sir", "lord", "king", "prince", "duke", "father", "brother"
    };
    static const std::unordered_set<std::string> female_titles = {
        "mrs", "ms", "miss", "lady", "queen", "princess", "dame", "mother", "sister"
    };
    const bool title_male = male_titles.count(title) > 0;
    const bool title_female = female_titles.count(title) > 0;
    if (!title_male && !title_female) return 0.0;

    const std::string pronoun = toLowerString(rec.pronoun);
    const bool pronoun_male = pronoun == "he" || pronoun == "him" || pronoun == "his";
    const bool pronoun_female = pronoun == "she" || pronoun == "her" || pronoun == "hers";
    if (!pronoun_male && !pronoun_female) return 0.0;
    return (title_male == pronoun_male) ? 1.0 : -1.0;
}

inline std::vector<double> extractPriorPronounLinkFeatures(
    const std::string& candidate_name,
    long long candidate_offset,
    const GapRecord& rec,
    const std::vector<TextSpan>& sentences)
{
    const std::string target = toLowerString(rec.pronoun);
    const bool masculine = target == "he" || target == "him" || target == "his";
    const bool feminine = target == "she" || target == "her" || target == "hers";
    if (!masculine && !feminine) return {0.0, 0.0};

    static const std::unordered_set<std::string> masculine_pronouns = {"he", "him", "his"};
    static const std::unordered_set<std::string> feminine_pronouns = {"she", "her", "hers"};
    const auto& compatible = masculine ? masculine_pronouns : feminine_pronouns;
    std::vector<std::pair<size_t, size_t>> prior_pronouns;
    for (size_t i = 0; i < rec.text.size();) {
        if (!std::isalpha(static_cast<unsigned char>(rec.text[i]))) { ++i; continue; }
        const size_t start = i;
        while (i < rec.text.size() && std::isalpha(static_cast<unsigned char>(rec.text[i]))) ++i;
        if (static_cast<long long>(start) >= rec.pronoun_offset) break;
        if (compatible.count(toLowerString(rec.text.substr(start, i - start))) > 0)
            prior_pronouns.emplace_back(start, i);
    }

    const auto mentions = findAllMentionsLoose(rec.text, candidate_name, candidate_offset);
    const size_t target_sentence = findSentenceIndex(sentences, rec.pronoun_offset);
    bool linked = false;
    double most_recent_link = 0.0;
    for (const auto& pron : prior_pronouns) {
        const size_t pron_sentence = findSentenceIndex(sentences, static_cast<long long>(pron.first));
        bool candidate_precedes = false;
        for (long long mention : mentions) {
            if (mention >= static_cast<long long>(pron.first)) break;
            if (findSentenceIndex(sentences, mention) == pron_sentence) {
                candidate_precedes = true;
                break;
            }
        }
        if (candidate_precedes) {
            linked = true;
            const size_t sentence_gap = target_sentence >= pron_sentence
                ? target_sentence - pron_sentence : 0;
            most_recent_link = std::max(most_recent_link,
                                        1.0 / (1.0 + static_cast<double>(sentence_gap)));
        }
    }
    return {linked ? 1.0 : 0.0, most_recent_link};
}

inline std::vector<double> extractNearbyPriorPronounChainFeatures(
    const std::string& candidate_name,
    long long candidate_offset,
    const GapRecord& rec,
    const std::vector<TextSpan>& sentences)
{
    const std::string target = toLowerString(rec.pronoun);
    const bool masculine = target == "he" || target == "him" || target == "his";
    const bool feminine = target == "she" || target == "her" || target == "hers";
    if (!masculine && !feminine) return {0.0, 0.0};
    static const std::unordered_set<std::string> masculine_pronouns = {"he", "him", "his"};
    static const std::unordered_set<std::string> feminine_pronouns = {"she", "her", "hers"};
    const auto& compatible = masculine ? masculine_pronouns : feminine_pronouns;

    std::vector<std::pair<size_t, size_t>> prior_pronouns;
    for (size_t i = 0; i < rec.text.size();) {
        if (!std::isalpha(static_cast<unsigned char>(rec.text[i]))) { ++i; continue; }
        const size_t start = i;
        while (i < rec.text.size() && std::isalpha(static_cast<unsigned char>(rec.text[i]))) ++i;
        if (static_cast<long long>(start) >= rec.pronoun_offset) break;
        if (compatible.count(toLowerString(rec.text.substr(start, i - start))) > 0)
            prior_pronouns.emplace_back(start, i);
    }

    const auto mentions = findAllMentionsLoose(rec.text, candidate_name, candidate_offset);
    double closest_link_score = 0.0;
    double linked_pronoun_count = 0.0;
    for (const auto& pron : prior_pronouns) {
        long long latest_mention = -1;
        for (long long mention : mentions) {
            if (mention >= static_cast<long long>(pron.first)) break;
            latest_mention = mention;
        }
        if (latest_mention < 0) continue;
        const size_t mention_sentence = findSentenceIndex(sentences, latest_mention);
        const size_t pronoun_sentence = findSentenceIndex(sentences, static_cast<long long>(pron.first));
        if (pronoun_sentence < mention_sentence || pronoun_sentence - mention_sentence > 1) continue;
        const double char_gap = static_cast<double>(static_cast<long long>(pron.first) - latest_mention);
        const double score = 1.0 / (1.0 + char_gap / 80.0);
        closest_link_score = std::max(closest_link_score, score);
        linked_pronoun_count += 1.0;
    }
    return {closest_link_score, std::log1p(linked_pronoun_count)};
}

inline bool isRoleSubjectPosition(const std::string& text, long long mention_offset,
                                  size_t mention_length, const TextSpan& sentence) {
    if (mention_offset < 0) return false;
    const size_t start = static_cast<size_t>(mention_offset);
    const size_t end = std::min(sentence.end, start + mention_length);
    if (start < sentence.start || end > sentence.end) return false;
    const std::string previous = roleWordBefore(text, start, sentence.start);
    const std::string next = roleWordAfter(text, end, sentence.end);
    const bool subject_start = previous.empty() || previous == "and" || previous == "but" || previous == "or" ||
                               previous == "while" || previous == "although" || previous == "because";
    const bool subject_determiner = previous == "the" || previous == "a" || previous == "an" ||
                                    previous == "this" || previous == "that" || previous == "each";
    return isCommonFiniteVerb(next) && (subject_start || subject_determiner);
}

inline bool isRoleObjectPosition(const std::string& text, long long mention_offset,
                                 size_t mention_length, const TextSpan& sentence) {
    if (mention_offset < 0) return false;
    const size_t start = static_cast<size_t>(mention_offset);
    const size_t end = std::min(sentence.end, start + mention_length);
    if (start < sentence.start || end > sentence.end) return false;
    const std::string previous = roleWordBefore(text, start, sentence.start);
    return isCommonPreposition(previous) || isCommonFiniteVerb(previous);
}

inline std::vector<double> extractCandidateRoleFeatures(
    const std::string& candidate_name,
    long long candidate_offset,
    const GapRecord& rec,
    const std::vector<TextSpan>& sentences)
{
    const size_t candidate_sentence = findSentenceIndex(sentences, candidate_offset);
    const auto mentions = findAllMentionsLoose(rec.text, candidate_name, candidate_offset);
    long long latest_prior = candidate_offset;
    for (long long mention : mentions) {
        if (mention < rec.pronoun_offset && mention > latest_prior) latest_prior = mention;
    }
    const size_t latest_sentence = findSentenceIndex(sentences, latest_prior);
    const bool current_subject = isRoleSubjectPosition(rec.text, candidate_offset,
                                                       candidate_name.size(), sentences[candidate_sentence]);
    const bool current_object = isRoleObjectPosition(rec.text, candidate_offset,
                                                     candidate_name.size(), sentences[candidate_sentence]);
    const bool recent_subject = isRoleSubjectPosition(rec.text, latest_prior,
                                                      candidate_name.size(), sentences[latest_sentence]);
    const bool recent_object = isRoleObjectPosition(rec.text, latest_prior,
                                                    candidate_name.size(), sentences[latest_sentence]);
    const std::string pronoun = toLowerString(rec.pronoun);
    const bool pronoun_subject = pronoun == "he" || pronoun == "she";
    const bool pronoun_object = pronoun == "him" || pronoun == "her";

    return {
        current_subject ? 1.0 : 0.0,
        current_object ? 1.0 : 0.0,
        recent_subject ? 1.0 : 0.0,
        recent_object ? 1.0 : 0.0,
        (current_subject && pronoun_subject) ? 1.0 : 0.0,
        (current_object && pronoun_object) ? 1.0 : 0.0
    };
}

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
    const NameGenderLexicon* lexicon = nullptr,
    const ContextPathLexicon* path_lexicon = nullptr,
    const FoldIdfLexicon* idf_lexicon = nullptr)
{
    auto sents = getSentences(rec.text);
    auto all_tokens = tokenize(rec.text);
    double text_len = static_cast<double>(std::max<size_t>(rec.text.size(), 1));
    double total_tokens = static_cast<double>(std::max<size_t>(all_tokens.size(), 1));

    // Base 18 candidate features
    std::vector<double> feat_a = extractCandidateV2Base(rec.a, rec.a_offset, rec, sents, all_tokens, cfg.use_v2x_matcher);
    std::vector<double> feat_b = extractCandidateV2Base(rec.b, rec.b_offset, rec, sents, all_tokens, cfg.use_v2x_matcher);

    if (cfg.use_him_distance_interaction) {
        const bool is_him = toLowerString(rec.pronoun) == "him";
        feat_a.push_back(is_him ? static_cast<double>(rec.pronoun_offset - rec.a_offset) / text_len : 0.0);
        feat_b.push_back(is_him ? static_cast<double>(rec.pronoun_offset - rec.b_offset) / text_len : 0.0);
    }

    if (cfg.use_subject_for_object_pronoun) {
        const std::string pronoun = toLowerString(rec.pronoun);
        const bool object_pronoun = pronoun == "him" || pronoun == "her";
        auto add_subject_object_cues = [&](const std::string& name, long long offset,
                                           std::vector<double>& candidate_features) {
            const auto role = extractCandidateRoleFeatures(name, offset, rec, sents);
            candidate_features.push_back(object_pronoun ? role[0] : 0.0);
            candidate_features.push_back(object_pronoun ? role[2] : 0.0);
        };
        add_subject_object_cues(rec.a, rec.a_offset, feat_a);
        add_subject_object_cues(rec.b, rec.b_offset, feat_b);
    }

    if (cfg.use_prior_pronoun_links) {
        const auto link_a = extractPriorPronounLinkFeatures(rec.a, rec.a_offset, rec, sents);
        const auto link_b = extractPriorPronounLinkFeatures(rec.b, rec.b_offset, rec, sents);
        feat_a.insert(feat_a.end(), link_a.begin(), link_a.end());
        feat_b.insert(feat_b.end(), link_b.begin(), link_b.end());
    }

    if (cfg.use_nearby_prior_pronoun_chain) {
        const auto chain_a = extractNearbyPriorPronounChainFeatures(rec.a, rec.a_offset, rec, sents);
        const auto chain_b = extractNearbyPriorPronounChainFeatures(rec.b, rec.b_offset, rec, sents);
        feat_a.insert(feat_a.end(), chain_a.begin(), chain_a.end());
        feat_b.insert(feat_b.end(), chain_b.begin(), chain_b.end());
    }

    // g1: URL Match (2 features per candidate: has_mention, is_page_entity)
    if (cfg.use_url) {
        double m_a = 0.0, e_a = 0.0;
        double m_b = 0.0, e_b = 0.0;
        checkUrlFeatures(rec.url, rec.a, m_a, e_a, cfg.strict_url_boundaries);
        checkUrlFeatures(rec.url, rec.b, m_b, e_b, cfg.strict_url_boundaries);
        feat_a.push_back(m_a);
        feat_a.push_back(e_a);
        feat_b.push_back(m_b);
        feat_b.push_back(e_b);
    }

    // Explicit honorifics are sparse but high-precision cues for pronoun gender.
    if (cfg.use_title_gender) {
        feat_a.push_back(explicitTitlePronounAgreement(rec, rec.a_offset));
        feat_b.push_back(explicitTitlePronounAgreement(rec, rec.b_offset));
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

    // g7: repeat the syntax cues at each candidate's nearest prior mention,
    // which may differ from the annotated candidate occurrence.
    if (cfg.use_recent_syntax) {
        auto add_recent_syntax = [&](const std::string& cand_name,
                                     long long cand_offset,
                                     std::vector<double>& feat) {
            auto mentions = findAllMentionsLoose(rec.text, cand_name, cand_offset);
            long long mention_offset = cand_offset;
            bool found_prior = false;
            for (long long m : mentions) {
                if (m < rec.pronoun_offset && (!found_prior || m > mention_offset)) {
                    mention_offset = m;
                    found_prior = true;
                }
            }
            if (mention_offset < 0 || static_cast<size_t>(mention_offset) >= rec.text.size()) {
                mention_offset = std::max<long long>(0, cand_offset);
            }

            auto name_tokens = tokenize(cand_name);
            std::vector<std::string> patterns = {cand_name};
            if (!name_tokens.empty()) {
                patterns.push_back(name_tokens.front());
                patterns.push_back(name_tokens.back());
            }
            std::sort(patterns.begin(), patterns.end(), [](const std::string& a, const std::string& b) {
                return a.size() > b.size();
            });
            patterns.erase(std::unique(patterns.begin(), patterns.end()), patterns.end());
            const std::string text_lower = toLowerString(rec.text);
            size_t mention_length = cand_name.size();
            for (const auto& pattern : patterns) {
                if (pattern.empty() || static_cast<size_t>(mention_offset) + pattern.size() > text_lower.size()) continue;
                const size_t start = static_cast<size_t>(mention_offset);
                const size_t end = start + pattern.size();
                const bool before_boundary = start == 0 ||
                    !std::isalnum(static_cast<unsigned char>(rec.text[start - 1]));
                const bool after_boundary = end == rec.text.size() ||
                    !std::isalnum(static_cast<unsigned char>(rec.text[end]));
                if (before_boundary && after_boundary &&
                    text_lower.compare(start, pattern.size(), toLowerString(pattern)) == 0) {
                    mention_length = pattern.size();
                    break;
                }
            }

            const size_t sent_idx = findSentenceIndex(sents, mention_offset);
            const size_t sent_start = sents[sent_idx].start;
            const size_t sent_end = sents[sent_idx].end;
            const size_t mention_end = std::min(rec.text.size(),
                static_cast<size_t>(mention_offset) + mention_length);

            double is_possessive = 0.0;
            if (mention_end + 2 <= rec.text.size() && rec.text.substr(mention_end, 2) == "'s") {
                is_possessive = 1.0;
            } else if (mention_end + 4 <= rec.text.size() &&
                       rec.text.substr(mention_end, 4) == "\xe2\x80\x99s") {
                is_possessive = 1.0;
            }
            feat.push_back(is_possessive);

            bool prior_capitalized_name = false;
            bool in_word = false;
            size_t word_start = 0;
            for (size_t k = sent_start; k < static_cast<size_t>(mention_offset); ++k) {
                if (std::isalpha(static_cast<unsigned char>(rec.text[k]))) {
                    if (!in_word) { in_word = true; word_start = k; }
                } else if (in_word) {
                    if (std::isupper(static_cast<unsigned char>(rec.text[word_start]))) {
                        const auto word = toLowerString(rec.text.substr(word_start, k - word_start));
                        static const std::unordered_set<std::string> starters = {
                            "the", "a", "an", "in", "on", "at", "after", "before", "during",
                            "however", "meanwhile", "later", "then", "although", "while",
                            "when", "since", "because", "he", "she", "it", "they"
                        };
                        if (!starters.count(word)) { prior_capitalized_name = true; break; }
                    }
                    in_word = false;
                }
            }
            const double is_sentence_first = prior_capitalized_name ? 0.0 : 1.0;
            feat.push_back(is_sentence_first);

            const std::string before = rec.text.substr(0, static_cast<size_t>(mention_offset));
            const auto before_tokens = tokenize(before);
            static const std::unordered_set<std::string> prepositions = {
                "of", "with", "to", "by", "for", "from", "about", "against", "at", "on", "in"
            };
            const bool after_preposition = !before_tokens.empty() && prepositions.count(before_tokens.back());
            feat.push_back(after_preposition ? 1.0 : 0.0);

            size_t open_parens = 0;
            for (size_t k = sent_start; k < static_cast<size_t>(mention_offset); ++k) {
                if (rec.text[k] == '(') ++open_parens;
                else if (rec.text[k] == ')' && open_parens > 0) --open_parens;
            }
            feat.push_back(open_parens > 0 && mention_end <= sent_end ? 1.0 : 0.0);

            const auto pron_lower = toLowerString(rec.pronoun);
            const bool pron_possessive = pron_lower == "his" || pron_lower == "her" || pron_lower == "hers";
            feat.push_back(is_possessive > 0.5 && pron_possessive ? 1.0 : 0.0);

            const bool candidate_initial =
                (mention_offset - static_cast<long long>(sent_start) <= 5) || is_sentence_first > 0.5;
            const size_t pron_sent = findSentenceIndex(sents, rec.pronoun_offset);
            const bool pronoun_initial =
                rec.pronoun_offset - static_cast<long long>(sents[pron_sent].start) <= 5;
            const bool pronoun_subject = pron_lower == "he" || pron_lower == "she";
            feat.push_back(candidate_initial && (pronoun_initial || pronoun_subject) ? 1.0 : 0.0);
        };
        add_recent_syntax(rec.a, rec.a_offset, feat_a);
        add_recent_syntax(rec.b, rec.b_offset, feat_b);
    }

    // g8: Shallow grammatical-role cues around the candidate mention and
    // the closest prior mention. These are candidate-block features so the
    // normal swap augmentation/TTA permutation remains well-defined.
    if (cfg.use_role_features) {
        const auto role_a = extractCandidateRoleFeatures(rec.a, rec.a_offset, rec, sents);
        const auto role_b = extractCandidateRoleFeatures(rec.b, rec.b_offset, rec, sents);
        feat_a.insert(feat_a.end(), role_a.begin(), role_a.end());
        feat_b.insert(feat_b.end(), role_b.begin(), role_b.end());
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

    if (cfg.use_path_lexicon) {
        feat_a.push_back(path_lexicon != nullptr ? path_lexicon->score(rec, rec.a, rec.a_offset) : 0.0);
        feat_b.push_back(path_lexicon != nullptr ? path_lexicon->score(rec, rec.b, rec.b_offset) : 0.0);
    }

    if (cfg.use_tfidf_overlap) {
        feat_a.push_back(idf_lexicon != nullptr ? idf_lexicon->sentenceCosine(rec, rec.a_offset) : 0.0);
        feat_b.push_back(idf_lexicon != nullptr ? idf_lexicon->sentenceCosine(rec, rec.b_offset) : 0.0);
    }

    if (cfg.use_local_tfidf_overlap) {
        feat_a.push_back(idf_lexicon != nullptr ? idf_lexicon->localContextCosine(rec, rec.a, rec.a_offset) : 0.0);
        feat_b.push_back(idf_lexicon != nullptr ? idf_lexicon->localContextCosine(rec, rec.b, rec.b_offset) : 0.0);
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
