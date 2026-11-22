#ifndef MSAR_DATA_LOADER_HPP
#define MSAR_DATA_LOADER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace msar {

struct GapRecord {
    std::string id;
    std::string text;
    std::string pronoun;
    long long pronoun_offset;
    std::string a;
    long long a_offset;
    bool a_coref;
    std::string b;
    long long b_offset;
    bool b_coref;
    std::string url;
    int label; // 0: A, 1: B, 2: Neither
};

struct PartitionStats {
    size_t total_samples;
    size_t count_a;
    size_t count_b;
    size_t count_neither;
};

inline std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

inline bool parseBool(const std::string& s) {
    std::string lower = trim(s);
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower == "true" || lower == "1";
}

inline std::vector<GapRecord> loadGapDataset(const std::string& tsv_path) {
    std::ifstream file(tsv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open TSV file: " + tsv_path);
    }

    std::vector<GapRecord> records;
    std::string line;
    
    // Read header line
    if (!std::getline(file, line)) {
        return records;
    }

    // Parse header to find column indices
    std::vector<std::string> headers;
    std::stringstream header_ss(line);
    std::string header_col;
    while (std::getline(header_ss, header_col, '\t')) {
        headers.push_back(trim(header_col));
    }

    auto getColIndex = [&](const std::string& col_name) -> int {
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == col_name) return static_cast<int>(i);
        }
        return -1;
    };

    int idx_id = getColIndex("ID");
    int idx_text = getColIndex("Text");
    int idx_pron = getColIndex("Pronoun");
    int idx_pron_off = getColIndex("Pronoun-offset");
    int idx_a = getColIndex("A");
    int idx_a_off = getColIndex("A-offset");
    int idx_a_coref = getColIndex("A-coref");
    int idx_b = getColIndex("B");
    int idx_b_off = getColIndex("B-offset");
    int idx_b_coref = getColIndex("B-coref");
    int idx_url = getColIndex("URL");

    if (idx_text == -1 || idx_pron == -1 || idx_a == -1 || idx_b == -1) {
        throw std::runtime_error("TSV header missing required columns in: " + tsv_path);
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string col;
        while (std::getline(ss, col, '\t')) {
            cols.push_back(col);
        }
        if (cols.size() < headers.size()) {
            continue; // Skip malformed lines
        }

        GapRecord rec;
        rec.id = idx_id >= 0 ? trim(cols[idx_id]) : "";
        rec.text = cols[idx_text];
        rec.pronoun = cols[idx_pron];
        rec.pronoun_offset = std::stoll(trim(cols[idx_pron_off]));
        rec.a = cols[idx_a];
        rec.a_offset = std::stoll(trim(cols[idx_a_off]));
        rec.a_coref = parseBool(cols[idx_a_coref]);
        rec.b = cols[idx_b];
        rec.b_offset = std::stoll(trim(cols[idx_b_off]));
        rec.b_coref = parseBool(cols[idx_b_coref]);
        rec.url = idx_url >= 0 && idx_url < static_cast<int>(cols.size()) ? trim(cols[idx_url]) : "";

        // Determine class label:
        // A-coref = True -> Class 0
        // B-coref = True -> Class 1
        // Both False -> Class 2 (Neither)
        if (rec.a_coref && !rec.b_coref) {
            rec.label = 0;
        } else if (!rec.a_coref && rec.b_coref) {
            rec.label = 1;
        } else if (!rec.a_coref && !rec.b_coref) {
            rec.label = 2;
        } else {
            // In case both are marked true (invalid in coreference choice)
            rec.label = 2;
        }

        records.push_back(rec);
    }

    return records;
}

inline PartitionStats computePartitionStats(const std::vector<GapRecord>& records) {
    PartitionStats stats = {records.size(), 0, 0, 0};
    for (const auto& r : records) {
        if (r.label == 0) stats.count_a++;
        else if (r.label == 1) stats.count_b++;
        else stats.count_neither++;
    }
    return stats;
}

} // namespace msar

#endif // MSAR_DATA_LOADER_HPP
