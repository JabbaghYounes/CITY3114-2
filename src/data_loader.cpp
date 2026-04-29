#include "data_loader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

const std::vector<DatasetSpec> ALL_DATASETS = {
    {"Wisconsin (Breast Cancer)", "data/wdbc.csv",
     true, 0, "M", 30, "Malignant", "Benign", true},
    {"Ionosphere", "data/ionosphere.data",
     false, -1, "g", 34, "Good", "Bad", true},
    {"Banknote Authentication", "data/banknote.txt",
     false, -1, "1", 4, "Forged", "Genuine", true},
    // Grid search disabled for Spambase (4601 samples × O(n²) SMO updates
    // would push runtime into hours). Default RBF + CV + kernel comparison
    // still run.
    {"Spambase", "data/spambase.data",
     false, -1, "1", 57, "Spam", "Ham", false},
};

static std::string trim(const std::string& s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

Dataset load_csv(const DatasetSpec& spec) {
    Dataset dataset;
    std::ifstream file(spec.filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + spec.filepath);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        if (trim(line).empty()) continue;

        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(trim(token));
        }

        if (spec.has_id_column) {
            tokens.erase(tokens.begin());
        }

        int label_idx = (spec.label_column == -1)
            ? static_cast<int>(tokens.size()) - 1
            : spec.label_column;

        if (label_idx < 0 || label_idx >= static_cast<int>(tokens.size())) {
            throw std::runtime_error(
                spec.name + ": label index " + std::to_string(label_idx) +
                " out of range on line " + std::to_string(line_num));
        }

        dataset.y.push_back(tokens[label_idx] == spec.positive_label ? 1 : -1);

        std::vector<double> features;
        features.reserve(tokens.size() - 1);
        for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
            if (i == label_idx) continue;
            features.push_back(std::stod(tokens[i]));
        }

        if (static_cast<int>(features.size()) != spec.expected_features) {
            throw std::runtime_error(
                spec.name + ": feature count mismatch on line " +
                std::to_string(line_num) + " — expected " +
                std::to_string(spec.expected_features) + ", got " +
                std::to_string(features.size()));
        }

        dataset.X.push_back(std::move(features));
    }

    return dataset;
}

void normalise(Dataset& dataset) {
    if (dataset.X.empty()) return;

    size_t n_features = dataset.X[0].size();
    size_t n_samples = dataset.X.size();

    for (size_t j = 0; j < n_features; ++j) {
        // Compute mean
        double mean = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            mean += dataset.X[i][j];
        }
        mean /= n_samples;

        // Compute standard deviation
        double std_dev = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            double diff = dataset.X[i][j] - mean;
            std_dev += diff * diff;
        }
        std_dev = std::sqrt(std_dev / n_samples);

        // Z-score normalisation
        if (std_dev > 1e-10) {
            for (size_t i = 0; i < n_samples; ++i) {
                dataset.X[i][j] = (dataset.X[i][j] - mean) / std_dev;
            }
        }
    }
}

void train_test_split(const Dataset& dataset,
                      Dataset& train, Dataset& test,
                      double test_ratio, unsigned int seed) {
    size_t n = dataset.X.size();

    // Create shuffled indices
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Simple Fisher-Yates shuffle with LCG random
    auto rng = seed;
    for (size_t i = n - 1; i > 0; --i) {
        rng = rng * 1103515245 + 12345;
        size_t j = (rng >> 16) % (i + 1);
        std::swap(indices[i], indices[j]);
    }

    size_t test_size = static_cast<size_t>(n * test_ratio);
    size_t train_size = n - test_size;

    train.X.reserve(train_size);
    train.y.reserve(train_size);
    test.X.reserve(test_size);
    test.y.reserve(test_size);

    for (size_t i = 0; i < train_size; ++i) {
        train.X.push_back(dataset.X[indices[i]]);
        train.y.push_back(dataset.y[indices[i]]);
    }
    for (size_t i = train_size; i < n; ++i) {
        test.X.push_back(dataset.X[indices[i]]);
        test.y.push_back(dataset.y[indices[i]]);
    }
}
