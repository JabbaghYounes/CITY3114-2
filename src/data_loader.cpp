#include "data_loader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

Dataset load_csv(const std::string& filepath) {
    Dataset dataset;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        // Skip ID column
        std::getline(ss, token, ',');

        // Read diagnosis: M -> +1, B -> -1
        std::getline(ss, token, ',');
        dataset.y.push_back(token == "M" ? 1 : -1);

        // Read 30 feature columns
        std::vector<double> features;
        while (std::getline(ss, token, ',')) {
            features.push_back(std::stod(token));
        }
        dataset.X.push_back(features);
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
