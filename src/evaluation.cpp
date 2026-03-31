#include "evaluation.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>

double accuracy(const std::vector<int>& predicted, const std::vector<int>& actual) {
    int correct = 0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i] == actual[i]) ++correct;
    }
    return static_cast<double>(correct) / predicted.size();
}

ConfusionMatrix confusion_matrix(const std::vector<int>& predicted, const std::vector<int>& actual) {
    ConfusionMatrix cm;
    for (size_t i = 0; i < predicted.size(); ++i) {
        if (actual[i] == 1 && predicted[i] == 1)       ++cm.tp;
        else if (actual[i] == -1 && predicted[i] == -1) ++cm.tn;
        else if (actual[i] == -1 && predicted[i] == 1)  ++cm.fp;
        else                                             ++cm.fn;
    }
    return cm;
}

double precision(const ConfusionMatrix& cm) {
    int denom = cm.tp + cm.fp;
    return denom > 0 ? static_cast<double>(cm.tp) / denom : 0.0;
}

double recall(const ConfusionMatrix& cm) {
    int denom = cm.tp + cm.fn;
    return denom > 0 ? static_cast<double>(cm.tp) / denom : 0.0;
}

double f1_score(const ConfusionMatrix& cm) {
    double p = precision(cm);
    double r = recall(cm);
    return (p + r > 0) ? 2.0 * p * r / (p + r) : 0.0;
}

double k_fold_cv(const Dataset& dataset, const Kernel& kernel,
                 double C, int k, unsigned int seed) {
    int n = static_cast<int>(dataset.X.size());

    // Shuffle indices
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    auto rng = seed;
    for (int i = n - 1; i > 0; --i) {
        rng = rng * 1103515245 + 12345;
        int j = (rng >> 16) % (i + 1);
        std::swap(indices[i], indices[j]);
    }

    double total_accuracy = 0.0;
    int fold_size = n / k;

    for (int fold = 0; fold < k; ++fold) {
        int test_start = fold * fold_size;
        int test_end = (fold == k - 1) ? n : test_start + fold_size;

        std::vector<std::vector<double>> train_X, test_X;
        std::vector<int> train_y, test_y;

        for (int i = 0; i < n; ++i) {
            int idx = indices[i];
            if (i >= test_start && i < test_end) {
                test_X.push_back(dataset.X[idx]);
                test_y.push_back(dataset.y[idx]);
            } else {
                train_X.push_back(dataset.X[idx]);
                train_y.push_back(dataset.y[idx]);
            }
        }

        SVM svm(kernel, C);
        svm.train(train_X, train_y);
        std::vector<int> predictions = svm.predict(test_X);
        double fold_acc = accuracy(predictions, test_y);

        std::cout << "  Fold " << (fold + 1) << "/" << k
                  << ": accuracy = " << std::fixed << std::setprecision(4)
                  << fold_acc << std::endl;

        total_accuracy += fold_acc;
    }

    return total_accuracy / k;
}

void print_results(const std::vector<int>& predicted, const std::vector<int>& actual) {
    ConfusionMatrix cm = confusion_matrix(predicted, actual);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Evaluation Results ===" << std::endl;
    std::cout << "Accuracy:  " << accuracy(predicted, actual) << std::endl;
    std::cout << "Precision: " << precision(cm) << std::endl;
    std::cout << "Recall:    " << recall(cm) << std::endl;
    std::cout << "F1 Score:  " << f1_score(cm) << std::endl;

    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "                Predicted +1   Predicted -1" << std::endl;
    std::cout << "  Actual +1 (M)    " << std::setw(5) << cm.tp
              << std::setw(14) << cm.fn << std::endl;
    std::cout << "  Actual -1 (B)    " << std::setw(5) << cm.fp
              << std::setw(14) << cm.tn << std::endl;
}
