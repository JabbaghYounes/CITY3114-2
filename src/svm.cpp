#include "svm.hpp"

SVM::SVM(const Kernel& kernel, double C, double tol, int max_iter)
    : kernel_(kernel), C_(C), tol_(tol), max_iter_(max_iter), b_(0.0) {}

void SVM::train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    // TODO: implement SMO
}

int SVM::predict(const std::vector<double>& x) const {
    return decision_function(x) >= 0 ? 1 : -1;
}

std::vector<int> SVM::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> results;
    results.reserve(X.size());
    for (const auto& x : X) {
        results.push_back(predict(x));
    }
    return results;
}

int SVM::support_vector_count() const {
    return static_cast<int>(support_vectors_.size());
}

double SVM::decision_function(const std::vector<double>& x) const {
    double sum = 0.0;
    for (size_t i = 0; i < support_vectors_.size(); ++i) {
        sum += support_alphas_[i] * support_labels_[i] * kernel_.compute(support_vectors_[i], x);
    }
    return sum + b_;
}
