#ifndef SVM_HPP
#define SVM_HPP

#include <vector>
#include "kernel.hpp"

class SVM {
public:
    SVM(const Kernel& kernel, double C = 1.0, double tol = 1e-3, int max_iter = 1000);

    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    int predict(const std::vector<double>& x) const;

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    int support_vector_count() const;

private:
    Kernel kernel_;
    double C_;
    double tol_;
    int max_iter_;
    double b_;

    std::vector<double> alphas_;
    std::vector<std::vector<double>> support_vectors_;
    std::vector<int> support_labels_;
    std::vector<double> support_alphas_;

    double decision_function(const std::vector<double>& x) const;
};

#endif
