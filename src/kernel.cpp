#include "kernel.hpp"
#include <cmath>
#include <stdexcept>

Kernel::Kernel(KernelType type, double gamma, double coef0, int degree)
    : type_(type), gamma_(gamma), coef0_(coef0), degree_(degree) {}

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double Kernel::compute(const std::vector<double>& a, const std::vector<double>& b) const {
    switch (type_) {
        case KernelType::LINEAR:
            // K(a, b) = a · b
            return dot(a, b);

        case KernelType::RBF: {
            // K(a, b) = exp(-gamma * ||a - b||^2)
            double dist_sq = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                double diff = a[i] - b[i];
                dist_sq += diff * diff;
            }
            return std::exp(-gamma_ * dist_sq);
        }

        case KernelType::POLYNOMIAL:
            // K(a, b) = (gamma * a · b + coef0)^degree
            return std::pow(gamma_ * dot(a, b) + coef0_, degree_);
    }
    throw std::runtime_error("Unknown kernel type");
}

std::string Kernel::name() const {
    switch (type_) {
        case KernelType::LINEAR:     return "Linear";
        case KernelType::RBF:        return "RBF";
        case KernelType::POLYNOMIAL: return "Polynomial";
    }
    return "Unknown";
}
