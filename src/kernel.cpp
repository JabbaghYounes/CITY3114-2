#include "kernel.hpp"

Kernel::Kernel(KernelType type, double gamma, double coef0, int degree)
    : type_(type), gamma_(gamma), coef0_(coef0), degree_(degree) {}

double Kernel::compute(const std::vector<double>& a, const std::vector<double>& b) const {
    // TODO: implement
    return 0.0;
}

std::string Kernel::name() const {
    // TODO: implement
    return "";
}
