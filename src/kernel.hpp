#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <vector>
#include <string>

enum class KernelType { LINEAR, RBF, POLYNOMIAL };

class Kernel {
public:
    Kernel(KernelType type, double gamma = 0.1, double coef0 = 0.0, int degree = 3);

    double compute(const std::vector<double>& a, const std::vector<double>& b) const;

    std::string name() const;

private:
    KernelType type_;
    double gamma_;
    double coef0_;
    int degree_;
};

#endif
