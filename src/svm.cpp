#include "svm.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

SVM::SVM(const Kernel& kernel, double C, double tol, int max_iter)
    : kernel_(kernel), C_(C), tol_(tol), max_iter_(max_iter), b_(0.0) {}

void SVM::train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    int n = static_cast<int>(X.size());
    alphas_.assign(n, 0.0);
    b_ = 0.0;

    // Precompute kernel matrix for performance
    std::vector<std::vector<double>> K(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            K[i][j] = kernel_.compute(X[i], X[j]);
            K[j][i] = K[i][j];
        }
    }

    // Error cache
    std::vector<double> E(n);
    for (int i = 0; i < n; ++i) {
        double f_i = 0.0;
        for (int j = 0; j < n; ++j) {
            f_i += alphas_[j] * y[j] * K[j][i];
        }
        f_i += b_;
        E[i] = f_i - y[i];
    }

    for (int iter = 0; iter < max_iter_; ++iter) {
        int num_changed = 0;

        for (int i = 0; i < n; ++i) {
            double r_i = E[i] * y[i];

            // Check KKT violation
            if ((r_i < -tol_ && alphas_[i] < C_) || (r_i > tol_ && alphas_[i] > 0)) {
                // Select j using second choice heuristic: maximise |E_i - E_j|
                int j = -1;
                double max_delta_E = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == i) continue;
                    double delta = std::abs(E[i] - E[k]);
                    if (delta > max_delta_E) {
                        max_delta_E = delta;
                        j = k;
                    }
                }
                if (j == -1) continue;

                double alpha_i_old = alphas_[i];
                double alpha_j_old = alphas_[j];

                // Compute bounds L and H
                double L, H;
                if (y[i] != y[j]) {
                    L = std::max(0.0, alphas_[j] - alphas_[i]);
                    H = std::min(C_, C_ + alphas_[j] - alphas_[i]);
                } else {
                    L = std::max(0.0, alphas_[i] + alphas_[j] - C_);
                    H = std::min(C_, alphas_[i] + alphas_[j]);
                }
                if (std::abs(L - H) < 1e-10) continue;

                // Compute eta
                double eta = 2.0 * K[i][j] - K[i][i] - K[j][j];
                if (eta >= 0) continue;

                // Update alpha_j
                alphas_[j] -= y[j] * (E[i] - E[j]) / eta;
                alphas_[j] = std::min(H, std::max(L, alphas_[j]));

                if (std::abs(alphas_[j] - alpha_j_old) < 1e-5) continue;

                // Update alpha_i
                alphas_[i] += y[i] * y[j] * (alpha_j_old - alphas_[j]);

                // Update bias
                double b1 = b_ - E[i]
                    - y[i] * (alphas_[i] - alpha_i_old) * K[i][i]
                    - y[j] * (alphas_[j] - alpha_j_old) * K[i][j];

                double b2 = b_ - E[j]
                    - y[i] * (alphas_[i] - alpha_i_old) * K[i][j]
                    - y[j] * (alphas_[j] - alpha_j_old) * K[j][j];

                if (alphas_[i] > 0 && alphas_[i] < C_) {
                    b_ = b1;
                } else if (alphas_[j] > 0 && alphas_[j] < C_) {
                    b_ = b2;
                } else {
                    b_ = (b1 + b2) / 2.0;
                }

                // Recompute error cache
                for (int k = 0; k < n; ++k) {
                    double f_k = 0.0;
                    for (int m = 0; m < n; ++m) {
                        f_k += alphas_[m] * y[m] * K[m][k];
                    }
                    f_k += b_;
                    E[k] = f_k - y[k];
                }

                ++num_changed;
            }
        }

        if (num_changed == 0) break;
    }

    // Extract support vectors (samples with alpha > 0)
    support_vectors_.clear();
    support_labels_.clear();
    support_alphas_.clear();

    for (int i = 0; i < n; ++i) {
        if (alphas_[i] > 1e-8) {
            support_vectors_.push_back(X[i]);
            support_labels_.push_back(y[i]);
            support_alphas_.push_back(alphas_[i]);
        }
    }

    std::cout << "Training complete. Support vectors: " << support_vectors_.size()
              << "/" << n << std::endl;
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
