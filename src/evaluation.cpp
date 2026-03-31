#include "evaluation.hpp"

double accuracy(const std::vector<int>& predicted, const std::vector<int>& actual) {
    // TODO: implement
    return 0.0;
}

ConfusionMatrix confusion_matrix(const std::vector<int>& predicted, const std::vector<int>& actual) {
    // TODO: implement
    return {};
}

double precision(const ConfusionMatrix& cm) {
    // TODO: implement
    return 0.0;
}

double recall(const ConfusionMatrix& cm) {
    // TODO: implement
    return 0.0;
}

double f1_score(const ConfusionMatrix& cm) {
    // TODO: implement
    return 0.0;
}

double k_fold_cv(const Dataset& dataset, const Kernel& kernel,
                 double C, int k, unsigned int seed) {
    // TODO: implement
    return 0.0;
}

void print_results(const std::vector<int>& predicted, const std::vector<int>& actual) {
    // TODO: implement
}
