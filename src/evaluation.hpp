#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <vector>
#include "svm.hpp"
#include "data_loader.hpp"

struct ConfusionMatrix {
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
};

double accuracy(const std::vector<int>& predicted, const std::vector<int>& actual);

ConfusionMatrix confusion_matrix(const std::vector<int>& predicted, const std::vector<int>& actual);

double precision(const ConfusionMatrix& cm);
double recall(const ConfusionMatrix& cm);
double f1_score(const ConfusionMatrix& cm);

double k_fold_cv(const Dataset& dataset, const Kernel& kernel,
                 double C, int k = 5, unsigned int seed = 42);

void print_results(const std::vector<int>& predicted, const std::vector<int>& actual);

#endif
