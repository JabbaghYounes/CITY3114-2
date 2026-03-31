#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <string>
#include <vector>

struct Dataset {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
};

Dataset load_csv(const std::string& filepath);

void normalise(Dataset& dataset);

void train_test_split(const Dataset& dataset,
                      Dataset& train, Dataset& test,
                      double test_ratio = 0.2,
                      unsigned int seed = 42);

#endif
