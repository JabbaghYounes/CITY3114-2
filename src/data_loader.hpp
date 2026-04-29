#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <string>
#include <vector>

struct Dataset {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
};

struct DatasetSpec {
    std::string name;
    std::string filepath;
    bool has_id_column;
    int label_column;            // index after ID removal; -1 means last
    std::string positive_label;  // token mapped to +1; everything else -> -1
    int expected_features;
    std::string positive_name;   // human label, e.g. "Forged"
    std::string negative_name;   // human label, e.g. "Genuine"
    bool run_grid_search = true; // disable for very large datasets
};

extern const std::vector<DatasetSpec> ALL_DATASETS;

Dataset load_csv(const DatasetSpec& spec);

void normalise(Dataset& dataset);

void train_test_split(const Dataset& dataset,
                      Dataset& train, Dataset& test,
                      double test_ratio = 0.2,
                      unsigned int seed = 42);

#endif
