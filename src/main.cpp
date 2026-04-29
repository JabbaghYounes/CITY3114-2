#include <iostream>
#include <iomanip>
#include "data_loader.hpp"
#include "kernel.hpp"
#include "svm.hpp"
#include "evaluation.hpp"

void run_pipeline(const DatasetSpec& spec) {
    std::cout << "\n\n================================================================\n";
    std::cout << "=== DATASET: " << spec.name << "\n";
    std::cout << "================================================================" << std::endl;

    // 1. Load dataset
    std::cout << "\nLoading dataset..." << std::endl;
    Dataset dataset = load_csv(spec);
    std::cout << "Samples: " << dataset.X.size()
              << ", Features: " << dataset.X[0].size() << std::endl;

    int positive = 0, negative = 0;
    for (int label : dataset.y) {
        if (label == 1) ++positive;
        else ++negative;
    }
    std::cout << spec.positive_name << " (+1): " << positive
              << ", " << spec.negative_name << " (-1): " << negative << std::endl;

    // 2. Normalise features
    std::cout << "\nNormalising features (z-score)..." << std::endl;
    normalise(dataset);

    // 3. Train/test split
    Dataset train, test;
    train_test_split(dataset, train, test, 0.2, 42);
    std::cout << "Train: " << train.X.size()
              << ", Test: " << test.X.size() << std::endl;

    // 4. Train and evaluate with RBF kernel
    double C = 1.0;
    double gamma = 0.1;

    std::cout << "\n--- RBF Kernel (C=" << C << ", gamma=" << gamma << ") ---" << std::endl;
    Kernel rbf(KernelType::RBF, gamma);
    SVM svm_rbf(rbf, C);
    svm_rbf.train(train.X, train.y);

    std::vector<int> predictions = svm_rbf.predict(test.X);
    print_results(predictions, test.y, spec.positive_name, spec.negative_name);

    // 5. K-fold cross-validation with RBF
    std::cout << "\n--- 5-Fold Cross-Validation (RBF) ---" << std::endl;
    double cv_acc = k_fold_cv(dataset, rbf, C, 5, 42);
    std::cout << "Mean CV Accuracy: " << std::fixed << std::setprecision(4)
              << cv_acc << std::endl;

    // 6. Compare all 3 kernels with default params
    std::cout << "\n=== Kernel Comparison (Default Params) ===" << std::endl;

    struct KernelConfig {
        std::string name;
        Kernel kernel;
        double C;
    };

    std::vector<KernelConfig> configs = {
        {"Linear",     Kernel(KernelType::LINEAR),                    1.0},
        {"RBF",        Kernel(KernelType::RBF, 0.1),                 1.0},
        {"Polynomial", Kernel(KernelType::POLYNOMIAL, 0.1, 1.0, 3),  1.0},
    };

    for (auto& cfg : configs) {
        std::cout << "\n--- " << cfg.name << " Kernel ---" << std::endl;

        SVM svm(cfg.kernel, cfg.C);
        svm.train(train.X, train.y);

        std::vector<int> preds = svm.predict(test.X);
        print_results(preds, test.y, spec.positive_name, spec.negative_name);
    }

    // 7. Hyperparameter optimisation via grid search (skip for huge datasets)
    if (!spec.run_grid_search) {
        std::cout << "\n=== Grid Search SKIPPED for " << spec.name
                  << " (dataset too large for full grid search with current SMO) ==="
                  << std::endl;
        return;
    }

    std::vector<double> C_values = {0.1, 1, 10, 100};
    std::vector<double> gamma_values = {0.001, 0.01, 0.1, 1};
    std::vector<int> degree_values = {2, 3, 4};

    struct {
        std::string name;
        KernelType type;
    } kernel_types[] = {
        {"Linear",     KernelType::LINEAR},
        {"RBF",        KernelType::RBF},
        {"Polynomial", KernelType::POLYNOMIAL},
    };

    for (auto& kt : kernel_types) {
        std::cout << "\n=== Grid Search (" << kt.name << ") ===" << std::endl;

        GridSearchResult result = grid_search(dataset, kt.type,
                                              C_values, gamma_values,
                                              degree_values, 5, 42);

        // Retrain on full training set with best params
        std::cout << "\n--- " << kt.name << " (Optimised) ---" << std::endl;
        Kernel best_kernel(kt.type, result.best_gamma, 1.0, result.best_degree);
        SVM best_svm(best_kernel, result.best_C);
        best_svm.train(train.X, train.y);

        std::vector<int> preds = best_svm.predict(test.X);
        print_results(preds, test.y, spec.positive_name, spec.negative_name);
    }
}

int main() {
    std::cout << "=== SVM Classifier (Multi-Dataset) ===" << std::endl;

    for (const auto& spec : ALL_DATASETS) {
        run_pipeline(spec);
    }

    return 0;
}
