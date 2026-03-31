#include <iostream>
#include <iomanip>
#include "data_loader.hpp"
#include "kernel.hpp"
#include "svm.hpp"
#include "evaluation.hpp"

int main() {
    std::cout << "=== SVM Classifier - Breast Cancer Wisconsin ===" << std::endl;

    // 1. Load dataset
    std::cout << "\nLoading dataset..." << std::endl;
    Dataset dataset = load_csv("data/wdbc.csv");
    std::cout << "Samples: " << dataset.X.size()
              << ", Features: " << dataset.X[0].size() << std::endl;

    int malignant = 0, benign = 0;
    for (int label : dataset.y) {
        if (label == 1) ++malignant;
        else ++benign;
    }
    std::cout << "Malignant (M=+1): " << malignant
              << ", Benign (B=-1): " << benign << std::endl;

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
    print_results(predictions, test.y);

    // 5. K-fold cross-validation with RBF
    std::cout << "\n--- 5-Fold Cross-Validation (RBF) ---" << std::endl;
    double cv_acc = k_fold_cv(dataset, rbf, C, 5, 42);
    std::cout << "Mean CV Accuracy: " << std::fixed << std::setprecision(4)
              << cv_acc << std::endl;

    // 6. Compare all 3 kernels
    std::cout << "\n=== Kernel Comparison ===" << std::endl;

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
        print_results(preds, test.y);
    }

    return 0;
}
