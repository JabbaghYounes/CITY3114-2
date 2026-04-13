# SVM Classifier

A Support Vector Machine (SVM) classifier built from scratch in C++ using the SMO (Sequential Minimal Optimization) algorithm. No external libraries and all components implemented manually.

## Features

- SVM with SMO training algorithm
- Three kernel functions: Linear, RBF, Polynomial
- Z-score feature normalisation
- Evaluation: accuracy, confusion matrix, precision, recall, F1 score
- 5-fold cross-validation
- Kernel comparison

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — 569 samples, 30 features, binary classification (Malignant / Benign).

Download the dataset and place it at `data/wdbc.csv`:
```bash
mkdir -p data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" -o data/wdbc.csv
```

## Build and Run

Requires CMake and a C++17 compiler.

```bash
cmake -B build
cmake --build build
./build/svm_classifier | tee run_output.txt
```

### Generate Figures

After running the classifier, generate publication-quality plots from the output:

```bash
python3 scripts/plot_results.py
```

This reads `run_output.txt` and saves six PNG figures to `figures/` (confusion matrices, CV folds, kernel comparisons, grid search heatmaps, default vs optimised accuracy). Requires Python 3 with `matplotlib` and `numpy`.

## Documentation

See the [`docs/`](docs/) folder for detailed documentation:

- [Architecture](docs/architecture.md) — project structure and module responsibilities
- [Algorithm](docs/algorithm.md) — SVM dual formulation, SMO training, kernel functions
- [Usage](docs/usage.md) — build, run, and output walk-through

The full assignment report is in [`resources/assignment-report.md`](resources/assignment-report.md).

## License

[MIT](LICENSE)
