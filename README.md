# SVM Classifier

A Support Vector Machine (SVM) classifier built from scratch in C++ using the SMO (Sequential Minimal Optimization) algorithm.

## Features

- SVM with SMO training algorithm
- Three kernel functions: Linear, RBF, Polynomial
- Z-score feature normalisation
- Evaluation: accuracy, confusion matrix, precision, recall, F1 score
- 5-fold cross-validation
- Kernel comparison

## Datasets

The classifier runs the full pipeline (load → normalise → split → train → 5-fold CV → kernel comparison → grid search → optimised retrain) on four UCI binary-classification datasets:

| Dataset | Samples | Features | Positive / Negative |
|---|---|---|---|
| [Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | 30 | Malignant / Benign |
| [Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere) | 351 | 34 | Good / Bad radar return |
| [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) | 1372 | 4 | Forged / Genuine |
| [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) | 4601 | 57 | Spam / Ham |

Grid search runs on the first three; Spambase skips it (a full grid over C/gamma/degree at 4601 samples would take hours with the current SMO).

Download all four datasets:

```bash
mkdir -p data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" -o data/wdbc.csv
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data" -o data/ionosphere.data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt" -o data/banknote.txt
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data" -o data/spambase.data
```

## Build and Run

Requires CMake and a C++17 compiler.

```bash
cmake -B build
cmake --build build
./build/svm_classifier | tee resources/run_output.txt
```

### Generate Figures

After running the classifier, generate publication-quality plots from the output:

```bash
python3 scripts/plot_results.py
```

This reads `resources/run_output.txt` and saves six PNG figures to `figures/` (confusion matrices, CV folds, kernel comparisons, grid search heatmaps, default vs optimised accuracy). Requires Python 3 with `matplotlib` and `numpy`.

> **Note:** the plotting script currently parses a single-dataset section and treats the first dataset's results as the canonical run. With four datasets in the output it'll plot Wisconsin's results and ignore the others. Multi-dataset plotting is a follow-up.

## Documentation

See the [`docs/`](docs/) folder for detailed documentation:

- [Architecture](docs/architecture.md) — project structure and module responsibilities
- [Algorithm](docs/algorithm.md) — SVM dual formulation, SMO training, kernel functions
- [Usage](docs/usage.md) — build, run, and output walk-through

The full assignment report is in [`resources/assignment-report.md`](resources/assignment-report.md).

## License

[MIT](LICENSE)
