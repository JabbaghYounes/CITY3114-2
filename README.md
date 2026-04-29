# SVM Classifier

A Support Vector Machine (SVM) classifier built from scratch in C++ using the SMO (Sequential Minimal Optimization) algorithm.

## Features

- SVM with SMO training algorithm and incremental error-cache update (O(n) per pair update)
- Three kernel functions: Linear, RBF, Polynomial
- Z-score feature normalisation
- Evaluation: accuracy, confusion matrix, precision, recall, F1 score
- 5-fold cross-validation
- Kernel comparison
- Hyperparameter grid search
- Multi-dataset pipeline driver (runs over all configured `DatasetSpec` entries)

## Datasets

The classifier runs the full pipeline (load → normalise → split → train → 5-fold CV → kernel comparison → grid search → optimised retrain) on four UCI binary-classification datasets:

| Dataset | Samples | Features | Positive / Negative |
|---|---|---|---|
| [Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | 30 | Malignant / Benign |
| [Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere) | 351 | 34 | Good / Bad radar return |
| [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication) | 1372 | 4 | Forged / Genuine |
| [Spambase](https://archive.ics.uci.edu/dataset/94/spambase) | 4601 | 57 | Spam / Ham |

Grid search runs on the first three; Spambase skips it (set `run_grid_search=false` in its `DatasetSpec`) — a 400-fit grid at 4601 samples is still too long even with the incremental SMO. The full 4-dataset pipeline runs in roughly 2:45 on a typical desktop CPU.

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

This reads `resources/run_output.txt`, splits it by `=== DATASET:` markers, and saves a per-dataset set of figures to `figures/` with the dataset slug as filename prefix:

| Pattern | Contents |
|---|---|
| `<slug>_confusion_matrices.png` | 2×2 heatmaps for initial RBF and the three optimised kernels |
| `<slug>_cv_folds.png` | 5-fold cross-validation accuracy bar chart with mean line |
| `<slug>_kernel_comparison_default.png` | Grouped bars: accuracy, precision, recall, F1 per kernel (default params) |
| `<slug>_kernel_comparison_optimised.png` | Same layout for optimised parameters |
| `<slug>_grid_search_heatmaps.png` | C × γ heatmaps per kernel (Polynomial split by degree) |
| `<slug>_default_vs_optimised.png` | Side-by-side accuracy comparison showing tuning impact |

Datasets that skip grid search (currently Spambase) only get `cv_folds` and `kernel_comparison_default`. Wisconsin / Ionosphere / Banknote each get the full set of six. Requires Python 3 with `matplotlib` and `numpy`.

## Documentation

See the [`docs/`](docs/) folder for detailed documentation:

- [Architecture](docs/architecture.md) — project structure and module responsibilities
- [Algorithm](docs/algorithm.md) — SVM dual formulation, SMO training, kernel functions
- [Usage](docs/usage.md) — build, run, and output walk-through

The full assignment report is in [`resources/assignment-report.md`](resources/assignment-report.md).

## License

[MIT](LICENSE)
