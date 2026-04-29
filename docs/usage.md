# Usage

How to build, run, and interpret the output of the SVM classifier. For the repository layout see [architecture.md](architecture.md); for the mathematics see [algorithm.md](algorithm.md).

## Requirements

- A C++17 compiler (GCC 9+, Clang 10+, or equivalent)
- CMake 3.14 or later
- `curl` (for downloading the dataset)

No third-party C++ libraries are needed — the project intentionally uses only the standard library.

## Datasets

The classifier iterates over three UCI binary-classification datasets defined in `ALL_DATASETS` (in `src/data_loader.cpp`). All three files must be present before running:

```bash
mkdir -p data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"            -o data/ionosphere.data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt" -o data/banknote.txt
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"                -o data/spambase.data
```

| Dataset | Path | Samples × Features | Positive class | ID column | Label position |
|---|---|---|---|---|---|
| Ionosphere | `data/ionosphere.data` | 351 × 34 | g (good radar return) | no | last column |
| Banknote Authentication | `data/banknote.txt` | 1372 × 4 | 1 (forged) | no | last column |
| Spambase | `data/spambase.data` | 4601 × 57 | 1 (spam) | no | last column |

`load_csv(spec)` in `src/data_loader.cpp` honours each dataset's `DatasetSpec` (`has_id_column`, `label_column`, `positive_label`, `expected_features`) so the parser is the same for all four.

## Build

From the repository root:

```bash
cmake -B build
cmake --build build
```

This produces a single executable at `build/svm_classifier`. No installation step is required.

To do a clean rebuild:

```bash
rm -rf build
cmake -B build
cmake --build build
```

## Run

```bash
./build/svm_classifier
```

The binary takes no command-line flags. `main()` loops over `ALL_DATASETS` and calls `run_pipeline(spec)` for each. Per dataset, the pipeline does:

1. Load the CSV via `load_csv(spec)` and print the sample count + class distribution.
2. Z-score normalise every feature column.
3. Split 80 / 20 into train and test sets using seed `42`.
4. Train an RBF-kernel SVM with default `C = 1.0`, `γ = 0.1`.
5. Evaluate on the test set (accuracy, precision, recall, F1, confusion matrix).
6. Run 5-fold cross-validation on the full normalised dataset with the same RBF configuration.
7. Train Linear and Polynomial kernels at default hyperparameters for a three-kernel comparison.
8. **If `spec.run_grid_search` is true:** grid search over `C ∈ {0.1, 1, 10, 100}`, `γ ∈ {0.001, 0.01, 0.1, 1}`, and (Polynomial only) `degree ∈ {2, 3, 4}`. Then retrain each kernel with its grid winner and print the final optimised results.
9. Otherwise print a `=== Grid Search SKIPPED for <dataset> ===` line and continue.

Spambase is the only dataset with `run_grid_search=false` by default — its 4601 samples × 5-fold × 80-combo grid search would take well over an hour even with the incremental error-cache update.

Total runtime for the full 3-dataset run is around 90 seconds on a typical desktop CPU. Every step writes to stdout — use `tee` to both see the output and save it for the plotting script:

```bash
./build/svm_classifier | tee resources/run_output.txt
```

## Generate Figures

A Python script (`scripts/plot_results.py`) splits the captured output on `=== DATASET:` markers and generates one set of figures per dataset, named with the dataset slug as a prefix:

```bash
python3 scripts/plot_results.py
```

Requires Python 3 with `matplotlib` and `numpy`. Figures are saved to `figures/` at 300 DPI. Per dataset the script writes:

| Pattern | Contents |
|---------|----------|
| `<slug>_confusion_matrices.png` | 2×2 heatmaps for initial RBF and all three optimised kernels |
| `<slug>_cv_folds.png` | 5-fold cross-validation accuracy bar chart with mean line |
| `<slug>_kernel_comparison_default.png` | Grouped bars: accuracy, precision, recall, F1 per kernel (default params) |
| `<slug>_kernel_comparison_optimised.png` | Same layout for optimised parameters |
| `<slug>_grid_search_heatmaps.png` | C × γ heatmaps per kernel (Polynomial split by degree) |
| `<slug>_default_vs_optimised.png` | Side-by-side accuracy comparison showing tuning impact |

The slug is the first word of the dataset name, lowercased and stripped to `[a-z0-9]+` — so `ionosphere`, `banknote`, `spambase`. Datasets that skip grid search (currently Spambase) only get `cv_folds` and `kernel_comparison_default`. The full run produces 14 figures (6 + 6 + 2).

The script also supports piping directly from the classifier:

```bash
./build/svm_classifier | python3 scripts/plot_results.py --stdin
```

Or an explicit input file:

```bash
python3 scripts/plot_results.py --input path/to/output.txt
```

## Expected Output (Abbreviated)

The run prints `=== SVM Classifier (Multi-Dataset) ===` once at the top, then a per-dataset block with the same phase structure for each of the four datasets:

- **Dataset header:**
  ```
  ================================================================
  === DATASET: Ionosphere
  ================================================================

  Loading dataset...
  Samples: 351, Features: 34
  Good (+1): 225, Bad (-1): 126

  Normalising features (z-score)...
  Train: 281, Test: 70
  ```

- **Default RBF baseline (`C = 1.0`, `γ = 0.1`):** an `Evaluation Results` block with accuracy, precision, recall, F1, and a confusion matrix. The matrix carries a class-name legend on the title line — e.g. `Confusion Matrix (+1=Good, -1=Bad):` — so the same `+1`/`-1` row labels stay aligned across datasets.

- **5-fold cross-validation:** one line per fold, then the mean accuracy.

- **Three-kernel comparison (default hyperparameters):** three `Evaluation Results` blocks, one per kernel.

- **Grid search (skipped for Spambase):** one line per hyperparameter combination, with the best configuration reported at the end of each kernel's search.

- **Optimised retraining (skipped for Spambase):** three final `Evaluation Results` blocks showing the tuned performance of each kernel.

`resources/assignment-report.md` is the cross-dataset comparison study built on this run output.

## Reproducibility

All random shuffles use a seeded linear congruential generator (`a = 1103515245`, `c = 12345`, default seed `42`), so train/test splits and k-fold partitions are bitwise reproducible across platforms. Running the binary twice on the same machine produces identical output.

The numerical solver is *not* strictly bit-for-bit portable: the SMO error cache is updated incrementally (see [algorithm.md](algorithm.md)), and floating-point ordering can shift slightly between compilers/CPUs (FMA vs no-FMA, libm differences). In practice the test-set accuracy/F1 numbers in `resources/assignment-report.md` are stable; the grid-search CV column may shift at the third decimal between toolchains.

If you want to change the seed, modify the default `seed = 42` in the train/test split call in `src/main.cpp`, or the seed argument to `k_fold_cv` / `grid_search`.

## Troubleshooting

**`Cannot open file: data/<filename>`** — one of the four dataset files is missing. Run the four `curl` commands in the Datasets section above. The file paths are resolved relative to the current working directory, so run the binary from the repository root.

**`<dataset>: feature count mismatch on line N`** — the CSV doesn't have the expected number of columns. The most common cause is downloading an HTML error page instead of the data file; re-download with the exact `curl` URL from the Datasets section.

**Linker errors during build** — make sure you have a C++17 compiler. On older systems you may need to explicitly select one:

```bash
cmake -B build -DCMAKE_CXX_COMPILER=g++-9
```

**No output for tens of seconds** — Banknote's polynomial grid search (48 combinations × 5 folds = 240 SVM fits at 1372 samples) is the longest single phase in the run. The full 3-dataset pipeline takes ~90 seconds in total.
