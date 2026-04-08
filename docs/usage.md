# Usage

How to build, run, and interpret the output of the SVM classifier. For the repository layout see [architecture.md](architecture.md); for the mathematics see [algorithm.md](algorithm.md).

## Requirements

- A C++17 compiler (GCC 9+, Clang 10+, or equivalent)
- CMake 3.14 or later
- `curl` (for downloading the dataset)

No third-party C++ libraries are needed — the project intentionally uses only the standard library.

## Dataset

Download the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI ML Repository and place it at `data/wdbc.csv`:

```bash
mkdir -p data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" \
    -o data/wdbc.csv
```

The file should contain 569 lines, one sample per line. Each line is:

```
<id>,<diagnosis>,<feature_1>,<feature_2>,...,<feature_30>
```

where `<diagnosis>` is `M` (malignant) or `B` (benign). `load_csv` in `src/data_loader.cpp` skips the ID column and converts the diagnosis to `+1`/`−1`.

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

The binary takes no command-line flags. A single run performs the full pipeline:

1. Loads `data/wdbc.csv` and prints the sample count and class distribution.
2. Z-score normalises every feature column.
3. Splits 80 / 20 into train and test sets using seed `42`.
4. Trains an RBF-kernel SVM with default `C = 1.0`, `γ = 0.1` on the training set.
5. Evaluates that model on the test set and prints accuracy, precision, recall, F1, and the confusion matrix.
6. Runs 5-fold cross-validation on the full normalised dataset with the same RBF configuration.
7. Trains and evaluates Linear and Polynomial kernels for a three-kernel comparison.
8. Runs grid search over `C ∈ {0.1, 1, 10, 100}`, `γ ∈ {0.001, 0.01, 0.1, 1}`, and (for Polynomial only) `degree ∈ {2, 3, 4}`.
9. Retrains each kernel with its grid-search winner and prints the final optimised results.

Total runtime is a few seconds on a typical desktop CPU. Every step writes to stdout — redirect to a file if you want to capture the full trace:

```bash
./build/svm_classifier > run.log
```

## Expected Output (Abbreviated)

The run prints output grouped by phase. The key sections are:

- **Load and preprocess:**
  ```
  === SVM Classifier - Breast Cancer Wisconsin ===

  Loading dataset...
  Samples: 569, Features: 30
  Malignant (M=+1): 212, Benign (B=-1): 357

  Normalising features (z-score)...
  Train: 456, Test: 113
  ```

- **Default RBF baseline (`C = 1.0`, `γ = 0.1`):** an `Evaluation Results` block with accuracy, precision, recall, F1, and a confusion matrix.

- **5-fold cross-validation:** one line per fold, then the mean accuracy.

- **Three-kernel comparison (default hyperparameters):** three `Evaluation Results` blocks, one per kernel.

- **Grid search:** one line per hyperparameter combination, with the best configuration reported at the end of each kernel's search.

- **Optimised retraining:** three final `Evaluation Results` blocks showing the tuned performance of each kernel.

The `resources/assignment-report.md` file contains the full tables of numbers produced by this pipeline.

## Reproducibility

All random shuffles use a seeded linear congruential generator, so the run is deterministic. Running the binary twice produces identical output, and the numbers in the assignment report are exactly reproducible from a clean build.

If you want to change the seed, modify the default `seed = 42` in the train/test split call in `src/main.cpp`, or the seed argument to `k_fold_cv` / `grid_search`.

## Troubleshooting

**`Cannot open file: data/wdbc.csv`** — the dataset is missing. Run the `curl` command in the Dataset section above. The file path is resolved relative to the current working directory, so run the binary from the repository root.

**Linker errors during build** — make sure you have a C++17 compiler. On older systems you may need to explicitly select one:

```bash
cmake -B build -DCMAKE_CXX_COMPILER=g++-9
```

**No output for several seconds** — the grid search for the polynomial kernel (48 combinations × 5 folds = 240 SVM fits) dominates the runtime. This is expected.
