# Architecture

This document describes the repository layout and how the modules fit together. For the mathematical background see [algorithm.md](algorithm.md); for build and run instructions see [usage.md](usage.md).

## Repository Layout

```
.
├── CMakeLists.txt          # CMake build configuration
├── README.md               # Project overview
├── LICENSE                 # MIT license
├── run_output.txt          # Captured stdout from a classifier run
├── scripts/                # Utility scripts
│   └── plot_results.py     # Generates figures from run output
├── src/                    # C++ source
│   ├── main.cpp            # Pipeline entry point
│   ├── data_loader.hpp/cpp # CSV parsing, normalisation, train/test split
│   ├── kernel.hpp/cpp      # Linear, RBF, Polynomial kernels
│   ├── svm.hpp/cpp         # SVM class with SMO solver
│   └── evaluation.hpp/cpp  # Metrics, k-fold CV, grid search
├── docs/                   # Project documentation (this folder)
│   ├── architecture.md     # Repo layout and module responsibilities
│   ├── algorithm.md        # SVM and SMO mathematics
│   └── usage.md            # Build, run, and output walk-through
├── resources/              # Assignment deliverables
│   ├── assignment-report.md    # Full report for Task 1 and Task 2
│   └── assignment-report.docx  # DOCX export of the report
├── figures/                # Generated plots (gitignored — run plot_results.py)
├── data/                   # Dataset (gitignored — download separately)
└── build/                  # CMake build output (gitignored)
```

## Module Responsibilities

The program is split into five translation units. Each module has a single concern, and the headers form an implicit dependency graph:

```
main.cpp
  ├─ data_loader (Dataset, load_csv, normalise, train_test_split)
  ├─ kernel      (Kernel, KernelType)
  ├─ svm         (SVM) ─→ depends on kernel
  └─ evaluation  (metrics, k_fold_cv, grid_search) ─→ depends on svm + data_loader
```

### `data_loader` (`src/data_loader.{hpp,cpp}`)

Owns the `Dataset` struct (`std::vector<std::vector<double>> X`, `std::vector<int> y`) and three free functions:

- **`load_csv(filepath)`** — reads `data/wdbc.csv`, skips the patient-ID column, maps the diagnosis column `M`/`B` to integer labels `+1`/`−1`, and parses the 30 remaining numeric features per row.
- **`normalise(dataset)`** — z-score normalises each feature column in place using population mean and population standard deviation (dividing by `n`, not `n − 1`). A feature with near-zero variance is left untouched to avoid division by zero.
- **`train_test_split(dataset, train, test, test_ratio, seed)`** — deterministically shuffles sample indices with a seeded linear congruential generator (`a = 1103515245`, `c = 12345`), then partitions into train and test sets. The same LCG constants are reused for k-fold shuffling in `evaluation` so that all runs are bitwise reproducible.

### `kernel` (`src/kernel.{hpp,cpp}`)

Defines `enum class KernelType { LINEAR, RBF, POLYNOMIAL }` and a single `Kernel` class. The `Kernel` constructor takes the type plus the three hyperparameters any kernel could need (`gamma`, `coef0`, `degree`); unused fields are ignored per kernel type. The single entry point is:

```cpp
double Kernel::compute(const std::vector<double>& a,
                       const std::vector<double>& b) const;
```

A `switch` on `type_` dispatches to the correct closed-form expression. This design keeps the SVM solver completely kernel-agnostic: `SVM` only ever sees `Kernel::compute()`, so adding a new kernel is a matter of adding one enum value and one case in the switch.

### `svm` (`src/svm.{hpp,cpp}`)

The core learning algorithm. The `SVM` class holds:

- **Configuration:** the `Kernel` (stored by value, so the trainer owns its copy), regularisation `C`, KKT tolerance `tol`, and iteration cap `max_iter`.
- **Model state after training:** the bias `b_`, the Lagrange multipliers `alphas_`, and the extracted support vectors (`support_vectors_`, `support_labels_`, `support_alphas_`).

The public API is small:

```cpp
void train(const std::vector<std::vector<double>>& X,
           const std::vector<int>& y);
int  predict(const std::vector<double>& x) const;
std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
int  support_vector_count() const;
```

Training runs SMO with a precomputed kernel matrix, an error cache, and the second-choice heuristic for pair selection. After convergence, any training sample with `α_i > 1e-8` is extracted as a support vector, and the original training set is not retained. Prediction is `sign(Σ α_i y_i K(x_i, x) + b)`.

Full mathematical details — KKT conditions, update bounds, bias rule — live in [algorithm.md](algorithm.md).

### `evaluation` (`src/evaluation.{hpp,cpp}`)

Metrics and model-selection helpers. This module never stores state; every function takes its inputs explicitly.

- **`ConfusionMatrix`** — plain struct with `tp`, `tn`, `fp`, `fn` counters.
- **`accuracy`, `precision`, `recall`, `f1_score`** — standard definitions, each a pure function over predictions and labels.
- **`k_fold_cv(dataset, kernel, C, k, seed, verbose)`** — shuffles the dataset using the same LCG as `data_loader`, splits into `k` folds, trains a fresh `SVM` per fold, and returns the mean fold accuracy.
- **`grid_search(dataset, type, C_values, gamma_values, degree_values, k, seed)`** — runs the full Cartesian product of the provided hyperparameter values, calling `k_fold_cv` for each combination, and returns the best configuration by CV accuracy. The `degree_values` vector is ignored for Linear and RBF kernels (forced to `{3}` internally).
- **`print_results(predicted, actual)`** — formats accuracy, precision, recall, F1, and the confusion matrix for stdout. This is where the tables in the report come from.

### `main` (`src/main.cpp`)

A fixed, linear pipeline that ties everything together. There are no command-line flags — running the binary performs every step of the demonstration in one pass:

1. Load `data/wdbc.csv` via `load_csv`.
2. Count malignant vs. benign samples and report the class distribution.
3. Z-score normalise the features.
4. Split 80/20 into train/test with seed `42`.
5. Train an RBF-kernel SVM with default `C = 1.0`, `γ = 0.1` and print the baseline evaluation.
6. Run 5-fold cross-validation on the full dataset with the same RBF configuration.
7. Train and evaluate Linear and Polynomial kernels for a three-kernel comparison.
8. Run grid search over `C ∈ {0.1, 1, 10, 100}`, `γ ∈ {0.001, 0.01, 0.1, 1}`, `degree ∈ {2, 3, 4}` for each of the three kernels.
9. Retrain each kernel on the 80/20 split using its grid-search winner and print the final evaluation.

The deliberate choice to hard-code the pipeline (instead of exposing CLI flags) keeps the demonstration self-contained: a single `./build/svm_classifier` invocation reproduces every result in the assignment report.

### `scripts/plot_results.py`

A standalone Python script that parses the classifier's stdout and generates six publication-quality PNG figures. It uses regex-based line-by-line parsing with a state machine to extract metrics, confusion matrices, CV fold results, and grid search data from the text output. No modifications to the C++ code are needed — the script works entirely as a post-processing step.

Dependencies are `matplotlib` and `numpy` only. The script supports three input modes: reading `run_output.txt` (default), an explicit `--input PATH`, or `--stdin` for piping directly from the classifier. Figures are saved to `figures/` at 300 DPI.

## Build System

`CMakeLists.txt` is intentionally minimal:

- Requires CMake 3.14+ and a C++17 compiler.
- Single target `svm_classifier` linking all five `.cpp` files.
- `src/` added as a private include directory so internal headers can use unqualified `#include`.
- No dependencies — not even a unit-test framework. The program *is* its own test: a successful run ending with the optimised-kernel table is the acceptance criterion.

## Reproducibility Notes

Every source of randomness in the codebase is seeded:

- `train_test_split` defaults to `seed = 42`.
- `k_fold_cv` defaults to `seed = 42`.
- `grid_search` forwards its seed to every inner CV call.
- The shuffles use a plain LCG (not `<random>`), which means the sequence is portable across platforms and compilers — the same seed produces the same shuffle on any machine.

As a result, every accuracy, precision, recall, and F1 value reported in `resources/assignment-report.md` is exactly reproducible from a clean build.
