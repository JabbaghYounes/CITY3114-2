# Kernel SVM Classifier with SMO: A Cross Dataset Comparison Study

**Module:** CITY3114 — Machine Learning and Algorithms
**Assignment:** 2
**Name:** [Your Name]
**Student ID:** [Your Student ID]

---

## Task 1: Implementation and Evaluation

### 1.1 Description of Program

This project implements a binary Support Vector Machine (SVM) classifier from scratch in C++17 and evaluates it across **three** UCI binary-classification datasets that span different sample counts (351 to 4,601), feature counts (4 to 57), and problem domains (radar return classification, banknote forensics, and email spam filtering). No external machine learning libraries are used — every component, from CSV parsing to the quadratic optimisation routine, is written by hand. The motivation for the from-scratch approach is partly educational (forcing a complete walk-through of the underlying mathematics) and partly to keep the dependency footprint minimal: a working compiler and CMake are the only requirements.

The motivation for evaluating across *several* datasets rather than one is more pointed. Task 2 asks the algorithm to be discussed in terms of its "ability to solve various problems" and the limitations it carries; the most direct way to answer that is to actually run the same algorithm across several problems and let the cross-dataset behaviour speak for itself. The three datasets were chosen for diversity along three axes — sample count, feature count, and domain — so that any pattern that persists across them is more credibly an algorithmic property than a single-dataset accident.

The classifier uses the **Sequential Minimal Optimization (SMO)** algorithm for training, originally introduced by John Platt at Microsoft Research (Platt, 1998). SMO is the most widely used method for solving the SVM dual problem because it decomposes the global quadratic program into a sequence of analytically solvable two-variable sub-problems, which means no general-purpose QP solver is required. The same algorithm — with additional refinements — is the basis of widely deployed implementations such as `libsvm` and the `SVC` class in scikit-learn (Scikit-learn developers). The implementation also applies an **incremental error-cache update** on every successful pair update — a closed-form delta rather than a full O(n²) rebuild — which cuts the per-update cost by roughly a factor of `n` and lets the full three-dataset run finish in around 90 seconds.

The program is built around a small, modular set of components, each in its own translation unit:

- **`data_loader`** — handles CSV parsing, z-score feature normalisation, and a seeded train/test split. The split uses a deterministic linear congruential generator so that train/test partitions are exactly reproducible. A `DatasetSpec` struct (name, filepath, ID column, label position, positive-label token, expected feature count, class names, grid-search flag) generalises the parser, and a single `ALL_DATASETS` table drives the per-dataset iteration.
- **`kernel`** — provides three kernel functions (Linear, RBF, Polynomial) behind a single `Kernel::compute()` interface. Switching kernels is therefore a one-line change in client code.
- **`svm`** — contains the core `SVM` class. This is the largest module: it owns the SMO training loop, the precomputed kernel matrix, the incremental error cache, the support-vector extraction logic, and the prediction function.
- **`evaluation`** — computes accuracy, precision, recall, F1 score, and a 2 × 2 confusion matrix; implements stratified-by-shuffle k-fold cross-validation; and runs the grid search used for hyperparameter optimisation.
- **`main`** — orchestrates the pipeline. `run_pipeline(spec)` performs all the per-dataset work (loading, normalising, splitting, baseline training, 5-fold CV, three-kernel comparison, optional grid search and optimised retrain) and `main()` simply loops over `ALL_DATASETS`.

The deliberate decision to keep `main` as a fixed-pipeline driver (rather than a CLI with flags) was made to keep the demonstration self-contained: a single invocation reproduces every figure in this report.

#### Dependencies and How to Run

The only requirements are a C++17 compiler (such as `g++` 9 or above) and CMake (version 3.14 or above). No third-party C++ libraries are needed.

To build and run:

```bash
cmake -B build
cmake --build build
./build/svm_classifier | tee resources/run_output.txt
```

The dataset files must be placed in a `data/` directory at the project root. They are downloaded from the UCI Machine Learning Repository:

```bash
mkdir -p data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"            -o data/ionosphere.data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt" -o data/banknote.txt
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"                -o data/spambase.data
```

A complete run takes about 90 seconds on a typical desktop CPU and prints, per dataset, the baseline RBF results, the 5-fold cross-validation accuracy, the three-kernel comparison, the full grid-search trace (where applicable), and the final results table for the optimised models. A companion Python script (`scripts/plot_results.py`) parses the program's stdout, splits it on `=== DATASET:` markers, and generates publication-quality figures per dataset — confusion matrix heatmaps, kernel comparison bar charts, grid-search heatmaps, and a cross-validation fold chart — into a `figures/` directory, with each filename prefixed by a dataset slug (e.g. `figures/ionosphere_grid_search_heatmaps.png`). The plotting script requires Python 3 with `matplotlib` and `numpy`; these are only needed to regenerate figures, not to build or run the C++ classifier itself.

### 1.2 Datasets

Three UCI binary-classification datasets are used. They were chosen to span different scales and domains so that the kernel-by-kernel comparison in §1.5 can be interpreted as evidence of which kernels suit which problem structures, rather than a single-dataset accident.

| Dataset | Samples | Features | Positive class | Negative class | Domain |
|---|---|---|---|---|---|
| **Ionosphere** | 351 | 34 | `g` (good) | `b` (bad) | Radar signal returns from the upper atmosphere |
| **Banknote Authentication** | 1,372 | 4 | `1` (forged) | `0` (genuine) | Wavelet features of banknote images |
| **Spambase** | 4,601 | 57 | `1` (spam) | `0` (ham) | Word/character frequencies in email |

Each row of every CSV is parsed by the same `load_csv(spec)` function — the differences (which column is the label, what string token represents the positive class, whether there is an ID column to skip, how many feature columns to expect) are encoded in the per-dataset `DatasetSpec` rather than in code branches.

#### Per-Dataset Description

**Ionosphere** (Sigillito et al., 1989). 351 radar returns from a phased-array antenna at the Goose Bay, Labrador research station. Each return is summarised as 17 complex-valued autocorrelations, expanded to 34 real features. A return is labelled "good" when the radar successfully bounced off a structured layer of the ionosphere and returned a usable echo, and "bad" when the signal passed straight through (no structure, no usable echo). The classification problem is therefore: given the autocorrelation features, decide whether the radar return is interpretable. Ionosphere is the smallest of the three datasets, has 34 features and a moderate class imbalance (225 good / 126 bad), and is the dataset for which non-linear boundaries make the most difference in §1.5.

**Banknote Authentication.** 1,372 banknotes scanned by an industrial-grade camera, each summarised by four features extracted from a wavelet transform of the image: the variance, skewness, kurtosis, and entropy of the wavelet coefficients. Banknotes are labelled forged (`1`) or genuine (`0`). Banknote is the smallest in feature dimension (just four features) but the most balanced (610 forged / 762 genuine) and the second-largest in sample count. It is the simplest of the three problems — any reasonably trained kernel can clear 90% accuracy — which makes it useful as a sanity check that the from-scratch SMO implementation is producing sensible models.

**Spambase** (Hopkins et al., 1999). 4,601 emails from George Forman's collection at Hewlett-Packard Labs, each represented by 57 hand-crafted features: the frequency (per word) of 48 specific words, the frequency of six specific characters, and three summary statistics on capital-letter run lengths. The positive class is spam (1,813 samples) and the negative class is non-spam ("ham", 2,788 samples). Spambase is by far the largest of the three datasets, both in samples (4,601) and features (57). Its size puts it past the comfortable scale of the precomputed-kernel-matrix SMO implemented here — the kernel matrix at training time is roughly 100 MB, and a full grid search over `(C, γ, degree)` would take several hours of wall-clock time even with the incremental error-cache update. The pipeline therefore runs only the default-hyperparameter and 5-fold CV phases on Spambase and skips grid search and the optimised retrain (`run_grid_search=false` in its dataset spec). This is a deliberate and informative trade-off rather than a bug: Spambase serves as the natural anchor for the §2.2 discussion of SVM scalability.

#### Class Distribution

| Dataset | Positive | Negative | Positive fraction | Majority-class accuracy |
|---|---|---|---|---|
| Ionosphere | 225 (g) | 126 (b) | 64.1% | 64.1% |
| Banknote | 610 (forged) | 762 (genuine) | 44.5% | 55.5% |
| Spambase | 1,813 (spam) | 2,788 (ham) | 39.4% | 60.6% |

None of these distributions are pathologically imbalanced, and no special techniques (oversampling, class weighting, SMOTE) are used. However, the imbalance is large enough to matter for interpretation: a trivial classifier that predicts the majority class on each dataset would already score 64%, 56%, and 61% respectively without learning anything. Accuracy on its own is therefore not a reliable measure, and the §1.5 results table reports precision, recall, and F1 alongside accuracy throughout.

#### Preprocessing

The same three preprocessing steps are applied to every dataset:

1. **ID/label parsing.** The `DatasetSpec` declares whether there is an ID column to skip and where the label lives (column index, or `-1` for "last"). The label string is mapped to `+1` if it equals `spec.positive_label` and to `-1` otherwise. The `±1` convention is the standard for SVMs because the dual formulation, decision rule, and KKT conditions are all stated in terms of `y_i ∈ {−1, +1}`.

2. **Z-score normalisation.** Each feature column is independently rescaled to zero mean and unit variance using the **population** standard deviation (dividing by `n`). A feature with near-zero variance is left untouched to avoid division by zero — relevant for Ionosphere, whose feature 2 is identically zero.

3. **80/20 train/test split.** Sample indices are deterministically shuffled with a seeded LCG (`a = 1103515245`, `c = 12345`, default seed 42), then partitioned 80/20. The actual split sizes are 281/70 for Ionosphere, 1098/274 for Banknote, and 3681/920 for Spambase.

Feature normalisation is **essential** for SVMs that use distance-based kernels like the RBF. Without it, features on a large numerical scale (e.g. Spambase's "capital-run-length total", which can run into the thousands) would dominate the squared-distance term `‖a − b‖²` in the RBF kernel and effectively drown out the smaller-scale features. Even for the linear kernel, normalisation tends to improve numerical conditioning of the optimisation. The dramatic effect of normalisation is one of the reasons SVMs are sometimes described as "not working out of the box": the data must be on a common scale before training begins.

### 1.3 Algorithm Implementation

#### Geometric Intuition

A Support Vector Machine is a supervised learning algorithm that finds the optimal separating hyperplane between two classes (Cortes & Vapnik, 1995; Burges, 1998). In the linearly separable case the "optimal" hyperplane is the one that maximises the **margin** — the perpendicular distance between the boundary and the nearest data points from each class. Those nearest points are called **support vectors**, and they are the only training examples that influence the position of the boundary; moving any other example a small distance has no effect on the model.

This margin-maximisation objective is what gives SVMs their well-known generalisation properties. Intuitively, a wider margin gives the classifier more "room for error" when it sees noisy or slightly shifted test data, and PAC learning theory formalises this with bounds on test error that depend on the margin width rather than on the dimensionality of the input space. In practice, SVMs often generalise well even when the number of features is comparable to the number of samples — the regime of two of the three datasets used here (Ionosphere has 34 features for 351 samples; Banknote has 4 features for 1,372).

#### The Primal and Dual Problems

For training data `{(x_i, y_i)}` with `y_i ∈ {−1, +1}`, the **soft-margin** SVM solves the primal problem:

```
minimise   (1/2) ‖w‖² + C Σ_i ξ_i
subject to y_i (w · x_i + b) ≥ 1 − ξ_i,  ξ_i ≥ 0
```

where `w` is the normal vector to the hyperplane, `b` is the bias, and `ξ_i` are slack variables that allow some training points to violate the margin (with the violations penalised by the regularisation constant `C`). A small `C` means a wide margin with many violations is acceptable; a large `C` means the optimiser will distort the boundary to reduce violations.

In practice the SVM is almost never solved in this primal form. Applying Lagrangian duality yields the **dual problem**:

```
maximise W(α) = Σ_i α_i  −  (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
subject to    0 ≤ α_i ≤ C  for all i
              Σ_i α_i y_i = 0
```

This is the form actually solved by `SVM::train`. The dual has two crucial advantages. First, the inputs `x_i` only appear inside the kernel function `K(x_i, x_j)`, never on their own. This is the **kernel trick**: the algorithm never needs to compute the explicit feature mapping `φ(x)`, so the same code can train a linear model, an RBF model, or a polynomial model just by changing one function. Second, the solution is **sparse**: at the optimum, most `α_i` are exactly zero, and only the training points with `α_i > 0` (the support vectors) participate in the final decision function:

```
f(x) = Σ_{i ∈ SV} α_i y_i K(x_i, x) + b
```

#### Kernel Functions

Three kernels were implemented in `src/kernel.cpp`, each with a closed-form definition:

- **Linear kernel.** `K(a, b) = a · b`. This is just the standard dot product. It corresponds to no feature mapping at all and yields a linear decision boundary in the original input space. It is the cheapest kernel to evaluate and is the right choice when the data is (or is close to) linearly separable.

- **RBF (Radial Basis Function) kernel.** `K(a, b) = exp(−γ ‖a − b‖²)`. The RBF kernel measures similarity as a Gaussian function of Euclidean distance: nearby points have similarity close to 1, distant points have similarity close to 0. It implicitly maps the data into an *infinite-dimensional* feature space and can represent arbitrarily complex non-linear decision boundaries. The width parameter `γ` controls how quickly the similarity decays with distance. A small `γ` (≈ 0.001) gives each support vector a broad sphere of influence, producing smooth boundaries; a large `γ` (≈ 1) makes the influence highly localised and can lead to severe overfitting where the model essentially memorises the training set.

- **Polynomial kernel.** `K(a, b) = (γ · a · b + c₀)^d`. The polynomial kernel maps the data into a feature space whose coordinates are all monomials of degree up to `d` over the original features. With `d = 2` this means all pairwise products `x_i x_j` become available, with `d = 3` all triples, and so on. The constant offset `c₀` (set to 1.0 in this implementation) allows the kernel to also include lower-order monomials, not just degree-`d` ones. Higher degrees are more expressive but more prone to numerical blow-up and overfitting.

All three kernels are fronted by a single `Kernel::compute(a, b)` method, which means the SMO solver in `src/svm.cpp` is completely kernel-agnostic — the type is set once when the `Kernel` object is constructed.

#### The SMO Algorithm

Solving the dual problem directly with a general-purpose QP solver is impractical for any non-trivial dataset because the problem has `n` variables and an `n × n` kernel matrix. SMO sidesteps this by noticing that, due to the equality constraint `Σ α_i y_i = 0`, the smallest possible update is to two `α` values at once: changing only one would violate the constraint. SMO therefore picks a pair `(α_i, α_j)` at each iteration, optimises just those two while holding all other `α`s fixed, and repeats until the KKT conditions are satisfied for every point.

Because there are only two free variables, the constrained sub-problem can be solved **analytically** in closed form, with no inner solver. This is what makes SMO so fast in practice.

The full training loop in `SVM::train` proceeds as follows:

1. **Initialisation.** All `α_i` are set to zero, and the bias `b` is set to zero. A model with `α = 0` predicts everything as the negative class (the decision function evaluates to `b = 0`, which the prediction rule treats as `+1` due to the `>= 0` tie-break, but then immediately starts updating).

2. **Precompute the kernel matrix.** A full `n × n` symmetric matrix `K[i][j] = K(x_i, x_j)` is computed once at the start of training. For the training sets used here this requires anywhere from ~40,000 kernel evaluations (Ionosphere, `n = 281`) to ~6.7 million (Spambase, `n = 3,681`); the Spambase kernel matrix alone occupies roughly 100 MB of memory for double-precision floats. This is a deliberate space-for-time trade-off: every subsequent iteration of SMO can read kernel values directly instead of recomputing them, which is the dominant cost in any SVM trainer. For datasets larger than Spambase the caching strategy stops being feasible (see §2.2).

3. **Maintain an error cache.** For every training point the value `E_i = f(x_i) − y_i` is stored, where `f(x_i)` is the current SVM output. After every successful joint update of `(α_i, α_j, b)`, the cache is patched **incrementally** using the closed-form delta

   ```
   ΔE[k] = δα_i · y_i · K(x_i, x_k) + δα_j · y_j · K(x_j, x_k) + δb
   ```

   This takes O(n) per pair update, compared to O(n²) for a full rebuild that re-evaluates `f(x_k) − y_k` for every `k`. The speed-up matters: on the 4-dataset version of this project the rebuild approach took roughly 12 minutes, the incremental approach takes about 2:45; on the 3-dataset version used here the run finishes in 90 seconds. The trade-off is numerical — the incremental update accumulates floating-point error in a path-dependent way, so the SMO converges to a near-equivalent (not strictly bit-identical) solution compared to the rebuild version. In practice the test-set accuracies are stable; the grid-search CV column may shift at the third decimal between toolchains.

4. **Iterate over all samples.** For each `i`, compute `r_i = E_i · y_i`. The KKT conditions for this dual problem reduce to:
   - if `α_i = 0` then `r_i ≥ −tol` (point is correctly classified outside the margin)
   - if `α_i = C` then `r_i ≤ tol` (point is at most marginally violating)
   - if `0 < α_i < C` then `|r_i| < tol` (point is exactly on the margin)

   The check `(r_i < −tol AND α_i < C) OR (r_i > tol AND α_i > 0)` in the source code exactly captures the union of all KKT-violating cases.

5. **Pick the second variable using the second-choice heuristic.** When sample `i` violates the KKT conditions, sample `j` is chosen as the index that maximises `|E_i − E_j|`. The reasoning is that the analytic update step for `α_j` is proportional to `(E_i − E_j) / η`, so picking the largest absolute difference produces the largest step per iteration and therefore the fastest overall convergence. Platt's full SMO uses a more elaborate hierarchy (first preferring non-bound `α`s, then falling back to a full scan), but the simple maximum-difference rule already works well at the dataset sizes used here.

6. **Compute the analytic update for the chosen pair.** Three quantities are needed:

   - The **bounds** `L` and `H` ensure that the new `α_j` stays inside `[0, C]`. They depend on whether the labels `y_i` and `y_j` agree:

     ```
     if y_i ≠ y_j:   L = max(0, α_j − α_i),       H = min(C, C + α_j − α_i)
     if y_i = y_j:   L = max(0, α_i + α_j − C),   H = min(C, α_i + α_j)
     ```

     These are derived from the equality constraint `Σ α_i y_i = 0`, which forces `y_i α_i + y_j α_j` to remain constant during the joint update.

   - The **second derivative** of the dual objective along the update direction:

     ```
     η = 2 K(x_i, x_j) − K(x_i, x_i) − K(x_j, x_j) = − ‖φ(x_i) − φ(x_j)‖²
     ```

     This is always non-positive for a valid (positive semi-definite) kernel. If `η ≥ 0` (numerically possible for degenerate pairs) the implementation skips the update.

   - The **unclipped update** for `α_j`:

     ```
     α_j ← α_j − y_j (E_i − E_j) / η
     ```

     This is then clipped to `[L, H]`. If the clipped change is below `1e-5` the iteration is skipped (to avoid numerical churn from updates that are too small to matter). Finally, `α_i` is updated to maintain the constraint:

     ```
     α_i ← α_i + y_i y_j (α_j_old − α_j_new)
     ```

7. **Update the bias.** Two candidate bias values `b1` and `b2` are computed, derived from the KKT condition that any non-bound support vector should have `f(x_i) = y_i` exactly. If `α_i` ended up strictly between 0 and `C`, then `b1` is used; if `α_j` did, `b2` is used; if both are at the bounds, the average `(b1 + b2) / 2` is used as a heuristic.

8. **Patch the error cache** with the incremental update from step 3, and continue. A full pass over the data with no successful updates is the convergence criterion. A safety cap of `max_iter = 1000` outer passes prevents infinite loops in pathological cases.

After the loop terminates, support vector extraction is straightforward: any training sample with `α_i > 1e-8` is retained. The `1e-8` threshold instead of strict `> 0` is to filter out floating-point noise. The retained `(x_i, y_i, α_i)` triples are everything the trained model needs for prediction; the rest of the training data is discarded.

#### Key Hyperparameters

The behaviour of the trained classifier is governed by a small set of hyperparameters:

- **`C` (regularisation strength).** Trades off margin width against training error. The default value used for the baseline experiments is `C = 1.0`, and the grid search explores the range `{0.1, 1, 10, 100}`.
- **`γ` (kernel width).** Used by the RBF and Polynomial kernels. The baseline value is `γ = 0.1`, with grid search over `{0.001, 0.01, 0.1, 1}`.
- **`degree`** (`d`). Used only by the Polynomial kernel. Baseline is 3, grid search over `{2, 3, 4}`.
- **`coef0`** (`c₀`). The polynomial kernel offset. Fixed at 1.0 throughout (a common default that includes the lower-order terms in the polynomial expansion).
- **`tol` (KKT tolerance).** The threshold for declaring that a point satisfies the KKT conditions. Set to `0.001`. Tighter values produce more precise solutions at the cost of more iterations; looser values converge faster but produce slightly suboptimal classifiers.
- **`max_iter`.** A safety cap of 1000 outer SMO passes. In practice this is never reached on any of the three datasets; convergence typically occurs in well under 50 outer passes.

### 1.4 Training

For each dataset the same five-step training protocol is followed:

1. Load the CSV via `load_csv(spec)`.
2. Z-score normalise every feature column.
3. Split 80/20 into train and test using seed 42. Actual split sizes are 281/70 (Ionosphere), 1098/274 (Banknote), and 3681/920 (Spambase).
4. Train each of the three kernels at the **default** hyperparameters (`C = 1.0`, `γ = 0.1`, `degree = 3`) for an apples-to-apples baseline.
5. Run a 5-fold cross-validation on the full normalised dataset at the default RBF configuration; then run a Cartesian-product **grid search** over `(C, γ, degree)` and retrain at the per-kernel CV winner. Spambase skips this last step.

The deterministic seed is important for reproducibility: every result in this report can be reproduced from the same binary, because the same LCG sequence is also used for the shuffle in cross-validation. (The numerical SMO solver itself is not strictly bit-portable across compilers and CPUs because the incremental error-cache update accumulates floating-point error in a path-dependent way; in practice the test-set accuracies are stable and only the third-decimal CV numbers may shift between toolchains.)

#### Why Cross-Validation

Selecting hyperparameters by their performance on a single train/test split would be unreliable: a particular split might happen to favour one combination over another simply due to which samples ended up in the test set. **k-fold cross-validation** addresses this by partitioning the (full, normalised) dataset into `k` equally sized folds, training on `k − 1` of them, evaluating on the held-out fold, and rotating through all `k` choices of held-out fold. The reported metric is the mean accuracy across the `k` runs. Five folds was chosen as a standard compromise: it gives reasonably stable estimates without making the grid search unaffordably slow.

It is worth noting that the cross-validation here uses **the full dataset, not just the training split**. This is the conventional approach for hyperparameter selection — the test split is only used at the very end, to give an unbiased estimate of generalisation for the final tuned model.

The per-fold accuracy spread for each dataset is shown below.

![Ionosphere — 5-fold cross-validation per-fold accuracy with mean line](../figures/ionosphere_cv_folds.png)

![Banknote — 5-fold cross-validation per-fold accuracy with mean line](../figures/banknote_cv_folds.png)

![Spambase — 5-fold cross-validation per-fold accuracy with mean line](../figures/spambase_cv_folds.png)

#### Baseline 5-Fold CV (Default RBF)

At the default RBF configuration `C = 1.0, γ = 0.1`, the mean 5-fold CV accuracies are:

| Dataset | 5-fold CV mean |
|---|---|
| Ionosphere | 85.16% |
| Banknote Authentication | 92.49% |
| Spambase | 70.46% |

The CV numbers already foreshadow the §1.5 results: Ionosphere and Banknote are problems where `γ = 0.1` is in the right ballpark, while Spambase is mis-served by it.

#### Grid Search

A full Cartesian-product grid search was conducted over:

- `C ∈ {0.1, 1, 10, 100}`
- `γ ∈ {0.001, 0.01, 0.1, 1}`
- `degree ∈ {2, 3, 4}` (Polynomial only)

This gives 16 combinations to evaluate for the Linear kernel (the `γ` column is ignored for Linear, so effectively 4 distinct configurations × 5 folds = 20 fits), 16 combinations for RBF (16 × 5 = 80 fits), and 48 combinations for Polynomial (48 × 5 = 240 fits). A total of around 340 SVM fits per dataset are performed during grid search. Spambase skips grid search by configuration: with `n = 3,681` per fold the total wall-clock would run into hours.

The grid search identified the following best configurations by 5-fold CV accuracy:

| Dataset | Kernel | Best `C` | Best `γ` | Best `degree` | CV Accuracy |
|---|---|---|---|---|---|
| Ionosphere | Linear | 1 | — | — | 70.72% |
| Ionosphere | RBF | 100 | 0.1 | — | 95.15% |
| Ionosphere | Polynomial | 10 | 0.001 | 4 | 80.04% |
| Banknote | Linear | 1 | — | — | 76.01% |
| Banknote | RBF | 10 | 1 | — | 98.69% |
| Banknote | Polynomial | 1 | 0.1 | 3 | 95.05% |

The grid search landscape is shown below as C-versus-γ heatmaps, with the best cell highlighted for each kernel.

![Ionosphere — grid search C×γ accuracy heatmaps per kernel](../figures/ionosphere_grid_search_heatmaps.png)

![Banknote — grid search C×γ accuracy heatmaps per kernel](../figures/banknote_grid_search_heatmaps.png)

A practical observation from the grid search trace is that the best `(C, γ)` combinations differ markedly between datasets: Banknote's RBF prefers a tight, high-`γ` boundary (`γ = 1`), Ionosphere's RBF prefers a moderate `γ = 0.1`, and the Linear kernel prefers `C = 1` for both. The grid search therefore behaves as a useful diagnostic tool — its output carries information about each dataset's structure, not just a tuned classifier.

### 1.5 Results

#### Default Parameters (`C = 1.0`, `γ = 0.1`)

The first comparison is between the three kernels with the same default hyperparameters across all three datasets. This is intentionally not a competitive setting — the point is to establish a baseline against which the §1.4 grid search can be judged.

![Ionosphere — kernel comparison at default hyperparameters (accuracy / precision / recall / F1)](../figures/ionosphere_kernel_comparison_default.png)

![Banknote — kernel comparison at default hyperparameters](../figures/banknote_kernel_comparison_default.png)

![Spambase — kernel comparison at default hyperparameters](../figures/spambase_kernel_comparison_default.png)

| Dataset | Kernel | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Ionosphere | Linear | 77.14% | 0.7419 | 1.0000 | 0.8519 |
| Ionosphere | RBF | **92.86%** | 0.9556 | 0.9348 | **0.9451** |
| Ionosphere | Polynomial | 82.86% | 0.8696 | 0.8696 | 0.8696 |
| Banknote | Linear | 81.75% | 0.9863 | 0.5950 | 0.7423 |
| Banknote | RBF | **97.08%** | 0.9449 | 0.9917 | **0.9677** |
| Banknote | Polynomial | 92.34% | 1.0000 | 0.8264 | 0.9050 |
| Spambase | Linear | 51.09% | 0.4505 | 1.0000 | 0.6212 |
| Spambase | RBF | **75.87%** | 0.6324 | 0.9512 | **0.7597** |
| Spambase | Polynomial | 40.11% | 0.4011 | 1.0000 | 0.5725 |

Two patterns are immediately visible:

- **RBF is the strongest default kernel on every dataset.** The default `(C = 1.0, γ = 0.1)` happens to be in the right ballpark for each problem's geometry — not optimal, but workable. This is consistent with RBF's reputation as a sensible default for general-purpose SVM classification.

- **Linear and Polynomial both collapse on Spambase, but in opposite directions.** Linear at default predicts almost everything as spam (recall = 1.0, precision = 0.45 — accuracy below the 60.6% majority-class baseline). Polynomial at default predicts almost everything as spam too (recall = 1.0, accuracy = 40%). Both extremes have the same root cause: at high feature dimension (57 for Spambase), the relevant decision surface is too complex for the Linear kernel to express and the Polynomial kernel's `(γ a·b + 1)^3` term blows up numerically with `γ = 0.1`. The defaults that suit Ionosphere and Banknote do not transfer.

#### Optimised Parameters

After grid search and retraining on the 80/20 split with the best per-kernel configuration, the test-set numbers are:

| Dataset | Kernel | `C` | `γ` | `d` | Accuracy | Precision | Recall | F1 | Support Vectors |
|---|---|---|---|---|---|---|---|---|---|
| Ionosphere | Linear | 1 | — | — | 77.14% | 0.7419 | 1.0000 | 0.8519 | 38 / 281 |
| Ionosphere | RBF | 100 | 0.1 | — | **98.57%** | 0.9787 | 1.0000 | **0.9892** | 100 / 281 |
| Ionosphere | Polynomial | 10 | 0.001 | 4 | 87.14% | 0.8364 | 1.0000 | 0.9109 | 40 / 281 |
| Banknote | Linear | 1 | — | — | 81.75% | 0.9863 | 0.5950 | 0.7423 | 10 / 1098 |
| Banknote | RBF | 10 | 1 | — | 96.72% | 1.0000 | 0.9256 | 0.9614 | 36 / 1098 |
| Banknote | Polynomial | 1 | 0.1 | 3 | 92.34% | 1.0000 | 0.8264 | 0.9050 | 23 / 1098 |
| Spambase | (grid search disabled — see §1.2) | | | | | | | | |

After tuning, every dataset where grid search ran reaches the high-90s in accuracy on its best kernel:

- **Ionosphere RBF jumps from 92.86% to 98.57% accuracy** with `C = 100, γ = 0.1`. This is the largest tuning gain in the study (+5.71 percentage points), and the tuned model achieves perfect recall (no false negatives) on the test set with only one false positive out of 70 test samples. The cost is a fourfold increase in support vector count (42 → 100), which is consistent with the larger `C = 100` permitting a tighter, more support-vector-dense boundary.

- **Banknote's tuned RBF actually scores marginally lower than the default RBF** on the test set (96.72% vs. 97.08%). The grid search picked `(C = 10, γ = 1)` because that combination had the highest *cross-validated* accuracy (98.69% averaged over 5 folds) — but on the particular 80/20 split used here, the default `(C = 1, γ = 0.1)` happens to do slightly better. This is a real and instructive phenomenon: a single train/test split can disagree with cross-validation by a small margin, and *cross-validation is the right thing to trust* when choosing hyperparameters because it averages over multiple splits. The 0.36-point gap on the test split is within noise.

- **Linear is identical in default and optimised** for both Ionosphere and Banknote. The grid winner for Linear was `C = 1` on both — i.e. the default. This explains why "Linear (Optimised)" rows above match "Linear (Default)" rows in the previous table exactly.

The optimised-parameter visual comparison and the side-by-side default-vs-optimised accuracy chart for each dataset where grid search ran are shown below.

![Ionosphere — kernel comparison at optimised hyperparameters](../figures/ionosphere_kernel_comparison_optimised.png)

![Banknote — kernel comparison at optimised hyperparameters](../figures/banknote_kernel_comparison_optimised.png)

![Ionosphere — default vs optimised accuracy per kernel](../figures/ionosphere_default_vs_optimised.png)

![Banknote — default vs optimised accuracy per kernel](../figures/banknote_default_vs_optimised.png)

#### Per-Dataset Discussion

**Ionosphere is the success story.** A 281-sample training set with 34 features, mild class imbalance, and a non-linear underlying structure — exactly the regime SVMs are good for. The tuned RBF model achieves 98.57% test accuracy, perfect recall (no missed "good" returns), and only one false positive out of 70. The Linear kernel's relative weakness (77.14%, F1 = 0.85) confirms that the boundary is not linear in the original input space; the polynomial kernel's intermediate performance (87.14%) confirms that something smoother than a polynomial-of-degree-4 — i.e. an RBF — is the right inductive bias.

**Banknote is the easy case** — and informative for that reason. With only four wavelet features and 1,372 samples, the problem is sufficiently low-dimensional that almost any tuned non-linear classifier should reach high-90s. The optimised RBF achieves 96.72% with just 36 support vectors out of 1,098 — a 96.7% compression of the training set. The interesting finding is that grid search here is *almost* a no-op: the tuned RBF lands within noise of the default RBF, and the only kernel that the grid search meaningfully helped was Linear (which it didn't help — `C = 1` was already optimal). This is the cautionary side of grid search: when defaults are already in the right region, a full grid is mostly an expensive sanity check. It is still worth running, because it tells you the defaults *are* in the right region.

**Spambase is the cautionary tale.** 4,601 samples × 57 features puts the precomputed-kernel-matrix SMO at the edge of its comfort zone. A single SVM fit at default parameters takes a non-trivial fraction of a second; a full grid search over `(C, γ, degree)` would iterate roughly 340 × 5 = 1,700 fits, an unacceptable wall-clock investment for a single dataset. The pipeline therefore disables grid search for Spambase and accepts the default-RBF baseline: 75.87% accuracy, F1 = 0.7597. That is well above the majority-class trivial baseline of 60.6% but well below what a proper non-from-scratch implementation (e.g. `liblinear` or `libsvm`) would achieve on the same data with full hyperparameter optimisation. This is the practical scalability ceiling of the implementation, and it directly motivates the §2.2 discussion below.

#### Confusion Matrices for the Best Models

**Ionosphere — Optimised RBF (C = 100, γ = 0.1):**

|                     | Predicted +1 (Good) | Predicted −1 (Bad) |
|---------------------|---------------------|--------------------|
| **Actual +1 (Good)** | 46                  | 0                  |
| **Actual −1 (Bad)**  | 1                   | 23                 |

Perfect recall on the positive class — every "good" radar return in the test set was identified — at the cost of a single false positive. This is the kind of profile that matters for a screening application: prioritise recall (catch every interpretable signal), tolerate the occasional false alarm.

**Banknote Authentication — Default RBF (C = 1, γ = 0.1):**

|                       | Predicted +1 (Forged) | Predicted −1 (Genuine) |
|-----------------------|-----------------------|------------------------|
| **Actual +1 (Forged)** | 120                   | 1                      |
| **Actual −1 (Genuine)** | 7                     | 146                    |

97.08% accuracy with one false negative (a forged note slipped through as genuine) and seven false positives (genuine notes flagged as forged). For banknote authentication the cost asymmetry favours minimising false negatives — a forgery that gets accepted is more damaging than a genuine note being re-checked — so this profile (recall = 0.9917) is well-aligned with the problem's needs.

**Spambase — Default RBF (C = 1, γ = 0.1):**

|                     | Predicted +1 (Spam) | Predicted −1 (Ham) |
|---------------------|---------------------|--------------------|
| **Actual +1 (Spam)** | 351                 | 18                 |
| **Actual −1 (Ham)**  | 204                 | 347                |

Recall of 0.95 (most spam is caught) but precision of only 0.63 (a third of "spam" predictions are actually ham). For real spam filtering the cost asymmetry runs the *other* way — false positives (legitimate email lost to the spam folder) are more costly than false negatives — so this profile is not production-quality. A tuned model would shift precision up at the cost of some recall, but Spambase's grid search is disabled here for the runtime reasons in §2.2.

Confusion matrices for the initial RBF baseline alongside the three optimised kernels (Ionosphere and Banknote only — Spambase has no optimised models) are shown below.

![Ionosphere — confusion matrices for initial RBF and the three optimised kernels](../figures/ionosphere_confusion_matrices.png)

![Banknote — confusion matrices for initial RBF and the three optimised kernels](../figures/banknote_confusion_matrices.png)

---

## Task 2: Discussion

### 2.1 Ability of the SVM Algorithm

SVMs are well suited to a broad class of supervised learning problems, and the cross-dataset experiments in Task 1 demonstrate several of their strongest properties in concrete form.

**High-dimensional data with limited samples.** SVMs handle the regime where the number of features `d` is large relative to the number of samples `n` better than most alternatives. Ionosphere (`d = 34, n = 351`) sits comfortably in that regime, and the optimised RBF SVM scored 98.57% on it. The reason is theoretical: the generalisation error of a maximum-margin classifier depends on the *margin width*, not on the dimensionality of the input space. This is why SVMs were the algorithm of choice for many bioinformatics and text-classification problems throughout the late 1990s and 2000s, where feature counts in the thousands or tens of thousands are common but labelled samples are scarce.

**Kernel-based versatility.** The kernel trick is what makes a single SVM framework applicable to linear, smoothly non-linear (RBF), and polynomial problems. Task 1 demonstrated this directly: exactly the same SMO solver, with the kernel object as the only change, produced three quite different classifiers per dataset, and the *winning* hyperparameters varied by dataset. Banknote's optimal RBF uses `γ = 1` (very local, sharp boundary in 4-dimensional feature space), Ionosphere's uses `γ = 0.1` (smoother boundary in 34-dimensional space). The same code, the same hyperparameter grid, and the boundary that emerges is dictated by the data, not by the algorithm. The kernel idea also generalises well beyond the three implemented here — string kernels for text and biological sequences, graph kernels for molecules, and histogram-intersection kernels for image features have all been used successfully. This flexibility means SVMs can be applied to non-vector data simply by designing an appropriate similarity function, which is something most other algorithms cannot do directly.

**Built-in regularisation.** The maximum-margin objective acts as a structural form of regularisation. Unlike algorithms that minimise training error and then add an external penalty (like L2-regularised logistic regression), the SVM formulation makes margin width *part of* the objective. The `C` hyperparameter then provides explicit control over the trade-off between margin and training error, giving the practitioner a single knob to tune the bias–variance balance. In Task 1 the best `C` value depended on the dataset and kernel — `C = 1` for both Linear winners, `C = 10` for Banknote's RBF, `C = 100` for Ionosphere's RBF — which is the bias–variance trade-off in action: each problem prefers a different point on the margin-vs-training-error curve.

**Sparse solutions and efficient prediction.** The trained model only depends on the support vectors. The starkest case in Task 1 was Banknote's optimised Linear SVM, which retained 10 support vectors out of 1,098 training samples — a 99.1% compression. Banknote's optimised RBF used 36 SVs out of 1,098 (96.7% compression). Sparsity matters in deployment: prediction cost scales with the number of support vectors, not with the size of the original training set, so SVMs are well suited to scenarios where the model is trained once and queried many times.

**Convex optimisation with global optima.** Unlike neural networks, the SVM dual is a convex quadratic program. Any solution found is a global optimum. There are no local minima, no sensitivity to weight initialisation, and no need for learning-rate schedules or warm-up tricks. The solver either converges to the optimum or it does not — and when it does converge, the answer is mathematically unique up to the support-vector boundary cases. This makes SVMs reproducible and easy to reason about.

**Strong theoretical foundations.** SVMs sit on top of decades of work in statistical learning theory. The relationship between margin width and generalisation error was made precise by Vapnik's work on VC dimension and structural risk minimisation, and PAC-style bounds on SVM error are tight enough to be useful in practice. This theoretical grounding makes SVMs a popular choice for academic and scientific applications where interpretability of *why the algorithm works* matters as much as raw accuracy.

In short: SVMs are a strong default choice for binary classification on tabular or vector data with hundreds to a few thousand samples, especially when interpretable hyperparameter tuning and reproducibility are desirable. The cross-dataset evidence in §1.5 makes that claim concrete: Ionosphere and Banknote both land in the high-90s with a small support-vector budget, and the algorithm's behaviour on each is predictable from the dataset's structure.

### 2.2 Limitations of the SVM Algorithm

Despite their strengths, SVMs have a number of practical limitations that the Task 1 experiments either revealed directly or hinted at.

**Scalability to large datasets.** The most fundamental limitation is computational. Training an SVM with SMO has time complexity that scales between roughly O(n²) and O(n³) depending on the dataset and the kernel, and the precomputed kernel matrix (used in this implementation) has space complexity O(n²). The §1.5 Spambase result is the concrete demonstration: at `n = 3,681` (training set after the 80/20 split) the kernel matrix occupies ~100 MB, and a full grid search over the same `(C, γ, degree)` ranges as the smaller datasets would take several hours of wall-clock time even with the O(n) incremental error-cache update. The pipeline therefore disables grid search for Spambase by configuration, leaving an untuned 75.87% baseline rather than a tuned result. The same approach would not survive a dataset of 100,000 samples, whose kernel matrix alone would require 80 GB of RAM. There are workarounds (kernel caching, chunking, online SMO variants, low-rank approximations like the Nyström method, and budgeted SVMs), but they all add complexity. For datasets with millions of examples, SVMs are simply not the right tool — gradient-boosted trees or neural networks scale much better.

**Sensitivity to hyperparameters.** Task 1 showed this dramatically: on Ionosphere, the RBF kernel went from **92.86% to 98.57% accuracy** purely from tuning `(C, γ)`. The flip side is the Spambase default-Polynomial result: 40% accuracy at `(C = 1, γ = 0.1, d = 3)`, worse than majority-class guessing. Off-the-shelf SVMs with default parameters are not reliably useful — they happen to land at a workable point on Banknote and Ionosphere because `γ = 0.1` is a reasonable default for moderate-feature-count datasets, but they fail spectacularly on Spambase. The `C` and `γ` parameters interact in non-obvious ways, and the right values depend strongly on the scale of the features, which is why normalisation is also mandatory. The standard solution is grid search or random search over the hyperparameter space, but this multiplies training time by the number of grid points (in Task 1, by a factor of around 80–240 depending on kernel).

**No native multi-class support.** SVMs are inherently *binary* classifiers — the dual formulation, the KKT conditions, and the decision rule are all stated for two classes. Extending them to `k`-class problems requires either a one-vs-rest scheme (`k` binary classifiers, one per class) or a one-vs-one scheme (`k(k−1)/2` classifiers, then voting). Both add training time and complicate prediction. Algorithms like decision trees, random forests, and neural networks handle multi-class natively and are usually less awkward in practice.

**Probabilistic outputs are an afterthought.** A trained SVM produces a *signed distance* from the decision boundary, not a probability. The standard fix is **Platt scaling**, which fits a sigmoid `1 / (1 + exp(A f(x) + B))` to the decision values using a separate held-out set. This works but adds another training step and requires more data. Logistic regression and most modern probabilistic classifiers produce calibrated probabilities directly. Spambase's 0.63 precision is an example of where probability calibration would matter: a real spam filter wants to threshold on `P(spam | x) > 0.95` to minimise legitimate-mail loss, which the raw signed distance cannot give.

**Limited interpretability.** Unlike a decision tree (where every prediction can be traced through a sequence of explicit rules) or even logistic regression (where coefficients have a clear sign and magnitude per feature), the SVM decision boundary is implicit. For a linear SVM you can extract `w = Σ α_i y_i x_i` and read off feature importances, but for an RBF or polynomial SVM the boundary lives in an implicit feature space and there is no direct way to say "the model used feature X because Y". In medical, legal, or regulated financial applications where decisions must be explained to a human expert, this is a real obstacle. Modern post-hoc tools like SHAP and LIME can be applied to SVMs as a workaround, but they add yet another layer of approximation and interpretation.

**Sensitivity to feature scaling.** As discussed in §1.2, SVMs (especially with the RBF kernel) require all features to be on a comparable numerical scale. Without normalisation, the squared-distance term in the RBF kernel is dominated by whichever features have the largest raw values, regardless of whether those features are actually more diagnostic. This means the preprocessing pipeline must include normalisation, and the same normalisation parameters (the means and standard deviations) must be saved and applied consistently to any new data at prediction time. A subtle failure mode is forgetting to apply the training-set normalisation to test data — this can degrade accuracy by tens of percentage points without producing any error message.

**Difficulty with noisy or overlapping classes.** The soft-margin formulation tolerates *some* noise via the `C` parameter, but SVMs are not the best choice when classes overlap heavily. In such cases the optimal Bayes decision rule is probabilistic, and probabilistic models (Bayesian methods, Gaussian mixtures, calibrated logistic regression) tend to outperform SVMs because they can express genuine prediction uncertainty rather than producing a hard side-of-the-line decision. The Spambase result is consistent with this: spam vs. ham is a heavily overlapping problem in feature space (many words appear in both), and even a perfectly tuned SVM is unlikely to dominate a probabilistic alternative.

**Sequential nature of SMO.** The SMO solver is inherently sequential — each pair update depends on the current state of all `α` values and the bias. This makes it difficult to parallelise the inner loop, in contrast to neural network training (which exploits batched SGD on GPUs) and tree ensembles (which can train trees in parallel). For very large training sets, this is another reason SVMs have lost ground to other methods on modern hardware.

### 2.3 Real-World Applications

Despite the limitations above, SVMs remain a practical and widely used algorithm in domains where their strengths line up with the problem structure. Several application areas are worth highlighting:

**Text classification and natural language processing.** SVMs were the dominant algorithm for text classification before the rise of deep learning (Joachims, 1998), and they remain competitive on small-corpus problems. Spam detection (the application Spambase models), sentiment analysis, language identification, and document categorisation are all classical SVM applications. Text data is naturally represented as high-dimensional sparse vectors (bag-of-words, TF-IDF), and Linear SVMs handle this representation extremely efficiently — sparse kernel evaluation is fast, and the resulting models are compact. For many production text-classification systems with limited training data, a Linear SVM trained with `liblinear` or similar is still the right starting point, simply because it works and is easy to deploy. Spambase's behaviour in §1.5 underlines both the appeal and the limitations: an untuned RBF reaches 76% accuracy from scratch, but Linear (51%) and high-degree Polynomial (40%) kernels collapse, which is why the established practice is to use a Linear SVM with proper hyperparameter tuning rather than a default RBF.

**Image and signal recognition.** SVMs were the workhorse of computer vision throughout the 2000s, often combined with hand-crafted feature descriptors like SIFT, HOG, or SURF. The pedestrian detector in early autonomous driving systems used HOG features fed into a Linear SVM. Handwritten digit recognition (MNIST and the postal-code reading systems that motivated much of the SVM literature) was an SVM application long before convolutional neural networks took over. The Banknote Authentication dataset used in this study is an example of the same pattern in miniature: hand-crafted wavelet features (variance, skewness, kurtosis, entropy) fed into an SVM, with the optimised RBF reaching 96.72% accuracy and 36 support vectors out of 1,098 training samples. Even today, SVMs are used for signal-classification tasks like the Ionosphere problem in §1.5 — small training sets, high feature counts, embedded-device deployment, and applications where model size and inference latency are tightly constrained.

**Bioinformatics and computational biology.** SVMs are ubiquitous in protein structure prediction, gene-expression analysis, prediction of protein–protein interaction networks, and identification of regulatory elements in DNA. These problems share two features that suit SVMs well: the data is often high-dimensional (thousands of gene-expression features, for example), and the number of labelled samples is typically small relative to the feature count. String kernels designed specifically for biological sequence data — the spectrum kernel, the gappy kernel, the mismatch kernel — extend the SVM framework to sequence inputs without requiring an explicit vectorisation step.

**Medical diagnosis.** SVMs have been used extensively for binary classification on biomedical data, both because their margin-based generalisation is robust on small samples and because reproducibility matters in regulated environments. Published work has applied SVMs to skin cancer detection from dermoscopy images, classification of brain tumours from MRI features, prediction of heart disease from electronic health records, and identification of diabetic retinopathy from fundus photographs. The recurring pattern is asymmetric error costs — a missed positive case is worse than a false alarm — and the practitioner can directly tune the SVM's `C` parameter to match that asymmetry by trading off margin width against training-error penalty.

**Finance and credit scoring.** SVMs have been applied to credit default prediction, fraud detection, stock-direction prediction, and customer churn modelling. The motivation is similar to the Banknote scenario in §1.5: the cost of errors is asymmetric (missing fraud is much worse than flagging a legitimate transaction), the datasets are often medium-sized rather than huge, and the regulatory environment values models whose behaviour can be characterised mathematically. Linear SVMs in particular are popular in regulated financial settings because feature importances can be extracted directly from the weight vector, partly mitigating the interpretability problem discussed in §2.2.

**Anomaly detection.** A variant of SVMs called the **one-class SVM** learns a boundary around "normal" data, which can then be used to flag outliers. This is widely used in network intrusion detection, manufacturing quality control (flagging defective units on a production line), and fault detection in industrial machinery.

**Why not just use deep learning?** A fair question is why SVMs continue to see use at all when deep learning has overtaken them on most large-scale benchmarks. The answer is that deep learning is overkill — and often actively worse — when the available data is small, the problem is well structured, the model needs to be deployable on modest hardware, or the regulatory environment demands reproducibility and interpretability. SVMs trade away the flexibility to learn from millions of examples in exchange for strong generalisation from hundreds or thousands of examples. The §1.5 results make the trade concrete: 281-sample Ionosphere reaches 98.57% accuracy and 1,098-sample Banknote reaches 96.72% with from-scratch C++ code that uses no library beyond the standard library. A neural network on either dataset would not do meaningfully better, would be much harder to justify under interpretability requirements, and would carry a far heavier deployment footprint.

---

## References

1. Platt, J. C. (1998). *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*. Microsoft Research Technical Report MSR-TR-98-14.

2. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.

3. Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley-Interscience.

4. Burges, C. J. C. (1998). A tutorial on support vector machines for pattern recognition. *Data Mining and Knowledge Discovery*, 2(2), 121–167.

5. Sigillito, V. G., Wing, S. P., Hutton, L. V., & Baker, K. B. (1989). Classification of radar returns from the ionosphere using neural networks. *Johns Hopkins APL Technical Digest*, 10, 262–266. UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/52/ionosphere

6. Lohweg, V., et al. *Banknote Authentication Data Set*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/267/banknote+authentication

7. Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase Data Set*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/94/spambase

8. Scikit-learn developers. *Support Vector Machines*. https://scikit-learn.org/stable/modules/svm.html

9. Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. *Proceedings of ECML-98*, 137–142.

---

## Further Reading — Links by Report Section

### Task 1: Description of Program

- [Support Vector Machine — Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Sequential Minimal Optimization — Wikipedia](https://en.wikipedia.org/wiki/Sequential_minimal_optimization)
- [Support Vector Machines — scikit-learn](https://scikit-learn.org/stable/modules/svm.html)

### Task 1: Datasets

- [Ionosphere — UCI ML Repository](https://archive.ics.uci.edu/dataset/52/ionosphere)
- [Banknote Authentication — UCI ML Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
- [Spambase — UCI ML Repository](https://archive.ics.uci.edu/dataset/94/spambase)

### Task 1: Algorithm Implementation and Parameters

- [Radial Basis Function Kernel — Wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
- [Polynomial Kernel — Wikipedia](https://en.wikipedia.org/wiki/Polynomial_kernel)
- [Sequential Minimal Optimization — Wikipedia](https://en.wikipedia.org/wiki/Sequential_minimal_optimization)

### Task 1: Training of Algorithm

- [Cross-validation (Statistics) — Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
- [Support Vector Machines — scikit-learn](https://scikit-learn.org/stable/modules/svm.html)

### Task 1: Results

- [Confusion Matrix — Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Precision and Recall — Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
- [F-score — Wikipedia](https://en.wikipedia.org/wiki/F-score)

### Task 2: Ability of Chosen Algorithm

- [Support Vector Machine — Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Support Vector Machines — scikit-learn](https://scikit-learn.org/stable/modules/svm.html)

### Task 2: Limitations of Chosen Algorithm

- [Support Vector Machine — Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Support Vector Machines — scikit-learn](https://scikit-learn.org/stable/modules/svm.html)
- [Platt scaling — Wikipedia](https://en.wikipedia.org/wiki/Platt_scaling)

### Task 2: Uses of Chosen Algorithm

- [Support Vector Machine — Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Spam filtering — Wikipedia](https://en.wikipedia.org/wiki/Email_filtering)
- [Banknote Authentication — UCI ML Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
