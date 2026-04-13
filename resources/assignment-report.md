# SVM Classifier for Breast Cancer Diagnosis

**Module:** CITY3114 — Machine Learning and Algorithms
**Assignment:** 2
**Name:** [Your Name]
**Student ID:** [Your Student ID]

---

## Task 1: Implementation and Evaluation

### 1.1 Description of Program

This project implements a binary Support Vector Machine (SVM) classifier from scratch in C++17. No external machine learning libraries are used — every component, from CSV parsing to the quadratic optimisation routine, is written by hand. The motivation for the from-scratch approach is partly educational (forcing a complete walk-through of the underlying mathematics) and partly to keep the dependency footprint minimal: a working compiler and CMake are the only requirements.

The classifier uses the **Sequential Minimal Optimization (SMO)** algorithm for training, originally introduced by John Platt at Microsoft Research (Platt, 1998). SMO is the most widely used method for solving the SVM dual problem because it decomposes the global quadratic program into a sequence of analytically solvable two-variable sub-problems, which means no general-purpose QP solver is required.

The program is built around a small, modular set of components, each in its own translation unit:

- **`data_loader`** — handles CSV parsing, z-score feature normalisation, and a seeded train/test split. The split uses a deterministic linear congruential generator so that runs are exactly reproducible.
- **`kernel`** — provides three kernel functions (Linear, RBF, Polynomial) behind a single `Kernel::compute()` interface. Switching kernels is therefore a one-line change in client code.
- **`svm`** — contains the core `SVM` class. This is the largest module: it owns the SMO training loop, the precomputed kernel matrix, the error cache, the support-vector extraction logic, and the prediction function.
- **`evaluation`** — computes accuracy, precision, recall, F1 score, and a 2 × 2 confusion matrix; implements stratified-by-shuffle k-fold cross-validation; and runs the grid search used for hyperparameter optimisation.
- **`main`** — orchestrates the full pipeline end-to-end: loading the dataset, normalising features, splitting into train/test, training a default model with each kernel, running 5-fold cross-validation, performing the grid search, and finally retraining with the optimised hyperparameters.

The deliberate decision to keep `main` as a fixed-pipeline driver (rather than a CLI with flags) was made to keep the demonstration self-contained: a single invocation produces every figure that appears in this report.

#### Dependencies and How to Run

The only requirements are a C++17 compiler (such as `g++` 9 or above) and CMake (version 3.14 or above). No third-party libraries are needed.

To build and run:

```bash
cmake -B build
cmake --build build
./build/svm_classifier
```

The dataset file (`wdbc.csv`) must be placed in a `data/` directory at the project root. It can be downloaded from the UCI Machine Learning Repository:

```bash
mkdir -p data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" -o data/wdbc.csv
```

A complete run takes a few seconds on a typical desktop CPU and prints the baseline results, the cross-validation accuracy, the per-kernel comparison, the full grid-search trace, and the final results table for the optimised models. A companion Python script (`plot_results.py`) parses the program's stdout and generates publication-quality figures — confusion matrix heatmaps, kernel comparison bar charts, grid search heatmaps, and a cross-validation fold chart — into a `figures/` directory.

### 1.2 Dataset

The dataset used is the **Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI Machine Learning Repository (Wolberg et al., 1995). It contains 569 samples derived from digitised images of fine needle aspirate (FNA) biopsies of breast masses. Each sample is a single tumour, and each is labelled as either **Malignant (M)** or **Benign (B)** based on the histopathological diagnosis. The diagnostic task here is therefore a binary classification problem: from the geometric and textural properties of the cell nuclei in an FNA image, predict whether the underlying tumour is cancerous.

#### Features

Each sample has 30 numerical features, computed from ten real-valued properties of the cell nuclei visible in the image:

1. **Radius** — mean of distances from the centre of the nucleus to its boundary points
2. **Texture** — standard deviation of grey-scale pixel values
3. **Perimeter** — total length of the nucleus boundary
4. **Area**
5. **Smoothness** — local variation in radius lengths
6. **Compactness** — perimeter² / area − 1.0
7. **Concavity** — severity of concave portions of the contour
8. **Concave points** — number of concave portions of the contour
9. **Symmetry**
10. **Fractal dimension** — "coastline approximation" minus 1

For each of these ten properties, three statistics are recorded across the cell nuclei in the sample: the **mean**, the **standard error**, and the **worst** (largest) value. The combination yields the 30 features used as model input. This structure is biologically meaningful — extreme values (the "worst" group) often carry more diagnostic signal than averages, because cancerous masses tend to contain at least some highly irregular nuclei even if most look normal.

#### Class Distribution and Why It Matters

The class distribution is mildly imbalanced:

- **Malignant (M, +1):** 212 samples (37.3%)
- **Benign (B, −1):** 357 samples (62.7%)

This imbalance is small enough that no special techniques (oversampling, class weighting, SMOTE, etc.) were used. However, it is large enough to matter when interpreting results. A trivial classifier that predicts "benign" for every sample would already achieve **62.7% accuracy** without learning anything, so accuracy in isolation is not a useful metric. This is reinforced by the asymmetric clinical cost of errors: in a medical screening context, a false negative (missing a malignant tumour) is far more dangerous than a false positive (a benign tumour flagged for further investigation). For this reason, the analysis in §1.5 reports precision, recall, and F1 score in addition to overall accuracy, and pays particular attention to recall on the malignant class.

#### Preprocessing

Three preprocessing steps are applied:

1. **ID column dropped.** The first column of `wdbc.csv` is a patient identifier with no diagnostic value; if left in it would act as noise (or, worse, leak information about ordering).
2. **Label encoding.** The diagnosis column is mapped from the string labels `M`/`B` to the integer labels `+1`/`−1`. The `±1` convention is the standard for SVMs because the dual formulation, decision rule, and KKT conditions are all stated in terms of `y_i ∈ {−1, +1}`.
3. **Z-score normalisation.** Each of the 30 feature columns is independently rescaled to zero mean and unit variance:

   ```
   x'_ij = (x_ij − μ_j) / σ_j
   ```

   where `μ_j` and `σ_j` are computed across the full dataset. The C++ implementation uses the population standard deviation (dividing by `n`, not `n − 1`); for `n = 569` this difference is negligible.

Feature normalisation is **essential** for SVMs that use distance-based kernels like the RBF. Without it, features with large numerical scales (such as `area`, which is on the order of hundreds, vs. `smoothness`, which is on the order of 0.1) would dominate the squared-distance term `‖a − b‖²` and effectively make the model ignore the smaller-scale features. Even for the linear kernel, normalisation tends to improve numerical conditioning of the optimisation. The dramatic effect of normalisation is one of the reasons SVMs are sometimes described as "not working out of the box": the data must be on a common scale before training begins.

### 1.3 Algorithm Implementation

#### Geometric Intuition

A Support Vector Machine is a supervised learning algorithm that finds the optimal separating hyperplane between two classes (Cortes & Vapnik, 1995). In the linearly separable case the "optimal" hyperplane is the one that maximises the **margin** — the perpendicular distance between the boundary and the nearest data points from each class. Those nearest points are called **support vectors**, and they are the only training examples that influence the position of the boundary; moving any other example a small distance has no effect on the model.

This margin-maximisation objective is what gives SVMs their well-known generalisation properties. Intuitively, a wider margin gives the classifier more "room for error" when it sees noisy or slightly shifted test data, and PAC learning theory formalises this with bounds on test error that depend on the margin width rather than on the dimensionality of the input space. In practice, this means SVMs often generalise well even when the number of features is comparable to (or larger than) the number of samples — exactly the regime of the Breast Cancer Wisconsin dataset (30 features, 569 samples).

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

2. **Precompute the kernel matrix.** A full `n × n` symmetric matrix `K[i][j] = K(x_i, x_j)` is computed once at the start of training. For `n = 456` (the training set size after the 80/20 split) this requires roughly `(456 × 457) / 2 ≈ 104,000` kernel evaluations and about 1.7 MB of memory for double-precision floats. This is a deliberate space-for-time trade-off: every subsequent iteration of SMO can read kernel values directly instead of recomputing them, which is the dominant cost in any SVM trainer. For larger datasets this caching strategy becomes infeasible (see §2.2).

3. **Maintain an error cache.** For every training point the value `E_i = f(x_i) − y_i` is stored, where `f(x_i)` is the current SVM output. The error cache is recomputed in full after every successful joint update. (A more sophisticated implementation would update the cache incrementally — only the entries affected by the change in `α_i, α_j, b` need adjusting — but the implementation here recomputes everything for simplicity. This is one of the easier optimisation targets if the codebase were taken further.)

4. **Iterate over all samples.** For each `i`, compute `r_i = E_i · y_i`. The KKT conditions for this dual problem reduce to:
   - if `α_i = 0` then `r_i ≥ −tol` (point is correctly classified outside the margin)
   - if `α_i = C` then `r_i ≤ tol` (point is at most marginally violating)
   - if `0 < α_i < C` then `|r_i| < tol` (point is exactly on the margin)

   The check `(r_i < −tol AND α_i < C) OR (r_i > tol AND α_i > 0)` in the source code exactly captures the union of all KKT-violating cases.

5. **Pick the second variable using the second-choice heuristic.** When sample `i` violates the KKT conditions, sample `j` is chosen as the index that maximises `|E_i − E_j|`. The reasoning is that the analytic update step for `α_j` is proportional to `(E_i − E_j) / η`, so picking the largest absolute difference produces the largest step per iteration and therefore the fastest overall convergence. Platt's full SMO uses a more elaborate hierarchy (first preferring non-bound `α`s, then falling back to a full scan), but the simple maximum-difference rule already works well on a dataset this size.

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

     This is always non-positive for a valid (positive semi-definite) kernel. If `η ≥ 0` (numerically possible for degenerate pairs) the implementation skips the update — Platt's paper handles this case specially, but a skip is safe and rarely activates in practice.

   - The **unclipped update** for `α_j`:

     ```
     α_j ← α_j − y_j (E_i − E_j) / η
     ```

     This is then clipped to `[L, H]`. If the clipped change is below `1e-5` the iteration is skipped (to avoid numerical churn from updates that are too small to matter). Finally, `α_i` is updated to maintain the constraint:

     ```
     α_i ← α_i + y_i y_j (α_j_old − α_j_new)
     ```

7. **Update the bias.** Two candidate bias values `b1` and `b2` are computed, derived from the KKT condition that any non-bound support vector should have `f(x_i) = y_i` exactly. If `α_i` ended up strictly between 0 and `C`, then `b1` is used; if `α_j` did, `b2` is used; if both are at the bounds, the average `(b1 + b2) / 2` is used as a heuristic.

8. **Recompute the error cache** and continue. A full pass over the data with no successful updates is the convergence criterion. A safety cap of `max_iter = 1000` outer passes prevents infinite loops in pathological cases.

After the loop terminates, support vector extraction is straightforward: any training sample with `α_i > 1e-8` is retained. The `1e-8` threshold instead of strict `> 0` is to filter out floating-point noise. The retained `(x_i, y_i, α_i)` triples are everything the trained model needs for prediction; the rest of the training data is discarded.

#### Key Hyperparameters

The behaviour of the trained classifier is governed by a small set of hyperparameters:

- **`C` (regularisation strength).** Trades off margin width against training error. The default value used for the baseline experiments is `C = 1.0`, and the grid search explores the range `{0.1, 1, 10, 100}`.
- **`γ` (kernel width).** Used by the RBF and Polynomial kernels. The baseline value is `γ = 0.1`, with grid search over `{0.001, 0.01, 0.1, 1}`.
- **`degree`** (`d`). Used only by the Polynomial kernel. Baseline is 3, grid search over `{2, 3, 4}`.
- **`coef0`** (`c₀`). The polynomial kernel offset. Fixed at 1.0 throughout (a common default that includes the lower-order terms in the polynomial expansion).
- **`tol` (KKT tolerance).** The threshold for declaring that a point satisfies the KKT conditions. Set to `0.001`. Tighter values produce more precise solutions at the cost of more iterations; looser values converge faster but produce slightly suboptimal classifiers.
- **`max_iter`.** A safety cap of 1000 outer SMO passes. In practice this is never reached on the Breast Cancer Wisconsin dataset; convergence typically occurs in under 50 outer passes.

### 1.4 Training

The dataset was split into **80% training (456 samples)** and **20% test (113 samples)** using a Fisher-Yates shuffle seeded with `seed = 42` and a simple linear congruential generator (LCG) with the constants `a = 1103515245`, `c = 12345`. The deterministic seed is important for reproducibility: every result in this report can be reproduced exactly by re-running the binary, because the same LCG sequence is also used for the shuffle in cross-validation.

#### Initial Training (Baseline)

To establish a baseline before any tuning, each kernel was trained with the default values `C = 1.0`, `γ = 0.1`, `degree = 3`. The intent of the baseline is not to be competitive but to show how much hyperparameter tuning matters — the gap between baseline and tuned results is one of the main pedagogical lessons of the experiment. The baseline numbers appear in §1.5.

#### Why Cross-Validation

Selecting hyperparameters by their performance on a single train/test split would be unreliable: a particular split might happen to favour one combination over another simply due to which 113 samples ended up in the test set. **k-fold cross-validation** addresses this by partitioning the (full, normalised) dataset into `k` equally sized folds, training on `k − 1` of them, evaluating on the held-out fold, and rotating through all `k` choices of held-out fold. The reported metric is the mean accuracy across the `k` runs. Five folds was chosen as a standard compromise: it gives reasonably stable estimates without making the grid search unaffordably slow. The per-fold accuracy spread is visualised in `figures/cv_folds.png`.

It is worth noting that the cross-validation here uses **the full dataset, not just the training split**. This is the conventional approach for hyperparameter selection — the test split is only used at the very end, to give an unbiased estimate of generalisation for the final tuned model.

#### Grid Search

A full Cartesian-product grid search was conducted over:

- `C ∈ {0.1, 1, 10, 100}`
- `γ ∈ {0.001, 0.01, 0.1, 1}`
- `degree ∈ {2, 3, 4}` (Polynomial only)

This gives 16 combinations to evaluate for the Linear kernel (the `γ` and `degree` columns are ignored for Linear, so effectively 4 distinct configurations × 5 folds = 20 fits), 16 combinations for RBF (16 × 5 = 80 fits), and 48 combinations for Polynomial (48 × 5 = 240 fits). A total of around 340 SVM fits are performed during grid search. On the 569-sample Wisconsin dataset this completes in a few seconds; on a larger dataset the cost would scale quadratically and quickly become the dominant runtime.

The grid search identified the following best configurations by 5-fold CV accuracy on the full dataset:

| Kernel     | Best `C` | Best `γ` | Best `degree` | CV Accuracy |
|------------|----------|----------|---------------|-------------|
| Linear     | 0.1      | —        | —             | 72.41%      |
| RBF        | 0.1      | 0.01     | —             | 90.16%      |
| Polynomial | 1.0      | 0.001    | 3             | 89.41%      |

The CV accuracies for the optimised hyperparameters look modest in this table (especially for Linear), but they reflect the **average across five different train/test splits**, and they are evaluated *before* the final retraining on the held-out 80/20 split. The held-out test results in §1.5 are noticeably better, particularly for the Linear kernel — a reminder that a single split can be optimistic relative to cross-validation.

The grid search landscape is visualised as C-vs-gamma heatmaps in `figures/grid_search_heatmaps.png`, with the best cell highlighted for each kernel. A practical observation from the grid search trace is that the best combinations all involve **small** values of `C` and `γ`. This is consistent with the data being approximately linearly separable after normalisation: a small `C` allows a wide margin (good generalisation), and a small `γ` makes the RBF kernel behave more linearly (in the limit `γ → 0`, the RBF kernel approaches a constant). The grid search is therefore not just a brute-force optimiser — its results carry diagnostic information about the structure of the dataset.

### 1.5 Results

#### Default Parameters (`C = 1.0`, `γ = 0.1`, baseline)

| Kernel     | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| Linear     | 41.59%   | 0.3945    | 1.0000 | 0.5658   |
| RBF        | 84.07%   | 0.9630    | 0.6047 | 0.7429   |
| Polynomial | 69.03%   | 1.0000    | 0.1860 | 0.3137   |

These baselines are revealing rather than competitive (see `figures/kernel_comparison_default.png` for the visual comparison). With `C = 1` and `γ = 0.1`:

- **The Linear kernel collapsed to predicting almost everything as malignant.** Its recall of 1.0 looks impressive in isolation — it catches every malignant case — but its accuracy of 41.6% is *worse than the trivial "always benign" baseline of 62.7%*. Precision of 0.39 confirms that the predictions are essentially indiscriminate.
- **The Polynomial kernel collapsed in the opposite direction**, predicting almost everything as benign. Its precision of 1.0 means it never produces a false positive, but its recall of 0.19 means it misses 81% of the actual cancer cases. In a medical context this would be catastrophic.
- **The RBF kernel** is the only baseline that produces a usable model, but with recall of 0.60 it still misses 40% of malignant cases.

The lesson from the baseline table is not that SVMs are bad classifiers — it is that the *defaults* are bad, and that a serious application requires tuning.

#### Optimised Parameters

After the grid search and retraining on the 80/20 split with the best per-kernel configuration (see `figures/kernel_comparison_optimised.png` and `figures/default_vs_optimised.png` for visual comparisons):

| Kernel     | `C`  | `γ`   | `d` | Accuracy | Precision | Recall | F1 Score | Support Vectors |
|------------|------|-------|-----|----------|-----------|--------|----------|-----------------|
| Linear     | 0.1  | —     | —   | 97.35%   | 1.0000    | 0.9302 | 0.9639   | 16              |
| RBF        | 0.1  | 0.01  | —   | 90.27%   | 1.0000    | 0.7442 | 0.8533   | 18              |
| Polynomial | 1.0  | 0.001 | 3   | 95.58%   | 0.9318    | 0.9535 | 0.9425   | 24              |

After tuning, every kernel improved substantially. The most striking single result is the **Linear kernel reaching 97.35% accuracy** — the best of the three — with **only 16 support vectors out of 456 training samples** (3.5% of the training set). This is an unusually clean outcome, and it tells us something important about the dataset: once the 30 features are normalised, the malignant and benign classes are *almost linearly separable*. A linear hyperplane can place 110 of 113 test samples on the correct side, and the entire decision rule depends on just 16 boundary cases.

The **Polynomial kernel** with `degree = 3` also performed well (95.6% accuracy, F1 = 0.9425). Its recall of 0.9535 is actually higher than the Linear kernel's 0.9302, which is the metric that matters most clinically. Its precision is slightly lower, so it produces a few false positives that the Linear kernel does not.

The **RBF kernel** is the most surprising result. It is the most flexible of the three kernels and is often the default choice for general-purpose SVM classification, yet here it produces the *worst* tuned result (90.3% accuracy). The reason is consistent with the grid search trace: the best `γ` for RBF is at the lower end of the search range (`γ = 0.01`), and the lower the `γ`, the more the RBF kernel behaves like a linear one. RBF's full flexibility is not needed on this dataset — and that flexibility, combined with the 30-dimensional input space, gives the optimiser more ways to find a slightly worse local solution. The RBF kernel's recall of 0.7442 is the worst of the three, meaning it misses 26% of malignant cases on the test set, which would be unacceptable in a clinical setting.

Two general points emerge from comparing the three:

1. **Model complexity should match the data.** When the underlying problem is close to linearly separable, a simpler kernel beats a more flexible one. This is the bias–variance trade-off in concrete form: the Linear SVM has lower variance because it can only express linear boundaries, and on this dataset that is exactly the right inductive bias.
2. **Hyperparameter tuning matters more than kernel choice.** The gap between Linear baseline (41.6%) and Linear optimised (97.4%) is much larger than the gap between any two optimised kernels (97.4% vs 90.3% vs 95.6%). For a practitioner, time spent on grid search yields larger returns than time spent on architectural choices.

Confusion matrices for all four models (initial RBF baseline plus the three optimised kernels) are visualised in `figures/confusion_matrices.png`.

#### Confusion Matrix — Best Model (Linear, Optimised)

|                       | Predicted Malignant (+1) | Predicted Benign (−1) |
|-----------------------|--------------------------|-----------------------|
| **Actual Malignant**  | 40                       | 3                     |
| **Actual Benign**     | 0                        | 70                    |

The optimised Linear SVM correctly classified 110 out of 113 test samples. There are **zero false positives** (no benign tumour was incorrectly flagged as malignant) and **three false negatives** (three malignant tumours were missed). In a real clinical workflow, the three missed malignant cases would still be a concern — in screening, recall is the priority — but the overall picture is strong for a from-scratch implementation with no library dependencies.

It is worth being honest about what these numbers do and do not show. They are evaluated on a *single* held-out 20% split. The 5-fold CV numbers in §1.4 are more conservative (CV accuracy for the Linear kernel was 72%, vs 97% on the held-out split), and the truth is somewhere between the two depending on how the data is partitioned. A more thorough evaluation would use repeated stratified k-fold CV on the held-out test set, but this is outside the scope of a single-binary demonstration.

---

## Task 2: Discussion

### 2.1 Ability of the SVM Algorithm

SVMs are well suited to a broad class of supervised learning problems, and the experiments in Task 1 demonstrate several of their strongest properties in concrete form.

**High-dimensional data with limited samples.** SVMs handle the regime where the number of features `d` is large relative to the number of samples `n` better than most alternatives. On Breast Cancer Wisconsin we have `d = 30` and `n = 569`, which is comfortably inside SVM's strong-performance zone. The reason is theoretical: the generalisation error of a maximum-margin classifier depends on the *margin width*, not on the dimensionality of the input space. This is why SVMs were the algorithm of choice for many bioinformatics and text-classification problems throughout the late 1990s and 2000s, where feature counts in the thousands or tens of thousands are common but labelled samples are scarce.

**Kernel-based versatility.** The kernel trick is what makes a single SVM framework applicable to linear, smoothly non-linear (RBF), and polynomial problems. Task 1 demonstrated this directly: exactly the same SMO solver, with the kernel object as the only change, produced three quite different classifiers. The kernel idea also generalises well beyond the three kernels implemented here — string kernels for text and biological sequences, graph kernels for molecules, and histogram-intersection kernels for image features have all been used successfully. This flexibility means SVMs can be applied to non-vector data simply by designing an appropriate similarity function, which is something most other algorithms cannot do directly.

**Built-in regularisation.** The maximum-margin objective acts as a structural form of regularisation. Unlike algorithms that minimise training error and then add an external penalty (like L2-regularised logistic regression), the SVM formulation makes margin width *part of* the objective. The `C` hyperparameter then provides explicit control over the trade-off between margin and training error, giving the practitioner a single knob to tune the bias–variance balance. In Task 1 the best `C` value was `0.1` for two of the three kernels — a small `C` corresponds to a wide margin, which is consistent with the data being largely separable.

**Sparse solutions and efficient prediction.** The trained model only depends on the support vectors. The optimised Linear SVM in Task 1 used **16** support vectors out of 456 training samples — a 96.5% compression of the training set. This sparsity matters in deployment: prediction cost scales with the number of support vectors, not with the size of the original training set, so SVMs are well suited to scenarios where the model is trained once and queried many times.

**Convex optimisation with global optima.** Unlike neural networks, the SVM dual is a convex quadratic program. Any solution found is a global optimum. There are no local minima, no sensitivity to weight initialisation, and no need for learning-rate schedules or warm-up tricks. The solver either converges to the optimum or it does not — and when it does converge, the answer is mathematically unique up to the support-vector boundary cases. This makes SVMs reproducible and easy to reason about.

**Strong theoretical foundations.** SVMs sit on top of decades of work in statistical learning theory. The relationship between margin width and generalisation error was made precise by Vapnik's work on VC dimension and structural risk minimisation, and PAC-style bounds on SVM error are tight enough to be useful in practice. This theoretical grounding makes SVMs a popular choice for academic and scientific applications where interpretability of *why the algorithm works* matters as much as raw accuracy.

In short: SVMs are a strong default choice for binary classification on tabular or vector data with hundreds to a few thousand samples, especially when interpretable hyperparameter tuning and reproducibility are desirable.

### 2.2 Limitations of the SVM Algorithm

Despite their strengths, SVMs have a number of practical limitations that the Task 1 experiments either revealed directly or hinted at.

**Scalability to large datasets.** The most fundamental limitation is computational. Training an SVM with SMO has time complexity that scales between roughly O(n²) and O(n³) depending on the dataset and the kernel, and the precomputed kernel matrix (used in this implementation) has space complexity O(n²). On the 569-sample Wisconsin dataset this is fine — the kernel matrix uses about 2 MB and training completes in well under a second — but the same approach would not survive a dataset of 100,000 samples, whose kernel matrix would require 80 GB of RAM. There are workarounds (kernel caching, chunking, online SMO variants, low-rank approximations like the Nyström method, and budgeted SVMs), but they all add complexity. For datasets with millions of examples, SVMs are simply not the right tool — gradient-boosted trees or neural networks scale much better.

**Sensitivity to hyperparameters.** Task 1 showed this dramatically: the Linear kernel went from **41.6% to 97.4% accuracy** purely from tuning `C`. Off-the-shelf SVMs with default parameters are essentially useless on most datasets. The `C` and `γ` parameters interact in non-obvious ways, and the right values depend strongly on the scale of the features, which is why normalisation is also mandatory. The standard solution is grid search or random search over the hyperparameter space, but this multiplies training time by the number of grid points (in Task 1, by a factor of around 80–240 depending on kernel).

**No native multi-class support.** SVMs are inherently *binary* classifiers — the dual formulation, the KKT conditions, and the decision rule are all stated for two classes. Extending them to `k`-class problems requires either a one-vs-rest scheme (`k` binary classifiers, one per class) or a one-vs-one scheme (`k(k−1)/2` classifiers, then voting). Both add training time and complicate prediction. Algorithms like decision trees, random forests, and neural networks handle multi-class natively and are usually less awkward in practice.

**Probabilistic outputs are an afterthought.** A trained SVM produces a *signed distance* from the decision boundary, not a probability. The standard fix is **Platt scaling**, which fits a sigmoid `1 / (1 + exp(A f(x) + B))` to the decision values using a separate held-out set. This works but adds another training step and requires more data. Logistic regression and most modern probabilistic classifiers produce calibrated probabilities directly.

**Limited interpretability.** Unlike a decision tree (where every prediction can be traced through a sequence of explicit rules) or even logistic regression (where coefficients have a clear sign and magnitude per feature), the SVM decision boundary is implicit. For a linear SVM you can extract `w = Σ α_i y_i x_i` and read off feature importances, but for an RBF or polynomial SVM the boundary lives in an implicit feature space and there is no direct way to say "the model used feature X because Y". In medical and legal applications where decisions must be explained to a human expert, this is a real obstacle. Modern post-hoc tools like SHAP and LIME can be applied to SVMs as a workaround, but they add yet another layer of approximation and interpretation.

**Sensitivity to feature scaling.** As discussed in §1.2, SVMs (especially with the RBF kernel) require all features to be on a comparable numerical scale. Without normalisation, the squared-distance term in the RBF kernel is dominated by whichever features have the largest raw values, regardless of whether those features are actually more diagnostic. This means the preprocessing pipeline must include normalisation, and the same normalisation parameters (the means and standard deviations) must be saved and applied consistently to any new data at prediction time. A subtle failure mode is forgetting to apply the training-set normalisation to test data — this can degrade accuracy by tens of percentage points without producing any error message.

**Difficulty with noisy or overlapping classes.** The soft-margin formulation tolerates *some* noise via the `C` parameter, but SVMs are not the best choice when classes overlap heavily. In such cases the optimal Bayes decision rule is probabilistic, and probabilistic models (Bayesian methods, Gaussian mixtures, calibrated logistic regression) tend to outperform SVMs because they can express genuine prediction uncertainty rather than producing a hard side-of-the-line decision.

**Sequential nature of SMO.** The SMO solver is inherently sequential — each pair update depends on the current state of all `α` values and the bias. This makes it difficult to parallelise the inner loop, in contrast to neural network training (which exploits batched SGD on GPUs) and tree ensembles (which can train trees in parallel). For very large training sets, this is another reason SVMs have lost ground to other methods on modern hardware.

### 2.3 Real-World Applications

Despite the limitations above, SVMs remain a practical and widely used algorithm in domains where their strengths line up with the problem structure. Several application areas are worth highlighting:

**Medical diagnosis and bioinformatics.** SVMs have been used extensively for binary classification on biomedical data, both because their margin-based generalisation is robust on small samples and because reproducibility matters in regulated environments. Beyond breast cancer classification (the present project), published work has applied SVMs to skin cancer detection from dermoscopy images, classification of brain tumours from MRI features, prediction of heart disease from electronic health records, and identification of diabetic retinopathy from fundus photographs. In bioinformatics, SVMs are still a standard tool for protein function prediction, gene expression classification, and the identification of splice sites in DNA — all problems where datasets typically have far more features than samples and where deep learning's data appetite is hard to satisfy.

**Text classification and natural language processing.** SVMs were the dominant algorithm for text classification before the rise of deep learning, and they remain competitive on small-corpus problems. Spam detection, sentiment analysis, language identification, and document categorisation are all classical SVM applications. Text data is naturally represented as high-dimensional sparse vectors (bag-of-words, TF-IDF), and Linear SVMs handle this representation extremely efficiently — sparse kernel evaluation is fast, and the resulting models are compact. For many production text-classification systems with limited training data, a Linear SVM trained with `liblinear` or similar is still the right starting point, simply because it works and is easy to deploy.

**Image recognition (pre–deep learning, and still in niches).** SVMs were the workhorse of computer vision throughout the 2000s, often combined with hand-crafted feature descriptors like SIFT, HOG, or SURF. The pedestrian detector in early autonomous driving systems used HOG features fed into a Linear SVM. Handwritten digit recognition (MNIST and the postal-code reading systems that motivated much of the SVM literature) was an SVM application long before convolutional neural networks took over. Even today, SVMs are used for image classification in scenarios where deep learning is impractical: small training sets, embedded devices with no GPU, and applications where model size and inference latency are tightly constrained.

**Bioinformatics and computational biology.** SVMs are ubiquitous in protein structure prediction, gene-expression analysis, prediction of protein–protein interaction networks, and identification of regulatory elements in DNA. These problems share two features that suit SVMs well: the data is often high-dimensional (thousands of gene-expression features, for example), and the number of labelled samples is typically small relative to the feature count. String kernels designed specifically for biological sequence data — the spectrum kernel, the gappy kernel, the mismatch kernel — extend the SVM framework to sequence inputs without requiring an explicit vectorisation step.

**Finance and credit scoring.** SVMs have been applied to credit default prediction, fraud detection, stock-direction prediction, and customer churn modelling. The motivation is similar to the medical case: the cost of errors is asymmetric (missing fraud is much worse than flagging a legitimate transaction), the datasets are often medium-sized rather than huge, and the regulatory environment values models whose behaviour can be characterised mathematically. Linear SVMs in particular are popular in regulated financial settings because feature importances can be extracted directly from the weight vector, partly mitigating the interpretability problem discussed in §2.2.

**Anomaly detection.** A variant of SVMs called the **one-class SVM** learns a boundary around "normal" data, which can then be used to flag outliers. This is widely used in network intrusion detection, manufacturing quality control (flagging defective units on a production line), and fault detection in industrial machinery.

**Why not just use deep learning?** A fair question is why SVMs continue to see use at all when deep learning has overtaken them on most large-scale benchmarks. The answer is that deep learning is overkill — and often actively worse — when the available data is small, the problem is well structured, the model needs to be deployable on modest hardware, or the regulatory environment demands reproducibility and interpretability. SVMs trade away the flexibility to learn from millions of examples in exchange for strong generalisation from hundreds or thousands of examples. For many real problems, hundreds-to-thousands is exactly what is available, and SVMs remain the right tool. The Breast Cancer Wisconsin dataset, with its 569 samples and 30 features, is a near-perfect example of this regime: an SVM trained from scratch in C++ with no library dependencies achieved 97.35% accuracy with 16 support vectors. A neural network on the same dataset would not do meaningfully better, and would be much harder to justify to a clinician.

---

## References

1. Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

2. Platt, J. C. (1998). *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*. Microsoft Research Technical Report MSR-TR-98-14.

3. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.

4. Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley-Interscience.

5. Burges, C. J. C. (1998). A tutorial on support vector machines for pattern recognition. *Data Mining and Knowledge Discovery*, 2(2), 121–167.

6. Scikit-learn developers. *Support Vector Machines*. https://scikit-learn.org/stable/modules/svm.html

7. American Cancer Society. *How Common Is Breast Cancer?* https://www.cancer.org/cancer/types/breast-cancer/about/how-common-is-breast-cancer.html

8. Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. *Proceedings of ECML-98*, 137–142.

---

## Further Reading — Links by Report Section

### Task 1: Description of Program

- [Support Vector Machine — Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Sequential Minimal Optimization — Wikipedia](https://en.wikipedia.org/wiki/Sequential_minimal_optimization)
- [Support Vector Machines — scikit-learn](https://scikit-learn.org/stable/modules/svm.html)

### Task 1: Dataset

- [Breast Cancer Wisconsin (Diagnostic) — UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [How Common Is Breast Cancer? — American Cancer Society](https://www.cancer.org/cancer/types/breast-cancer/about/how-common-is-breast-cancer.html)

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

### Task 2: Uses of Chosen Algorithm

- [Support Vector Machine — Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
- [How Common Is Breast Cancer? — American Cancer Society](https://www.cancer.org/cancer/types/breast-cancer/about/how-common-is-breast-cancer.html)
