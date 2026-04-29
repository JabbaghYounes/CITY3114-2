# Algorithm

This document describes the mathematics of the SVM and the SMO training routine as implemented in `src/svm.cpp`. For the module layout see [architecture.md](architecture.md); for build and run instructions see [usage.md](usage.md).

## Problem Setup

Given training data `{(x_i, y_i)}` for `i = 1, …, n` with `x_i ∈ ℝ^d` and `y_i ∈ {−1, +1}`, the soft-margin SVM learns a decision function

```
f(x) = w · φ(x) + b
```

that assigns the class `sign(f(x))` to a new input `x`. The feature map `φ` is implicit — the algorithm never evaluates it directly, only the kernel `K(a, b) = φ(a) · φ(b)`.

## Primal Problem

The standard soft-margin formulation minimises a combination of the margin-width penalty and the training-error penalty:

```
minimise   (1/2) ‖w‖² + C Σ_i ξ_i
subject to y_i (w · φ(x_i) + b) ≥ 1 − ξ_i
           ξ_i ≥ 0
```

`C > 0` controls the trade-off: small `C` allows a wide margin with many slack variables `ξ_i > 0`, large `C` forces the optimiser to reduce training errors at the cost of a narrower margin.

## Dual Problem

The primal is almost never solved directly. Applying Lagrangian duality yields the dual:

```
maximise W(α) = Σ_i α_i − (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
subject to 0 ≤ α_i ≤ C          (box constraints)
           Σ_i α_i y_i = 0      (equality constraint)
```

This is the form actually solved in `SVM::train`. The dual has two critical properties:

1. **The kernel trick.** Inputs `x_i` only appear inside `K(x_i, x_j)`, so no explicit feature map is ever computed. The same solver works for linear, RBF, or polynomial kernels.
2. **Sparse solutions.** At the optimum, most `α_i` are exactly zero. Only samples with `α_i > 0` — the support vectors — contribute to the final decision function.

The trained decision function is:

```
f(x) = Σ_{i : α_i > 0} α_i y_i K(x_i, x) + b
```

## KKT Conditions

At the dual optimum, each training sample must satisfy one of three Karush–Kuhn–Tucker conditions:

| Condition      | Meaning                                        |
|----------------|------------------------------------------------|
| `α_i = 0`      | `y_i f(x_i) ≥ 1` — correctly classified, outside the margin |
| `0 < α_i < C`  | `y_i f(x_i) = 1` — exactly on the margin       |
| `α_i = C`      | `y_i f(x_i) ≤ 1` — inside the margin or misclassified |

The SMO loop in `SVM::train` uses `E_i = f(x_i) − y_i` as its error cache and tests `r_i = E_i · y_i` against the `tol` parameter (default `1e-3`). The KKT-violation check becomes:

```cpp
if ((r_i < -tol && alphas_[i] < C_) ||
    (r_i > tol && alphas_[i] > 0)) { ... }
```

which is the union of all cases where sample `i` is not yet at its optimal `α`.

## SMO Algorithm

Sequential Minimal Optimization (Platt, 1998) decomposes the `n`-variable QP into a sequence of 2-variable sub-problems. Two variables at a time is the smallest possible update: because of the equality constraint `Σ α_i y_i = 0`, changing any single `α` alone would violate the constraint. Updating *two* at once keeps the constraint satisfied and has a closed-form analytic solution.

### Training Loop Overview

`SVM::train` in `src/svm.cpp` implements SMO as follows:

1. **Initialisation.** Set `α_i = 0` for all `i` and `b = 0`.
2. **Precompute the kernel matrix.** Compute `K[i][j] = K(x_i, x_j)` once for all training pairs and store it as a symmetric `n × n` matrix. Every subsequent iteration reads from this cache instead of re-evaluating the kernel.
3. **Build the error cache.** For every sample, store `E_i = f(x_i) − y_i` initially.
4. **Iterate until no KKT violations remain.** Each outer pass loops over all `n` samples. For each violating `i`, pick a partner `j`, compute the analytic update, apply it, update the bias, and recompute the error cache. A pass with zero successful updates is the convergence signal. A safety cap of `max_iter = 1000` outer passes prevents infinite loops.

### Picking the Second Variable

Given a KKT-violating index `i`, the second index `j` is chosen by the **second-choice heuristic**: pick `j ≠ i` that maximises `|E_i − E_j|`. The update step for `α_j` is proportional to `(E_i − E_j) / η`, so this choice produces the largest step per iteration and accelerates convergence.

Platt's original paper uses a more elaborate hierarchy (first prefer non-bound support vectors, fall back to a full scan only if that fails), but the simpler maximum-difference rule converges well on a dataset of this size.

### Bounds on `α_j`

The two-variable update must respect both box constraints (`0 ≤ α ≤ C`) and the equality constraint. Because the equality fixes `y_i α_i + y_j α_j` throughout the joint update, the new `α_j` is constrained to a line segment `[L, H]`:

| Case            | `L`                       | `H`                         |
|-----------------|---------------------------|-----------------------------|
| `y_i ≠ y_j`     | `max(0, α_j − α_i)`       | `min(C, C + α_j − α_i)`     |
| `y_i = y_j`     | `max(0, α_i + α_j − C)`   | `min(C, α_i + α_j)`         |

These bounds are derived from combining the box constraints with the equality `y_i α_i^new + y_j α_j^new = y_i α_i^old + y_j α_j^old`. If `L == H`, no progress is possible and the pair is skipped.

### Second Derivative `η`

The second derivative of the dual objective along the update direction is:

```
η = 2 K(x_i, x_j) − K(x_i, x_i) − K(x_j, x_j)
  = − ‖φ(x_i) − φ(x_j)‖²
```

For any valid (positive semi-definite) kernel this is always non-positive. If `η ≥ 0` the update direction is degenerate and the pair is skipped — Platt's paper handles this case with special bound evaluations, but skipping is safe and rarely needed in practice.

### Update Equations

The unclipped new value of `α_j` is:

```
α_j^new = α_j^old − y_j (E_i − E_j) / η
```

This is then clipped to the feasible range:

```
α_j^new ← max(L, min(H, α_j^new))
```

If the magnitude of the change is below `1e-5`, the update is skipped to avoid numerical churn. Otherwise `α_i` is updated to preserve the equality constraint:

```
α_i^new = α_i^old + y_i y_j (α_j^old − α_j^new)
```

### Bias Update

After a successful pair update, two candidate bias values are computed:

```
b1 = b − E_i
       − y_i (α_i^new − α_i^old) K(x_i, x_i)
       − y_j (α_j^new − α_j^old) K(x_i, x_j)

b2 = b − E_j
       − y_i (α_i^new − α_i^old) K(x_i, x_j)
       − y_j (α_j^new − α_j^old) K(x_j, x_j)
```

Both come from the KKT condition that any non-bound support vector must satisfy `f(x_i) = y_i` exactly. The actual choice is:

- If `0 < α_i^new < C`, use `b = b1`.
- Else if `0 < α_j^new < C`, use `b = b2`.
- Otherwise (both at a bound), use `b = (b1 + b2) / 2`.

### Error Cache Update

After every successful pair update the cache is updated *incrementally*, in O(n). The change in `f(x_k)` from the joint update is

```
Δf(x_k) = δα_i · y_i · K(x_i, x_k) + δα_j · y_j · K(x_j, x_k) + δb
```

so each entry is bumped by

```
E[k] ← E[k] + Δf(x_k)
```

instead of being rebuilt from the full sum `Σ_m α_m y_m K(x_m, x_k) + b − y_k`. That drops the per-update cost from O(n²) to O(n), which is the difference between the full 4-dataset pipeline taking ~12 minutes and ~2:45 on a typical desktop CPU. The trade-off is numerical: the incremental update accumulates floating-point error across the entire training run instead of recomputing fresh each iteration, so SMO converges to a near-equivalent (not bit-identical) solution. Test-set accuracies on Wisconsin and Ionosphere are unchanged from the rebuild-every-iteration version; Banknote Polynomial's grid winner shifts from `degree=2` to `degree=3` (a CV-tie reshuffle).

## Support Vector Extraction

After the main loop terminates, any training sample with `α_i > 1e-8` is copied into the model state as a support vector. The `1e-8` threshold filters out floating-point noise; a strict `> 0` check would retain near-zero multipliers that contribute nothing to predictions. The rest of the training set is discarded.

## Prediction

Prediction is implemented as `sign(f(x))`, with ties (exactly `f(x) = 0`) broken as `+1`:

```cpp
int SVM::predict(const std::vector<double>& x) const {
    return decision_function(x) >= 0 ? 1 : -1;
}

double SVM::decision_function(const std::vector<double>& x) const {
    double sum = 0.0;
    for (size_t i = 0; i < support_vectors_.size(); ++i) {
        sum += support_alphas_[i] * support_labels_[i]
             * kernel_.compute(support_vectors_[i], x);
    }
    return sum + b_;
}
```

The prediction cost is `O(|SV| · d)` per input, independent of the original training set size.

## Kernel Functions

Three kernels are implemented in `src/kernel.cpp`:

- **Linear:** `K(a, b) = a · b`. The cheapest kernel, produces a linear boundary in the original input space.
- **RBF (Radial Basis Function):** `K(a, b) = exp(−γ ‖a − b‖²)`. Smooth non-linear boundaries. `γ` controls locality — small `γ` gives smooth, linear-like behaviour; large `γ` makes the model highly local and prone to overfitting.
- **Polynomial:** `K(a, b) = (γ · a · b + c₀)^d`. Degree-`d` polynomial surfaces in feature space. `c₀` (fixed at `1.0` in grid search) ensures lower-order monomials are included.

All three kernels go through the same `Kernel::compute()` interface, so the SMO solver is completely kernel-agnostic.

## Hyperparameters at a Glance

| Parameter   | Role                                       | Default in `SVM::SVM` | Grid search range          |
|-------------|--------------------------------------------|------------------------|----------------------------|
| `C`         | Regularisation strength                    | `1.0`                  | `{0.1, 1, 10, 100}`        |
| `γ`         | Kernel width (RBF, Polynomial)             | `0.1`                  | `{0.001, 0.01, 0.1, 1}`    |
| `degree`    | Polynomial degree                          | `3`                    | `{2, 3, 4}`                |
| `coef0`     | Polynomial constant offset                 | `1.0` (grid search)    | fixed                      |
| `tol`       | KKT violation tolerance                    | `1e-3`                 | fixed                      |
| `max_iter`  | Outer SMO pass cap                         | `1000`                 | fixed                      |

## References

- Platt, J. C. (1998). *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*. Microsoft Research Technical Report MSR-TR-98-14.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Burges, C. J. C. (1998). A tutorial on support vector machines for pattern recognition. *Data Mining and Knowledge Discovery*, 2(2), 121–167.
