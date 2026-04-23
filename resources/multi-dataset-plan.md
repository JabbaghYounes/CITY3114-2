# Implementation Plan: Multi-Dataset SVM Support

## Goal

Extend the SVM classifier to run the full pipeline (load → normalise → split → train/eval → CV → kernel comparison → grid search → optimised retrain) on four binary-classification datasets: Wisconsin (existing), Ionosphere, Spambase, Banknote.

## Design decisions

- **One binary, loops all datasets in `main()`.** Simpler than CLI flags; produces a single comprehensive report output.
- **Parameterised CSV loader** via a `DatasetSpec` struct. Keeps the parser in one place, avoids per-dataset loader functions.
- **No changes to modelling code** (`kernel`, `svm`, `evaluation`). They're already generic.
- **Shared grid search ranges** across datasets (current ranges span enough magnitudes). Revisit only if results are poor.
- **Pipeline wrapped in a function**, called once per dataset. Current `main()` becomes `run_pipeline(spec)`.

## Dataset summary

| Dataset | ID col | Label pos | Label format | Features | Samples |
|---|---|---|---|---|---|
| Wisconsin | yes | col 2 | M/B | 30 | 569 |
| Ionosphere | no | last | g/b | 34 | 351 |
| Spambase | no | last | 1/0 | 57 | 4601 |
| Banknote | no | last | 0/1 | 4 | 1372 |

## File-by-file changes

### `src/data_loader.hpp`

- Add `DatasetSpec` struct: `name`, `filepath`, `has_id_column`, `label_column` (-1 = last), `positive_label`, `expected_features`.
- Change signature: `Dataset load_csv(const DatasetSpec& spec);`
- Add `extern const std::vector<DatasetSpec> ALL_DATASETS;` or move to a new header.

### `src/data_loader.cpp`

- Rewrite `load_csv` to:
  1. Tokenise line into `std::vector<std::string>`.
  2. If `has_id_column`, drop index 0.
  3. Resolve label index (last if `-1`), extract label token, map to `+1` if equals `positive_label` else `-1`.
  4. Parse remaining tokens as `double`, collect as features.
  5. Assert `features.size() == expected_features` (clear error message on mismatch).
- Define `ALL_DATASETS` with the four specs.
- Keep `normalise` and `train_test_split` untouched.

### `src/main.cpp`

- Extract current body into `void run_pipeline(const DatasetSpec& spec)`.
- Replace the hardcoded header (`"=== SVM Classifier - Breast Cancer Wisconsin ==="`) with `spec.name`.
- Replace `"Malignant (M=+1) / Benign (B=-1)"` labels with generic `"Positive (+1) / Negative (-1)"` plus the dataset's mapping in parentheses.
- New `main()`: loop `for (const auto& spec : ALL_DATASETS) run_pipeline(spec);`.

### `data/` directory

Add download entries for the three new datasets (curl commands in README). All gitignored, same as Wisconsin.

### `README.md`

- Add download instructions for Ionosphere, Spambase, Banknote under existing Wisconsin instructions.
- Note expected file paths: `data/ionosphere.data`, `data/spambase.data`, `data/banknote.txt`.
- Update "what the binary does" section to say it runs the pipeline on four datasets.

### `resources/assignment-report.md` (optional, after results are in)

- Task 1 still focuses on Wisconsin as the primary dataset.
- Task 2 section uses the multi-dataset results as empirical evidence for the "ability to solve various problems" discussion (different domains, sample sizes 351–4601, feature counts 4–57).

### `CLAUDE.md`

- Update project overview to reflect four datasets.
- Update the "Build and Run" section (no new flags, but clarify the binary now iterates).
- Update the hyperparameter table or add a note that grid search runs on all four.

## Implementation order

1. **Parser refactor** — add `DatasetSpec`, rewrite `load_csv`, verify Wisconsin still loads and produces identical results to current behaviour. This is the only risk point; lock it in first.
2. **Add dataset specs and download three new files** — test each loads cleanly with a tiny debug print of sample count / feature count / class balance before running the full pipeline.
3. **Refactor `main()` into `run_pipeline()`** — verify output identical for Wisconsin.
4. **Loop all four** — run end-to-end, capture output to `resources/run_output.txt`.
5. **Time and review** — Spambase likely dominates runtime (4601² kernel matrix, grid search × 5-fold × 3 kernels). If unacceptable, option to skip grid search on Spambase or reduce folds.
6. **Update README and CLAUDE.md.**
7. **(Optional) Extend report** with cross-dataset comparison table.

## Risk points

- **Spambase runtime.** 4601 samples → ~170MB kernel matrix, SMO convergence slower. Plan: time a single-kernel default-params run first; if grid search becomes prohibitive, add a per-dataset `run_grid_search` flag to `DatasetSpec`.
- **Ionosphere feature 2 is all zeros.** Z-score normalisation's `std_dev > 1e-10` guard already handles this — feature stays at 0 post-normalise. Worth a comment in the report.
- **Spambase gamma scale.** With 57 features, the current `gamma ∈ {0.001, 0.01, 0.1, 1}` should still include a workable value (likely `0.01` or `0.001`). If best gamma lands at a grid edge, extend the grid.
- **Label mapping for Banknote/Spambase.** Numeric labels (`0`/`1`) work fine — `positive_label = "1"` matches the string token.
- **Output volume.** Four datasets × verbose grid search = long log. Consider adding a terminal summary table at the end with best params and test accuracy per dataset/kernel.

## Dataset download URLs

- Wisconsin — already documented in README.
- Ionosphere — `https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data`
- Spambase — `https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data`
- Banknote — `https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt`

## Deliverables

- Refactored `data_loader.{hpp,cpp}` with `DatasetSpec`.
- Refactored `main.cpp` with `run_pipeline()`.
- Three new dataset files in `data/` (gitignored).
- Updated `README.md` with download commands.
- Updated `CLAUDE.md` reflecting multi-dataset scope.
- Fresh `resources/run_output.txt` covering all four datasets.
- (Optional) Cross-dataset comparison section in the report.
