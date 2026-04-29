# Implementation Plan: Drop Wisconsin, Restructure as Cross-Dataset Comparison

## Goal

Convert the project from a Wisconsin-primary case study with multi-dataset extension into a **3-dataset comparison study** across:

- **Ionosphere** — 351 × 34, radar return classification
- **Banknote Authentication** — 1372 × 4, banknote forensics
- **Spambase** — 4601 × 57, email spam filtering (grid search disabled)

Wisconsin (Breast Cancer) is removed from the pipeline, the captured run output, the figures, the docs, and the assignment report.

## Why

- The assignment brief asks to "evaluate one algorithm on a chosen dataset" — running the same algorithm across three datasets still satisfies this and answers Task 2's "ability to solve various problems" question with concrete data instead of prose.
- The current report's primary-dataset framing is an artefact of historical build order, not a brief requirement.
- A 3-way comparison gives a stronger Task 1 §1.5 results section (cross-dataset table + commentary) and tighter linkage between Task 1 evidence and Task 2 discussion.

## Scope (file-by-file)

| File | Change | Effort |
|---|---|---|
| `src/data_loader.cpp` | Remove Wisconsin entry from `ALL_DATASETS` | trivial |
| `src/main.cpp` | No change (already loops over `ALL_DATASETS`) | none |
| `data/wdbc.csv` | Leave on disk (gitignored, not pushed) | none |
| `resources/run_output.txt` | Regenerate after re-run | small |
| `figures/wisconsin_*.png` | Delete six PNGs | trivial |
| `README.md` | Drop Wisconsin row + curl line; update count to 3 | small |
| `docs/usage.md` | Same: drop Wisconsin from dataset table + downloads | small |
| `docs/architecture.md` | Sweep for Wisconsin-specific examples | small |
| `docs/algorithm.md` | Sweep for `n = 456`, "Wisconsin" mentions | small |
| `CLAUDE.md` | Update tech-stack line + hyperparameter table | small |
| `resources/assignment-report.md` | **Major rewrite** — see below | LARGE |
| `resources/multi-dataset-plan.md` | Mark superseded (kept as historical reference) | trivial |
| `resources/drop-wisconsin-plan.md` | This file | — |

## Implementation order

1. **Code & data.** Remove Wisconsin from `ALL_DATASETS`. Confirm the binary builds clean and runs end-to-end on the three remaining datasets.
2. **Re-run pipeline** to capture a clean `resources/run_output.txt` covering only the three datasets. Time it (expect ~2 min — Wisconsin grid was a chunk of the previous 2:45).
3. **Regenerate figures.** Delete `figures/wisconsin_*.png`. Run `scripts/plot_results.py`, confirm 14 PNGs (6 for Ionosphere, 6 for Banknote, 2 for Spambase).
4. **Sweep README + docs** for Wisconsin references. Wisconsin-specific numbers (e.g. "n = 456", "30 features") become Ionosphere or generic.
5. **Update CLAUDE.md** tech-stack line, hyperparameter table, and design-constraints sweep.
6. **Rewrite the assignment report** (largest chunk — see Report rewrite below).
7. **Commit and push.** Probably one commit for code/run/figures/docs, separate commit for the report rewrite (the report diff is large enough to warrant its own commit for review).

## Report rewrite

### Title & framing

- **Title:** `Kernel SVM Classifier with SMO: A Cross Dataset Comparison Study` (confirmed).
- **Header lines** (Module, Assignment, Name, Student ID): unchanged.
- **Opening framing** in §1.1: "This project evaluates a from-scratch C++ kernel SVM across three UCI binary-classification datasets spanning very different sample sizes (351–4601), feature counts (4–57), and problem domains (radar return, banknote forensics, email spam). The motivation for the cross-dataset evaluation is to make Task 2's discussion of strengths and limitations concrete rather than abstract."

### Section-level changes

| Section | Action |
|---|---|
| §1.1 Description of Program | Minor edit: replace "primary dataset" framing with "three datasets". Update runtime claim (≈ 2 min, three-dataset run). |
| §1.2 **Dataset → Datasets** | Full rewrite. Comparative table (sample / features / class balance / domain). One short paragraph per dataset describing what each is and why it's interesting (e.g., Spambase is the scalability cautionary tale). Shared preprocessing (z-score, 80/20 split, seed 42) explained once. |
| §1.3 Algorithm Implementation | Keep the maths intact. Remove Wisconsin-specific examples — "for `n = 456`" becomes generic ("for `n` in the hundreds-to-low-thousands"). Drop the breast-cancer-specific motivation paragraphs. |
| §1.4 Training | Replace single grid-search table with a two-table layout: (i) per-dataset best `C, γ, d` per kernel, (ii) per-dataset CV accuracy at the chosen winner. Keep the discussion of why CV matters and what 5-fold buys. |
| §1.5 Results | This becomes the centrepiece. Two big results tables: default-params (3 datasets × 3 kernels = 9 cells of accuracy / F1) and optimised-params (same shape). Followed by **per-dataset commentary** (one or two paragraphs each) explaining which kernel wins and why. Confusion-matrix subsection: pick the best model per dataset (Ionosphere RBF, Banknote RBF, Spambase RBF-default) and present the three matrices. |
| §1.6 Extended Evaluation Across Additional Datasets | **Delete** — the entire report is now this. |
| §2.1 Ability | Rewrite the "Kernel-based versatility" paragraph around the cross-dataset evidence. Replace the "Sparse solutions" example (was Wisconsin's 16 SVs) with whichever dataset's optimised model has the most striking sparsity. |
| §2.2 Limitations | Already cites Spambase concretely. Light edit only. |
| §2.3 Real-World Applications | Drop the breast-cancer-specific paragraphs. Keep the broader application categories (medical, NLP, finance, anomaly detection). The closing "Why not deep learning" paragraph stays but uses Banknote (~96.7% with 4 features) as the worked example instead of Wisconsin. |
| References | Drop Wolberg et al. 1995. Add UCI dataset references for Ionosphere (Sigillito et al. 1989), Banknote (Lohweg / UCI), Spambase (Hopkins / UCI) — already in the README links, just need formal citation lines. |
| Further Reading | Drop the "How Common Is Breast Cancer" link and the UCI Wisconsin link. Replace with the three new dataset UCI pages. |

### Risks

- **Word count.** The current report is around 8,000 words. Removing Wisconsin-specific narrative cuts maybe 1,500 words; the new cross-dataset commentary adds ~1,000 — net should land at a similar length. If the brief specifies a target (it might — `assignment-brief.md` is on disk locally), I'll check before drafting.
- **Tone shift.** The original report leans on the medical-screening framing for emotional weight ("a false negative could miss a malignant tumour"). The new datasets don't have that same gravitas. Replace with practitioner-grounded framing: "the cost of misclassifying a forged banknote", "the user-irritation cost of a spam false positive", "false negatives in radar interpretation".
- **Spambase asymmetry.** Two datasets get the full grid-search treatment, Spambase doesn't. The narrative needs to handle this gracefully — Spambase becomes the natural anchor for the §2.2 scalability discussion, not a deficiency.
- **Numerical drift.** With Wisconsin gone, only the three-dataset numbers matter. Those are already known from the current run (Ionosphere RBF 98.57%, Banknote RBF 96.72%, Spambase default RBF 75.87%) — no fresh hyperparameter surprises expected.

## Deliverables

- `src/data_loader.cpp` with three-entry `ALL_DATASETS`.
- Fresh `resources/run_output.txt` covering Ionosphere / Banknote / Spambase only.
- `figures/` containing 14 PNGs (no `wisconsin_*`).
- Updated `README.md`, `docs/*.md`, `CLAUDE.md`.
- Rewritten `resources/assignment-report.md` framed as a cross-dataset comparison.
- Two commits (code+infrastructure, then report rewrite) pushed to `main`.

## Pre-flight notes

- **Title:** `Kernel SVM Classifier with SMO: A Cross Dataset Comparison Study` (confirmed).
- **Word-count target:** none — `resources/assignment-brief.md` does not specify a length cap.
- **Citations:** keep existing SVM theory references (Cortes & Vapnik / Platt / Burges / Vapnik) and the scikit-learn doc link. Add UCI dataset references for Ionosphere (Sigillito et al. 1989), Banknote, Spambase using the existing numbered-list style.
