#!/usr/bin/env python3
"""Parse SVM classifier output and generate publication-quality figures."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Constants ---
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "figures"
DEFAULT_INPUT = SCRIPT_DIR / "run_output.txt"
DPI = 300

# Colorblind-friendly palette
BLUE = "#2196F3"
GREEN = "#4CAF50"
ORANGE = "#FF9800"
RED = "#F44336"
PURPLE = "#9C27B0"
TEAL = "#009688"
METRIC_COLORS = [BLUE, GREEN, ORANGE, RED]
KERNEL_COLORS = [BLUE, ORANGE, GREEN]


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────

def parse_metrics(lines, i):
    """Parse an === Evaluation Results === block. Returns (metrics_dict, next_index)."""
    metrics = {}
    cm = {}
    while i < len(lines):
        line = lines[i].strip()
        if m := re.match(r"Accuracy:\s+([\d.]+)", line):
            metrics["accuracy"] = float(m.group(1))
        elif m := re.match(r"Precision:\s+([\d.]+)", line):
            metrics["precision"] = float(m.group(1))
        elif m := re.match(r"Recall:\s+([\d.]+)", line):
            metrics["recall"] = float(m.group(1))
        elif m := re.match(r"F1 Score:\s+([\d.]+)", line):
            metrics["f1"] = float(m.group(1))
        elif m := re.match(r"Actual \+1 \(M\)\s+(\d+)\s+(\d+)", line):
            cm["tp"] = int(m.group(1))
            cm["fn"] = int(m.group(2))
        elif m := re.match(r"Actual -1 \(B\)\s+(\d+)\s+(\d+)", line):
            cm["fp"] = int(m.group(1))
            cm["tn"] = int(m.group(2))
            metrics["cm"] = cm
            return metrics, i + 1
        i += 1
    metrics["cm"] = cm
    return metrics, i


def parse_output(text):
    """Parse full program output into structured dict."""
    lines = text.splitlines()
    data = {
        "initial_rbf": {},
        "cv_folds": [],
        "cv_mean": 0.0,
        "kernel_comparison": {},
        "grid_search": {},
        "optimised": {},
    }

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip noise
        if not line or line.startswith("Training complete") or line.startswith("Loading") or line.startswith("Samples") or line.startswith("Malignant") or line.startswith("Normalising") or line.startswith("Train:"):
            i += 1
            continue

        # Initial RBF result
        if re.match(r"--- RBF Kernel \(C=", line):
            i += 1
            # find next === Evaluation Results ===
            while i < len(lines) and "=== Evaluation Results ===" not in lines[i]:
                i += 1
            data["initial_rbf"], i = parse_metrics(lines, i)
            continue

        # 5-Fold CV
        if "5-Fold Cross-Validation" in line:
            i += 1
            while i < len(lines):
                fline = lines[i].strip()
                if fline.startswith("Training complete"):
                    i += 1
                    continue
                if m := re.match(r"Fold (\d+)/(\d+): accuracy = ([\d.]+)", fline):
                    data["cv_folds"].append({
                        "fold": int(m.group(1)),
                        "accuracy": float(m.group(3)),
                    })
                    i += 1
                    continue
                if m := re.match(r"Mean CV Accuracy:\s+([\d.]+)", fline):
                    data["cv_mean"] = float(m.group(1))
                    i += 1
                    break
                i += 1
            continue

        # Kernel Comparison (Default Params)
        if "Kernel Comparison (Default Params)" in line:
            i += 1
            while i < len(lines):
                cline = lines[i].strip()
                if cline.startswith("=== Grid Search") or cline.startswith("=== SVM"):
                    break
                if m := re.match(r"--- (\w+) Kernel ---", cline):
                    kernel_name = m.group(1)
                    i += 1
                    while i < len(lines) and "=== Evaluation Results ===" not in lines[i]:
                        i += 1
                    data["kernel_comparison"][kernel_name], i = parse_metrics(lines, i)
                    continue
                i += 1
            continue

        # Grid Search sections
        if m := re.match(r"=== Grid Search \((\w+)\) ===", line):
            kernel_name = m.group(1)
            gs = {"results": [], "best": {}}
            i += 1
            while i < len(lines):
                gline = lines[i].strip()
                if gline.startswith("Training complete"):
                    i += 1
                    continue
                # Grid search result line (with degree)
                if gm := re.match(r"C=([\d.]+),\s*gamma=([\d.]+),\s*degree=(\d+)\s*-> CV accuracy:\s*([\d.]+)", gline):
                    gs["results"].append({
                        "C": float(gm.group(1)),
                        "gamma": float(gm.group(2)),
                        "degree": int(gm.group(3)),
                        "accuracy": float(gm.group(4)),
                    })
                    i += 1
                    continue
                # Grid search result line (no degree)
                if gm := re.match(r"C=([\d.]+),\s*gamma=([\d.]+)\s*-> CV accuracy:\s*([\d.]+)", gline):
                    gs["results"].append({
                        "C": float(gm.group(1)),
                        "gamma": float(gm.group(2)),
                        "accuracy": float(gm.group(3)),
                    })
                    i += 1
                    continue
                # Best line
                if gm := re.match(r"Best: C=([\d.]+),\s*gamma=([\d.]+)(?:,\s*degree=(\d+))?\s*-> CV accuracy:\s*([\d.]+)", gline):
                    gs["best"] = {
                        "C": float(gm.group(1)),
                        "gamma": float(gm.group(2)),
                        "accuracy": float(gm.group(4)),
                    }
                    if gm.group(3):
                        gs["best"]["degree"] = int(gm.group(3))
                    data["grid_search"][kernel_name] = gs
                    i += 1
                    break
                i += 1
            continue

        # Optimised results
        if m := re.match(r"--- (\w+) \(Optimised\) ---", line):
            kernel_name = m.group(1)
            i += 1
            while i < len(lines) and "=== Evaluation Results ===" not in lines[i]:
                i += 1
            data["optimised"][kernel_name], i = parse_metrics(lines, i)
            continue

        i += 1

    return data


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────

def setup_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    })


def save_fig(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Figure 1: Confusion Matrices
# ──────────────────────────────────────────────

def plot_confusion_matrices(data):
    panels = [
        ("Initial RBF\n(Default)", data["initial_rbf"]["cm"]),
        ("Linear\n(Optimised)", data["optimised"]["Linear"]["cm"]),
        ("RBF\n(Optimised)", data["optimised"]["RBF"]["cm"]),
        ("Polynomial\n(Optimised)", data["optimised"]["Polynomial"]["cm"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)

    for ax, (title, cm) in zip(axes, panels):
        matrix = np.array([[cm["tp"], cm["fn"]],
                           [cm["fp"], cm["tn"]]])
        ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred +1", "Pred -1"], fontsize=9)
        ax.set_yticklabels(["Actual +1", "Actual -1"], fontsize=9)
        ax.set_title(title, fontsize=11, pad=8)

        max_val = matrix.max()
        for row in range(2):
            for col in range(2):
                val = matrix[row, col]
                color = "white" if val > max_val * 0.6 else "black"
                ax.text(col, row, str(val), ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)

    fig.tight_layout()
    save_fig(fig, "confusion_matrices.png")


# ──────────────────────────────────────────────
# Figure 2: 5-Fold CV
# ──────────────────────────────────────────────

def plot_cv_folds(data):
    folds = data["cv_folds"]
    mean = data["cv_mean"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = [f["fold"] for f in folds]
    y = [f["accuracy"] for f in folds]

    bars = ax.bar(x, y, color=BLUE, alpha=0.8, edgecolor="white", width=0.6)

    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.axhline(y=mean, color=RED, linestyle="--", linewidth=2, label=f"Mean = {mean:.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("5-Fold Cross-Validation (RBF Kernel)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x])
    ax.set_ylim(0.80, 1.02)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "cv_folds.png")


# ──────────────────────────────────────────────
# Figure 3 & 4: Kernel Comparison (Default / Optimised)
# ──────────────────────────────────────────────

def plot_kernel_comparison(data, which, title_suffix, filename):
    source = data[which]
    kernels = ["Linear", "RBF", "Polynomial"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(kernels))
    width = 0.18
    offsets = np.arange(len(metrics)) - (len(metrics) - 1) / 2

    fig, ax = plt.subplots(figsize=(10, 6))

    for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [source[k][metric] for k in kernels]
        bars = ax.bar(x + offsets[j] * width, vals, width,
                      label=label, color=METRIC_COLORS[j], alpha=0.85,
                      edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(kernels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title(f"Kernel Comparison — {title_suffix}", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, filename)


# ──────────────────────────────────────────────
# Figure 5: Grid Search Heatmaps
# ──────────────────────────────────────────────

def plot_grid_search_heatmaps(data):
    gs = data["grid_search"]
    C_vals = sorted({r["C"] for r in gs["RBF"]["results"]})
    gamma_vals = sorted({r["gamma"] for r in gs["RBF"]["results"]})

    # Build 2D grids
    def build_grid(results, degree=None):
        grid = np.zeros((len(C_vals), len(gamma_vals)))
        for r in results:
            if degree is not None and r.get("degree") != degree:
                continue
            ci = C_vals.index(r["C"])
            gi = gamma_vals.index(r["gamma"])
            grid[ci, gi] = r["accuracy"]
        return grid

    # Polynomial degrees
    poly_degrees = sorted({r["degree"] for r in gs["Polynomial"]["results"]})

    # Layout: top row = Linear, RBF, (empty); bottom row = Poly deg 2, 3, 4
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle("Grid Search — CV Accuracy (C vs Gamma)", fontsize=16,
                 fontweight="bold")

    # Manual gridspec for better spacing
    gs_layout = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                                  left=0.06, right=0.92, top=0.92, bottom=0.06)

    # Gather all panels: (ax, grid, title, best_dict)
    panels = []
    # Linear
    ax_lin = fig.add_subplot(gs_layout[0, 0])
    panels.append((ax_lin, build_grid(gs["Linear"]["results"]),
                   "Linear", gs["Linear"]["best"]))
    # RBF
    ax_rbf = fig.add_subplot(gs_layout[0, 1])
    panels.append((ax_rbf, build_grid(gs["RBF"]["results"]),
                   "RBF", gs["RBF"]["best"]))

    # Polynomial by degree
    for idx, deg in enumerate(poly_degrees):
        ax_p = fig.add_subplot(gs_layout[1, idx])
        best = gs["Polynomial"]["best"]
        panels.append((ax_p, build_grid(gs["Polynomial"]["results"], degree=deg),
                       f"Polynomial (degree={deg})", best if best.get("degree") == deg else None))

    c_labels = [str(v) for v in C_vals]
    g_labels = [str(v) for v in gamma_vals]
    last_im = None

    for ax, grid, title, best in panels:
        last_im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(gamma_vals)))
        ax.set_xticklabels(g_labels, fontsize=9)
        ax.set_yticks(range(len(C_vals)))
        ax.set_yticklabels(c_labels, fontsize=9)
        ax.set_xlabel("Gamma")
        ax.set_ylabel("C")
        ax.set_title(title, fontsize=11, pad=6)

        max_val = grid.max()
        for ci in range(len(C_vals)):
            for gi in range(len(gamma_vals)):
                val = grid[ci, gi]
                color = "white" if val > (0.4 + (max_val - 0.4) * 0.6) else "black"
                ax.text(gi, ci, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        # Mark best cell
        if best is not None:
            bc = C_vals.index(best["C"])
            bg = gamma_vals.index(best["gamma"])
            ax.add_patch(Rectangle((bg - 0.5, bc - 0.5), 1, 1,
                                   fill=False, edgecolor="lime", linewidth=3))

    # Colorbar in the empty top-right slot
    cbar_ax = fig.add_subplot(gs_layout[0, 2])
    cbar_ax.set_visible(False)
    cbar = fig.colorbar(last_im, ax=cbar_ax, shrink=0.9, label="CV Accuracy",
                        fraction=0.8, pad=0.0)

    save_fig(fig, "grid_search_heatmaps.png")


# ──────────────────────────────────────────────
# Figure 6: Default vs Optimised
# ──────────────────────────────────────────────

def plot_default_vs_optimised(data):
    kernels = ["Linear", "RBF", "Polynomial"]
    default_acc = [data["kernel_comparison"][k]["accuracy"] for k in kernels]
    opt_acc = [data["optimised"][k]["accuracy"] for k in kernels]

    x = np.arange(len(kernels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_d = ax.bar(x - width / 2, default_acc, width, label="Default",
                    color="#90CAF9", edgecolor="white")
    bars_o = ax.bar(x + width / 2, opt_acc, width, label="Optimised",
                    color=BLUE, edgecolor="white")

    for bar, val in zip(bars_d, default_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    for bar, d, o in zip(bars_o, default_acc, opt_acc):
        delta = o - d
        label = f"{o:.2f}\n(+{delta:.2f})" if delta > 0 else f"{o:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                label, ha="center", va="bottom", fontsize=9,
                color="black", linespacing=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(kernels, fontsize=12)
    ax.set_ylabel("Accuracy")
    ax.set_title("Impact of Hyperparameter Tuning", fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "default_vs_optimised.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate plots from SVM classifier output")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to captured output file (default: run_output.txt)")
    parser.add_argument("--stdin", action="store_true",
                        help="Read from stdin instead of a file")
    args = parser.parse_args()

    if args.stdin:
        text = sys.stdin.read()
    else:
        if not args.input.exists():
            print(f"Error: {args.input} not found. Run the classifier first:")
            print(f"  ./build/svm_classifier | tee run_output.txt")
            sys.exit(1)
        text = args.input.read_text()

    print("Parsing output...")
    data = parse_output(text)

    # Validate parse
    if not data["cv_folds"]:
        print("Error: Could not parse CV fold data")
        sys.exit(1)
    if not data["kernel_comparison"]:
        print("Error: Could not parse kernel comparison data")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_style()

    print("Generating figures...")
    plot_confusion_matrices(data)
    plot_cv_folds(data)
    plot_kernel_comparison(data, "kernel_comparison", "Default Parameters",
                          "kernel_comparison_default.png")
    plot_kernel_comparison(data, "optimised", "Optimised Parameters",
                          "kernel_comparison_optimised.png")
    plot_grid_search_heatmaps(data)
    plot_default_vs_optimised(data)

    print(f"\nDone. {len(list(OUTPUT_DIR.glob('*.png')))} figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
