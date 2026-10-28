"""
evaluation/plot_metrics.py
===========================
Generates all performance metric charts and saves them to
  outputs/evaluation_charts/

Charts produced
---------------
  1. Littering detection confusion matrix + ROC curve
  2. Waste classification 6x6 confusion matrix (heatmap)
  3. Per-class precision / recall / F1 bar chart
  4. Tracking metrics bar chart (MOTA, MOTP, IDF1, Det-P, Det-R)
  5. Report generation time distribution
  6. Test suite pass/fail pie chart
  7. Summary radar / spider chart (one metric per module)

Run:  python evaluation/plot_metrics.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT   = Path(__file__).parent.parent.resolve()
OUT_DIR = ROOT / "outputs" / "evaluation_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from evaluation.metrics_engine import (
    evaluate_littering_detection,
    evaluate_waste_classification,
    evaluate_person_tracking,
    evaluate_report_generation,
    evaluate_test_suite,
)
from config import TRASHNET_CLASSES

PALETTE   = ["#00b4d8", "#0077b6", "#90e0ef", "#023e8a", "#48cae4", "#ade8f4"]
BG_COLOR  = "#0e1117"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2f3e"

def _style_ax(ax, title=""):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=10)

def _fig(rows=1, cols=1, w=10, h=6):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h),
                             facecolor=BG_COLOR, squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)
    return fig, axes


# ── 1. Littering Detection ─────────────────────────────────────────────────────

def plot_littering(data: dict) -> Path:
    fig = plt.figure(figsize=(14, 5), facecolor=BG_COLOR)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG_COLOR)
    cm  = np.array(data["confusion_matrix"]["matrix"])
    im  = ax1.imshow(cm, cmap="Blues", aspect="auto")
    labs = data["confusion_matrix"]["labels"]
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labs, color=TEXT_COLOR, fontsize=9)
    ax1.set_yticklabels(labs, color=TEXT_COLOR, fontsize=9)
    ax1.set_xlabel("Predicted", color=TEXT_COLOR)
    ax1.set_ylabel("Actual",    color=TEXT_COLOR)
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black",
                     fontsize=14, fontweight="bold")
    _style_ax(ax1, "Littering Detection – Confusion Matrix")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    # Metric bars
    ax2  = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG_COLOR)
    keys = ["accuracy", "precision", "recall", "f1_score", "specificity", "auc_roc"]
    vals = [data[k] for k in keys]
    bars = ax2.barh(keys, vals, color=PALETTE[:len(keys)], edgecolor=GRID_COLOR)
    ax2.set_xlim(0, 1.05)
    for bar, val in zip(bars, vals):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color=TEXT_COLOR, fontsize=9)
    ax2.axvline(x=0.5, color="#555", linestyle="--", linewidth=0.8)
    _style_ax(ax2, "Littering Detection – Key Metrics")
    ax2.set_xlabel("Score (0 – 1)", color=TEXT_COLOR)

    path = OUT_DIR / "01_littering_detection.png"
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight",
                                    facecolor=BG_COLOR); plt.close()
    print(f"  Saved: {path.name}")
    return path


# ── 2. Waste Classification Confusion Matrix ───────────────────────────────────

def plot_classification_cm(data: dict) -> Path:
    cm  = np.array(data["confusion_matrix"])
    n   = len(TRASHNET_CLASSES)
    fig, axes = _fig(1, 1, 9, 7)
    ax  = axes[0, 0]

    # Normalise row-wise for colour
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    im  = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(TRASHNET_CLASSES, rotation=35, ha="right",
                       color=TEXT_COLOR, fontsize=9)
    ax.set_yticklabels(TRASHNET_CLASSES, color=TEXT_COLOR, fontsize=9)
    ax.set_xlabel("Predicted Class",  color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("True Class",       color=TEXT_COLOR, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if cm_norm[i, j] > 0.5 else TEXT_COLOR)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    cbar.set_label("Normalised proportion", color=TEXT_COLOR, fontsize=9)
    _style_ax(ax, f"Waste Classification – 6×6 Confusion Matrix  "
                  f"(Top-1 Acc = {data['top1_accuracy']:.1%})")

    path = OUT_DIR / "02_waste_classification_cm.png"
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight",
                                    facecolor=BG_COLOR); plt.close()
    print(f"  Saved: {path.name}")
    return path


# ── 3. Per-class P / R / F1 ──────────────────────────────────────────────────

def plot_classification_prf(data: dict) -> Path:
    fig, axes = _fig(1, 1, 10, 5)
    ax  = axes[0, 0]
    cls = TRASHNET_CLASSES
    pc  = data["per_class"]
    x   = np.arange(len(cls))
    w   = 0.26

    b1 = ax.bar(x - w, [pc[c]["precision"] for c in cls], w, label="Precision",
                color=PALETTE[0], edgecolor=GRID_COLOR)
    b2 = ax.bar(x,     [pc[c]["recall"]    for c in cls], w, label="Recall",
                color=PALETTE[1], edgecolor=GRID_COLOR)
    b3 = ax.bar(x + w, [pc[c]["f1_score"]  for c in cls], w, label="F1-Score",
                color=PALETTE[2], edgecolor=GRID_COLOR)

    ax.set_xticks(x); ax.set_xticklabels(cls, color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.axhline(y=data["macro_f1"], color="#ff9f43", linestyle="--",
               linewidth=1, label=f"Macro F1 = {data['macro_f1']:.3f}")
    leg = ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                    labelcolor=TEXT_COLOR, fontsize=9)
    _style_ax(ax, "Waste Classification – Per-Class Precision / Recall / F1")
    ax.set_ylabel("Score", color=TEXT_COLOR)

    # Value labels on F1 bars
    for bar in b3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", color=TEXT_COLOR, fontsize=8)

    path = OUT_DIR / "03_waste_classification_prf.png"
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight",
                                    facecolor=BG_COLOR); plt.close()
    print(f"  Saved: {path.name}")
    return path


# ── 4. Tracking Metrics ───────────────────────────────────────────────────────

def plot_tracking(data: dict) -> Path:
    fig, axes = _fig(1, 2, 13, 5)

    # Bar chart of main metrics
    ax1  = axes[0, 0]
    keys = ["MOTA", "MOTP", "IDF1", "detection_precision", "detection_recall"]
    labs = ["MOTA", "MOTP", "IDF1", "Det Precision", "Det Recall"]
    vals = [data[k] for k in keys]
    bars = ax1.bar(labs, vals, color=PALETTE, edgecolor=GRID_COLOR)
    ax1.set_ylim(0, 1.1)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha="center", color=TEXT_COLOR, fontsize=9)
    ax1.axhline(0.5, color="#555", linestyle="--", linewidth=0.8)
    _style_ax(ax1, "Person Tracking – Main Metrics")
    ax1.set_ylabel("Score (0 – 1)", color=TEXT_COLOR)
    ax1.tick_params(axis="x", colors=TEXT_COLOR, labelsize=9)

    # Count stats
    ax2  = axes[0, 1]
    ckeys = ["gt_tracks", "predicted_tracks", "id_switches",
             "track_fragmentations", "false_positives", "false_negatives"]
    clabs = ["GT Tracks", "Pred Tracks", "ID Switches",
             "Fragmentations", "False Pos", "False Neg"]
    cvals = [data[k] for k in ckeys]
    colours = [PALETTE[0], PALETTE[1], "#e74c3c", "#e67e22", "#e74c3c", "#e67e22"]
    bars2 = ax2.bar(clabs, cvals, color=colours, edgecolor=GRID_COLOR)
    for bar, val in zip(bars2, cvals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                 str(val), ha="center", color=TEXT_COLOR, fontsize=9)
    _style_ax(ax2, "Person Tracking – Count Statistics")
    ax2.set_ylabel("Count", color=TEXT_COLOR)
    ax2.tick_params(axis="x", colors=TEXT_COLOR, labelsize=8, rotation=20)

    path = OUT_DIR / "04_person_tracking.png"
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight",
                                    facecolor=BG_COLOR); plt.close()
    print(f"  Saved: {path.name}")
    return path


# ── 5. Report Generation Time ─────────────────────────────────────────────────

def plot_report_gen(data: dict) -> Path:
    fig, axes = _fig(1, 2, 12, 4)

    t   = data["generation_time_ms"]
    rng2 = np.random.default_rng(42)
    # Simulate individual trial times around the reported mean/std
    samples = rng2.normal(t["mean"], max(t["std"], 0.1), size=data["trials"])
    samples = np.clip(samples, 0.1, None)

    ax1 = axes[0, 0]
    ax1.hist(samples, bins=10, color=PALETTE[0], edgecolor=GRID_COLOR, alpha=0.85)
    ax1.axvline(t["mean"], color="#ff9f43", linestyle="--",
                linewidth=1.5, label=f"Mean = {t['mean']:.1f} ms")
    ax1.axvline(t["p95"],  color="#e74c3c", linestyle=":",
                linewidth=1.5, label=f"P95  = {t['p95']:.1f} ms")
    ax1.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    _style_ax(ax1, "Report Generation – Time Distribution")
    ax1.set_xlabel("Time (ms)", color=TEXT_COLOR)
    ax1.set_ylabel("Frequency", color=TEXT_COLOR)

    ax2  = axes[0, 1]
    skeys = ["success_rate", "field_completeness"]
    slabs = ["Success Rate", "Field Completeness"]
    svals = [data[k] for k in skeys]
    bars  = ax2.bar(slabs, svals, color=[PALETTE[0], PALETTE[2]], edgecolor=GRID_COLOR)
    for bar, val in zip(bars, svals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.1%}", ha="center", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    _style_ax(ax2, "Report Generation – Quality Metrics")
    ax2.tick_params(axis="x", colors=TEXT_COLOR)

    path = OUT_DIR / "05_report_generation.png"
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight",
                                    facecolor=BG_COLOR); plt.close()
    print(f"  Saved: {path.name}")
    return path


# ── 6. Test Suite Pie ─────────────────────────────────────────────────────────

def plot_test_suite(data: dict) -> Path:
    fig, axes = _fig(1, 1, 6, 5)
    ax  = axes[0, 0]
    vals  = [data["passed"], data["failed"], data["skipped"]]
    labs  = [f"Passed ({data['passed']})", f"Failed ({data['failed']})",
             f"Skipped ({data['skipped']})"]
    cols  = ["#2ecc71", "#e74c3c", "#f39c12"]
    non_zero = [(v, l, c) for v, l, c in zip(vals, labs, cols) if v > 0]
    if non_zero:
        vs, ls, cs = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(
            vs, labels=ls, colors=cs, autopct="%1.0f%%",
            pctdistance=0.75, startangle=90,
            wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
        )
        for t in texts + autotexts:
            t.set_color(TEXT_COLOR)
            t.set_fontsize(10)
    ax.set_facecolor(BG_COLOR)
    _style_ax(ax, f"Test Suite – {data['total']} Tests  "
                  f"(Pass rate: {data['pass_rate']:.0%})")

    path = OUT_DIR / "06_test_suite.png"
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight",
                                    facecolor=BG_COLOR); plt.close()
    print(f"  Saved: {path.name}")
    return path


# ── 7. Summary Radar Chart ────────────────────────────────────────────────────

def plot_radar(lit, clf, trk, rep, tst) -> Path:
    labels = [
        "Littering\nF1", "Littering\nAUC",
        "Waste\nTop-1 Acc", "Waste\nMacro F1",
        "Tracking\nMOTA", "Tracking\nIDF1",
        "Report\nSuccess", "Test\nPass Rate",
    ]
    values = [
        lit["f1_score"], lit["auc_roc"],
        clf["top1_accuracy"], clf["macro_f1"],
        trk["MOTA"],      trk["IDF1"],
        rep["success_rate"], tst["pass_rate"],
    ]
    N    = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
                           subplot_kw={"polar": True}, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels,
                      color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       color="#888", fontsize=8)
    ax.yaxis.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5)
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5)

    ax.plot(angles, values_plot, "o-", linewidth=2, color=PALETTE[0])
    ax.fill(angles, values_plot, alpha=0.25, color=PALETTE[0])

    for angle, val in zip(angles[:-1], values):
        ax.text(angle, val + 0.06, f"{val:.2f}", ha="center", va="center",
                color=PALETTE[0], fontsize=8, fontweight="bold")

    ax.set_title("GAVA AI – Overall Performance Radar\n(Synthetic Benchmark)",
                 color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=20)

    path = OUT_DIR / "07_radar_summary.png"
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight",
                                    facecolor=BG_COLOR); plt.close()
    print(f"  Saved: {path.name}")
    return path


# ── Master runner ─────────────────────────────────────────────────────────────

def generate_all_charts(results: dict) -> dict:
    print("Generating charts ...\n")
    paths = {}
    paths["littering"]       = plot_littering(results["littering"])
    paths["classification_cm"]  = plot_classification_cm(results["classification"])
    paths["classification_prf"] = plot_classification_prf(results["classification"])
    paths["tracking"]        = plot_tracking(results["tracking"])
    paths["report"]          = plot_report_gen(results["report"])
    paths["test_suite"]      = plot_test_suite(results["tests"])
    paths["radar"]           = plot_radar(
        results["littering"], results["classification"],
        results["tracking"],  results["report"], results["tests"],
    )
    print(f"\nAll charts saved to: {OUT_DIR}")
    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":
    from evaluation.metrics_engine import run_all_evaluations
    results = run_all_evaluations()
    generate_all_charts(results)
