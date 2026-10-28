"""
evaluation/generate_report.py
==============================
Generates a complete Markdown + plain-text performance report
from the metrics engine results, and saves chart images.

Run:  python evaluation/generate_report.py
"""

from __future__ import annotations
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

OUT_DIR    = ROOT / "outputs" / "evaluation_charts"
REPORT_DIR = ROOT / "outputs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def generate_markdown_report(results: dict, chart_paths: dict) -> str:
    r    = results
    lit  = r["littering"]
    clf  = r["classification"]
    trk  = r["tracking"]
    rep  = r["report"]
    tst  = r["tests"]
    meta = r["_meta"]
    ts   = meta["timestamp"]

    lines = []
    def h(n, t): lines.append(f"{'#'*n} {t}\n")
    def p(*args): lines.append(" ".join(str(a) for a in args) + "\n")
    def sep(): lines.append("---\n")
    def blank(): lines.append("\n")

    h(1, "GAVA AI — Full Performance Evaluation Report")
    p(f"**Generated:** {ts}  |  **Python:** {meta['python']}")
    p("**Institution:** Muthoot Institute of Technology and Science (MITS), Kerala")
    p("**Department:** Computer Science and Engineering")
    p(f"**Evaluation time:** {meta['eval_time_s']} s")
    blank()
    p("> ⚠️ **Data Source Note:** All metrics are computed on **synthetic benchmark")
    p("> datasets** with known ground truth. Real-world performance will differ")
    p("> once actual labelled campus footage and trained models are available.")
    sep()

    # ── Test Suite ────────────────────────────────────────────────────────────
    h(2, "0. Test Suite Results (Live)")
    lines.append(f"""
| Metric | Value |
|--------|-------|
| Total Tests | {tst['total']} |
| Passed | **{tst['passed']}** ✅ |
| Failed | {tst['failed']} |
| Skipped | {tst['skipped']} |
| Pass Rate | **{tst['pass_rate']:.1%}** |
| Data Source | {tst['data_source']} |
""")
    sep()

    # ── Module 1 ──────────────────────────────────────────────────────────────
    h(2, "1. Littering Detection (Member 1)")
    p(f"**Data:** {lit['data_source']}")
    p(f"*{lit['note']}*")
    blank()
    lines.append(f"""
| Metric | Value |
|--------|-------|
| Accuracy | **{lit['accuracy']:.4f}** |
| Precision | {lit['precision']:.4f} |
| Recall (Sensitivity) | {lit['recall']:.4f} |
| F1-Score | **{lit['f1_score']:.4f}** |
| Specificity | {lit['specificity']:.4f} |
| False Positive Rate | {lit['false_positive_rate']:.4f} |
| False Negative Rate | {lit['false_negative_rate']:.4f} |
| AUC-ROC | **{lit['auc_roc']:.4f}** |
| Samples | {lit['n_samples']} |
| Positive Rate | {lit['positive_rate']:.1%} |
""")
    cm = lit["confusion_matrix"]
    lines.append(f"""
**Confusion Matrix:**

|  | Pred: No Litter | Pred: Littering |
|--|-----------------|-----------------|
| **Actual: No Litter** | TN = {cm['TN']} | FP = {cm['FP']} |
| **Actual: Littering** | FN = {cm['FN']} | TP = {cm['TP']} |
""")
    lat = lit["latency_ms"]
    lines.append(f"""
**Detection Latency (per frame, ms):**

| Mean | Std | Min | Max | P95 |
|------|-----|-----|-----|-----|
| {lat['mean']} | {lat['std']} | {lat['min']} | {lat['max']} | {lat['p95']} |
""")
    sep()

    # ── Module 2 ──────────────────────────────────────────────────────────────
    h(2, "2. Waste Classification (Member 2)")
    p(f"**Data:** {clf['data_source']}")
    p(f"*{clf['note']}*")
    blank()
    lines.append(f"""
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **{clf['top1_accuracy']:.4f}** ({clf['top1_accuracy']:.1%}) |
| Top-2 Accuracy | {clf['top2_accuracy']:.4f} ({clf['top2_accuracy']:.1%}) |
| Cohen's Kappa | **{clf['cohen_kappa']:.4f}** |
| Macro Precision | {clf['macro_precision']:.4f} |
| Macro Recall | {clf['macro_recall']:.4f} |
| Macro F1 | **{clf['macro_f1']:.4f}** |
| Weighted F1 | {clf['weighted_f1']:.4f} |
| Classes | {clf['n_classes']} |
| Samples | {clf['n_samples']} |
| Feature Extractor | {clf['feature_extractor']} |
| Classifier | {clf['classifier']} |
""")
    lines.append("\n**Per-Class Breakdown:**\n")
    lines.append("| Class | Precision | Recall | F1-Score | Support |\n")
    lines.append("|-------|-----------|--------|----------|---------|\n")
    for cls in clf["classes"]:
        pc = clf["per_class"][cls]
        lines.append(f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} "
                     f"| {pc['f1_score']:.4f} | {pc['support']} |\n")
    blank()
    sep()

    # ── Module 3 ──────────────────────────────────────────────────────────────
    h(2, "3. Person Tracking (Member 3)")
    p(f"**Data:** {trk['data_source']}")
    p(f"*{trk['note']}*")
    blank()
    lines.append(f"""
| Metric | Value | Notes |
|--------|-------|-------|
| MOTA | **{trk['MOTA']:.4f}** | Multi-Object Tracking Accuracy |
| MOTP | {trk['MOTP']:.4f} | Multi-Object Tracking Precision (mean IoU) |
| IDF1 | **{trk['IDF1']:.4f}** | ID F1-Score |
| Detection Precision | {trk['detection_precision']:.4f} | |
| Detection Recall | {trk['detection_recall']:.4f} | |
| ID Switches | {trk['id_switches']} | Lower is better |
| Track Fragmentations | {trk['track_fragmentations']} | Lower is better |
| Track Completeness | {trk['track_completeness']:.4f} | |
| GT Tracks | {trk['gt_tracks']} | |
| Predicted Tracks | {trk['predicted_tracks']} | |
| False Positives | {trk['false_positives']} | |
| False Negatives | {trk['false_negatives']} | |
| Sequence Frames | {trk['sequence_frames']} | |
| Detector | {trk['detector']} | |
| Tracker | {trk['tracker']} | |
""")
    sep()

    # ── Module 4 ──────────────────────────────────────────────────────────────
    h(2, "4. Fine Report Generation (Member 4)")
    p(f"**Data:** {rep['data_source']}")
    p(f"*{rep['note']}*")
    blank()
    t = rep["generation_time_ms"]
    lines.append(f"""
| Metric | Value |
|--------|-------|
| Success Rate | **{rep['success_rate']:.1%}** |
| Report Format | {rep['report_format']} |
| Field Completeness | {rep['field_completeness']:.1%} |
| Mean Generation Time | {t['mean']:.2f} ms |
| Std Dev | {t['std']:.2f} ms |
| Min | {t['min']:.2f} ms |
| Max | {t['max']:.2f} ms |
| 95th Percentile | {t['p95']:.2f} ms |
| Trials | {rep['trials']} |
""")
    sep()

    # ── Summary Table ─────────────────────────────────────────────────────────
    h(2, "5. Summary — Key Metrics Across All Modules")
    lines.append(f"""
| Module | Key Metric | Value | Benchmark Source |
|--------|-----------|-------|-----------------|
| Littering Detection | F1-Score | {lit['f1_score']:.4f} | Synthetic (n={lit['n_samples']}) |
| Littering Detection | AUC-ROC | {lit['auc_roc']:.4f} | Synthetic (n={lit['n_samples']}) |
| Littering Detection | Accuracy | {lit['accuracy']:.4f} | Synthetic (n={lit['n_samples']}) |
| Waste Classification | Top-1 Accuracy | {clf['top1_accuracy']:.4f} | Synthetic (n={clf['n_samples']}) |
| Waste Classification | Macro F1 | {clf['macro_f1']:.4f} | Synthetic (n={clf['n_samples']}) |
| Waste Classification | Cohen's Kappa | {clf['cohen_kappa']:.4f} | Synthetic (n={clf['n_samples']}) |
| Person Tracking | MOTA | {trk['MOTA']:.4f} | Synthetic sequence |
| Person Tracking | IDF1 | {trk['IDF1']:.4f} | Synthetic sequence |
| Report Generation | Success Rate | {rep['success_rate']:.1%} | Live ({rep['trials']} trials) |
| Test Suite | Pass Rate | {tst['pass_rate']:.1%} | Live pytest run |
""")
    sep()

    # ── Limitations ───────────────────────────────────────────────────────────
    h(2, "6. Limitations & Honest Assessment")
    lines.append("""
| Limitation | Impact |
|-----------|--------|
| Metrics on synthetic data only | Real-world accuracy may differ significantly |
| Classifier not trained on real TrashNet | Demo predictions are simulated |
| Motion heuristic only | No deep learned littering detector |
| No real labelled video | Tracking evaluated on procedural sequences |
| HOG detector (fallback) | Lower detection accuracy than YOLOv8n |
| Identity matching disabled | Cannot link tracks to real persons |
| Single camera | No cross-camera tracking tested |

**To obtain real metrics:**
1. Download TrashNet → train classifier → re-evaluate on held-out test split
2. Collect labelled campus footage → fine-tune YOLOv8 for littering detection
3. Install ultralytics (`pip install ultralytics`) for full YOLOv8n tracking
""")
    sep()
    p(f"*Report generated by GAVA AI Evaluation Suite · {ts}*")
    p("*MITS Kerala · Department of Computer Science and Engineering*")

    return "".join(lines)


def save_report(results: dict, chart_paths: dict) -> str:
    md   = generate_markdown_report(results, chart_paths)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"GAVA_AI_Evaluation_{ts}.md"
    path.write_text(md, encoding="utf-8")

    # Also save raw JSON
    json_path = REPORT_DIR / f"GAVA_AI_Metrics_{ts}.json"
    safe = {k: v for k, v in results.items() if k != "tests" or
            (k == "tests" and "raw_output" not in v)}
    for key in safe:
        if isinstance(safe[key], dict) and "raw_output" in safe[key]:
            safe[key] = {k2: v2 for k2, v2 in safe[key].items() if k2 != "raw_output"}
    json_path.write_text(json.dumps(safe, indent=2), encoding="utf-8")

    print(f"\nMarkdown report : {path}")
    print(f"JSON metrics    : {json_path}")
    return str(path)


if __name__ == "__main__":
    from evaluation.metrics_engine import run_all_evaluations
    from evaluation.plot_metrics   import generate_all_charts

    results     = run_all_evaluations()
    chart_paths = generate_all_charts(results)
    save_report(results, chart_paths)
    print("\nDone. Check outputs/reports/ and outputs/evaluation_charts/")
