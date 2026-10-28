# GAVA AI — Full Performance Evaluation Report
**Generated:** 2026-10-24 21:47:34  |  **Python:** 3.13.9
**Institution:** Muthoot Institute of Technology and Science (MITS), Kerala
**Department:** Computer Science and Engineering
**Evaluation time:** 5.3882 s

> ⚠️ **Data Source Note:** All metrics are computed on **synthetic benchmark
> datasets** with known ground truth. Real-world performance will differ
> once actual labelled campus footage and trained models are available.
---
## 0. Test Suite Results (Live)

| Metric | Value |
|--------|-------|
| Total Tests | 79 |
| Passed | **79** ✅ |
| Failed | 0 |
| Skipped | 0 |
| Pass Rate | **100.0%** |
| Data Source | Live pytest run |
---
## 1. Littering Detection (Member 1)
**Data:** Synthetic benchmark (n=300, 30% positive rate)
*Motion-heuristic evaluated on procedurally generated sequences with known GT*


| Metric | Value |
|--------|-------|
| Accuracy | **0.8100** |
| Precision | 0.6293 |
| Recall (Sensitivity) | 0.8391 |
| F1-Score | **0.7192** |
| Specificity | 0.7981 |
| False Positive Rate | 0.2019 |
| False Negative Rate | 0.1609 |
| AUC-ROC | **1.0000** |
| Samples | 300 |
| Positive Rate | 29.0% |

**Confusion Matrix:**

|  | Pred: No Litter | Pred: Littering |
|--|-----------------|-----------------|
| **Actual: No Litter** | TN = 170 | FP = 43 |
| **Actual: Littering** | FN = 14 | TP = 73 |

**Detection Latency (per frame, ms):**

| Mean | Std | Min | Max | P95 |
|------|-----|-----|-----|-----|
| 8.3935 | 2.2071 | 3.0 | 13.6606 | 12.1598 |
---
## 2. Waste Classification (Member 2)
**Data:** Synthetic benchmark (n=480, 80 samples/class)
*Calibrated to published histogram-feature MLP baselines on TrashNet-6*


| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **0.7750** (77.5%) |
| Top-2 Accuracy | 0.8706 (87.1%) |
| Cohen's Kappa | **0.7300** |
| Macro Precision | 0.7808 |
| Macro Recall | 0.7750 |
| Macro F1 | **0.7747** |
| Weighted F1 | 0.7747 |
| Classes | 6 |
| Samples | 480 |
| Feature Extractor | Colour histogram (R,G,B 256-bin) + edge density |
| Classifier | scikit-learn MLP (256→128 hidden, max_iter=300) |

**Per-Class Breakdown:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| cardboard | 0.8769 | 0.7125 | 0.7862 | 80 |
| glass | 0.8088 | 0.6875 | 0.7432 | 80 |
| metal | 0.7907 | 0.8500 | 0.8193 | 80 |
| paper | 0.7303 | 0.8125 | 0.7692 | 80 |
| plastic | 0.7191 | 0.8000 | 0.7574 | 80 |
| trash | 0.7590 | 0.7875 | 0.7730 | 80 |

---
## 3. Person Tracking (Member 3)
**Data:** Synthetic sequence (120 frames, 4 GT tracks, HOG+IoU sim)
*MOTA/MOTP computed on procedurally generated GT with 12% miss rate, 5% FP*


| Metric | Value | Notes |
|--------|-------|-------|
| MOTA | **-0.7219** | Multi-Object Tracking Accuracy |
| MOTP | 0.7725 | Multi-Object Tracking Precision (mean IoU) |
| IDF1 | **0.0727** | ID F1-Score |
| Detection Precision | 0.0430 | |
| Detection Recall | 0.2351 | |
| ID Switches | 0 | Lower is better |
| Track Fragmentations | 4 | Lower is better |
| Track Completeness | 0.5000 | |
| GT Tracks | 4 | |
| Predicted Tracks | 8 | |
| False Positives | 289 | |
| False Negatives | 231 | |
| Sequence Frames | 120 | |
| Detector | HOG pedestrian (fallback) / YOLOv8n (if installed) | |
| Tracker | Centroid-IoU tracker (max_age=30) | |
---
## 4. Fine Report Generation (Member 4)
**Data:** Live benchmark (20 report generation trials)
*Measured on actual IncidentReportGenerator with demo incidents*


| Metric | Value |
|--------|-------|
| Success Rate | **100.0%** |
| Report Format | PDF (ReportLab) |
| Field Completeness | 100.0% |
| Mean Generation Time | 8.49 ms |
| Std Dev | 1.57 ms |
| Min | 6.96 ms |
| Max | 14.65 ms |
| 95th Percentile | 9.68 ms |
| Trials | 20 |
---
## 5. Summary — Key Metrics Across All Modules

| Module | Key Metric | Value | Benchmark Source |
|--------|-----------|-------|-----------------|
| Littering Detection | F1-Score | 0.7192 | Synthetic (n=300) |
| Littering Detection | AUC-ROC | 1.0000 | Synthetic (n=300) |
| Littering Detection | Accuracy | 0.8100 | Synthetic (n=300) |
| Waste Classification | Top-1 Accuracy | 0.7750 | Synthetic (n=480) |
| Waste Classification | Macro F1 | 0.7747 | Synthetic (n=480) |
| Waste Classification | Cohen's Kappa | 0.7300 | Synthetic (n=480) |
| Person Tracking | MOTA | -0.7219 | Synthetic sequence |
| Person Tracking | IDF1 | 0.0727 | Synthetic sequence |
| Report Generation | Success Rate | 100.0% | Live (20 trials) |
| Test Suite | Pass Rate | 100.0% | Live pytest run |
---
## 6. Limitations & Honest Assessment

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
---
*Report generated by GAVA AI Evaluation Suite · 2026-10-24 21:47:34*
*MITS Kerala · Department of Computer Science and Engineering*
