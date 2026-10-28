# GAVA AI — Smart Waste Monitoring & Automated Fine Generation System

> **Research prototype** developed for MITS Kerala (CSE Department).  
> Not for real enforcement. All incidents require authorised human review.

---

## 📁 Project Location

```
c:\Users\VICTUS\OneDrive\Desktop\waste_gava\
```

## 🗂️ Project Structure

```
waste_gava/
├── app.py                          # Streamlit dashboard (8 pages)
├── config.py                       # Central configuration
├── requirements.txt
├── pytest.ini
│
├── modules/
│   ├── littering_detection.py      # Member 1 – Littering detection
│   ├── waste_classification.py     # Member 2 – Waste classification
│   ├── person_tracking.py          # Member 3 – Person tracking
│   └── fine_report.py              # Member 4 – Fine report generation
│
├── utils/
│   ├── dataset_utils.py            # Dataset discovery & image loaders
│   └── integration.py             # Full pipeline orchestrator
│
├── datasets/                       # Manually downloaded datasets go here
│   ├── trashnet/
│   ├── mivia_iwdd/
│   ├── taco/
│   └── market1501/
│
├── models/                         # Saved model weights
├── outputs/
│   ├── reports/                    # Generated PDF/TXT incident reports
│   └── evidence/                   # Saved evidence frames
│
├── sample_data/
│   ├── generate_samples.py         # Generates synthetic test data
│   ├── synthetic_waste_images/     # 6 synthetic class images
│   ├── synthetic_video.avi         # 60-frame synthetic video
│   └── sample_incident.json        # Sample incident for report testing
│
└── tests/
    ├── test_littering_detection.py
    ├── test_waste_classification.py
    ├── test_person_tracking.py
    ├── test_fine_report.py
    ├── test_integration.py
    └── test_dataset_utils.py
```

---

## ⚡ Quick Start

### 1. Install dependencies

```powershell
cd "c:\Users\VICTUS\OneDrive\Desktop\waste_gava"
pip install -r requirements.txt
```

### 2. Generate sample data (synthetic, no real datasets needed)

```powershell
python sample_data/generate_samples.py
```

### 3. Launch the Streamlit dashboard

```powershell
streamlit run app.py
```

> Enable **Demo Mode** in the sidebar (on by default) to explore all features  
> without real datasets or videos.

---

## 🧪 Running Tests

```powershell
# All tests
python -m pytest

# Individual module
python -m pytest tests/test_waste_classification.py -v
python -m pytest tests/test_littering_detection.py  -v
python -m pytest tests/test_person_tracking.py      -v
python -m pytest tests/test_fine_report.py          -v
python -m pytest tests/test_integration.py          -v
```

---

## 🔧 Running Modules Directly

```powershell
# Member 1 – Littering Detection (demo)
python modules/littering_detection.py --demo

# Member 2 – Waste Classification (demo)
python modules/waste_classification.py --demo

# Member 2 – Waste Classification (real image)
python modules/waste_classification.py --image sample_data/synthetic_waste_images/plastic_sample.png

# Member 3 – Person Tracking (demo)
python modules/person_tracking.py --demo

# Member 4 – Fine Report Generation (demo)
python modules/fine_report.py --demo

# Full integration pipeline (demo)
python utils/integration.py --demo

# Check dataset & model status
python utils/dataset_utils.py
```

---

## 📦 Datasets (Manual Download Required)

| Dataset | Module | Download |
|---------|--------|----------|
| TrashNet | Waste Classification | https://github.com/garythung/trashnet |
| MIVIA-IWDD | Littering Detection | https://github.com/MiviaLab/IllegalWasteDumping |
| TACO | Optional litter data | https://github.com/pedropro/TACO |
| Market-1501 | Person Re-ID (optional) | https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html |

Place each dataset in the corresponding `datasets/<name>/` sub-folder.

---

## 🤖 Optional: Deep-Learning Backends

**MobileNetV2 features** (better waste classification accuracy):
```powershell
pip install torch torchvision
```

**YOLOv8n person detection** (better tracking):
```powershell
pip install ultralytics
# yolov8n.pt (~6 MB) downloads automatically on first use
```

Both are optional — all modules fall back gracefully without them.

---

## 👥 Team Modules

| Member | Module | File |
|--------|--------|------|
| Member 1 | Littering Detection | `modules/littering_detection.py` |
| Member 2 | Waste Classification | `modules/waste_classification.py` |
| Member 3 | Person Tracking | `modules/person_tracking.py` |
| Member 4 | Fine Report Generation | `modules/fine_report.py` |

---

## ⚠️ Ethical & Legal Disclaimers

- **No student data** is collected, stored, or processed by this system.  
- **Track IDs are anonymous** — no identity matching is performed.  
- **All incidents require authorised human review** before any action.  
- This is a **research prototype** — not production-ready enforcement software.  
- Complies with applicable privacy regulations.

---

## 📋 Limitations

- Demo predictions are simulated (classifier not trained on real data by default).
- Motion heuristic has a high false-positive rate on real video.
- Single-camera, batch processing only (no real-time streaming).
- Requires manual dataset download for full functionality.

---

*MITS Kerala · Department of Computer Science and Engineering · GAVA AI Project*
