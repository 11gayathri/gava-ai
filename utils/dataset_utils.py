"""
utils/dataset_utils.py
========================
Dataset discovery helpers – no large files are auto-downloaded.
Provides status checks, image loaders, and manual download instructions.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    TRASHNET_CLASSES,
    TRASHNET_DIR, MIVIA_IWDD_DIR, TACO_DIR, MARKET1501_DIR,
    MODELS_DIR,
)

# ── Image extensions ──────────────────────────────────────────────────────────
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# ── Core loaders ──────────────────────────────────────────────────────────────

def load_image_rgb(path) -> Optional[np.ndarray]:
    """Load an image from *path* as a uint8 HxWx3 NumPy array, or None on failure."""
    try:
        from PIL import Image
        img = Image.open(str(path)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def resize_image(img: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize *img* (HxWx3 uint8) to *size* = (W, H) using PIL LANCZOS."""
    from PIL import Image
    pil = Image.fromarray(img)
    pil = pil.resize(size, Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


def list_images(directory) -> List[Path]:
    """Return all image files directly inside *directory* (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def list_videos(directory) -> List[Path]:
    """Return all video files directly inside *directory* (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _VIDEO_EXTS)


# ── TrashNet discovery ────────────────────────────────────────────────────────

def discover_trashnet(root=None) -> Dict[str, List[Path]]:
    """
    Return a dict {class_name: [list_of_image_paths]} for TrashNet.
    Returns an empty dict if the dataset root does not exist.
    """
    root = Path(root) if root else TRASHNET_DIR
    if not root.is_dir():
        return {}
    result: Dict[str, List[Path]] = {}
    for cls in TRASHNET_CLASSES:
        cls_dir = root / cls
        result[cls] = list_images(cls_dir) if cls_dir.is_dir() else []
    return result


# ── Dataset status ────────────────────────────────────────────────────────────

def check_dataset_status(datasets_dir=None) -> Dict[str, dict]:
    """Return availability info for every expected dataset."""
    base = Path(datasets_dir) if datasets_dir else TRASHNET_DIR.parent

    datasets = {
        "trashnet": {
            "name":     "TrashNet",
            "path":     base / "trashnet",
            "url":      "https://github.com/garythung/trashnet",
            "purpose":  "Waste classification (Member 2)",
        },
        "mivia_iwdd": {
            "name":     "MIVIA-IWDD",
            "path":     base / "mivia_iwdd",
            "url":      "https://github.com/MiviaLab/IllegalWasteDumping",
            "purpose":  "Littering / illegal dumping detection (Member 1)",
        },
        "taco": {
            "name":     "TACO",
            "path":     base / "taco",
            "url":      "https://github.com/pedropro/TACO",
            "purpose":  "Additional litter detection annotations (optional)",
        },
        "market1501": {
            "name":     "Market-1501",
            "path":     base / "market1501",
            "url":      "https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html",
            "purpose":  "Person re-identification (Member 3, optional)",
        },
    }

    for key, info in datasets.items():
        p = info["path"]
        info["available"] = p.is_dir() and any(p.iterdir()) if p.is_dir() else False
        info["file_count"] = sum(1 for _ in p.rglob("*") if _.is_file()) if p.is_dir() else 0

    return datasets


def check_model_status(models_dir=None) -> Dict[str, dict]:
    """Return availability of saved model weights."""
    base = Path(models_dir) if models_dir else MODELS_DIR
    models = {
        "waste_classifier": {
            "name": "Waste Classifier (scikit-learn pipeline)",
            "path": base / "waste_classifier.joblib",
        },
        "yolo": {
            "name": "YOLOv8n person detector",
            "path": base / "yolov8n.pt",
        },
    }
    for key, info in models.items():
        info["available"] = Path(info["path"]).exists()
    return models


# ── Manual download instructions ──────────────────────────────────────────────

DOWNLOAD_INSTRUCTIONS = """
DATASET MANUAL DOWNLOAD INSTRUCTIONS
======================================
No dataset or large model weight is downloaded automatically.
Follow the steps below for each dataset you want to use.

1. TrashNet  (Waste Classification – Member 2)
   Source : https://github.com/garythung/trashnet
   Steps  : Download the ZIP → extract so that you have:
              datasets/trashnet/cardboard/   (images)
              datasets/trashnet/glass/
              datasets/trashnet/metal/
              datasets/trashnet/paper/
              datasets/trashnet/plastic/
              datasets/trashnet/trash/

2. MIVIA-IWDD  (Littering Detection – Member 1)
   Source : https://github.com/MiviaLab/IllegalWasteDumping
   Steps  : Clone or download → place video clips under:
              datasets/mivia_iwdd/videos/
            Label CSVs (if any) under:
              datasets/mivia_iwdd/annotations/

3. TACO  (Optional additional litter data)
   Source : https://github.com/pedropro/TACO
   Steps  : Follow TACO README → place images under:
              datasets/taco/images/
            COCO annotations under:
              datasets/taco/annotations/

4. Market-1501  (Person Re-ID – Member 3, optional)
   Source : https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
   Steps  : Download zip → extract to:
              datasets/market1501/

YOLOv8n weights (person tracking):
   Install ultralytics:  pip install ultralytics
   The 6 MB yolov8n.pt will be downloaded automatically on first run.
"""


if __name__ == "__main__":
    print("=== GAVA AI – Dataset & Model Status ===\n")
    ds = check_dataset_status()
    for key, info in ds.items():
        status = "✓ AVAILABLE" if info["available"] else "✗ NOT FOUND"
        print(f"[{status}] {info['name']}")
        print(f"   Path    : {info['path']}")
        print(f"   Files   : {info['file_count']}")
        print(f"   Purpose : {info['purpose']}")
        if not info["available"]:
            print(f"   Download: {info['url']}")
        print()

    print("=== Model Weights ===\n")
    mdl = check_model_status()
    for key, info in mdl.items():
        status = "✓" if info["available"] else "✗"
        print(f"  [{status}] {info['name']}")
        print(f"       {info['path']}")
    print()
    print(DOWNLOAD_INSTRUCTIONS)
