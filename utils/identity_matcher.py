"""
utils/identity_matcher.py
==========================
Authorised gallery identity matcher for GAVA AI.

Method
------
Uses a combined descriptor of:
  1. Colour histograms (HSV — 16 bins per channel → 4096-D)
  2. HOG texture descriptor (if OpenCV available → 324-D)
  Concatenated → L2-normalised → cosine similarity.

This runs entirely on CPU with NumPy + OpenCV (no deep learning weights
needed), while still discriminating between individuals with different
clothing/appearance patterns.

MobileNetV2 upgrade path
------------------------
If `torch` is installed, the matcher transparently switches to
MobileNetV2 penultimate-layer (1280-D) embeddings for much higher
accuracy — no code changes required in the callers.

Public API
----------
  IdentityMatcher(gallery_dir)   – class
    .load_gallery()              – reads all images from directory
    .match(crop_img)             – returns MatchResult dict
  match_result_unknown()         – returns an "unverified" dict
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    _TORCH = True
except ImportError:
    _TORCH = False

# ── Constants ─────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.70   # cosine similarity above this → "verified"
GALLERY_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractors
# ─────────────────────────────────────────────────────────────────────────────

def _colour_hist_feature(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract a flat HSV colour histogram (16 bins × 3 channels).
    Returns L2-normalised 1-D float32 vector.
    """
    if not _CV2:
        # Pure NumPy fallback: RGB histogram only
        feats = []
        for ch in range(3):
            h = np.histogram(img_rgb[:, :, ch], bins=32, range=(0, 256))[0]
            feats.append(h.astype(np.float32))
        v = np.concatenate(feats)
    else:
        hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        feats = []
        for ch, bins in enumerate([16, 16, 16]):
            ranges = (0, 180) if ch == 0 else (0, 256)
            h = cv2.calcHist([hsv], [ch], None, [bins], list(ranges))
            feats.append(h.flatten())
        v = np.concatenate(feats).astype(np.float32)

    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def _hog_feature(img_rgb: np.ndarray, size: Tuple = (64, 128)) -> np.ndarray:
    """HOG texture feature (requires OpenCV). Returns L2-normalised vector."""
    if not _CV2:
        return np.zeros(324, dtype=np.float32)
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, size)
    hog     = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    feat    = hog.compute(resized).flatten().astype(np.float32)
    norm    = np.linalg.norm(feat)
    return feat / (norm + 1e-8)


# Lazy MobileNetV2 singleton
_mobilenet_model  = None
_mobilenet_transform = None

def _mobilenet_feature(img_rgb: np.ndarray) -> np.ndarray:
    """1280-D MobileNetV2 penultimate feature (requires torch)."""
    global _mobilenet_model, _mobilenet_transform
    if _mobilenet_model is None:
        weights = MobileNet_V2_Weights.DEFAULT
        model   = mobilenet_v2(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()
        _mobilenet_model = model
        _mobilenet_transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), antialias=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    from PIL import Image
    pil  = Image.fromarray(img_rgb)
    tens = _mobilenet_transform(pil).unsqueeze(0)
    with torch.no_grad():
        feat = _mobilenet_model(tens).squeeze().numpy()
    norm = np.linalg.norm(feat)
    return (feat / (norm + 1e-8)).astype(np.float32)


def extract_feature(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract the best available feature vector from *img_rgb*.
    Priority: MobileNetV2 > HOG+Colour > Colour only.
    """
    if _TORCH:
        try:
            return _mobilenet_feature(img_rgb)
        except Exception:
            pass
    if _CV2:
        c = _colour_hist_feature(img_rgb)
        h = _hog_feature(img_rgb)
        v = np.concatenate([c, h])
    else:
        v = _colour_hist_feature(img_rgb)
    norm = np.linalg.norm(v)
    return (v / (norm + 1e-8)).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ─────────────────────────────────────────────────────────────────────────────
# Gallery entry & matcher
# ─────────────────────────────────────────────────────────────────────────────

def match_result_unknown(track_id: int = 0) -> dict:
    """Return a standard 'unverified' result."""
    return {
        "identity_status": "unverified",
        "matched_id":       f"Track-ID-{track_id}",
        "student_name":     "Unknown Person",
        "match_confidence": 0.0,
        "reid_active":      False,
        "disclaimer":       "Identity matching not performed or no match found.",
    }


class IdentityMatcher:
    """
    Loads a gallery of authorised reference images and matches
    incoming bounding-box crops using feature cosine similarity.
    """

    def __init__(self, gallery_dir: str,
                 threshold: float = SIMILARITY_THRESHOLD):
        self.gallery_dir = Path(gallery_dir)
        self.threshold   = threshold
        self.gallery: List[Dict] = []   # list of {roll_id, name, feature}
        self._feature_method = "unknown"
        self._loaded = False

    # ── Gallery loading ───────────────────────────────────────────────────────

    def load_gallery(self) -> int:
        """
        Load all images from gallery_dir.
        Reads metadata from manifest.json if present, otherwise infers
        name / ID from filename (e.g. student_01_Anoop.jpg → Anoop).

        Returns number of gallery entries loaded.
        """
        if not self.gallery_dir.is_dir():
            return 0

        manifest = self._load_manifest()
        self.gallery = []

        for img_path in sorted(self.gallery_dir.iterdir()):
            if img_path.suffix.lower() not in GALLERY_EXTENSIONS:
                continue

            img = self._load_img(img_path)
            if img is None:
                continue

            # Metadata
            meta = manifest.get(img_path.name, {})
            if not meta:
                parts   = img_path.stem.split("_")
                roll_id = f"MITS-STU-{parts[1] if len(parts) > 1 else '??'}"
                name    = parts[-1].capitalize() if parts else "Unknown"
            else:
                roll_id = meta.get("roll_id", img_path.stem)
                name    = meta.get("name",    img_path.stem)

            feat = extract_feature(img)
            self.gallery.append({
                "filename": img_path.name,
                "roll_id":  roll_id,
                "name":     name,
                "feature":  feat,
            })

        self._loaded = True
        if self.gallery:
            self._feature_method = (
                "MobileNetV2" if _TORCH else
                "HOG+ColorHist" if _CV2 else "ColorHist"
            )
        return len(self.gallery)

    def _load_manifest(self) -> dict:
        mpath = self.gallery_dir / "manifest.json"
        if mpath.exists():
            try:
                data = json.loads(mpath.read_text(encoding="utf-8"))
                return {entry["filename"]: entry for entry in data}
            except Exception:
                pass
        return {}

    @staticmethod
    def _load_img(path: Path) -> Optional[np.ndarray]:
        try:
            from PIL import Image
            return np.array(Image.open(path).convert("RGB").resize((224, 224)))
        except Exception:
            return None

    # ── Matching ──────────────────────────────────────────────────────────────

    def match(self, crop_rgb: np.ndarray,
              track_id: int = 0) -> dict:
        """
        Compare *crop_rgb* against every gallery entry.

        Returns a dict with identity_status, matched_id, student_name,
        match_confidence, and reid_active fields.
        """
        if not self._loaded:
            self.load_gallery()

        if not self.gallery:
            return match_result_unknown(track_id)

        if crop_rgb is None or crop_rgb.size == 0:
            return match_result_unknown(track_id)

        # Ensure RGB uint8
        if crop_rgb.dtype != np.uint8:
            crop_rgb = np.clip(crop_rgb, 0, 255).astype(np.uint8)

        try:
            query_feat = extract_feature(crop_rgb)
        except Exception:
            return match_result_unknown(track_id)

        # Find best match
        best_score, best_entry = 0.0, None
        for entry in self.gallery:
            score = cosine_similarity(query_feat, entry["feature"])
            if score > best_score:
                best_score, best_entry = score, entry

        if best_score >= self.threshold and best_entry is not None:
            return {
                "identity_status": "verified",
                "matched_id":       best_entry["roll_id"],
                "student_name":     best_entry["name"],
                "match_confidence": round(float(best_score), 4),
                "matched_file":     best_entry["filename"],
                "reid_active":      True,
                "feature_method":   self._feature_method,
                "disclaimer": (
                    "Match based on visual similarity only. "
                    "Requires authorised human review before any action."
                ),
            }
        else:
            result = match_result_unknown(track_id)
            result["reid_active"]      = True
            result["match_confidence"] = round(float(best_score), 4)
            result["feature_method"]   = self._feature_method
            result["closest_match"]    = (
                best_entry["name"] if best_entry else "None"
            )
            return result

    # ── Batch match from track crop box ──────────────────────────────────────

    def match_from_frame(self, frame_rgb: np.ndarray,
                         box: Tuple,
                         track_id: int = 0) -> dict:
        """
        Crop *box* (x1,y1,x2,y2) from *frame_rgb* and match.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return match_result_unknown(track_id)
        crop = frame_rgb[y1:y2, x1:x2]
        return self.match(crop, track_id)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_entries(self) -> int:
        return len(self.gallery)

    @property
    def loaded(self) -> bool:
        return self._loaded


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GAVA AI – Identity Matcher")
    parser.add_argument("--gallery", required=True, help="Path to gallery directory")
    parser.add_argument("--image",   default="",   help="Image to match")
    args = parser.parse_args()

    matcher = IdentityMatcher(args.gallery)
    n = matcher.load_gallery()
    print(f"Gallery loaded: {n} entries from {args.gallery}")
    for e in matcher.gallery:
        print(f"  {e['roll_id']}  {e['name']}  feat_shape={e['feature'].shape}")

    if args.image and Path(args.image).exists():
        img = matcher._load_img(Path(args.image))
        if img is not None:
            result = matcher.match(img)
            import json
            print("\nMatch result:")
            print(json.dumps({k: v for k, v in result.items()
                              if k != "feature"}, indent=2))
