"""
modules/person_tracking.py  –  Member 3
=========================================
Tracks persons in video using:
  • YOLOv8n detector + centroid/IoU tracker  (when ultralytics installed)
  • HOG pedestrian detector fallback         (requires opencv-python)
  • Demo mode                                (no video required)

Identity matching is DISABLED. Track IDs are anonymous.

Public API
----------
  process_video(video_path, ...) -> dict
  CentroidTracker                – class
  demo_track()                   -> dict
  reid_placeholder(track_id)     -> dict
"""

from __future__ import annotations
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    FRAME_SAMPLE_RATE,
    MAX_FRAMES_TO_SCAN,
    TRACKER_IOU_THRESHOLD,
    TRACKER_MAX_AGE,
)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ── IoU helper ────────────────────────────────────────────────────────────────

def _iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Intersection-over-Union for two boxes (x1, y1, x2, y2).
    Returns a float in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


# ── Centroid / IoU tracker ────────────────────────────────────────────────────

class CentroidTracker:
    """
    Simple IoU-based multi-object tracker.
    Assigns persistent integer track IDs across frames.
    Identity is NEVER inferred – identity_status is always 'unverified'.
    """

    def __init__(self, iou_threshold: float = TRACKER_IOU_THRESHOLD,
                 max_age: int = TRACKER_MAX_AGE):
        self.iou_threshold  = iou_threshold
        self.max_age        = max_age
        self._tracks: Dict  = {}   # track_id -> track_state
        self._next_id: int  = 1

    def reset(self):
        self._tracks = {}
        self._next_id = 1

    def update(self, detections: List[Tuple]) -> List[Dict]:
        """
        Update tracker with a list of (x1, y1, x2, y2) bounding boxes.
        Returns a list of active track dicts.
        """
        # Age all existing tracks
        for tid in list(self._tracks):
            self._tracks[tid]["missed"] += 1
            if self._tracks[tid]["missed"] > self.max_age:
                del self._tracks[tid]

        if not detections:
            return list(self._tracks.values())

        unmatched = list(range(len(detections)))

        # Match detections to existing tracks
        for tid, trk in list(self._tracks.items()):
            best_iou, best_det = 0.0, -1
            for di in unmatched:
                score = _iou(trk["box"], detections[di])
                if score > best_iou:
                    best_iou, best_det = score, di
            if best_iou >= self.iou_threshold and best_det >= 0:
                self._tracks[tid]["box"]    = detections[best_det]
                self._tracks[tid]["missed"] = 0
                unmatched.remove(best_det)

        # Create new tracks for unmatched detections
        for di in unmatched:
            self._tracks[self._next_id] = {
                "track_id":        self._next_id,
                "box":             detections[di],
                "missed":          0,
                "identity_status": "unverified",
            }
            self._next_id += 1

        return list(self._tracks.values())


# ── ReID placeholder ──────────────────────────────────────────────────────────

def reid_placeholder(track_id: int) -> dict:
    """
    Stub for re-identification.
    Identity matching is permanently disabled unless authorised data is provided.
    """
    return {
        "track_id":        track_id,
        "reid_active":     False,
        "identity_status": "unverified",
        "reason":          "Identity matching disabled. Authorised data not supplied.",
    }


# ── Person detectors ──────────────────────────────────────────────────────────

def _detect_yolo(frame_bgr: np.ndarray) -> List[Tuple]:
    """YOLOv8n person detections – returns list of (x1,y1,x2,y2)."""
    model = _detect_yolo._model  # type: ignore[attr-defined]
    results = model(frame_bgr, classes=[0], verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
    return boxes

_detect_yolo._model = None  # lazy-loaded singleton

'''def _detect_hog(frame):
    """Fallback person detector using HOG, with contour fallback if HOG is absent."""
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]
    except (AttributeError, Exception):
        # Fallback: simple foreground/contour detection if HOGDescriptor is missing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((x, y, w, h))
        return boxes if boxes else [(50, 50, 100, 200)]  # dummy bounding box fallback'''
def _detect_hog(frame_bgr: np.ndarray) -> List[Tuple]:
    """HOG pedestrian detector (opencv fallback)."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rects, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return [(x, y, x + w, y + h) for (x, y, w, h) in rects]


def _init_yolo():
    global _detect_yolo
    if _detect_yolo._model is None:
        _detect_yolo._model = _YOLO("yolov8n.pt")


# ── Main public API ───────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    frame_sample_rate: int = FRAME_SAMPLE_RATE,
    max_frames: int = MAX_FRAMES_TO_SCAN,
) -> dict:
    """
    Process *video_path* and return tracking results.

    Returns
    -------
    dict with keys: track_summary, frame_results, video_path,
                    total_frames_processed, demo_mode, model_used
    """
    vp = Path(video_path)
    if not vp.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "opencv-python is required for video tracking.\n"
            "Install: pip install opencv-python"
        )

    import cv2

    # Choose detector
    use_yolo = False
    if _YOLO_AVAILABLE:
        try:
            _init_yolo()
            use_yolo = True
        except Exception:
            use_yolo = False

    tracker       = CentroidTracker()
    cap           = cv2.VideoCapture(str(vp))
    fps           = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_results = []
    idx           = 0

    # Track lifetime accumulator
    track_data: Dict[int, dict] = {}

    while cap.isOpened() and idx < max_frames * frame_sample_rate:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_sample_rate == 0:
            ts = idx / fps
            dets = _detect_yolo(frame) if use_yolo else _detect_hog(frame)
            tracks = tracker.update(dets)

            for trk in tracks:
                tid = trk["track_id"]
                if tid not in track_data:
                    track_data[tid] = {
                        "track_id":        tid,
                        "first_seen":      round(ts, 2),
                        "last_seen":       round(ts, 2),
                        "frame_count":     1,
                        "identity_status": "unverified",
                    }
                else:
                    track_data[tid]["last_seen"]   = round(ts, 2)
                    track_data[tid]["frame_count"] += 1

            frame_results.append({
                "frame_idx": idx,
                "timestamp": round(ts, 2),
                "n_persons": len(tracks),
                "track_ids": [t["track_id"] for t in tracks],
            })
        idx += 1

    cap.release()

    return {
        "track_summary":           list(track_data.values()),
        "frame_results":           frame_results,
        "video_path":              str(vp),
        "total_frames_processed":  idx,
        "demo_mode":               False,
        "model_used":              "yolov8n" if use_yolo else "hog_pedestrian",
    }


def demo_track() -> dict:
    """Demo tracking results without a real video."""
    random.seed(7)
    tracks = []
    for i in range(1, 4):
        first = random.uniform(0.0, 5.0)
        last  = first + random.uniform(8.0, 20.0)
        tracks.append({
            "track_id":        i,
            "first_seen":      round(first, 2),
            "last_seen":       round(last, 2),
            "frame_count":     random.randint(4, 15),
            "identity_status": "unverified",
        })

    frames = []
    for f in range(0, 60, FRAME_SAMPLE_RATE):
        ts  = f / 15.0
        active = [t["track_id"] for t in tracks
                  if t["first_seen"] <= ts <= t["last_seen"]]
        frames.append({
            "frame_idx": f,
            "timestamp": round(ts, 2),
            "n_persons": len(active),
            "track_ids": active,
        })

    return {
        "track_summary":           tracks,
        "frame_results":           frames,
        "video_path":              "demo_mode",
        "total_frames_processed":  60,
        "demo_mode":               True,
        "demo_note":               "No real video processed. Results are simulated.",
        "model_used":              "demo_simulation",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="GAVA AI – Person Tracking (Member 3)")
    parser.add_argument("--video", default="", help="Path to video file")
    parser.add_argument("--demo",  action="store_true", help="Run demo without a video")
    args = parser.parse_args()

    if args.demo or not args.video:
        print("[DEMO MODE]")
        print(json.dumps(demo_track(), indent=2))
    else:
        print(f"[LIVE] Processing: {args.video}")
        print(json.dumps(process_video(args.video), indent=2))
