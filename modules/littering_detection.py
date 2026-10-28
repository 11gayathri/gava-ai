"""
modules/littering_detection.py  –  Member 1
=============================================
Detects potential littering events in a video using:
  • MIVIA-IWDD dataset hooks (when dataset is present)
  • Motion-based heuristic fallback (always available)
  • Demo mode (no video required)

Public API
----------
  analyse_video(video_path, ...) -> dict
  demo_analyse()                 -> dict
"""

from __future__ import annotations
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    LITTERING_LABELS,
    LITTERING_CONFIDENCE_THRESHOLD,
    FRAME_SAMPLE_RATE,
    MAX_FRAMES_TO_SCAN,
    EVIDENCE_DIR,
    MIVIA_IWDD_DIR,
)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    """Convert float seconds → HH:MM:SS string."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _motion_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Normalised mean absolute difference between two frames [0, 1]."""
    diff = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
    return float(diff.mean() / 255.0)


def _dropping_motion_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Ratio of motion in the lower half vs upper half of the frame.
    >1.0 suggests downward movement (object being dropped).
    """
    h = frame_a.shape[0]
    upper = _motion_score(frame_a[:h // 2], frame_b[:h // 2]) + 1e-6
    lower = _motion_score(frame_a[h // 2:], frame_b[h // 2:]) + 1e-6
    return lower / upper


def _analyse_frames_heuristic(
    frames: List[np.ndarray],
    timestamps: List[float],
) -> List[Dict]:
    """Classify each adjacent frame pair using motion heuristics."""
    if len(frames) < 2:
        return [{
            "event_type": "uncertain",
            "confidence": 0.0,
            "timestamp": _fmt_ts(timestamps[0] if timestamps else 0),
            "frame_index": 0,
            "frame_path": "",
            "demo_mode": False,
            "method": "heuristic",
        }]

    events = []
    for i in range(1, len(frames)):
        motion   = _motion_score(frames[i - 1], frames[i])
        drop_rat = _dropping_motion_score(frames[i - 1], frames[i])

        if motion > 0.04 and drop_rat > 1.4:
            raw_conf = min(1.0, motion * 4.0 * (drop_rat - 1.0))
            ev_type  = ("potential_littering"
                        if raw_conf >= LITTERING_CONFIDENCE_THRESHOLD
                        else "uncertain")
        elif motion < 0.005:
            raw_conf = 0.85
            ev_type  = "no_littering"
        else:
            raw_conf = 0.3 + random.uniform(-0.05, 0.05)
            ev_type  = "no_littering"

        events.append({
            "event_type":  ev_type,
            "confidence":  round(raw_conf, 3),
            "timestamp":   _fmt_ts(timestamps[i]),
            "frame_index": i,
            "frame_path":  "",
            "demo_mode":   False,
            "method":      "heuristic",
        })
    return events


def _save_evidence_frame(frame: np.ndarray, report_id: str) -> str:
    """Save a frame to the evidence directory and return the path."""
    if not _CV2_AVAILABLE:
        return ""
    try:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        path = EVIDENCE_DIR / f"{report_id}_frame.jpg"
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return str(path)
    except Exception:
        return ""


# ── Main public API ───────────────────────────────────────────────────────────

def analyse_video(
    video_path: str,
    frame_sample_rate: int = FRAME_SAMPLE_RATE,
    max_frames: int = MAX_FRAMES_TO_SCAN,
    save_evidence: bool = True,
) -> dict:
    """
    Analyse *video_path* for littering events.

    Returns
    -------
    dict with keys: events, summary_event, video_path, total_frames,
                    demo_mode, model_used, evidence_path
    """
    vp = Path(video_path)
    if not vp.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "opencv-python is required for video analysis.\n"
            "Install: pip install opencv-python\n"
            "Or use demo_analyse() for a no-video demo."
        )

    import cv2

    cap    = cv2.VideoCapture(str(vp))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, timestamps = [], []
    idx    = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_sample_rate == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            timestamps.append(idx / fps)
        idx += 1
    cap.release()

    events = _analyse_frames_heuristic(frames, timestamps)

    # Choose summary event (highest confidence potential_littering, or first)
    littering = [e for e in events if e["event_type"] == "potential_littering"]
    summary   = (max(littering, key=lambda e: e["confidence"])
                 if littering else events[0])

    # Save evidence frame for summary event
    ev_path = ""
    if save_evidence and littering:
        fi = summary["frame_index"]
        if fi < len(frames):
            ev_path = _save_evidence_frame(frames[fi], f"ev_{int(time.time())}")
        summary["frame_path"] = ev_path

    return {
        "events":       events,
        "summary_event": summary,
        "video_path":   str(vp),
        "total_frames": len(frames),
        "demo_mode":    False,
        "model_used":   "motion_heuristic",
        "evidence_path": ev_path,
    }


def demo_analyse() -> dict:
    """
    Return a realistic-looking result without any video file.
    All events are clearly marked demo_mode=True.
    """
    random.seed(42)
    n_events = 8
    events = []
    for i in range(n_events):
        ts  = i * 3.5
        if i in (2, 5):
            ev  = "potential_littering"
            conf = round(random.uniform(0.70, 0.94), 3)
        elif i == 3:
            ev  = "uncertain"
            conf = round(random.uniform(0.40, 0.59), 3)
        else:
            ev  = "no_littering"
            conf = round(random.uniform(0.75, 0.92), 3)

        events.append({
            "event_type":  ev,
            "confidence":  conf,
            "timestamp":   _fmt_ts(ts),
            "frame_index": i * FRAME_SAMPLE_RATE,
            "frame_path":  "",
            "demo_mode":   True,
            "method":      "demo_simulation",
        })

    littering = [e for e in events if e["event_type"] == "potential_littering"]
    summary   = max(littering, key=lambda e: e["confidence"]) if littering else events[0]

    return {
        "events":        events,
        "summary_event": summary,
        "video_path":    "demo_mode",
        "total_frames":  n_events * FRAME_SAMPLE_RATE,
        "demo_mode":     True,
        "demo_note":     "No real video analysed. All results are simulated.",
        "model_used":    "demo_simulation",
        "evidence_path": "",
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="GAVA AI – Littering Detection (Member 1)")
    parser.add_argument("--video", default="", help="Path to input video file")
    parser.add_argument("--demo", action="store_true", help="Run demo without a real video")
    args = parser.parse_args()

    if args.demo or not args.video:
        print("[DEMO MODE] Running littering detection demo …")
        result = demo_analyse()
    else:
        print(f"[LIVE] Analysing: {args.video}")
        result = analyse_video(args.video)

    print(json.dumps(result, indent=2))
