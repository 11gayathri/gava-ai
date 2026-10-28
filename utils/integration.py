"""
utils/integration.py
======================
Orchestrates all four GAVA AI modules into a single pipeline:
  1. Person Tracking   (Member 3)
  2. Littering Detection (Member 1)
  3. Waste Classification (Member 2)
  4. Temporal Association (event ↔ track)
  5. Fine Report Generation (Member 4)

Public API
----------
  run_pipeline(video_path, image_path, ...) -> dict
  run_demo_pipeline(generate_report)        -> dict
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from modules.littering_detection import analyse_video, demo_analyse
from modules.waste_classification import WasteClassifier, predict_image
from modules.person_tracking      import process_video, demo_track
from modules.fine_report          import IncidentReportGenerator, build_sample_incident
from config import INSTITUTION


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_to_seconds(ts: str) -> float:
    """Convert 'HH:MM:SS' → float seconds. Returns 0.0 on error."""
    try:
        parts = ts.split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return float(h * 3600 + m * 60 + s)
    except Exception:
        return 0.0


def _associate_event_to_track(
    event_ts: float,
    tracks: List[Dict],
    time_window: float = 5.0,
) -> Optional[Dict]:
    """
    Return the track whose active window [first_seen, last_seen]
    contains or is closest to *event_ts* within *time_window* seconds.
    Returns None if no track qualifies.
    """
    best, best_dist = None, float("inf")
    for trk in tracks:
        first = float(trk.get("first_seen", 0))
        last  = float(trk.get("last_seen",  0))
        if first <= event_ts <= last:
            dist = 0.0
        else:
            dist = min(abs(event_ts - first), abs(event_ts - last))
        if dist <= time_window and dist < best_dist:
            best      = trk
            best_dist = dist
    return best


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    image_path: Optional[str] = None,
    generate_report: bool = True,
    report_location: str = "Unknown",
) -> dict:
    """
    Run the complete GAVA AI pipeline on a real video file.

    Parameters
    ----------
    video_path      : path to the input video
    image_path      : optional path to a crop for waste classification
    generate_report : whether to produce a PDF report
    report_location : camera / location label for the report

    Returns
    -------
    dict with pipeline results and optional report path
    """
    errors   = []
    t_start  = time.time()
    result   = {
        "demo_mode":       False,
        "pipeline_status": "started",
        "errors":          errors,
        "elapsed_s":       0.0,
    }

    # 1. Person tracking
    try:
        tracking_result = process_video(video_path)
    except Exception as exc:
        errors.append(f"tracking: {exc}")
        tracking_result = {"track_summary": [], "demo_mode": False, "model_used": "error"}
    result["tracking_result"] = tracking_result

    # 2. Littering detection
    try:
        littering_result = analyse_video(video_path)
    except Exception as exc:
        errors.append(f"littering: {exc}")
        littering_result = demo_analyse()
        littering_result["note"] = "Fell back to demo due to error"
    result["littering_result"] = littering_result

    # 3. Waste classification
    waste_result = {"predicted_class": "unknown", "confidence": 0.0,
                    "waste_category": "unknown", "demo_mode": True}
    if image_path and Path(image_path).exists():
        try:
            from utils.dataset_utils import load_image_rgb
            img = load_image_rgb(image_path)
            if img is not None:
                waste_result = predict_image(img)
        except Exception as exc:
            errors.append(f"classification: {exc}")
    else:
        clf = WasteClassifier()
        waste_result = clf.demo_predict()
    result["waste_result"] = waste_result

    # 4. Association: link littering event to a track
    summary_ev  = littering_result.get("summary_event", {})
    event_ts    = _ts_to_seconds(summary_ev.get("timestamp", "00:00:00"))
    matched_trk = _associate_event_to_track(event_ts,
                                            tracking_result.get("track_summary", []))

    association = {
        "track_id":             matched_trk["track_id"] if matched_trk else None,
        "identity_status":      "unverified",
        "association_verified": False,
        "note":                 "Track IDs are anonymous. Identity matching disabled.",
    }
    result["association"] = association

    # 5. Report generation
    if generate_report:
        incident = {
            "littering_event": littering_result,
            "waste_result":    waste_result,
            "track_info":      matched_trk or {},
            "location":        report_location,
            "notes":           "Generated by GAVA AI automated pipeline.",
            "demo_mode":       False,
        }
        try:
            gen  = IncidentReportGenerator()
            rp   = gen.generate(incident)
            result["report_path"] = rp
        except Exception as exc:
            errors.append(f"report: {exc}")
            result["report_path"] = ""

    result["pipeline_status"] = "completed" if not errors else "completed_with_errors"
    result["elapsed_s"]       = round(time.time() - t_start, 2)
    return result


# ── Demo pipeline ─────────────────────────────────────────────────────────────

def run_demo_pipeline(generate_report: bool = True) -> dict:
    """
    Run the complete pipeline in demo mode (no video / image files required).
    All results are clearly marked demo_mode=True.
    """
    t_start  = time.time()
    errors   = []

    # 1. Demo tracking
    tracking_result  = demo_track()

    # 2. Demo littering
    littering_result = demo_analyse()

    # 3. Demo classification
    clf          = WasteClassifier()
    waste_result = clf.demo_predict()

    # 4. Association
    summary_ev  = littering_result.get("summary_event", {})
    event_ts    = _ts_to_seconds(summary_ev.get("timestamp", "00:00:00"))
    matched_trk = _associate_event_to_track(event_ts,
                                            tracking_result.get("track_summary", []))

    association = {
        "track_id":             matched_trk["track_id"] if matched_trk else None,
        "identity_status":      "unverified",
        "association_verified": False,
        "note":                 "Demo mode. Track IDs are anonymous.",
    }

    # 5. Report
    report_path = ""
    if generate_report:
        incident = build_sample_incident(demo=True)
        incident.update({
            "littering_event": littering_result,
            "waste_result":    waste_result,
            "track_info":      matched_trk or {},
        })
        try:
            gen         = IncidentReportGenerator()
            report_path = gen.generate(incident)
        except Exception as exc:
            errors.append(f"report: {exc}")

    return {
        "littering_result":  littering_result,
        "waste_result":      waste_result,
        "tracking_result":   tracking_result,
        "association":       association,
        "report_path":       report_path,
        "pipeline_status":   "demo_completed",
        "errors":            errors,
        "demo_mode":         True,
        "elapsed_s":         round(time.time() - t_start, 2),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="GAVA AI – Full Integration Pipeline")
    parser.add_argument("--video",    default="", help="Input video path")
    parser.add_argument("--image",    default="", help="Waste image path (optional)")
    parser.add_argument("--demo",     action="store_true", help="Run demo pipeline")
    parser.add_argument("--no_report", action="store_true", help="Skip report generation")
    args = parser.parse_args()

    if args.demo or not args.video:
        print("[DEMO PIPELINE]")
        result = run_demo_pipeline(generate_report=not args.no_report)
    else:
        print(f"[LIVE PIPELINE] video={args.video}")
        result = run_pipeline(args.video, args.image,
                              generate_report=not args.no_report)

    # Print summary (skip large frame_results lists)
    summary = {k: v for k, v in result.items()
               if k not in ("littering_result", "tracking_result")}
    print(json.dumps(summary, indent=2, default=str))
    if result.get("report_path"):
        print(f"\nReport: {result['report_path']}")
