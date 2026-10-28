"""
tests/test_littering_detection.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from modules.littering_detection import (
    demo_analyse, _motion_score, _dropping_motion_score,
    _fmt_ts, _analyse_frames_heuristic, LITTERING_LABELS,
)

@pytest.mark.parametrize("s,exp", [(0,"00:00:00"),(65,"00:01:05"),(3661,"01:01:01")])
def test_fmt_ts(s, exp):
    assert _fmt_ts(s) == exp

def test_motion_score_identical():
    f = np.zeros((240,320,3), dtype=np.uint8)
    assert _motion_score(f, f) == pytest.approx(0.0)

def test_motion_score_max():
    b = np.zeros((240,320,3), dtype=np.uint8)
    w = np.full((240,320,3), 255, dtype=np.uint8)
    assert _motion_score(b, w) == pytest.approx(1.0, abs=0.01)

def test_motion_score_range():
    a = np.random.randint(0,255,(240,320,3),dtype=np.uint8)
    b = np.random.randint(0,255,(240,320,3),dtype=np.uint8)
    assert 0.0 <= _motion_score(a,b) <= 1.0

def test_dropping_lower_half():
    a = np.zeros((240,320,3),dtype=np.uint8)
    b = np.zeros((240,320,3),dtype=np.uint8)
    b[120:] = 200
    assert _dropping_motion_score(a,b) > 1.0

def test_heuristic_returns_list():
    frames = [np.random.randint(0,255,(240,320,3),dtype=np.uint8) for _ in range(5)]
    result = _analyse_frames_heuristic(frames, [float(i) for i in range(5)])
    assert isinstance(result, list) and len(result) > 0

def test_heuristic_labels_valid():
    frames = [np.random.randint(0,255,(240,320,3),dtype=np.uint8) for _ in range(6)]
    for ev in _analyse_frames_heuristic(frames, list(range(6))):
        assert ev["event_type"] in LITTERING_LABELS

def test_demo_keys():
    r = demo_analyse()
    assert {"events","summary_event","demo_mode","video_path"}.issubset(r.keys())

def test_demo_mode_true():
    r = demo_analyse()
    assert r["demo_mode"] is True
    assert all(e["demo_mode"] for e in r["events"])

def test_demo_labels():
    for ev in demo_analyse()["events"]:
        assert ev["event_type"] in LITTERING_LABELS

def test_analyse_video_missing():
    from modules.littering_detection import analyse_video
    with pytest.raises(FileNotFoundError):
        analyse_video("no_such_file_xyz.mp4")
