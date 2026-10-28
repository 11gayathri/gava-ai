"""
tests/test_person_tracking.py
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from modules.person_tracking import CentroidTracker, _iou, demo_track, reid_placeholder

def test_iou_identical():
    b = (10,10,100,100)
    assert _iou(b,b) == pytest.approx(1.0)

def test_iou_no_overlap():
    assert _iou((0,0,10,10),(20,20,30,30)) == pytest.approx(0.0)

def test_iou_partial():
    v = _iou((0,0,20,20),(10,10,30,30))
    assert 0.0 < v < 1.0

def test_iou_symmetric():
    a,b = (5,5,25,25),(15,5,35,25)
    assert _iou(a,b) == pytest.approx(_iou(b,a))

def test_tracker_creates_tracks():
    t = CentroidTracker()
    assert len(t.update([(10,10,50,80),(200,10,240,80)])) == 2

def test_tracker_unique_ids():
    t = CentroidTracker()
    ids = [tr["track_id"] for tr in t.update([(10,10,50,80),(200,10,240,80)])]
    assert len(ids) == len(set(ids))

def test_tracker_identity_unverified():
    t = CentroidTracker()
    assert all(tr["identity_status"]=="unverified" for tr in t.update([(10,10,50,80)]))

def test_tracker_persists_id():
    t = CentroidTracker()
    t1 = t.update([(10,10,60,90)])
    t2 = t.update([(12,12,62,92)])
    assert t1[0]["track_id"] == t2[0]["track_id"]

def test_tracker_drops_stale():
    t = CentroidTracker(max_age=1)
    t.update([(10,10,50,80)])
    for _ in range(3):
        r = t.update([])
    assert r == []

def test_tracker_empty():
    assert CentroidTracker().update([]) == []

def test_tracker_reset():
    t = CentroidTracker()
    t.update([(10,10,50,80)])
    t.reset()
    assert t.update([(10,10,50,80)])[0]["track_id"] == 1

def test_demo_track_keys():
    assert {"track_summary","frame_results","demo_mode","video_path"}.issubset(demo_track().keys())

def test_demo_track_mode_true():
    assert demo_track()["demo_mode"] is True

def test_demo_track_has_tracks():
    assert len(demo_track()["track_summary"]) > 0

def test_demo_track_identity_unverified():
    for trk in demo_track()["track_summary"]:
        assert trk["identity_status"] == "unverified"

def test_reid_not_active():
    assert reid_placeholder(5)["reid_active"] is False

def test_reid_track_id():
    assert reid_placeholder(7)["track_id"] == 7

def test_process_video_missing():
    from modules.person_tracking import process_video
    with pytest.raises(FileNotFoundError):
        process_video("no_such_file.mp4")
