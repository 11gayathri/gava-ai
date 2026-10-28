"""
tests/test_integration.py
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from utils.integration import run_demo_pipeline, _associate_event_to_track, _ts_to_seconds

@pytest.mark.parametrize("ts,exp",[("00:00:12",12.0),("00:01:05",65.0),("01:01:01",3661.0)])
def test_ts_to_seconds(ts,exp):
    assert _ts_to_seconds(ts) == pytest.approx(exp)

def test_ts_malformed():
    assert _ts_to_seconds("bad") == 0.0
    assert _ts_to_seconds("") == 0.0

def _trk(tid, first, last):
    return {"track_id":tid,"first_seen":first,"last_seen":last,"identity_status":"unverified"}

def test_associate_finds_overlap():
    tracks = [_trk(1,5.0,15.0), _trk(2,20.0,30.0)]
    assert _associate_event_to_track(10.0, tracks, 5.0)["track_id"] == 1

def test_associate_no_match():
    assert _associate_event_to_track(5.0, [_trk(1,50.0,60.0)], 3.0) is None

def test_associate_empty():
    assert _associate_event_to_track(10.0, [], 5.0) is None

@pytest.fixture(scope="module")
def demo_res():
    return run_demo_pipeline(generate_report=False)

def test_demo_is_dict(demo_res):
    assert isinstance(demo_res, dict)

def test_demo_keys(demo_res):
    assert {"littering_result","waste_result","tracking_result",
            "association","pipeline_status","errors","demo_mode","elapsed_s"}.issubset(demo_res.keys())

def test_demo_mode_true(demo_res):
    assert demo_res["demo_mode"] is True

def test_demo_no_errors(demo_res):
    assert demo_res["errors"] == []

def test_demo_status(demo_res):
    assert demo_res["pipeline_status"] == "demo_completed"

def test_demo_elapsed(demo_res):
    assert demo_res["elapsed_s"] >= 0.0

def test_demo_association_not_verified(demo_res):
    assert demo_res["association"]["association_verified"] is False

def test_demo_identity_unverified(demo_res):
    assert demo_res["association"]["identity_status"] == "unverified"

def test_demo_with_report():
    res = run_demo_pipeline(generate_report=True)
    rp  = res.get("report_path","")
    if rp:
        assert Path(rp).exists()
