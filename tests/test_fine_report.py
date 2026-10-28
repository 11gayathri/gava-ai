"""
tests/test_fine_report.py
"""
import sys
import tempfile
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from modules.fine_report import IncidentReportGenerator, build_sample_incident, _RL_AVAILABLE

@pytest.fixture
def demo_inc():
    return build_sample_incident(demo=True)

@pytest.fixture
def gen():
    return IncidentReportGenerator()

def test_sample_incident_keys(demo_inc):
    assert {"littering_event","waste_result","track_info","location","demo_mode"}.issubset(demo_inc.keys())

def test_sample_demo_flag(demo_inc):
    assert demo_inc["demo_mode"] is True

def test_sample_waste_class_valid(demo_inc):
    from config import TRASHNET_CLASSES
    assert demo_inc["waste_result"]["predicted_class"] in TRASHNET_CLASSES

def test_report_id_starts_gava():
    assert IncidentReportGenerator._make_report_id().startswith("GAVA-")

def test_generate_returns_str(gen, demo_inc):
    with tempfile.TemporaryDirectory() as tmp:
        path = gen.generate(demo_inc, output_path=str(Path(tmp)/"r.pdf"))
        assert isinstance(path, str) and len(path) > 0

def test_generated_file_exists(gen, demo_inc):
    with tempfile.TemporaryDirectory() as tmp:
        path = gen.generate(demo_inc, output_path=str(Path(tmp)/"r.pdf"))
        assert Path(path).exists()

@pytest.mark.skipif(not _RL_AVAILABLE, reason="reportlab not installed")
def test_pdf_magic_bytes(gen, demo_inc):
    with tempfile.TemporaryDirectory() as tmp:
        path = gen.generate(demo_inc, output_path=str(Path(tmp)/"r.pdf"))
        assert open(path,"rb").read(4) == b"%PDF"

def test_generate_auto_path(gen, demo_inc):
    path = gen.generate(demo_inc)
    assert path and Path(path).exists()

def test_generate_empty_incident(gen):
    assert isinstance(gen.generate({}), str)

def test_fallback_contains_mits(gen, demo_inc):
    with tempfile.TemporaryDirectory() as tmp:
        path = gen._fallback_text_report(demo_inc, Path(tmp)/"r.txt")
        text = Path(path).read_text(encoding="utf-8")
        assert "MITS" in text or "Muthoot" in text
