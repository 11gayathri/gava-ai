"""
tests/test_waste_classification.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import TRASHNET_CLASSES, WASTE_CATEGORY_MAP
from modules.waste_classification import WasteClassifier, extract_features_histogram

@pytest.fixture
def img():
    return np.random.randint(0,255,(224,224,3),dtype=np.uint8)

def test_all_classes_mapped():
    for cls in TRASHNET_CLASSES:
        assert cls in WASTE_CATEGORY_MAP

def test_map_values_valid():
    valid = {"biodegradable","non_biodegradable","unknown"}
    for v in WASTE_CATEGORY_MAP.values():
        assert v in valid

def test_histogram_shape(img):
    feats = extract_features_histogram(img[np.newaxis])
    assert feats.ndim == 2 and feats.shape[0] == 1 and feats.shape[1] > 0

def test_histogram_finite(img):
    feats = extract_features_histogram(img[np.newaxis])
    assert np.all(np.isfinite(feats))

def test_histogram_empty_batch():
    feats = extract_features_histogram(np.zeros((0,224,224,3),dtype=np.uint8))
    assert feats.shape[0] == 0

def test_demo_predict_keys(img):
    r = WasteClassifier.demo_predict(img)
    assert {"predicted_class","waste_category","confidence","all_probabilities","demo_mode"}.issubset(r.keys())

def test_demo_predict_class_valid(img):
    assert WasteClassifier.demo_predict(img)["predicted_class"] in TRASHNET_CLASSES

def test_demo_predict_confidence_range(img):
    assert 0.0 <= WasteClassifier.demo_predict(img)["confidence"] <= 1.0

def test_demo_predict_marked_demo(img):
    assert WasteClassifier.demo_predict(img)["demo_mode"] is True

def test_demo_predict_none():
    r = WasteClassifier.demo_predict(None)
    assert "predicted_class" in r

def test_untrained_falls_back_to_demo(img):
    clf = WasteClassifier()
    clf.trained  = False
    clf.pipeline = None
    assert clf.predict_image(img).get("demo_mode") is True
