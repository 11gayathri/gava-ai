"""
tests/test_dataset_utils.py
"""
import sys, tempfile
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from utils.dataset_utils import (
    load_image_rgb, resize_image, list_images, list_videos,
    discover_trashnet, check_dataset_status, check_model_status,
)

def test_load_missing():
    assert load_image_rgb("no_such_file.png") is None

def test_load_valid_png():
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.fromarray(np.random.randint(0,255,(64,64,3),dtype=np.uint8)).save(f.name)
        path = f.name
    r = load_image_rgb(path)
    assert r is not None and r.ndim == 3 and r.shape[2] == 3

def test_resize_shape():
    img = np.random.randint(0,255,(480,640,3),dtype=np.uint8)
    assert resize_image(img,(224,224)).shape == (224,224,3)

def test_list_images(tmp_path):
    (tmp_path/"a.jpg").write_bytes(b"x")
    (tmp_path/"b.txt").write_bytes(b"x")
    names = {p.name for p in list_images(tmp_path)}
    assert "a.jpg" in names and "b.txt" not in names

def test_list_images_missing():
    assert list_images(Path("no_dir_xyz")) == []

def test_list_videos(tmp_path):
    (tmp_path/"v.mp4").write_bytes(b"x")
    (tmp_path/"d.pdf").write_bytes(b"x")
    names = {p.name for p in list_videos(tmp_path)}
    assert "v.mp4" in names and "d.pdf" not in names

def test_discover_trashnet_missing():
    assert discover_trashnet(Path("no_such_dir")) == {}

def test_discover_trashnet_partial(tmp_path):
    from PIL import Image
    for cls in ["plastic","glass"]:
        d = tmp_path/cls; d.mkdir()
        Image.fromarray(np.random.randint(0,255,(64,64,3),dtype=np.uint8)).save(str(d/"t.jpg"))
    r = discover_trashnet(tmp_path)
    assert "plastic" in r and len(r["plastic"]) == 1

def test_dataset_status_keys():
    from config import DATASETS_DIR
    s = check_dataset_status(DATASETS_DIR)
    assert {"trashnet","mivia_iwdd","taco","market1501"}.issubset(s.keys())

def test_dataset_status_available_bool():
    from config import DATASETS_DIR
    for info in check_dataset_status(DATASETS_DIR).values():
        assert isinstance(info["available"], bool)

def test_model_status_keys():
    from config import MODELS_DIR
    assert {"waste_classifier","yolo"}.issubset(check_model_status(MODELS_DIR).keys())
