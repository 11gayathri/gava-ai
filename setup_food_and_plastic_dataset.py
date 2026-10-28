"""
setup_food_and_plastic_dataset.py
==================================
Builds a specialized dataset at datasets/smart_waste_dataset/ covering:
  - food_waste (apples, peels, leftovers)
  - plastic_bag (polythene bags, wrappers)
  - plastic_rigid (bottles, containers)
  - paper_cardboard (papers, cartons)
"""

import shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).parent.resolve()
DEST_DIR = ROOT / "datasets" / "smart_waste_dataset"
CLASSES = ["food_waste", "plastic_bag", "plastic_rigid", "paper_cardboard"]

for c in CLASSES:
    (DEST_DIR / c).mkdir(parents=True, exist_ok=True)

# 1. Harvest Genuine Crops from Your Available Test Videos
print("[1/3] Extracting genuine video frames for food waste and plastic bags...")

# Point to your test video files in sample_data
video_map = {
    "food_waste": [
        ROOT / "sample_data" / "phone_litter_test6.mp4",  # Minha with apple
    ],
    "plastic_bag": [
        ROOT / "sample_data" / "phone_litter_test1.mp4",  # Plastic bag / wrapper
    ]
}

for label, vpaths in video_map.items():
    for vp in vpaths:
        if vp.exists():
            cap = cv2.VideoCapture(str(vp))
            fidx = 0
            saved = 0
            while cap.isOpened() and saved < 25:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                h, w = frame.shape[:2]
                if fidx % 4 == 0:
                    # Capture lower-half ground/flight areas
                    crop = frame[int(h * 0.40):int(h * 0.90), int(w * 0.25):int(w * 0.75)]
                    if crop.size > 0:
                        out_path = DEST_DIR / label / f"vid_{vp.stem}_{saved+1:02d}.jpg"
                        cv2.imwrite(str(out_path), cv2.resize(crop, (224, 224)))
                        saved += 1
                fidx += 1
            cap.release()
            print(f"  -> Extracted {saved} frames from {vp.name} into '{label}'")

# 2. Ingest existing TrashNet classes into rigid plastics and paper
print("\n[2/3] Ingesting TrashNet into plastic_rigid and paper_cardboard...")
trashnet = ROOT / "datasets" / "trashnet"

def copy_samples(src_cat, target_cat, limit=40):
    src = trashnet / src_cat
    if src.exists():
        files = list(src.glob("*.jpg")) + list(src.glob("*.png"))
        for i, f in enumerate(files[:limit]):
            shutil.copy2(f, DEST_DIR / target_cat / f"{src_cat}_{i:03d}.jpg")

copy_samples("plastic", "plastic_rigid", 45)
copy_samples("paper", "paper_cardboard", 30)
copy_samples("cardboard", "paper_cardboard", 30)

# 3. Synthesize Texture-Augmented Varied Plastic Bags and Organic Foods
print("\n[3/3] Generating realistic augmentations (crumpled bags & organic food)...")

def make_plastic_bag_texture(i):
    # Generates translucent, crumpled, folded bag patterns with specular crinkle lines
    canvas = np.ones((224, 224, 3), dtype=np.uint8) * np.random.randint(160, 210)
    for _ in range(8):
        pt1 = (np.random.randint(20, 200), np.random.randint(20, 200))
        pt2 = (np.random.randint(20, 200), np.random.randint(20, 200))
        cv2.line(canvas, pt1, pt2, (245, 245, 245), np.random.randint(2, 5))
    noise = np.random.normal(0, 15, (224, 224, 3)).astype(np.int16)
    return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def make_food_texture(i):
    # Generates biological textures (apple reds, citrus oranges, organic curves)
    canvas = np.ones((224, 224, 3), dtype=np.uint8) * np.random.randint(180, 220)
    center = (112 + np.random.randint(-20, 20), 112 + np.random.randint(-20, 20))
    radius = np.random.randint(55, 80)
    # Red/organic hue
    color = (np.random.randint(15, 45), np.random.randint(20, 60), np.random.randint(155, 230))
    cv2.circle(canvas, center, radius, color, -1)
    noise = np.random.normal(0, 18, (224, 224, 3)).astype(np.int16)
    return cv2.GaussianBlur(np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8), (3, 3), 0)

for i in range(35):
    cv2.imwrite(str(DEST_DIR / "plastic_bag" / f"synth_bag_{i+1:02d}.jpg"), make_plastic_bag_texture(i))
    cv2.imwrite(str(DEST_DIR / "food_waste" / f"synth_food_{i+1:02d}.jpg"), make_food_texture(i))

print("\n" + "=" * 60)
print(f"SMART WASTE DATASET READY: {DEST_DIR}")
for c in CLASSES:
    print(f"  • {c:<18}: {len(list((DEST_DIR / c).glob('*')))} images")
print("=" * 60)