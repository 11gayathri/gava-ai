"""
enroll_students.py – Biometric Database Generator for GAVA AI
==============================================================
1. Scans sample_data/student_db for enrolled student portraits.
2. Applies MTCNN face detection with 5-point landmark alignment.
3. Extracts 512-D L2-normalized embeddings using Inception-ResNet-v1 (VGGFace2).
4. Stores multiple reference view vectors per student + calculates average centroid.
5. Saves the final compiled database to models/student_face_embeddings.pkl.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pickle
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

ROOT = Path(__file__).parent.resolve()
DB_DIR = ROOT / "sample_data" / "student_db"
OUTPUT_PKL = ROOT / "models" / "student_face_embeddings.pkl"
OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_embeddings_for_image(
    image_path: Path, 
    mtcnn: MTCNN, 
    resnet: InceptionResnetV1, 
    device: torch.device
) -> List[np.ndarray]:
    """Loads image, detects faces, aligns, and outputs normalized 512-D vectors."""
    try:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return []
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Detect face & apply alignment
        face_tensor = mtcnn(pil_img)
        if face_tensor is None:
            # Fallback if strict detector misses (try with lower margin)
            return []

        # If a single face tensor is returned
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)

        face_tensor = face_tensor.to(device)
        with torch.no_grad():
            embs = resnet(face_tensor)
            # L2 normalization for accurate cosine distance
            embs = embs / torch.norm(embs, p=2, dim=1, keepdim=True)
            return [e.cpu().numpy().flatten() for e in embs]

    except Exception as e:
        print(f"    [Warning] Failed to extract from {image_path.name}: {e}")
        return []


def enroll_all_students():
    print("\n" + "=" * 65)
    print("      GAVA AI: MULTI-IMAGE BIOMETRIC ENROLLMENT ENGINE")
    print("=" * 65)

    if not DB_DIR.exists():
        print(f"[Error] Directory not found: {DB_DIR}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  • Compute Device    : {device}")
    print(f"  • Gallery Directory : {DB_DIR}")
    print(f"  • Target File       : {OUTPUT_PKL}\n")

    # MTCNN initialized with strict parameters to filter blurred/cut faces
    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        keep_all=False,
        min_face_size=30,
        select_largest=True,
        post_process=True,
        device=device
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    enrolled_db: Dict[str, Dict[str, Any]] = {}
    
    # Check whether subfolders or flat files are used
    subdirs = [p for p in DB_DIR.iterdir() if p.is_dir()]
    
    if subdirs:
        # ── Mode A: Subdirectories per student ──────────────────────────────
        for sdir in sorted(subdirs):
            student_name = sdir.name
            img_files = [f for f in sdir.iterdir() if f.suffix.lower() in VALID_EXTS]
            if not img_files:
                continue

            print(f"  [Processing] Student: {student_name:<25} ({len(img_files)} photos)")
            all_embs: List[np.ndarray] = []
            front_photo = str(img_files[0])

            for img_path in img_files:
                # Prefer files named 'front' for Figure B display in the report
                if "front" in img_path.stem.lower():
                    front_photo = str(img_path)

                embs = extract_embeddings_for_image(img_path, mtcnn, resnet, device)
                if embs:
                    all_embs.extend(embs)
                    print(f"    -> Extracted: {img_path.name}")
                else:
                    print(f"    -> [No Face Detected]: {img_path.name}")

            if all_embs:
                centroid = np.mean(all_embs, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                enrolled_db[student_name] = {
                    "embeddings": all_embs,
                    "centroid": centroid,
                    "front_photo": front_photo,
                    "image_count": len(all_embs)
                }
    else:
        # ── Mode B: Flat files with prefixes (e.g. Achsa_1.jpg, Achsa_specs.jpg) ──
        all_files = [f for f in DB_DIR.iterdir() if f.suffix.lower() in VALID_EXTS]
        student_groups: Dict[str, List[Path]] = {}
        for f in all_files:
            # Splits prefix before first '_' as name, or uses entire stem
            name = f.stem.split("_")[0] if "_" in f.stem else f.stem
            student_groups.setdefault(name, []).append(f)

        for student_name, img_files in sorted(student_groups.items()):
            print(f"  [Processing] Student: {student_name:<25} ({len(img_files)} photos)")
            all_embs: List[np.ndarray] = []
            front_photo = str(img_files[0])

            for img_path in img_files:
                if "front" in img_path.stem.lower():
                    front_photo = str(img_path)

                embs = extract_embeddings_for_image(img_path, mtcnn, resnet, device)
                if embs:
                    all_embs.extend(embs)
                    print(f"    -> Extracted: {img_path.name}")
                else:
                    print(f"    -> [No Face Detected]: {img_path.name}")

            if all_embs:
                centroid = np.mean(all_embs, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                enrolled_db[student_name] = {
                    "embeddings": all_embs,
                    "centroid": centroid,
                    "front_photo": front_photo,
                    "image_count": len(all_embs)
                }

    # Save to pickle
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(enrolled_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n" + "=" * 65)
    print("ENROLLMENT COMPLETE SUMMARY:")
    print(f"  • Total Enrolled Students : {len(enrolled_db)}")
    for name, data in enrolled_db.items():
        print(f"    - {name:<25}: {data['image_count']} valid face embeddings registered")
    print(f"  • Database File Saved to  : {OUTPUT_PKL}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    enroll_all_students()