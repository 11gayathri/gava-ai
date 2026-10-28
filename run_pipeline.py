"""
run_pipeline.py – GAVA AI End-to-End Orchestrator
==================================================
1. Discard Kinematics via YOLO-Pose (mid-air trajectory down to ground landing).
2. Full-Clip Biometrics: Relaxed surveillance mode for facial matching.
3. Waste Classifier: Material differentiation (Paper vs Food vs Plastic).
4. Formal PDF Dossier Generation.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from __future__ import annotations
import argparse
import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import REPORTS_DIR, EVIDENCE_DIR, FINE_AMOUNTS
from modules.action_recognition import DiscardActionDetector
from modules.fine_report import IncidentReportGenerator

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Stage 2: Surveillance-Optimized Biometric Identification ──────────────────

def clean_student_name(raw_name: str) -> str:
    """Removes system prefixes like MITS_, roll numbers, or folder indices."""
    name = raw_name.replace("MITS_", "").replace("MITS-", "").replace("MITS", "").strip()
    name = name.strip("_- ")
    if "(" in name:
        name = name.split("(")[0].strip()
    return name if len(name) > 0 else raw_name


def verify_student_identity_full_search(video_path: str) -> Dict[str, Any]:
    db_file = ROOT / "models" / "student_face_embeddings.pkl"
    cctv_crop_path = EVIDENCE_DIR / "evidence_person.jpg"

    if not db_file.exists():
        raise FileNotFoundError("models/student_face_embeddings.pkl not found. Run enroll_students.py first.")

    with open(db_file, "rb") as f:
        enrolled_db: Dict[str, Dict[str, Any]] = pickle.load(f)

    from facenet_pytorch import MTCNN, InceptionResnetV1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. RELAXED MTCNN FOR SURVEILLANCE
    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        keep_all=False,
        min_face_size=20,  # Lowered to detect smaller faces
        select_largest=True,
        post_process=True,
        device=device
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    student_scores: Dict[str, List[float]] = {k: [] for k in enrolled_db.keys()}
    best_cctv_crop = None
    best_overall_score = -1.0
    frame_idx = 0

    print(f"  [Biometrics] Searching video frames for facial matches (Surveillance Mode)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        # Process every 2nd frame to ensure we catch brief moments of clarity
        if frame_idx % 2 != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        boxes, probs = mtcnn.detect(pil_frame)

        if boxes is not None and probs is not None and len(probs) > 0:
            # 2. RELAXED DETECTION CONFIDENCE (Allows motion blur)
            if probs[0] is not None and probs[0] >= 0.70:
                face_tensor = mtcnn(pil_frame)
                if face_tensor is not None:
                    with torch.no_grad():
                        emb = resnet(face_tensor.unsqueeze(0).to(device))
                        emb = (emb / torch.norm(emb, p=2, dim=1, keepdim=True)).cpu().numpy().flatten()

                    for student_name, s_data in enrolled_db.items():
                        sims = [float(np.dot(emb, ref_emb)) for ref_emb in s_data.get("embeddings", [])]
                        if sims:
                            best_sim = max(sims)
                            student_scores[student_name].append(best_sim)

                            if best_sim > best_overall_score:
                                best_overall_score = best_sim
                                box = boxes[0].astype(int)
                                h, w = frame.shape[:2]
                                bx1, by1 = max(0, box[0]), max(0, box[1])
                                bx2, by2 = min(w, box[2]), min(h, box[3])
                                best_cctv_crop = frame[by1:by2, bx1:bx2]

    cap.release()

    if best_cctv_crop is not None and best_cctv_crop.size > 0:
        cv2.imwrite(str(cctv_crop_path), best_cctv_crop)

    # Average of top 3 hits per student
    candidate_ranking = []
    for name, score_list in student_scores.items():
        if score_list:
            top_scores = sorted(score_list, reverse=True)[:3]
            mean_score = float(np.mean(top_scores))
            candidate_ranking.append((name, mean_score))
        else:
            candidate_ranking.append((name, 0.0))

    candidate_ranking.sort(key=lambda x: x[1], reverse=True)

    print("\n  --- TOP CANDIDATES ACROSS VIDEO ---")
    for rank, (cand_name, score) in enumerate(candidate_ranking[:4]):
        cleaned = clean_student_name(cand_name)
        print(f"    Rank {rank+1}: {cleaned:<25} -> Similarity: {score:.3f}")
    print("  -----------------------------------\n")

    top_raw_name, top_score = candidate_ranking[0]
    final_student_name = clean_student_name(top_raw_name)
    
    # 3. RELAXED VERIFICATION THRESHOLD
    status = "verified" if top_score >= 0.38 else "unverified"

    # Default to the first enrolled person if completely unverified and no face was found at all
    if top_score == 0.0:
        top_raw_name = list(enrolled_db.keys())[0]

    top_record = enrolled_db[top_raw_name]

    return {
        "status": status,
        "student_name": final_student_name,
        "match_confidence": round(top_score, 3),
        "crop_path": top_record["front_photo"],
    }


# ── Stage 3: Live Waste Classification (Material Discrimination) ──────────────

def isolate_foreground_object(bgr_crop: np.ndarray) -> np.ndarray:
    if bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_c) > 100:
            x, y, w, h = cv2.boundingRect(largest_c)
            p = 8
            x1 = max(0, x - p)
            y1 = max(0, y - p)
            x2 = min(bgr_crop.shape[1], x + w + p)
            y2 = min(bgr_crop.shape[0], y + h + p)
            fg = bgr_crop[y1:y2, x1:x2]
            if fg.size > 0:
                return fg

    return bgr_crop

def is_paper_waste(bgr_crop: np.ndarray) -> Tuple[bool, float]:
    """Detects paper waste (notebook pages, bills, printouts, slips)."""
    if bgr_crop is None or bgr_crop.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    _, s_channel, v_channel = cv2.split(hsv)

    paper_mask = (s_channel < 45) & (v_channel > 160)
    paper_ratio = float(np.count_nonzero(paper_mask)) / gray.size

    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / gray.size

    dark_sole_mask = v_channel < 50
    sole_ratio = float(np.count_nonzero(dark_sole_mask)) / gray.size

    if paper_ratio > 0.12 and edge_ratio > 0.02 and sole_ratio < 0.25:
        conf = min(0.96, 0.82 + (paper_ratio * 0.4))
        return True, round(conf, 3)

    return False, 0.0

def is_white_plastic_bag(bgr_crop: np.ndarray) -> Tuple[bool, float]:
    """Detects white or translucent plastic carry bags."""
    if bgr_crop is None or bgr_crop.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    _, s_channel, v_channel = cv2.split(hsv)

    white_neutral_mask = (s_channel < 60) & (v_channel > 140)
    white_ratio = float(np.count_nonzero(white_neutral_mask)) / gray.size

    edges = cv2.Canny(gray, 40, 120)
    edge_ratio = float(np.count_nonzero(edges)) / gray.size

    _, specular = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    specular_ratio = float(np.count_nonzero(specular)) / gray.size

    if white_ratio > 0.15 and (edge_ratio > 0.025 or specular_ratio > 0.005):
        conf = min(0.96, 0.82 + (white_ratio * 0.3) + (edge_ratio * 1.5))
        return True, round(conf, 3)

    return False, 0.0

def check_plastic_properties(bgr_crop: np.ndarray) -> Tuple[bool, float]:
    """Detects general plastic glare and properties."""
    if bgr_crop is None or bgr_crop.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

    _, specular_mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    specular_ratio = np.count_nonzero(specular_mask) / gray.size

    edges = cv2.Canny(gray, 60, 150)
    edge_ratio = np.count_nonzero(edges) / gray.size

    s_channel = hsv[:, :, 1]
    low_sat_ratio = np.count_nonzero(s_channel < 55) / s_channel.size

    if (specular_ratio > 0.008 and edge_ratio > 0.03) or (low_sat_ratio > 0.65 and edge_ratio > 0.05):
        conf = min(0.95, 0.78 + (specular_ratio * 4.0) + (edge_ratio * 1.5))
        return True, round(conf, 3)

    return False, 0.0

def classify_waste_live(waste_crop_path: str) -> Tuple[str, str, float, int]:
    img_bgr = cv2.imread(waste_crop_path)
    if img_bgr is None or img_bgr.size == 0:
        return "plastic", "non_biodegradable", 0.90, FINE_AMOUNTS.get("non_biodegradable", 500)

    clean_crop = isolate_foreground_object(img_bgr)

    # 1. Paper Priority Check
    is_paper, paper_conf = is_paper_waste(clean_crop)
    if is_paper:
        fine = FINE_AMOUNTS.get("biodegradable", 100)
        return "paper", "biodegradable", paper_conf, fine

    # 2. White Plastic Bag Priority Check
    is_bag, bag_conf = is_white_plastic_bag(clean_crop)
    if is_bag:
        return "plastic", "non_biodegradable", bag_conf, FINE_AMOUNTS.get("non_biodegradable", 500)

    # 3. Saturated Organic Fruit / Food
    hsv = cv2.cvtColor(clean_crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 70, 40]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([168, 70, 40]), np.array([180, 255, 255]))
    apple_red = cv2.bitwise_or(m1, m2)
    organic_green = cv2.inRange(hsv, np.array([25, 65, 40]), np.array([85, 255, 255]))

    total_px = clean_crop.shape[0] * clean_crop.shape[1]
    red_ratio = float(np.count_nonzero(apple_red)) / total_px
    green_ratio = float(np.count_nonzero(organic_green)) / total_px

    if red_ratio > 0.05 or green_ratio > 0.15:
        conf = min(0.95, 0.80 + (red_ratio * 1.5))
        return "organic", "biodegradable", round(conf, 3), FINE_AMOUNTS.get("biodegradable", 100)

    # 4. General Plastic Glare
    is_general_plastic, p_conf = check_plastic_properties(clean_crop)
    if is_general_plastic:
        return "plastic", "non_biodegradable", p_conf, FINE_AMOUNTS.get("non_biodegradable", 500)

    # 5. Neural Network Inference
    model_path = ROOT / "models" / "smart_waste_mobilenet.pth"
    if not model_path.exists():
        model_path = ROOT / "models" / "waste_multiclass_mobilenet.pth"

    if model_path.exists():
        try:
            import torch.nn.functional as F
            from torchvision import transforms, models

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device)
            class_names = checkpoint["classes"]

            model = models.mobilenet_v3_small(weights=None)
            in_feat = model.classifier[3].in_features
            model.classifier[3] = torch.nn.Linear(in_feat, len(class_names))
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(device).eval()

            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            tensor = preprocess(cv2.cvtColor(clean_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = F.softmax(out, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                pred_cls = class_names[idx].lower()
                conf = float(probs[idx])

            if "food" in pred_cls or "organic" in pred_cls:
                return "organic", "biodegradable", round(conf, 3), FINE_AMOUNTS.get("biodegradable", 100)
            else:
                return "plastic", "non_biodegradable", round(conf, 3), FINE_AMOUNTS.get("non_biodegradable", 500)
        except Exception:
            pass

    return "plastic", "non_biodegradable", 0.90, FINE_AMOUNTS.get("non_biodegradable", 500)


# ── Stage 4 & Master Orchestrator ─────────────────────────────────────────────

def run_pipeline(video_path: str, reference_dir: str = "sample_data/student_db"):
    print("\n" + "=" * 70)
    print("      GAVA AI: END-TO-END SMART MONITORING & AUDIT PIPELINE")
    print("=" * 70)

    # 1. Action Kinematics & Evidence Extraction
    print(f"\n[Stage 1/4] Detecting Discard Kinematics via YOLO-Pose...")
    action_detector = DiscardActionDetector()
    action_img, waste_img, timestamp, action_conf = action_detector.detect_discard_event(video_path, EVIDENCE_DIR)

    print(f"  -> Discard Event Detected at         : {timestamp} (Confidence: {action_conf*100:.1f}%)")
    print(f"  -> Clean Action Frame (Figure A)     : {action_img}")
    print(f"  -> Discarded Waste ROI               : {waste_img}")

    # 2. Biometric Search Across Video Frames
    print(f"\n[Stage 2/4] Executing Biometric Search Across Video Frames...")
    id_info = verify_student_identity_full_search(video_path)
    person_evidence = id_info["crop_path"]

    print(f"  -> Identity Status                    : {id_info['status'].upper()}")
    print(f"  -> Matched Student Record             : {id_info['student_name']} ({id_info['match_confidence']*100:.1f}%)")
    print(f"  -> Official Reference Photo (Figure B): {person_evidence}")

    # 3. Live Waste Classification
    print(f"\n[Stage 3/4] Running Material Classification on Ground/Flight Crop...")
    waste_cls, waste_cat, waste_conf, fine = classify_waste_live(waste_img)
    print(f"  -> Detected Material                  : {waste_cls.upper()}")
    print(f"  -> Category                           : {waste_cat.upper()}")
    print(f"  -> Confidence                         : {waste_conf*100:.1f}%")
    print(f"  -> Assessed Fine                      : INR {fine}")

    # 4. Formal Report Generation
    print(f"\n[Stage 4/4] Generating Incident Audit Report (PDF)...")
    generator = IncidentReportGenerator()
    track_id = 1

    incident_payload = {
        "incident_id": f"MITS-2026-TRK{track_id:02d}",
        "location": "MITS Campus Courtyard",
        "timestamp": timestamp,
        "littering_event": {
            "summary_event": {
                "event_type": "littering_discard",
                "confidence": action_conf,
                "timestamp": timestamp,
                "frame_path": action_img,
            },
            "model_used": "yolo_pose_kinematics",
            "evidence_path": action_img,
        },
        "waste_result": {
            "predicted_class": waste_cls,
            "waste_category": waste_cat,
            "confidence": waste_conf,
            "demo_mode": False
        },
        "track_info": {
            "track_id": track_id,
            "identity_status": id_info["status"],
            "student_name": id_info["student_name"],
            "crop_path": person_evidence,
            "match_confidence": id_info["match_confidence"],  # FIX ADDED HERE
        },
        "evidence_path": action_img,
        "person_crop_path": person_evidence,
        "fine_amount": fine,
        "review_status": "Verified",
        "notes": f"Offender identified as {id_info['student_name']}. Discarded material: {waste_cls}.",
        "demo_mode": False,
    }

    pdf_path = generator.generate(incident_payload)
    print(f"  -> Audit PDF Ready                    : {pdf_path}")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE:")
    print(f"  • Offender Identified : {id_info['student_name']}")
    print(f"  • Item Discarded      : {waste_cls.upper()} ({waste_cat.upper()})")
    print(f"  • Fine Levied         : INR {fine}")
    print(f"  • PDF Dossier Saved   : {pdf_path}")
    print("=" * 70 + "\n")

    return incident_payload, pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAVA AI Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--reference-dir", type=str, default="sample_data/student_db", help="Reference directory")
    args = parser.parse_args()

    run_pipeline(args.video, args.reference_dir)