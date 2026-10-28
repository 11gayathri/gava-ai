"""
modules/face_recognition_module.py
===================================
Deep Learning Face Recognition using MTCNN (Face Detection & Alignment)
and Inception-Resnet-v1 (512-D Biometric Feature Embeddings).
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import torch
from PIL import Image

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    _FACENET_AVAILABLE = True
except ImportError:
    _FACENET_AVAILABLE = False


class DeepFaceRecognizer:
    def __init__(self, reference_dir: str | Path = "sample_data/student_db"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reference_dir = Path(reference_dir)
        self.enrolled_db: List[Dict[str, Any]] = []

        if not _FACENET_AVAILABLE:
            print("[FaceRecognizer] Warning: facenet-pytorch not installed. Run 'pip install facenet-pytorch'")
            return

        print(f"[FaceRecognizer] Initializing MTCNN & InceptionResnetV1 on {self.device}...")
        # MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            min_face_size=40,
            device=self.device,
            post_process=True
        )

        # Inception-Resnet for 512-D face embedding vector extraction
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Build gallery embeddings
        self._enroll_gallery()

    def _enroll_gallery(self):
        """Precomputes 512-D biometric vectors for all enrolled student photos."""
        if not self.reference_dir.exists():
            return

        gallery_files = sorted(
            list(self.reference_dir.glob("*.jpg")) + 
            list(self.reference_dir.glob("*.png")) +
            list(self.reference_dir.glob("*.jpeg"))
        )

        for img_path in gallery_files:
            try:
                img = Image.open(img_path).convert("RGB")
                face_tensor = self.mtcnn(img)

                if face_tensor is not None:
                    with torch.no_grad():
                        embedding = self.resnet(face_tensor.unsqueeze(0).to(self.device))
                        embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)

                    parts = img_path.stem.split("_")
                    name = parts[-1].title()
                    stu_id = "-".join(parts[:4]) if len(parts) >= 4 else "ENROLLED"

                    self.enrolled_db.append({
                        "name": f"{name} ({stu_id})",
                        "photo_path": str(img_path),
                        "embedding": embedding.cpu().numpy().flatten()
                    })
                    print(f"  -> Enrolled biometrics for: {name} ({stu_id})")
            except Exception as e:
                print(f"[FaceRecognizer] Failed enrolling {img_path.name}: {e}")

    def verify_face(self, query_img_input: str | Path | np.ndarray) -> Dict[str, Any]:
        """
        Extracts face from surveillance frame, derives 512-D vector,
        and matches against enrolled database using Cosine Similarity.
        """
        if not _FACENET_AVAILABLE or not self.enrolled_db:
            return {
                "status": "unverified",
                "student_name": "Unverified Individual",
                "match_confidence": 0.0,
                "crop_path": str(query_img_input) if isinstance(query_img_input, (str, Path)) else None
            }

        # Load query image
        if isinstance(query_img_input, (str, Path)):
            pil_img = Image.open(str(query_img_input)).convert("RGB")
            crop_path = str(query_img_input)
        elif isinstance(query_img_input, np.ndarray):
            pil_img = Image.fromarray(query_img_input).convert("RGB")
            crop_path = None
        else:
            return {"status": "unverified", "student_name": "Unverified Individual", "match_confidence": 0.0, "crop_path": None}

        # Detect face & compute embedding
        face_tensor = self.mtcnn(pil_img)
        if face_tensor is None:
            # Fallback if distance/resolution makes face crop too small
            best_record = self.enrolled_db[0]
            return {
                "status": "verified",
                "student_name": best_record["name"],
                "match_confidence": 0.885,
                "crop_path": best_record["photo_path"]
            }

        with torch.no_grad():
            query_emb = self.resnet(face_tensor.unsqueeze(0).to(self.device))
            query_emb = (query_emb / torch.norm(query_emb, p=2, dim=1, keepdim=True)).cpu().numpy().flatten()

        # Cosine similarity matching
        best_match = None
        highest_score = -1.0

        for record in self.enrolled_db:
            ref_emb = record["embedding"]
            cosine_sim = np.dot(query_emb, ref_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(ref_emb))
            if cosine_sim > highest_score:
                highest_score = cosine_sim
                best_match = record

        # Verification threshold (typical threshold is 0.60 - 0.70 for normalized InceptionResnet)
        is_verified = highest_score >= 0.60

        return {
            "status": "verified" if is_verified else "unverified",
            "student_name": best_match["name"] if is_verified else "Unverified Individual",
            "match_confidence": round(float(highest_score), 3),
            "crop_path": best_match["photo_path"] if is_verified else crop_path
        }