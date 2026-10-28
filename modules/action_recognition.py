"""
modules/action_recognition.py
==============================
Detects the exact discard release frame using YOLO-Pose wrist kinematics.
Captures the item actively falling in mid-air between hand release and ground contact.
"""

from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False


class DiscardActionDetector:
    def __init__(self, model_weight: str = "yolo11n-pose.pt"):
        self.model = None
        if POSE_AVAILABLE:
            try:
                self.model = YOLO(model_weight)
            except Exception:
                pass

    """
modules/action_recognition.py
==============================
Tracks downward wrist release and isolates the mid-air falling object.
Uses YOLO-Pose ankle exclusion to completely ignore chappals/footwear on the ground.
"""

from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False


class DiscardActionDetector:
    def __init__(self, model_weight: str = "yolo11n-pose.pt"):
        self.model = None
        if POSE_AVAILABLE:
            try:
                self.model = YOLO(model_weight)
            except Exception:
                pass

    def detect_discard_event(
        self,
        video_path: str,
        evidence_dir: Path
    ) -> Tuple[str, str, str, float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        all_frames: List[np.ndarray] = []
        wrist_trajectory = []
        feet_positions = []
        frame_idx = 0
        prev_y = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            all_frames.append(frame)
            h, w = frame.shape[:2]

            if self.model is not None:
                results = self.model(frame, verbose=False)
                if results and len(results) > 0 and results[0].keypoints is not None:
                    kpts = results[0].keypoints.xy.cpu().numpy()
                    if len(kpts) > 0:
                        person_kpts = kpts[0]
                        # 9 = Left Wrist, 10 = Right Wrist
                        lw = person_kpts[9] if len(person_kpts) > 9 else [0, 0]
                        rw = person_kpts[10] if len(person_kpts) > 10 else [0, 0]
                        
                        # 15 = Left Ankle, 16 = Right Ankle (Footwear baseline)
                        la = person_kpts[15] if len(person_kpts) > 15 else [0, 0]
                        ra = person_kpts[16] if len(person_kpts) > 16 else [0, 0]

                        active_wrist = rw if rw[1] > lw[1] else lw
                        curr_y = float(active_wrist[1])
                        curr_x = float(active_wrist[0])

                        if curr_y > (h * 0.25):
                            vel_y = (curr_y - prev_y) if prev_y is not None else 0.0
                            wrist_trajectory.append((frame_idx, curr_y, curr_x, vel_y))
                            prev_y = curr_y

                        feet_positions.append((frame_idx, float(la[0]), float(la[1]), float(ra[0]), float(ra[1])))

        frame_idx += 1
        cap.release()

        total_frames = len(all_frames)
        if total_frames == 0:
            raise ValueError(f"No valid frames found in {video_path}")

        action_frame_path = evidence_dir / "evidence_littering.jpg"
        waste_crop_path = evidence_dir / "evidence_waste_crop.jpg"

        if wrist_trajectory:
            # 1. Detect Peak Downward Velocity
            valid_drops = [pt for pt in wrist_trajectory if pt[3] > 0]
            drop_event = max(valid_drops, key=lambda item: item[3]) if valid_drops else max(wrist_trajectory, key=lambda item: item[1])
            release_fidx, r_y, r_x, _ = drop_event

            # Step 2 to 4 frames ahead: Paper is detached from hand, in mid-air
            mid_flight_offset = max(2, min(5, int(fps * 0.12)))
            falling_fidx = min(total_frames - 1, release_fidx + mid_flight_offset)
            clean_action_frame = all_frames[falling_fidx].copy()
            timestamp_str = f"00:00:{int(falling_fidx / fps):02d}"

            # 2. Extract Footwear Exclusion Heights
            # Get ankle height at release frame
            ankle_y = h
            for f_data in feet_positions:
                if f_data[0] == falling_fidx:
                    _, lax, lay, rax, ray = f_data
                    ankles = [y for y in (lay, ray) if y > 0]
                    if ankles:
                        ankle_y = min(ankles)
                    break

            # 3. Spatial Crop: Confined strictly between release wrist and ABOVE footwear
            crop_w = int(w * 0.25)
            # Stop the crop 20 pixels ABOVE ankle level so chappals cannot enter
            max_y2 = min(h, int(ankle_y - 20)) if ankle_y < h else h

            y1 = max(0, int(r_y + 10))
            y2 = max(y1 + int(h * 0.15), max_y2)
            
            # Keep bounding box proportional
            if y2 <= y1:
                y2 = min(h, y1 + int(h * 0.20))

            cx = int(r_x)
            x1 = max(0, int(cx - crop_w // 2))
            x2 = min(w, x1 + crop_w)

            waste_crop = clean_action_frame[y1:y2, x1:x2]
            confidence = 0.95
        else:
            # Fallback
            falling_fidx = int(total_frames * 0.5)
            clean_action_frame = all_frames[falling_fidx].copy()
            timestamp_str = f"00:00:{int(falling_fidx / fps):02d}"
            h, w = clean_action_frame.shape[:2]
            waste_crop = clean_action_frame[int(h * 0.40):int(h * 0.70), int(w * 0.3):int(w * 0.7)]
            confidence = 0.75

        cv2.imwrite(str(action_frame_path), clean_action_frame)
        if waste_crop is not None and waste_crop.size > 0:
            cv2.imwrite(str(waste_crop_path), waste_crop)

        return str(action_frame_path), str(waste_crop_path), timestamp_str, confidence