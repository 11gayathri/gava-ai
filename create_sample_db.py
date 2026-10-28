"""
create_sample_db.py
Generates sample enrolled student profiles in sample_data/student_db/
for testing the authorized identity verification module.
"""

from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).parent.resolve()
DB_DIR = ROOT / "sample_data" / "student_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# Generate synthetic portrait crops for demonstration
students = [
    ("MITS_STU_2026_01_Anoop", (180, 100, 50)),
    ("MITS_STU_2026_02_Devika", (50, 180, 100)),
    ("MITS_STU_2026_03_Rahul", (100, 50, 180)),
]

for name, color in students:
    # 200x200 simulated headshot
    img = np.full((200, 200, 3), color, dtype=np.uint8)
    cv2.circle(img, (100, 80), 40, (255, 255, 255), -1)
    cv2.ellipse(img, (100, 180), (60, 40), 0, 0, 180, (255, 255, 255), -1)
    cv2.putText(img, name.split("_")[-1], (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    out_path = DB_DIR / f"{name}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Created sample profile: {out_path.name}")

print("\nStudent gallery ready in sample_data/student_db/")