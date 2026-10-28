"""
debug_biometrics.py
===================
Inspects enrolled student embeddings and prints similarity matrices to find data mix-ups.
"""
import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.resolve()
DB_FILE = ROOT / "models" / "student_face_embeddings.pkl"

if not DB_FILE.exists():
    print(f"Error: {DB_FILE} not found. Run enroll_students.py first.")
    exit(1)

with open(DB_FILE, "rb") as f:
    db = pickle.load(f)

print("=" * 60)
print(f"ENROLLED STUDENTS DIAGNOSTIC ({len(db)} records)")
print("=" * 60)

student_keys = list(db.keys())
for idx, key in enumerate(student_keys):
    item = db[key]
    n_emb = len(item.get("embeddings", []))
    front = Path(item.get("front_photo", "")).name
    print(f"[{idx+1}] {key}")
    print(f"    Total Vectors : {n_emb}")
    print(f"    Front Image   : {front}")

print("\n" + "=" * 60)
print("CROSS-STUDENT SIMILARITY CHECK (Identifies mislabeled photos)")
print("=" * 60)

# Check if two different student profiles have near-identical vectors
for i in range(len(student_keys)):
    for j in range(i + 1, len(student_keys)):
        k1, k2 = student_keys[i], student_keys[j]
        embs1 = db[k1].get("embeddings", [])
        embs2 = db[k2].get("embeddings", [])
        
        max_sim = -1.0
        for e1 in embs1:
            for e2 in embs2:
                sim = float(np.dot(e1, e2))
                if sim > max_sim:
                    max_sim = sim
                    
        # If two different students have similarity > 0.75, their photos are likely mixed up!
        if max_sim > 0.70:
            print(f"  [CRITICAL WARNING] High overlap between:")
            print(f"    - {k1}")
            print(f"    - {k2}")
            print(f"    -> Max Cosine Similarity: {max_sim:.3f}")
            print(f"    -> Action: Check their image filenames in sample_data/student_db/\n")

print("Diagnostic complete.")