import os
import pickle
from deepface import DeepFace
import numpy as np

# Paths
KNOWN_DIR = "dataset/Known"  # VGGFace2 val dataset
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "known_embeddings.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

known_faces = []

print("🔍 Starting embedding generation...")

for identity_folder in os.listdir(KNOWN_DIR):
    identity_path = os.path.join(KNOWN_DIR, identity_folder)
    
    if not os.path.isdir(identity_path):
        continue

    for file_name in os.listdir(identity_path):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(identity_path, file_name)
            relative_path = os.path.join(identity_folder, file_name)
            print(f"🔍 Processing: {relative_path}")

            try:
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name="VGG-Face",
                    enforce_detection=False
                )[0]["embedding"]

                known_faces.append({
                    "name": identity_folder,
                    "embedding": embedding,
                    "filename": relative_path  # Needed for displaying image
                })

            except Exception as e:
                print(f"❌ Failed: {relative_path} → {e}")

# Save all embeddings
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(known_faces, f)

print(f"\n✅ Saved {len(known_faces)} embeddings to {OUTPUT_FILE}")
