import os
import numpy as np
import pickle
from deepface import DeepFace

# Cosine similarity distance function
def cosine_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return 1 - (dot_product / (norm_vec1 * norm_vec2))

# Directories
unknown_dir = "dataset/Unknown"  # Place your test images here
embedding_path = "embeddings/known_embeddings.pkl"  # Loaded from VGGFace2 embeddings

# Load known face embeddings
with open(embedding_path, "rb") as f:
    known_faces = pickle.load(f)

# Match unknown images
for file in os.listdir(unknown_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        test_path = os.path.join(unknown_dir, file)
        try:
            # Extract face embedding from DeepFace
            test_embedding = DeepFace.represent(test_path, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]

            # Compare with known embeddings
            best_match = None
            best_distance = float("inf")
            for person in known_faces:
                distance = cosine_distance(test_embedding, person["embedding"])
                if distance < best_distance:
                    best_distance = distance
                    best_match = person["name"]

            print(f"✅ MATCH FOUND:\n   Unknown: {file}\n   Match: {best_match}\n   Distance: {best_distance:.4f}\n")

        except Exception as e:
            print(f"❌ Error processing {file}: {str(e)}")
