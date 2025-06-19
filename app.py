import os
import shutil
import pickle
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from deepface import DeepFace

# --- Flask App Initialization ---
app = Flask(__name__)


DATASET_KNOWN = "dataset/Known"
STATIC_KNOWN = "static/Known"
UPLOAD_FOLDER = "static/uploads"
EMBEDDINGS_PATH = "embeddings/known_embeddings.pkl"

os.makedirs(STATIC_KNOWN, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    with open(EMBEDDINGS_PATH, "rb") as f:
        known_faces = pickle.load(f)
    print(f"✅ Successfully loaded {len(known_faces)} known face embeddings.")
except FileNotFoundError:
    known_faces = []
    print(f"🚨 WARNING: Embeddings file not found at '{EMBEDDINGS_PATH}'.")
    print("👉 Please run 'save_embeddings.py' first to create it.")

# --- Helper Function for Similarity ---
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# --- Main Application Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="Please select an image to upload.")

        fname = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, fname)
        file.save(upload_path)

        try:
            # Represent the uploaded image using the robust 'mtcnn' detector
            rep = DeepFace.represent(
                img_path=upload_path,
                model_name="VGG-Face",
                enforce_detection=True,
                detector_backend='mtcnn'
            )
            test_emb = rep[0]["embedding"]
        except ValueError:
            return render_template("result.html", error="No face detected in the uploaded image.", uploaded_img=fname)
        except Exception as e:
            return render_template("result.html", error=f"An error occurred: {e}", uploaded_img=fname)

        results = []
        for person in known_faces:
            sim = cosine_similarity(test_emb, person["embedding"])
            results.append({
                "name": person["name"],
                "filename": person["filename"],
                "similarity": sim
            })

        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:4]
        
        for r in results:
            src_path = os.path.join(DATASET_KNOWN, r["filename"])
            dst_path = os.path.join(STATIC_KNOWN, r["filename"])
            
            # Create sub-directory if it doesn't exist
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # Copy the file from dataset to static folder
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
            
            # --- THE FINAL FIX FOR WINDOWS ---
            # Convert Windows-style backslashes to web-style forward slashes for the HTML template.
            r["img_path"] = r["filename"].replace("\\", "/")
            r["similarity_percent"] = f"{r['similarity'] * 100:.2f}%"

        return render_template(
            "result.html",
            uploaded_img=fname,
            matches=results
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
