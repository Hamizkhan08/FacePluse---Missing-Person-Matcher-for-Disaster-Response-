
# 🧠 FacePluse – Missing Person Matcher for Disaster Response

This project helps rescue teams automatically match a photo of a missing person with a large local dataset of known individuals using face recognition (DeepFace). It’s designed for use in post-disaster scenarios for rapid identification and aid.


## 📂 Project Structure

```

FacePluse/
├── app.py                     # Flask web app (main interface)
├── match\_faces.py            # (Optional) standalone matching script
├── save\_embeddings.py        # Precompute embeddings from VGGFace2 val images
├── dataset/
│   └── Known/                # ⬇️ Contains VGGFace2 `val` set (renamed)
├── embeddings/
│   └── known\_embeddings.pkl  # 🧠 Face embeddings (generated)
├── static/
│   ├── Known/                # Publicly served known images
│   └── uploads/              # Uploaded unknown face images
├── templates/
│   ├── index.html            # Upload form UI
│   └── result.html           # Match results page
└── README.md                 # 📘 This file

````

---

## 📦 Setup Instructions

### 1. 🔽 Download Dataset (VGGFace2 `val`)
Use only the **validation (val)** subset (about ~3GB) from the original VGGFace2 dataset.

📁 Download Link:  
👉 [[https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/data.html](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/data.html)  ](https://www.kaggle.com/datasets/hearfool/vggface2)
> ⚠️ Only use val folder dataset for this project Extract and rename the folder to:  
```bash
dataset/Known
````

---

### 2. ⚙️ Install Requirements

Make sure you have Python 3.8+ and run:

```bash
pip install -r requirements.txt
```

`requirements.txt` should contain:

```txt
flask
deepface
numpy
werkzeug
```

---

### 3. 🧠 Precompute Face Embeddings

Before running the app, extract face embeddings from the dataset:

```bash
python save_embeddings.py
```

This will create:

```
embeddings/known_embeddings.pkl
```

---

### 4. 🚀 Run the App

```bash
python app.py
```

Visit your app at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📷 Usage

1. Upload an image of a missing person.
2. The system will:

   * Extract facial features using VGG-Face.
   * Compare with precomputed embeddings.
   * Show the **best match** with:

     * Matched person's image
     * Name (folder)
     * Confidence score

---

## 🛠 Technologies Used

* DeepFace (VGG-Face model)
* Flask (Web framework)
* NumPy (Embedding processing)
* HTML/CSS (Frontend)

---

## ⚡ Notes

* This uses cosine similarity to compare embeddings.
* You can modify the system to display **Top N matches** or use a threshold to declare "no match found".
* Ensure GPU is enabled (if possible) for faster embedding generation.

---

## 🤝 Contributing

PRs are welcome! Fork the repo, create a branch, and submit a pull request.

---

## 🧑‍💻 Author

Made by [Hamiz Khan](https://github.com/Hamizkhan08)

---

## 🧠 Acknowledgements

* VGGFace2 Dataset: © University of Oxford – VGG
* DeepFace Library: [https://github.com/serengil/deepface](https://github.com/serengil/deepface)

---

```

