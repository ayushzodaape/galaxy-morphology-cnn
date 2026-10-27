"""Galaxy Morphology CNN — Flask website.

Upload a galaxy image, get a morphology prediction from a small CNN trained
on the Galaxy10 DECaLS dataset (2,500-image CPU subset, 12 epochs,
test accuracy 48.00% on 375 test images).

Usage:
    pip install -r requirements.txt
    python app.py            # http://127.0.0.1:5000

For a later Render-style deploy:
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4

Preprocessing is exactly the training pipeline's: PIL BILINEAR resize to
64x64, pixel values / 255.0, channels-first, NO mean/std normalisation.
"""

import io

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image

from model import CLASS_NAMES, GalaxyCNN

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = GalaxyCNN(num_classes=10)
        state = torch.load("best_model.pt", map_location="cpu", weights_only=True)
        # The checkpoint wraps the weights: {"state_dict": ..., "class_names": ...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        MODEL.load_state_dict(state)
        MODEL.eval()
    return MODEL


def preprocess(pil_image):
    """Match the training pipeline exactly: BILINEAR 64x64, /255, CHW."""
    img = pil_image.convert("RGB").resize((64, 64), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def predict(pil_image):
    """Return (predicted class, confidence, top-3 list of (class, prob))."""
    model = get_model()
    with torch.no_grad():
        probs = F.softmax(model(preprocess(pil_image)), dim=1)[0]
    top = torch.topk(probs, 3)
    top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top.indices.tolist()]
    return top3[0][0], top3[0][1], top3


@app.route("/")
def home():
    return render_template("index.html", classes=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files or not request.files["image"].filename:
        return jsonify({"error": "No image uploaded."}), 400
    try:
        pil_image = Image.open(io.BytesIO(request.files["image"].read()))
    except Exception:
        return jsonify({"error": "That file is not a readable image."}), 400
    predicted, confidence, top3 = predict(pil_image)
    return jsonify({
        "predicted": predicted,
        "confidence": confidence,
        "top3": [{"class": name, "probability": p} for name, p in top3],
    })


if __name__ == "__main__":
    print("Loading model (best_model.pt)...")
    get_model()
    print("Galaxy website running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
