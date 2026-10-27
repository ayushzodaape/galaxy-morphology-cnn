# Galaxy Morphology CNN — Website

Upload a galaxy image, get a morphology prediction. A visitor uploads an
image, the page shows it, and the model answers with a predicted class plus
top-3 probabilities as bars.

## What the model is

A small convolutional neural network (3 conv blocks 3→32→64→128, then
Linear 8192→256→10) trained by Ayush on the **Galaxy10 DECaLS** dataset —
real telescope images, verified via Zenodo record 10845026 (SHA-256 checked).

Training was a **2,500-image CPU subset** (1,750 train / 375 val / 375 test),
12 epochs. Nothing here is inflated:

| Metric | Result |
|---|---|
| Test accuracy | **48.00%** (375 test images, 10 classes) |
| Best val accuracy | 52.00% |
| Final train accuracy | 58.86% |

Per-class test accuracy — strongest and weakest, honestly:

- **Strongest:** Edge-on without Bulge **83.33%**, Edge-on with Bulge 71.79%, Unbarred Tight Spiral 73.68%, Round Smooth 69.64%
- **Weakest:** Disturbed **0%** (23 test images — the model never gets this class right), Barred Spiral **9.09%**

So a prediction is a guess with a confidence score, not a fact — and a
photo that is not a galaxy at all will still get a label, which will be
meaningless. The website says this on the page itself.

## How the website works

- `app.py` — Flask app. `GET /` serves the page; `POST /predict` takes an
  uploaded image file and returns JSON: predicted class, confidence, top-3
  probabilities.
- `templates/index.html` — dark deep-space page: click or drag-and-drop
  upload, uploaded-image preview, predicted class prominently, top-3
  probability bars, a plain-language guide to all 10 classes, and the honest
  accuracy note.
- `model.py` — model definition, unchanged from the training project.
- `best_model.pt` — trained weights (8.7 MB).
- Preprocessing is exactly the training pipeline's: PIL BILINEAR resize to
  64×64, pixel values / 255.0, channels-first, NO mean/std normalisation.

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. For a Render-style deploy:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4
```
