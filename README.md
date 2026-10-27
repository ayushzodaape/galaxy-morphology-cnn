# Galaxy Morphology CNN

A small convolutional neural network (CNN) that looks at a telescope image of a
galaxy and predicts its shape — spiral, smooth, edge-on, merging, and so on.
This is the same kind of task astronomers use to sort the millions of galaxies
in sky surveys, where classifying every galaxy by eye is impossible.

## Dataset (real data, trusted source only)

- **Dataset:** Galaxy10 DECaLS — 17,736 real galaxy images (256×256×3, g/r/z
  bands) in 10 morphology classes. Images come from the DESI Legacy Imaging
  Surveys; labels come from Galaxy Zoo volunteers.
- **Source:** Zenodo record 10845026 (CERN's established open research
  repository — a trusted, verified source, no account/key/login needed):
  https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5
- **Integrity check:** the downloaded file's SHA-256 matches the hash published
  for that Zenodo record exactly:
  `19AEFC477C41BB7F77FF07599A6B82A038DC042F889A111B0D4D98BB755C1571`
  (file size 2,735,267,419 bytes). No data was synthesized or faked.
- **Subset used for training (CPU-friendly):** 2,500 images, picked with a
  stratified random sample (seed 42) so every class keeps its real share of
  the full dataset, resized from 256×256 to 64×64.
- **Split (also stratified, seed 42):** train 1,750 / validation 375 / test 375.

The 10 classes and their subset counts (train/val/test):

| Class | Train | Val | Test |
|---|---:|---:|---:|
| Disturbed | 106 | 23 | 23 |
| Merging | 183 | 39 | 39 |
| Round Smooth | 261 | 56 | 56 |
| In-between Round Smooth | 200 | 43 | 43 |
| Cigar Shaped Smooth | 33 | 7 | 7 |
| Barred Spiral | 201 | 43 | 44 |
| Unbarred Tight Spiral | 181 | 39 | 38 |
| Unbarred Loose Spiral | 259 | 55 | 56 |
| Edge-on without Bulge | 141 | 30 | 30 |
| Edge-on with Bulge | 185 | 40 | 39 |

## Model architecture (`model.py`)

A small 3-block CNN called `GalaxyCNN`:

1. Conv 3→32, ReLU, MaxPool (64×64 → 32×32)
2. Conv 32→64, ReLU, MaxPool (32×32 → 16×16)
3. Conv 64→128, ReLU, MaxPool (16×16 → 8×8)
4. Flatten (128×8×8 = 8192) → Linear 8192→256, ReLU, Dropout 0.3
5. Linear 256→10 (one score per galaxy class)

Training: Adam optimizer (learning rate 0.001), batch size 32, 12 epochs,
cross-entropy loss, random horizontal flips as augmentation on the training
set only. The checkpoint with the best validation accuracy is kept
(`outputs/best_model.pt`). Everything runs on CPU (2 threads).

## How to run it

```bash
pip install -r requirements.txt

# 1. Download the dataset (Zenodo, ~2.5 GB, no login) into data/
#    https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5
#    then verify: sha256sum data/Galaxy10_DECals.h5
#    (expect 19AEFC47...C1571, see above)

# 2. Build the 2,500-image subset + splits
python data_prep.py

# 3. Train (CPU) — saves best_model.pt, metrics.json, training_curves.png
python train.py

# 4. Sample predictions figure (8 correct + 4 wrong test images)
python predict_figure.py
```

## Results (actually measured, not claimed)

Run on CPU, 12 epochs, best-validation checkpoint:

- **Final train accuracy: 58.86%** (epoch 12)
- **Best validation accuracy: 52.00%** (epoch 12)
- **Test accuracy: 48.00%** (375 test images)

Per-class test accuracy:

| Class | Test accuracy |
|---|---:|
| Edge-on without Bulge | 83.33% |
| Unbarred Tight Spiral | 73.68% |
| Edge-on with Bulge | 71.79% |
| Round Smooth | 69.64% |
| In-between Round Smooth | 41.86% |
| Merging | 38.46% |
| Unbarred Loose Spiral | 37.50% |
| Cigar Shaped Smooth | 28.57% |
| Barred Spiral | 9.09% |
| Disturbed | 0.00% |

Honest reading: 48% on a 10-class problem is far above the 10% random-guess
baseline, and the model is genuinely good at edge-on galaxies and tight
spirals. It struggles where the classes genuinely look alike (Disturbed vs
Merging vs Loose Spiral) and where a class has very few examples — Disturbed
had only 23 test images and Cigar Shaped Smooth only 7, so those two scores
are noisy. More training images and more epochs would be the first things to
try for a better number. The full confusion matrix is in
`outputs/metrics.json`.

## Files

- `model.py` — the CNN definition + class names
- `data_prep.py` — subset + split builder (writes `data/subset.npz`,
  `data/subset_info.json`)
- `train.py` — training loop (writes `outputs/best_model.pt`,
  `outputs/metrics.json`, `outputs/training_curves.png`)
- `predict_figure.py` — sample predictions figure
  (`figures/sample_predictions.png`: 12 test images, green title = correct,
  red title = wrong)
- `data/Galaxy10_DECals.h5` — the full downloaded dataset (~2.5 GB, not
  needed for a re-run once `data/subset.npz` exists)
- `requirements.txt` — exact dependency versions used

## If someone asks you about this project (5 lines)

1. I trained a small neural network to classify galaxy images by their shape.
2. The data is real telescope images from the Galaxy10 DECaLS dataset, which comes from the DESI sky survey with labels from Galaxy Zoo volunteers.
3. There are 10 shape classes, like spiral, smooth, edge-on, and merging galaxies.
4. I trained on 2,500 images on CPU and got 48% accuracy on a held-out test set, which is well above the 10% you'd get by guessing.
5. The model is strongest on edge-on and tight spiral galaxies, and weakest on disturbed and merging ones, because those shapes genuinely look similar.
