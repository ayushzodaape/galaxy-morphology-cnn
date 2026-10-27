"""Download + prepare a Galaxy10 DECaLS subset for training.

What this script does:
  1. Opens the Galaxy10 DECaLS file (data/Galaxy10_DECals.h5).
     Download it first -- see README.md for the link (Zenodo, open access,
     no account needed). The full file is ~2.5 GB with 17,736 images.
  2. Picks a stratified random subset (default 2,500 images). "Stratified"
     means every class keeps its real share of the data, so the subset
     looks like the full dataset, just smaller.
  3. Resizes every image from 256x256 to 64x64 (much faster on CPU, and
     galaxy shapes are still clearly visible at that size).
  4. Splits the subset into train / validation / test (default 70/15/15),
     again stratified, with a fixed seed so the split is reproducible.
  5. Saves everything to data/subset.npz plus a data/subset_info.json
     summary (exact counts per class and split).

Usage:
    python data_prep.py
    python data_prep.py --n-images 3000 --img-size 64 --seed 42
"""

import argparse
import json
import os

import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from model import CLASS_NAMES

# Default paths are relative to this script's folder, so it works from anywhere.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resize_image(img_array, size):
    """Resize one uint8 RGB image (H x W x 3) to size x size with PIL."""
    img = Image.fromarray(img_array)
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser(description="Prepare a Galaxy10 subset.")
    parser.add_argument("--h5", default=os.path.join(BASE_DIR, "data", "Galaxy10_DECals.h5"),
                        help="Path to the downloaded Galaxy10_DECals.h5 file.")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "data", "subset.npz"),
                        help="Where to save the prepared subset (.npz).")
    parser.add_argument("--n-images", type=int, default=2500,
                        help="Total subset size (default: 2500).")
    parser.add_argument("--img-size", type=int, default=64,
                        help="Resize images to this size (default: 64).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits.")
    args = parser.parse_args()

    print(f"Opening {args.h5} ...")
    with h5py.File(args.h5, "r") as f:
        print(f"  datasets in file: {list(f.keys())}")
        labels_all = np.array(f["ans"])  # shape (17736,), values 0-9
        n_total = len(labels_all)
        print(f"  full dataset: {n_total} images, "
              f"{len(np.unique(labels_all))} classes")

        # --- Step 1: stratified subset -------------------------------------
        # train_test_split with stratify keeps each class's share.
        all_idx = np.arange(n_total)
        sub_idx, _ = train_test_split(
            all_idx, train_size=args.n_images,
            stratify=labels_all, random_state=args.seed)
        sub_labels = labels_all[sub_idx]
        print(f"  subset: {len(sub_idx)} images")

        # --- Step 2: read + resize only the subset images ------------------
        # (We never load the full 256x256 dataset into memory.)
        images = np.empty((len(sub_idx), args.img_size, args.img_size, 3),
                          dtype=np.uint8)
        for i, idx in enumerate(sub_idx):
            images[i] = resize_image(np.array(f["images"][idx]), args.img_size)
            if (i + 1) % 500 == 0:
                print(f"  resized {i + 1}/{len(sub_idx)} ...")

    # --- Step 3: stratified train / val / test split -----------------------
    idx = np.arange(len(sub_idx))
    train_idx, temp_idx = train_test_split(
        idx, train_size=0.70, stratify=sub_labels, random_state=args.seed)
    # Of the remaining 30%, half goes to val and half to test (15% each).
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5,
        stratify=sub_labels[temp_idx], random_state=args.seed)

    def counts(y):
        """Per-class counts as {class_name: count}."""
        uniq, c = np.unique(y, return_counts=True)
        return {CLASS_NAMES[int(k)]: int(v) for k, v in zip(uniq, c)}

    info = {
        "source": "Galaxy10 DECaLS (Zenodo record 10845026, "
                  "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5)",
        "source_note": "Images from DESI Legacy Imaging Surveys; labels from Galaxy Zoo.",
        "full_dataset_images": int(n_total),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "subset_images": int(len(sub_idx)),
        "image_size": [args.img_size, args.img_size, 3],
        "seed": args.seed,
        "splits": {
            "train": {"count": int(len(train_idx)), "per_class": counts(sub_labels[train_idx])},
            "val": {"count": int(len(val_idx)), "per_class": counts(sub_labels[val_idx])},
            "test": {"count": int(len(test_idx)), "per_class": counts(sub_labels[test_idx])},
        },
    }

    np.savez(
        args.out,
        X_train=images[train_idx], y_train=sub_labels[train_idx],
        X_val=images[val_idx], y_val=sub_labels[val_idx],
        X_test=images[test_idx], y_test=sub_labels[test_idx],
        class_names=np.array(CLASS_NAMES),
    )
    info_path = os.path.splitext(args.out)[0] + "_info.json"
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=2)

    print(f"\nSaved subset to {args.out}")
    print(f"Saved summary to {info_path}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {info['splits'][split]['count']} images")
    print(f"  per-class test counts: {info['splits']['test']['per_class']}")


if __name__ == "__main__":
    main()
