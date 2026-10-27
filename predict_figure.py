"""Make a sample-predictions figure from the trained model.

Loads outputs/best_model.pt and the test split, runs the model on the
test images, saves figures/sample_predictions.png, and opens the saved
image in the default image viewer so it pops up on screen: a 3x4 grid
of real test galaxies with their true label and the model's predicted
label. Green titles = model was right, red titles = model was wrong.
Showing mistakes on purpose: a real project shows where it still fails.

Usage:
    python predict_figure.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # file output only; the viewer popup is done by the OS
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import CLASS_NAMES, GalaxyCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    data = np.load(os.path.join(BASE_DIR, "data", "subset.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    checkpoint = torch.load(os.path.join(BASE_DIR, "outputs", "best_model.pt"),
                            map_location="cpu", weights_only=False)
    model = GalaxyCNN(num_classes=10)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Predict the whole test set (no augmentation here).
    with torch.no_grad():
        images = torch.from_numpy(X_test).float() / 255.0
        images = images.permute(0, 3, 1, 2)
        preds = []
        for i in range(0, len(images), 64):
            preds.extend(model(images[i:i + 64]).argmax(dim=1).tolist())

    rng = np.random.RandomState(0)
    correct_idx = [i for i in range(len(y_test)) if preds[i] == y_test[i]]
    wrong_idx = [i for i in range(len(y_test)) if preds[i] != y_test[i]]
    rng.shuffle(correct_idx)
    rng.shuffle(wrong_idx)
    # 8 correct + 4 wrong: an honest sample, mistakes included.
    chosen = correct_idx[:8] + wrong_idx[:4]
    rng.shuffle(chosen)

    os.makedirs(os.path.join(BASE_DIR, "figures"), exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, i in zip(axes.flat, chosen):
        ax.imshow(X_test[i])
        true_name = CLASS_NAMES[int(y_test[i])]
        pred_name = CLASS_NAMES[int(preds[i])]
        color = "green" if preds[i] == y_test[i] else "red"
        ax.set_title(f"True: {true_name}\nPred: {pred_name}",
                     fontsize=9, color=color)
        ax.axis("off")
    fig.suptitle("Sample test predictions (green = correct, red = wrong)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(BASE_DIR, "figures", "sample_predictions.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out} ({len(chosen)} images: "
          f"{sum(1 for i in chosen if preds[i] == y_test[i])} correct, "
          f"{sum(1 for i in chosen if preds[i] != y_test[i])} wrong)")
    # Pop the saved image open in the default viewer (Windows Photos).
    # This does not depend on matplotlib's backend, so it always shows.
    if os.name == "nt":
        print("Opening the image in your default viewer...")
        os.startfile(out)


if __name__ == "__main__":
    main()
