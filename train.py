"""Train the Galaxy Morphology CNN on the prepared subset.

What this script does:
  1. Loads data/subset.npz (made by data_prep.py).
  2. Trains the small CNN from model.py with Adam + cross-entropy loss.
     Training images get a random horizontal flip as augmentation --
     galaxies look the same mirrored, so this doubles the variety for free.
  3. After each epoch it measures validation accuracy and keeps the best
     model (best validation accuracy wins, not the last epoch).
  4. At the end it evaluates the best model on the test set and writes:
       outputs/best_model.pt     -- saved model weights
       outputs/metrics.json      -- train/val/test accuracy, per-class
                                      test accuracy, confusion matrix
       outputs/training_curves.png -- loss and accuracy per epoch

Usage:
    python train.py
    python train.py --epochs 12 --batch-size 32 --lr 0.001
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display needed; save figures to files
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from model import CLASS_NAMES, GalaxyCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class GalaxyDataset(Dataset):
    """Wraps numpy image arrays for PyTorch.

    Images are stored as uint8 (0-255). They are scaled to 0-1 here.
    If augment=True (training only), each image is randomly flipped
    left-right when it is loaded -- a safe augmentation for galaxies.
    """

    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment and np.random.rand() < 0.5:
            img = np.flip(img, axis=1)  # horizontal flip
        # To tensor: H x W x 3 -> 3 x H x W, scaled to 0-1.
        tensor = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        return tensor, int(self.labels[i])


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (accuracy, all_true_labels, all_predicted_labels)."""
    model.eval()
    correct, total = 0, 0
    all_true, all_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
    return correct / total, all_true, all_pred


def main():
    parser = argparse.ArgumentParser(description="Train the Galaxy CNN.")
    parser.add_argument("--data", default=os.path.join(BASE_DIR, "data", "subset.npz"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "outputs"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)  # this machine has 2 CPUs
    device = torch.device("cpu")
    print(f"Using device: {device}")

    data = np.load(args.data)
    print(f"Loaded {args.data}: "
          f"train={len(data['y_train'])}, val={len(data['y_val'])}, "
          f"test={len(data['y_test'])}")

    train_loader = DataLoader(
        GalaxyDataset(data["X_train"], data["y_train"], augment=True),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        GalaxyDataset(data["X_val"], data["y_val"]),
        batch_size=args.batch_size)
    test_loader = DataLoader(
        GalaxyDataset(data["X_test"], data["y_test"]),
        batch_size=args.batch_size)

    model = GalaxyCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += len(labels)
        train_loss = running_loss / total
        train_acc = correct / total
        val_acc, _, _ = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{args.epochs}: loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> new best validation accuracy, checkpoint saved")

    # Save the best model and evaluate it on the untouched test set.
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "class_names": CLASS_NAMES},
               os.path.join(args.out_dir, "best_model.pt"))
    test_acc, y_true, y_pred = evaluate(model, test_loader, device)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        row_total = int(cm[i].sum())
        per_class_acc[name] = round(float(cm[i, i] / row_total), 4) if row_total else None

    metrics = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "final_train_accuracy": round(history["train_acc"][-1], 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class_test_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_note": "Rows = true class, columns = predicted class, "
                                 "order = CLASS_NAMES in model.py.",
        "history": {k: [round(v, 4) for v in vals] for k, vals in history.items()},
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Training curves figure.
    epochs = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "training_curves.png"))
    plt.close(fig)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy (best model): {test_acc:.4f}")
    print(f"Per-class test accuracy: {per_class_acc}")
    print(f"Saved model, metrics and curves to {args.out_dir}")


if __name__ == "__main__":
    main()
