"""Galaxy Morphology CNN -- model definition.

A small convolutional neural network (CNN) for classifying galaxy images
into 10 morphology classes (Galaxy10 DECaLS).

Architecture, for an input image of shape 3 x 64 x 64 (RGB, 64x64 pixels):
  Block 1: Conv 3 -> 32, ReLU, MaxPool   -> 32 x 32 x 32
  Block 2: Conv 32 -> 64, ReLU, MaxPool -> 64 x 16 x 16
  Block 3: Conv 64 -> 128, ReLU, MaxPool -> 128 x 8 x 8
  Classifier: Flatten -> Linear 8192 -> 256, ReLU, Dropout -> Linear 256 -> 10

Why this shape? Each MaxPool halves the image size (64 -> 32 -> 16 -> 8)
while the number of channels grows, so the network first learns small
patterns (edges, bright cores) and later bigger shapes (spiral arms, discs).
"""

import torch.nn as nn

# Class names, in the same order as the Galaxy10 "ans" labels (0-9).
# Source: Galaxy10 DECaLS documentation (astroNN / Zenodo).
CLASS_NAMES = [
    "Disturbed",                # 0
    "Merging",                  # 1
    "Round Smooth",             # 2
    "In-between Round Smooth",  # 3
    "Cigar Shaped Smooth",      # 4
    "Barred Spiral",            # 5
    "Unbarred Tight Spiral",    # 6
    "Unbarred Loose Spiral",    # 7
    "Edge-on without Bulge",    # 8
    "Edge-on with Bulge",       # 9
]


class GalaxyCNN(nn.Module):
    """Small CNN for 10-class galaxy morphology classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Feature extractor: three convolution + pooling blocks.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Classifier: flatten the 128 x 8 x 8 feature map, then two dense layers.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
