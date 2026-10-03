"""
CNG403 Assignment 1 — Data Utilities
======================================
Shared utilities for data loading and feature extraction.
- Used by both reference.py and train.py
- HOG feature extraction with fixed parameters
"""

import random
import numpy as np
import torch
from skimage.feature import hog
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Feature extraction (HOG only)
# ---------------------------------------------------------------------------

def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """
    Convert a batch of grayscale images (N, H, W) to HOG feature vectors.

    Args:
        images: numpy array of shape (N, H, W), values in [0, 1].

    Returns:
        features: numpy array of shape (N, D).
    """
    # Fixed HOG parameters
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)

    features = []
    for img in images:
        feat = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            channel_axis=None,
        )
        features.append(feat)
    return np.stack(features)


def load_and_extract(data_root: str) -> tuple:
    """
    Download MNIST, extract HOG features, return torch tensors.

    Args:
        data_root: directory where MNIST should be stored/loaded.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test  — all torch.Tensor
    """
    # Fixed train/val split
    val_split = 0.1

    to_tensor = transforms.ToTensor()
    train_raw = datasets.MNIST(data_root, train=True, download=True, transform=to_tensor)
    test_raw = datasets.MNIST(data_root, train=False, download=True, transform=to_tensor)

    def dataset_to_numpy(ds):
        imgs = ds.data.numpy().astype(np.float32) / 255.0  # (N, 28, 28)
        labels = ds.targets.numpy()
        return imgs, labels

    train_imgs, train_labels = dataset_to_numpy(train_raw)
    test_imgs, test_labels = dataset_to_numpy(test_raw)

    print("Extracting HOG features (train)…")
    X_train_full = extract_hog_features(train_imgs)
    print("Extracting HOG features (test)…")
    X_test = extract_hog_features(test_imgs)

    # Standardise using train statistics
    mean = X_train_full.mean(axis=0)
    std = X_train_full.std(axis=0) + 1e-8
    X_train_full = (X_train_full - mean) / std
    X_test = (X_test - mean) / std

    # Train / validation split
    n_val = int(len(X_train_full) * val_split)
    n_train = len(X_train_full) - n_val
    idx = np.random.permutation(len(X_train_full))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X_train_full[train_idx], train_labels[train_idx]
    X_val, y_val = X_train_full[val_idx], train_labels[val_idx]

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    to_l = lambda a: torch.tensor(a, dtype=torch.long)

    return to_t(X_train), to_l(y_train), to_t(X_val), to_l(y_val), to_t(X_test), to_l(test_labels)
