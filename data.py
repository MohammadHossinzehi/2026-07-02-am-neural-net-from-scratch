"""
data.py - Synthetic dataset generator: the classic 3-armed spiral.

The spiral dataset is a standard, deliberately non-linearly-separable toy
problem (popularized by Stanford's CS231n) used to sanity check that a
classifier can learn a non-linear decision boundary that a linear model
cannot. It is generated purely with numpy, so no downloads or external
files are required to run this project.
"""

from __future__ import annotations

import numpy as np


def make_spiral(
    points_per_class: int = 150,
    n_classes: int = 3,
    noise: float = 0.2,
    seed: int = 0,
):
    """Generate a 2D spiral classification dataset.

    Returns (X, y) where X has shape (points_per_class * n_classes, 2)
    and y has shape (points_per_class * n_classes,) with integer labels
    in [0, n_classes).
    """
    rng = np.random.default_rng(seed)
    n = points_per_class
    X = np.zeros((n * n_classes, 2))
    y = np.zeros(n * n_classes, dtype=int)

    for class_idx in range(n_classes):
        r = np.linspace(0.0, 1.0, n)
        t = (
            np.linspace(class_idx * 4, (class_idx + 1) * 4, n)
            + rng.standard_normal(n) * noise
        )
        idx = slice(n * class_idx, n * (class_idx + 1))
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = class_idx

    return X, y


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def train_test_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_test = int(X.shape[0] * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
