"""
train.py - Train the from-scratch MLP on the spiral dataset and report
test accuracy. Optionally saves a decision boundary plot to
decision_boundary.png (requires matplotlib).

Usage:
    python train.py
    python train.py --epochs 2000 --lr 0.1 --hidden 64 --no-plot
"""

from __future__ import annotations

import argparse

import numpy as np

from data import make_spiral, one_hot, train_test_split
from nn import CrossEntropyLoss, Dense, ReLU, SGD, Sequential


def build_model(hidden: int, n_classes: int, seed: int = 1) -> Sequential:
    return Sequential(
        [
            Dense(2, hidden, seed=seed),
            ReLU(),
            Dense(hidden, hidden, seed=seed + 1),
            ReLU(),
            Dense(hidden, n_classes, seed=seed + 2),
        ]
    )


def accuracy(model: Sequential, X: np.ndarray, y: np.ndarray) -> float:
    logits = model.forward(X)
    preds = logits.argmax(axis=1)
    return float((preds == y).mean())


def train(epochs=1500, lr=0.1, hidden=32, batch_size=32, seed=0, verbose=True):
    n_classes = 3
    X, y = make_spiral(points_per_class=150, n_classes=n_classes, seed=seed)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=seed)
    y_train_oh = one_hot(y_train, n_classes)

    model = build_model(hidden, n_classes)
    loss_fn = CrossEntropyLoss()
    opt = SGD(lr=lr, momentum=0.9)

    n = X_train.shape[0]
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb, yb = X_train[batch_idx], y_train_oh[batch_idx]

            logits = model.forward(xb)
            epoch_loss += loss_fn(logits, yb) * len(batch_idx)
            model.backward(loss_fn.grad())
            opt.step(model)

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            train_acc = accuracy(model, X_train, y_train)
            test_acc = accuracy(model, X_test, y_test)
            print(
                f"epoch {epoch:5d}  loss {epoch_loss / n:.4f}  "
                f"train_acc {train_acc:.3f}  test_acc {test_acc:.3f}"
            )

    return model, (X_train, y_train, X_test, y_test)


def plot_decision_boundary(model, X, y, path="decision_boundary.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.forward(grid).argmax(axis=1).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, preds, alpha=0.3, cmap="viridis")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", s=20)
    plt.title("Decision boundary learned on the spiral dataset")
    plt.savefig(path, dpi=120)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    trained_model, (X_train, y_train, X_test, y_test) = train(
        epochs=args.epochs, lr=args.lr, hidden=args.hidden
    )

    final_acc = accuracy(trained_model, X_test, y_test)
    print(f"\nFinal test accuracy: {final_acc:.3f}")

    if not args.no_plot:
        X_all = np.concatenate([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        plot_decision_boundary(trained_model, X_all, y_all)
