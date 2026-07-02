"""
test_nn.py - Unit tests for the from-scratch neural network library.

Run with: pytest test_nn.py -v

Covers:
  - Shape correctness of forward passes.
  - Gradient checking: analytical gradients (from backward()) are compared
    against numerical gradients (central differences) to catch backprop
    bugs that would otherwise silently produce a network that "trains"
    but learns the wrong thing.
  - An end-to-end convergence test: the network should be able to drive
    training loss on a small fixed batch close to zero, a standard sanity
    check that the full forward/backward/optimizer loop is wired up
    correctly.
  - A shape/label sanity check on the synthetic spiral dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from data import make_spiral, one_hot
from nn import CrossEntropyLoss, Dense, ReLU, SGD, Sequential, numerical_gradient


def test_dense_forward_shape():
    layer = Dense(4, 3, seed=0)
    x = np.random.default_rng(0).standard_normal((5, 4))
    out = layer.forward(x)
    assert out.shape == (5, 3)


def test_relu_zeroes_negatives():
    relu = ReLU()
    x = np.array([[-1.0, 2.0, -3.0, 0.5]])
    out = relu.forward(x)
    assert np.array_equal(out, np.array([[0.0, 2.0, 0.0, 0.5]]))


def test_dense_gradient_matches_numerical():
    rng = np.random.default_rng(42)
    layer = Dense(3, 2, seed=1)
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 2))

    def loss():
        return float(np.sum(layer.forward(x) * upstream))

    layer.forward(x)
    layer.backward(upstream)
    analytical_dW = layer.dW

    numerical_dW = numerical_gradient(loss, layer.W)

    assert np.allclose(analytical_dW, numerical_dW, atol=1e-4)


def test_cross_entropy_gradient_matches_numerical():
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((6, 3))
    y = one_hot(rng.integers(0, 3, size=6), 3)
    loss_fn = CrossEntropyLoss()

    def loss():
        return loss_fn(logits, y)

    loss_fn(logits, y)
    analytical = loss_fn.grad()
    numerical = numerical_gradient(loss, logits)

    assert np.allclose(analytical, numerical, atol=1e-4)


def test_network_overfits_tiny_batch():
    # A network with enough capacity should be able to drive loss on a
    # small fixed batch close to zero. If backprop or the optimizer is
    # wired incorrectly, this is usually the first thing that fails.
    rng = np.random.default_rng(3)
    X = rng.standard_normal((16, 2))
    y = rng.integers(0, 3, size=16)
    y_oh = one_hot(y, 3)

    model = Sequential(
        [
            Dense(2, 32, seed=1),
            ReLU(),
            Dense(32, 32, seed=2),
            ReLU(),
            Dense(32, 3, seed=3),
        ]
    )
    loss_fn = CrossEntropyLoss()
    opt = SGD(lr=0.2, momentum=0.9)

    losses = []
    for _ in range(300):
        logits = model.forward(X)
        losses.append(loss_fn(logits, y_oh))
        model.backward(loss_fn.grad())
        opt.step(model)

    assert losses[-1] < 0.05
    assert losses[-1] < losses[0]


def test_spiral_dataset_shape():
    X, y = make_spiral(points_per_class=50, n_classes=3)
    assert X.shape == (150, 2)
    assert y.shape == (150,)
    assert set(np.unique(y)) == {0, 1, 2}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
