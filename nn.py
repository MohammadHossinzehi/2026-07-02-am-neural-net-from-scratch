"""
nn.py - A minimal, from-scratch neural network library.

Implements dense (fully connected) layers, a ReLU activation, a fused
softmax + categorical cross-entropy loss, a Sequential container, and an
SGD optimizer with momentum. No autodiff, no external ML frameworks -
every forward and backward pass is derived and coded by hand.
"""

from __future__ import annotations

import numpy as np


class Dense:
    """A fully connected layer: y = x @ W + b."""

    def __init__(self, n_in: int, n_out: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # He initialization, a good default for ReLU networks.
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))

        # Momentum buffers, used by the SGD optimizer.
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        # Cache for the backward pass.
        self._x = None

        # Gradients, populated by backward().
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dy: np.ndarray) -> np.ndarray:
        x = self._x
        self.dW = x.T @ dy
        self.db = dy.sum(axis=0, keepdims=True)
        dx = dy @ self.W.T
        return dx

    def params_and_grads(self):
        yield self.W, self.dW, self.vW
        yield self.b, self.db, self.vb


class ReLU:
    """Elementwise rectified linear unit."""

    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._mask

    def params_and_grads(self):
        return iter(())


class CrossEntropyLoss:
    """Categorical cross-entropy loss with a built in softmax.

    Call `loss_fn(logits, y_onehot)` to get the scalar loss, then
    `loss_fn.grad()` to get d(loss)/d(logits). Combining softmax and
    cross-entropy into one op keeps the gradient a simple, numerically
    stable expression: (softmax(logits) - y_onehot) / batch_size.
    """

    def __init__(self):
        self._probs = None
        self._y = None

    def __call__(self, logits: np.ndarray, y_onehot: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self._probs = probs
        self._y = y_onehot
        n = logits.shape[0]
        clipped = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(y_onehot * np.log(clipped)) / n)

    def grad(self) -> np.ndarray:
        n = self._y.shape[0]
        return (self._probs - self._y) / n


class Sequential:
    """A simple ordered stack of layers, run front to back on forward()
    and back to front on backward()."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def params_and_grads(self):
        for layer in self.layers:
            yield from layer.params_and_grads()


class SGD:
    """Stochastic gradient descent with momentum."""

    def __init__(self, lr: float = 0.05, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum

    def step(self, model: Sequential) -> None:
        for param, grad, velocity in model.params_and_grads():
            velocity *= self.momentum
            velocity -= self.lr * grad
            param += velocity


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central difference numerical gradient of a scalar function f at x.

    Used only in tests, to check the analytical backward() implementations
    against a slow but trustworthy finite-difference approximation.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        f_plus = f()
        x[idx] = original - eps
        f_minus = f()
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad
