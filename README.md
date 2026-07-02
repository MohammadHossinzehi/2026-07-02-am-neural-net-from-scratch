# Neural Net From Scratch

A feedforward neural network (multilayer perceptron) with backpropagation,
implemented from scratch in plain numpy. No PyTorch, no TensorFlow, no
autodiff library. Every forward pass, every gradient, and the optimizer
update rule are hand derived and hand coded, and checked against numerical
gradients in the test suite.

## What it does and why it's useful

Most "neural net from scratch" projects stop at a single hidden layer and
never verify the math. This one:

- Implements a `Dense` (fully connected) layer, `ReLU` activation, and a
  fused softmax + categorical cross-entropy loss, composed through a small
  `Sequential` container so you can stack layers like `Dense -> ReLU ->
  Dense -> ReLU -> Dense`.
- Trains with mini-batch SGD and momentum on a synthetic, non-linearly
  separable **spiral dataset** (the classic 3-armed spiral popularized by
  Stanford's CS231n) — a problem a linear classifier provably cannot solve,
  which makes it a genuine test of whether the hidden layers and backprop
  are actually working, not just memorizing.
- Ships with **gradient checking**: every analytical gradient produced by
  `backward()` is compared against a numerical (central difference)
  gradient in the test suite. This is the standard way to catch silent
  backprop bugs — code that runs without error but computes the wrong
  gradient, and therefore trains a network that never quite learns.
- Renders a decision boundary plot so you can visually confirm the network
  carved out three spiral-shaped regions rather than learning something
  degenerate.

The point isn't that this beats scikit-learn or PyTorch on the spiral
problem (it won't, and doesn't try to). The point is that every line
between "input" and "prediction" is visible, auditable, and unit tested,
which is exactly what you want from a project meant to demonstrate that
backpropagation is understood, not just imported.

## How to run it

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# Train the network and print train/test accuracy per epoch.
python train.py

# Faster run, no plot:
python train.py --epochs 500 --no-plot

# Tune the network:
python train.py --epochs 2000 --lr 0.1 --hidden 64
```

A typical run reaches **95-99% test accuracy** on the held-out spiral
points within 1000-1500 epochs, and writes `decision_boundary.png` showing
the learned regions.

Run the test suite:

```bash
pytest test_nn.py -v
```

## Project layout

```
nn.py         Core library: Dense, ReLU, CrossEntropyLoss, Sequential, SGD,
              and numerical_gradient (used only for testing).
data.py       Synthetic spiral dataset generator, one-hot encoding,
              train/test split. No external data files or downloads.
train.py      CLI entry point: builds a model, trains it, reports
              accuracy, and optionally plots the decision boundary.
test_nn.py    Unit tests, including numerical gradient checks and an
              overfit-a-tiny-batch convergence sanity check.
```

## Design decisions and testing

**Why the spiral dataset instead of MNIST or a CSV of real data?** The
spiral is generated in a few lines of numpy with a fixed seed, so the
project has zero external dependencies beyond numpy/matplotlib/pytest and
is fully reproducible from a clean clone — no downloading a dataset,
no network access required to run the tests or training script. It's also
specifically constructed to be non-linearly separable, which is the
property that actually exercises hidden layers and backprop, unlike many
toy datasets a single linear layer can already solve.

**Why fuse softmax into the loss instead of a separate `Softmax` layer?**
Computing `d(cross_entropy)/d(logits)` directly as `(softmax(logits) -
y_true) / batch_size` avoids ever forming the Jacobian of softmax
explicitly, which is both simpler and more numerically stable than
chaining a generic softmax layer's backward pass into a generic
cross-entropy layer's backward pass. This is the standard approach used
in most production frameworks internally.

**Why gradient checking instead of just "it trains, so it must be
correct"?** A network can appear to train — loss goes down, accuracy goes
up — even with a subtly wrong gradient, especially with momentum
smoothing out noise. Numerical gradient checking compares every analytical
gradient against a finite-difference approximation on small test cases,
which is the standard way (used throughout the original CS231n course and
most ML libraries' own test suites) to catch sign errors, transposition
bugs, and off-by-one mistakes in a backward pass before they hide inside a
"working" training curve.

**Why momentum SGD instead of Adam?** Momentum SGD has a two-line update
rule (`velocity = momentum * velocity - lr * grad; param += velocity`)
that's easy to verify by hand, which fits the project's goal of keeping
every component inspectable. Swapping in Adam would only mean adding two
more numpy arrays (first and second moment estimates) per parameter and a
bias-correction step; the `Dense.params_and_grads()` interface is already
generic enough to support it.
