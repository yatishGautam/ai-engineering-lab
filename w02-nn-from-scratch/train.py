"""
A complete 2-layer neural network, NumPy only. No autograd, no frameworks.

The story, in three jobs:
    forward   compute the answer, and REMEMBER the intermediate values
    backward  walk the same path in reverse, handing out blame (gradients)
    update    turn every knob a little bit against its blame

Data is sklearn's load_digits: 8x8 handwritten digits, so 64 inputs and
10 classes. (The lesson's 784 is full MNIST -- same code, bigger first number.)

Shapes. B = number of images in the batch, H = hidden units.
    X        (B, 64)     pixel brightness, scaled to 0..1
    W1, b1   (64, H), (H,)
    h_pre    (B, H)      match scores for H learned patterns
    h        (B, H)      same, with negatives zeroed by ReLU
    W2, b2   (H, 10), (10,)
    logits   (B, 10)     one raw score per class
    probs    (B, 10)     the same scores as percentages; each ROW sums to 1
    y        (B,)        integer labels 0..9

The rule that catches most bugs: every gradient has the SAME shape as the
thing it is a gradient of. dW1 is (64, H). db2 is (10,). No exceptions.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)          # seeded, so runs are reproducible

# --------------------------------------------------------------------- config
H     = 128        # hidden units: how many patterns the first layer may learn
LR    = 0.1        # learning rate: how far to step against the gradient
STEPS = 1000       # full-batch, so one step = one pass over all training data


# ------------------------------------------------------------------- the data
digits = load_digits()
X_all = digits.data / 16.0              # raw pixels are 0..16; scale to 0..1
y_all = digits.target                   # ints 0..9

X, X_test, y, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=0, stratify=y_all
)


# ---------------------------------------------------------------- the pieces
def relu(x):
    """Keep positives, zero the negatives. Elementwise, so shape is unchanged.

    np.maximum (elementwise, two args) -- NOT np.max (reduces to one number).
    """
    return np.maximum(0, x)


def softmax(z):
    """Turn each ROW of raw scores into percentages that sum to 1.

    Subtracting the row max first is pure overflow protection: shifting a whole
    row by a constant cannot change softmax's output, but it keeps exp() from
    blowing up to inf. axis=1 reduces across columns, i.e. within each image.
    keepdims=True leaves the result (B,1) so it broadcasts back across the row.
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ the knobs
# Weights random so neurons start different -- identical neurons would receive
# identical gradients and stay identical forever. Biases can safely start at 0.
W1 = rng.standard_normal((64, H)) * 0.01;  b1 = np.zeros(H)
W2 = rng.standard_normal((H, 10)) * 0.01;  b2 = np.zeros(10)


# --------------------------------------------------------------- the training
print(f"train {X.shape}  test {X_test.shape}")
print(f"step {'loss':>8}")

for step in range(STEPS):

    # ---- forward: compute, and keep everything we'll need for the blame ----
    h_pre  = X @ W1 + b1            # (B,64)@(64,H) -> (B,H)   b1 broadcasts down the rows
    h      = relu(h_pre)            # (B,H)   roughly half the entries become exactly 0.0
    logits = h @ W2 + b2            # (B,H)@(H,10) -> (B,10)   raw, unbounded class scores
    probs  = softmax(logits)        # (B,10)  now percentages; each row sums to 1

    # cross-entropy: the penalty for the probability given to the TRUE class.
    # probs[np.arange(B), y] pairs row i with column y[i] -> one number per image.
    loss = -np.log(probs[np.arange(len(y)), y]).mean()

    # ---- backward: the same path in reverse, handing out blame ----

    # dL/dlogits = (probs - onehot(y)) / B, built without materialising onehot.
    # Read it as "how wrong were my percentages": the true class gets pushed up
    # (negative), every wrong class gets pushed down. Each row sums to ~0.
    dlogits = probs.copy()                      # copy: the next line mutates in place
    dlogits[np.arange(len(y)), y] -= 1
    dlogits /= len(y)                           # loss used .mean(), so divide by B here

    # matmul rule:  C = A @ B  ->  dA = dC @ B.T   and   dB = A.T @ dC
    # bias rule:    forward broadcast  ->  backward sum over the batch axis
    dW2    = h.T @ dlogits                      # (H,B)@(B,10) -> (H,10), same shape as W2
    db2    = dlogits.sum(0)                     # (10,)
    dh     = dlogits @ W2.T                     # (B,10)@(10,H) -> (B,H)

    # ReLU gate: neurons that were silent going forward get zero blame coming back.
    dh_pre = dh * (h_pre > 0)                   # boolean array acts as 1.0 / 0.0

    dW1    = X.T @ dh_pre                       # (64,B)@(B,H) -> (64,H), same shape as W1
    db1    = dh_pre.sum(0)                      # (H,)

    # ---- update: step downhill. `p -= ...` mutates in place; `p = p - ...`
    # would rebind the loop variable and silently leave the weights untouched.
    for p, g in [(W1, dW1), (b1, db1), (W2, dW2), (b2, db2)]:
        p -= LR * g

    if step % 100 == 0:
        print(f"{step:4d} {loss:8.4f}")


# --------------------------------------------------------------- the verdict
def predict(inputs):
    """Forward pass only. No loss, no gradients, no update."""
    h = relu(inputs @ W1 + b1)
    return (h @ W2 + b2).argmax(axis=1)         # highest score wins

train_acc = (predict(X) == y).mean()
test_acc  = (predict(X_test) == y_test).mean()
print(f"\ntrain accuracy {train_acc:.3f}")
print(f"test  accuracy {test_acc:.3f}   <- the only one that counts")
