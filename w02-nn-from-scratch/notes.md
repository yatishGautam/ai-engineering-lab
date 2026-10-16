# Notes — W2 (neural nets from scratch)

## A neuron is a dot product with an attitude

### Plain language
A neuron scores evidence, then decides whether to speak up.

Should you go for a run? Inputs: temperature, tiredness, rain.
- **Weights** = how much you care about each input (rain −5, temperature +1).
- Multiply each input by how much you care, add it up → your score.
- **Bias** = baseline eagerness. High bias = "I'll go unless there's a reason not to."
- **Nonlinearity** = if the score is negative, you just don't go. Output 0.

Weigh, sum, offset, squash.

### The dot product intuition
`w · x` is a **match score**. `w` is a pattern the neuron looks for; `x` is what it got.
The dot product is large when `x` points the same direction as `w`.

> A neuron stores one pattern and reports how strongly the input matches it.
> The bias sets the bar; the nonlinearity enforces it.

With `w = [0.5, -1]`, `b = 1`:

```
x = [2, 3]  ->  z = 0.5(2) + (-1)(3) + 1 = -1  ->  ReLU(-1) = 0   (silent)
x = [4, 1]  ->  z = 0.5(4) + (-1)(1) + 1 =  2  ->  ReLU(2)  = 2   (fires)
```

Geometrically `w·x + b = 0` is a hyperplane cutting space in two. The neuron reports
which side you're on and how far. ReLU zeroes out one entire side.

### A layer is a matrix multiply
Many neurons, same input `x`, each with its own `w` and `b`. Stack the weight vectors
as rows of `W`:

```
math convention:   z = W x + b        W: (n_out x n_in), x a column vector
code convention:   Z = X @ W + b      X: (batch x n_in), W: (n_in x n_out)
a = ReLU(z)                            elementwise
```

Same operation, transposed. The R2 code uses the second form (`h_pre = X @ W1 + b1`,
`W1` is 784x128) because data arrives as rows. Don't let the flip confuse you: rows of
one are columns of the other, and either way each output number is one dot product of
one input with one neuron's weight vector.

Nothing new — just all the dot products at once. In code a layer is ONE matmul; that
is what the GPU is for. A network stacks them: each layer's activations are the next
layer's inputs. Layer 1 finds patterns in the data; layer 2 finds patterns in layer 1's
pattern-matches.

### Why the nonlinearity is non-negotiable
Drop the ReLU from two layers:

```
W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2) = W' x + b'
```

`W2 W1` is just another matrix. **100 linear layers collapse into 1.** Depth buys zero
expressive power; you can only ever draw one straight boundary.

Insert `max(0, ·)` and the collapse is algebraically impossible. Each ReLU neuron is a
**switch** — on/off by which side of its hyperplane the input landed. `n` neurons →
up to `2^n` on/off patterns, a different linear function in each region. The network is
**piecewise-linear**; enough pieces approximate any continuous function (universal
approximation). Depth grows region count ~exponentially, width ~linearly. That is the
whole case for "deep".

### Jargon, now that it's cheap
- `Wx + b` = **affine transform**. `z` = **pre-activation** / logit. `a = φ(z)` = **activation**.
- ReLU derivative is exactly 1 or 0 → on the backward pass it passes the gradient
  through untouched or blocks it. Sigmoid/tanh derivatives cap at 0.25 and shrink
  gradients through depth → **vanishing gradients**. That's why ReLU won.
- A neuron outputting 0 for every input has no gradient and never recovers: **dead ReLU**.
  (Leaky ReLU / GELU soften this.)
- A layer's activation vector is a **representation** / hidden state / embedding: the
  input re-described as "how much of each learned pattern is present".
- Architecture defines what is expressible; the weights decide what is expressed.

### Five sentences to remember
1. A neuron holds a pattern and reports how well the input matches it — the dot product.
2. The bias sets how eager it is to fire.
3. A layer is many neurons matching in parallel = one matrix multiply.
4. Without a nonlinearity, any stack of layers collapses into a single linear map.
5. ReLU makes each neuron a switch → piecewise-linear network → universal approximator.

### Check yourself
- Shapes: 2 inputs, 3 neurons → `W` is (3, 2), `b` is (3,), output is (3,).
- Prove the collapse yourself on paper before trusting it.
- In `train.py`, find the line that is `Wx + b` and the line that is `max(0, z)`.

---

## Where this shows up in the R2 code
- Forward:  `h_pre = X @ W1 + b1` -> the matmul.  `h = relu(h_pre)` -> the switch.
- Backward: `dh_pre = dh * (h_pre > 0)` -> **the same switch, on the way back**. Neurons
  that were silent forward get zero gradient backward. ReLU is a gate in both directions.
- Zero init is fatal for the same reason nonlinearity matters: if every neuron holds the
  same pattern `w`, every neuron computes the same thing and gets the same gradient
  forever. Symmetry never breaks. Small random init gives each neuron a different pattern
  to start refining.

## Interview drill this feeds (R2)
Follow-up to "explain backprop": **"Why do we need nonlinearities at all?"**
Answer in one breath: composing affine maps gives an affine map — `W2(W1x+b1)+b2` is
literally `W'x+b'` — so depth without a nonlinearity is algebraically one layer. ReLU
makes each neuron a switch, so the network is piecewise-linear over exponentially many
regions, which is what buys expressive power.
