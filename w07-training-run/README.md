# W7 — A real training run, with pathologies

**This week stands alone.** Nothing here imports from earlier weeks. You write a small
character-level GPT, train it properly on TinyShakespeare, then break it three ways on purpose
and explain each break. Every file in this folder is a skeleton: signatures, shapes and TODOs
are given; the bodies are yours to type.

## Milestones (commit after each)

1. **Data** — `python data.py` downloads TinyShakespeare (~1 MB), builds the character
   vocabulary, writes `data/train.bin` and `data/val.bin` (uint16, 90/10 split), prints the
   vocab size (~65) and token counts. `data/` is gitignored.
2. **Model** — `python model.py` builds the GPT and runs one forward pass on a random batch;
   the loss at init should be about `ln(65) ≈ 4.17`. If it is not, the wiring is wrong
   (R2's "loss 2.3 at init" rule, same idea).
3. **Overfit one batch** — `python train.py --overfit-one-batch --steps 300 --tag overfit`
   drives train loss toward 0 on a single fixed batch. If this cannot reach near-zero, the
   optimizer, the shapes or the loss are broken; fix before going further.
4. **Real run** — `python train.py --tag base` with the defaults. Expect val loss under 1.6
   within ~3000 steps on MPS, in under 15 minutes. Check `figures/base.png` and the samples
   it prints: by the end they should look like Shakespearean nonsense with real words.
5. **Pathologies** — three more runs, three more PNGs, three post-mortems below:
   - `python train.py --lr 1 --steps 500 --tag explode`
   - `python train.py --overfit-one-batch --steps 300 --tag overfit` (reuse milestone 3)
   - `python train.py --n-layer 8 --n-embd 384 --steps 6000 --tag diverge` (too much model
     for 1 MB of text; watch val turn upward while train keeps falling)
6. **README** — fill in the three sections below and the W7 row of the top-level index.

## What I built
<Two sentences. Plain language, no jargon padding.>

## The number

| metric | value | measured on |
| --- | --- | --- |
| val loss (char-level) | <0.00> | 10% held-out split of TinyShakespeare, step <N> |
| tokens/sec during training | <0> | <device> |

## What broke and why

Five lines each: symptom · cause · the signal that would have caught it · the fix · the rule.

### Pathology 1 — learning rate 1
<…>

### Pathology 2 — single-batch overfit
<…>

### Pathology 3 — val/train divergence
<…>

## Run it

```bash
source ../.venv/bin/activate
python data.py
python model.py
python train.py --tag base
```

Interview drill from the lesson, once all of this works: "your run went NaN at step 4,000 of
10,000 — debug it out loud." Your three post-mortems are the material.
