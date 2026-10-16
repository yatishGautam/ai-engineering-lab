"""The training loop: the lesson's five stages, end to end.

    data -> batching -> optimization -> evaluation -> checkpointing

    python train.py --tag base                              the real run
    python train.py --lr 1 --steps 500 --tag explode        pathology 1
    python train.py --overfit-one-batch --steps 300 --tag overfit   pathology 2
    python train.py --n-layer 8 --n-embd 384 --steps 6000 --tag diverge   pathology 3

Every run writes runs/<tag>.csv (step, train_loss, val_loss, lr) and figures/<tag>.png.
Checkpoints go to checkpoints/<tag>.pt (gitignored); figures are committed.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from data import decode, get_batch, load_split
from model import GPT, GPTConfig

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train a character GPT on TinyShakespeare")
    p.add_argument("--tag", default="base", help="name for this run's csv/png/checkpoint")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
    p.add_argument("--warmup", type=int, default=100, help="linear warmup steps")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--overfit-one-batch", action="store_true", help="pathology 2: same batch every step")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, peak: float, warmup: int, total: int, floor_ratio: float = 0.1) -> float:
    """Linear warmup to `peak`, then cosine decay to `peak * floor_ratio` at `total`."""
    # TODO(1): if step < warmup: return peak * (step + 1) / warmup
    #          progress = (step - warmup) / max(1, total - warmup)      # 0 -> 1
    #          cosine = 0.5 * (1 + math.cos(math.pi * progress))        # 1 -> 0
    #          return peak * floor_ratio + (peak - peak * floor_ratio) * cosine
    raise NotImplementedError


@torch.no_grad()
def estimate_loss(model: GPT, data, args, device: str) -> float:
    """Mean loss over `eval_batches` random batches. model.eval() before, model.train() after."""
    # TODO(2): switch to eval mode, average model(x, y)[1].item() over the batches, switch back.
    raise NotImplementedError


def sample(model: GPT, itos: dict[int, str], device: str, n_tokens: int = 200) -> str:
    """Generate from a single newline token so each sample starts a fresh line."""
    # TODO(3): start = torch.tensor([[stoi['\n']]]) -- or just token 0 -- on device;
    #          return decode(model.generate(start, n_tokens)[0].tolist(), itos)
    raise NotImplementedError


def save_plot(rows: list[dict], path: Path) -> None:
    """Train and val loss vs step. matplotlib; this PNG is the artifact you commit."""
    # TODO(4): plt.plot(step, train_loss) and (step, val_loss) for rows that have them;
    #          label axes, legend, title = path.stem, plt.savefig(path, dpi=120). Mkdir parents.
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device {device}  tag {args.tag}")

    # ---- data ----
    train_data, val_data = load_split("train"), load_split("val")
    # TODO(5): load the vocabulary data.py saved, build itos for decoding samples,
    #          and set vocab_size from it.
    vocab_size = 65
    itos: dict[int, str] = {}

    fixed = None
    if args.overfit_one_batch:
        # Pathology 2: the same offsets every step, so the model can memorize one batch.
        fixed = torch.randint(0, len(train_data) - args.block_size, (args.batch_size,)).numpy()

    # ---- model + optimizer ----
    cfg = GPTConfig(vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"{model.num_params() / 1e6:.2f}M parameters")
    # TODO(6): optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
    #          weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # ---- loop ----
    rows: list[dict] = []
    best_val = float("inf")
    ckpt_path = HERE / "checkpoints" / f"{args.tag}.pt"
    t0 = time.time()
    tokens_seen = 0

    for step in range(args.steps):
        # TODO(7): lr = lr_at(step, args.lr, args.warmup, args.steps); set it on every
        #          param group: for g in optimizer.param_groups: g["lr"] = lr
        #
        # TODO(8): x, y = get_batch(train_data, args.block_size, args.batch_size, device, fixed)
        #          logits, loss = model(x, y)
        #          optimizer.zero_grad(set_to_none=True); loss.backward()
        #          if args.grad_clip > 0: grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        #          optimizer.step(); tokens_seen += x.numel()
        #
        # TODO(9): every args.eval_every steps (and at the last step):
        #          val = estimate_loss(model, val_data, args, device)
        #          print step, train loss, val loss, lr, grad_norm, tokens/sec = tokens_seen / (time.time() - t0)
        #          print(sample(model, itos, device))
        #          rows.append({"step": step, "train_loss": loss.item(), "val_loss": val, "lr": lr})
        #          if val < best_val: best_val = val; mkdir; torch.save({"cfg": cfg, "model": model.state_dict(),
        #              "step": step, "val_loss": val}, ckpt_path)
        #          else (non-eval steps): rows.append({"step": step, "train_loss": loss.item(), "lr": lr})
        #
        # TODO(10): if the loss is NaN or inf, print the step and break -- that is pathology 1's
        #           signal, and you want the csv/png to show the moment it happened.
        raise NotImplementedError

    # ---- artifacts ----
    runs = HERE / "runs"; runs.mkdir(exist_ok=True)
    with (runs / f"{args.tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "train_loss", "val_loss", "lr"])
        w.writeheader(); w.writerows(rows)
    save_plot(rows, HERE / "figures" / f"{args.tag}.png")
    print(f"best val {best_val:.3f}  checkpoint {ckpt_path}  wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
