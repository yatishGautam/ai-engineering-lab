"""Verify the environment before week 2. Run: python check_setup.py"""
import importlib
import os
import sys

REQUIRED = ["numpy", "torch", "sklearn", "matplotlib"]
OPTIONAL = ["anthropic", "tiktoken", "rank_bm25"]


def check(name):
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__version__", "ok")
    except ImportError:
        return False, "missing"


def main():
    print(f"python {sys.version.split()[0]}\n")
    ok = True
    for name in REQUIRED:
        found, version = check(name)
        ok &= found
        print(f"  [{'x' if found else ' '}] {name:<12} {version}")
    print()
    for name in OPTIONAL:
        found, version = check(name)
        print(f"  [{'x' if found else ' '}] {name:<12} {version}  (optional until week 8)")

    print()
    if os.getenv("ANTHROPIC_API_KEY"):
        print("  [x] ANTHROPIC_API_KEY found in environment")
    else:
        print("  [ ] ANTHROPIC_API_KEY not set — needed from week 8 (see .env.example)")

    if torch_device() and ok:
        print(f"\nready. accelerator: {torch_device()}")
    elif ok:
        print("\nready (cpu only — use Colab for weeks 7, 12-15)")
    else:
        print("\nmissing required packages: pip install -r requirements.txt")
        sys.exit(1)


def torch_device():
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


if __name__ == "__main__":
    main()
