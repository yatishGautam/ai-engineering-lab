# AI Engineering Lab

My working notebook for becoming an AI engineer: one folder per week, each with a build, a
measured number, and an honest account of what broke. Everything here is written by me —
no tutorial reruns.

Curriculum: 34 lessons across transformer foundations, training, RAG, fine-tuning, RL,
agents, evals, serving, and cost. Flagship project lives separately in
[`mini-code`](https://github.com/yatishGautam/mini-code).

## The index

Numbers appear as I finish each week. Empty rows are scheduled, not abandoned.

**Weeks are independent from W7 onward.** Each folder is self-contained (its README lists
milestones; skeleton files give signatures, shapes and TODOs, the bodies are mine). W2–W6 are
recap builds and can be filled in any time; nothing later imports from them.

| Week | Build | Headline number |
| --- | --- | --- |
| [W2](w02-nn-from-scratch) | Neural net from scratch, NumPy only | test acc 0.956 on 360 held-out digits |
| [W3](w03-attention) | Scaled dot-product + multi-head attention | |
| [W4](w04-minigpt) | MiniGPT assembled and trained on characters | |
| [W5](w05-pytorch-ladder) | Same net three ways: raw tensors → nn.Module → MPS | |
| [W6](w06-modern-arch) | RMSNorm + SwiGLU + RoPE retrofit | |
| [W7](w07-training-run) | Full training run + three induced pathologies | |
| [W8](w08-research-agent) | Research agent, raw API, no frameworks | |
| [W9](w09-rag-baseline) | RAG in ~80 lines over my own corpus | |
| [W10](w10-rag-hybrid-rerank) | BM25 + RRF fusion + cross-encoder rerank | |
| [W11](w11-long-context) | Map-reduce vs recursive QA vs long context | |
| [W12](w12-lora-finetune) | LoRA fine-tune, structured extraction task | |
| [W13](w13-qlora-quantization) | QLoRA on free-tier GPU + GGUF quantisation | |
| [W14](w14-dpo-preference) | DPO on a synthetic preference set | |
| [W15](w15-rlvr-verifier) | Verifier + pass@k + GRPO-style advantages | |
| [W16](w16-rl-environment) | Procedurally generated RL environment | |
| [W17](w17-context-budget) | Context budgeting and result truncation | |
| [W18](w18-agent-memory) | Agent memory files with a salience rule | |
| [W19](w19-serving-benchmark) | Serving benchmark vs bandwidth ceiling | |
| [W21](w21-langgraph-mcp) | LangGraph port + MCP server over my notes | |
| [W23](w23-huggingface-publish) | Model + card + Space published | |
| [W24](w24-observability) | Traced agent, cost and latency per span | |
| [W25](w25-browser-agent) | Browser agent with per-step verification | |
| [W26](w26-eval-harness) | 60-case eval, calibrated judge, error analysis | |
| [W27](w27-multimodal-extractor) | Invoice extractor with arithmetic verification | |
| [W28](w28-redteam) | Red-team pass: attack success before/after | |
| [W29](w29-cost-memo) | Cost engineering memo | |
| W30 | Harness upgrade of mini-code (permissions, sandbox, checkpoints, skills) — lives in [`mini-code`](https://github.com/yatishGautam/mini-code) | |
| [papers](papers) | Paper notes, pass-2 and deeper | |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python check_setup.py
```

Copy `.env.example` to `.env` and add your API key before week 8. `.env` is gitignored.

Weeks 7 and 12-15 want a GPU — use Colab's free tier and commit the notebook plus results here.

## Conventions

Every week folder answers three questions in its README: **what I built**, **the number**,
**what broke and why**. Commit messages state the change and its effect
(`add RRF fusion, hit@5 0.61 -> 0.74`), never `update`.
