# Plan and decision log

Decisions are recorded with their reasons so the final report can explain why the project looks the way it does. Estimates are replaced by measurements as soon as receipts exist.

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-10-12 | Ladder of 25M (original) → 50M → 100M → 200M total parameters | Fits a 300 compute-unit budget with room for SFT, RL, ablations, and reruns; 300M was dropped to keep pretraining sessions short |
| 2026-10-12 | Keep the original architecture rhythm at every rung | Differences between rungs should be attributable to scale |
| 2026-10-12 | BPE tokenizer (16,384 vocabulary) for the new rungs | Byte-level sequences are about 4× longer; byte vs BPE becomes an ablation |
| 2026-10-12 | Work from VS Code on Colab runtimes; no files opened on Colab | Owner preference |
| 2026-10-12 | Code and receipts on GitHub; checkpoints and token shards on Hugging Face Hub (free tier) | GitHub rejects files over 100 MB; a 200M resume checkpoint is about 2.4 GB |
| 2026-10-12 | No paid services beyond the existing Colab Pro subscription | Owner constraint |
| 2026-10-12 | Everything in the repository is written in English | Owner preference |
| 2026-10-12 | Only the owner pushes to GitHub | Owner preference |
| open | Fine-tuning task | To be decided before Phase 4 |

## Compute budget (estimates until Phase 0 measurements)

Assumes about 30% model-FLOPs utilization after the Phase 0 fixes. Colab rates checked March 2026: L4 ≈ 1.71 units/hour, A100 40 GB ≈ 5.40 units/hour.

| Item | GPU | Estimate | Units |
|---|---|---|---:|
| Baseline reproduction and smoke tests | L4 | < 1 hour | ~2 |
| `s-50m` pretraining, ~1.5B tokens | L4 | 2–3 hours | ~5 |
| `m-100m` pretraining, ~3B tokens | A100 | 3–4 hours | ~20 |
| `l-200m` pretraining, ~4B tokens | A100 | 7–10 hours | ~40–55 |
| SFT, fine-tuning, RL across all rungs | mixed | — | ~60–80 |
| Ablations and seed replications | mixed | — | ~60 |
| Reserve | — | — | ~50+ |

L4 costs roughly the same per token as A100 (about 3× slower, about 3× cheaper per hour), so any run can fall back to L4 at the cost of wall-clock time.

## Session-loss resilience (Phase 1–2 requirement)

- Checkpoint weights, optimizer, scheduler, RNG states, and data position every 20–30 minutes; upload in a background thread.
- Keep the two most recent resume checkpoints; keep milestone weights (bf16 `safetensors`) permanently.
- Re-running a notebook resumes automatically from the latest checkpoint.
- Deterministic data order, proven by a test: 200 uninterrupted steps must equal 100 + resume + 100 steps.
- Every session is recorded in the run receipt.

## Security

- Hugging Face access through a fine-grained token scoped to this project's repositories only, entered at runtime and never written to a file; 2FA enabled on the account.
- Checkpoints stored as `safetensors` (no pickle, no code execution on load).
- SHA-256 of every checkpoint recorded in GitHub receipts and verified on load.
- All checkpoints are reproducible from configs and seeds in this repository.

## Open research questions

- **Double residual.** HybridBlock returns `x + delta` and HyperConnection adds it to the
  stream again, so the input is counted twice per layer (inherited from the original model,
  kept for parity). Ablation: blocks return only `delta`. Does it change loss or stability?
- **RoPE base.** 1,000 vs 10,000 vs 100,000 at 1024-token context.