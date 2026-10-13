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
| 2026-10-12 | Only the owner pushes to GitHub | Owner preference 

| 2026-10-13 | Phase 0 benchmarks measured MFU of 4–11% (L4 and A100), not the assumed ~30% | Profiling showed ~80% of GPU time in elementwise ops and dtype copies; the model is memory- and launch-bound at this scale |
| 2026-10-13 | Enable TF32 and `torch.compile` for all training | Measured 1.41× on L4 from compile; TF32 moves the float32 linear-attention core onto tensor cores |
| 2026-10-13 | Stop throughput optimisation for now | One more round (graph-break fix) gave +10% on m-100m only; further gains need deeper rewrites |
| 2026-10-13 | Same tokens-per-parameter ratio for every rung (20, on total parameters) | Rungs trained with different ratios cannot be compared fairly |
| 2026-10-13 | Defer `l-200m` until all phases (pretraining → SFT → fine-tuning → RL) are complete on `s-50m` and `m-100m` | Learn the full pipeline on cheap runs first; 200M was ~2/3 of the pretraining budget |
| 2026-10-13 | A100 for `m-100m` (and later `l-200m`); `s-50m` on either GPU | Measured cost per token is similar on L4 and A100; A100 needs ~3× fewer Colab sessions |
| open | Fine-tuning task | To be decided before Phase 4 |

## Compute budget (measured, A100 40 GB, torch.compile + TF32, commit 181c686)

| Rung | Tokens/s | Target tokens (20/param) | Hours | Units (+15% overhead) |
|---|---:|---:|---:|---:|
| `s-50m` | 117,117 | 1.06B | 2.5 | ~16 |
| `m-100m` | 82,307 | 1.92B | 6.5 | ~40 |
| `l-200m` (deferred) | 58,229 | 3.99B (or 1.99B at 10/param) | 19.0 (9.5) | ~118 (~59) |

| Item | Units |
|---|---:|
| `s-50m` + `m-100m` pretraining | ~56 |
| SFT, fine-tuning, RL on both rungs | ~60–80 |
| Ablations on `s-50m` (shortened runs) | ~40 |
| Remaining for `l-200m` and reserve | ~125–140 |

Raw measurements: [`artifacts/receipts/`](artifacts/receipts/).

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