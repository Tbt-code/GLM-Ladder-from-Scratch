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
| 2026-10-13 | Data mix by UTF-8 bytes: 60% English (FineWeb-Edu sample-10BT), 30% Python (codeparrot-clean), 10% Turkish (FineWeb2-HQ tur_Latn) | Code enables executable-reward RL; Turkish keeps a Turkish fine-tuning task possible, and must be in the tokenizer now because the tokenizer cannot change after pretraining |
| 2026-10-13 | Train the tokenizer locally; tokenise the full corpus on a Colab CPU runtime | Tokenizer training needs ~300 MB and minutes; keeping `tokenizer.json` in git avoids moving files out of Colab |
| 2026-10-14 | Pretraining recipe for every rung: 262,144-token batches, AdamW (0.9, 0.95), wd 0.1 on matrices, LR 6e-4 with 100 warmup steps and cosine decay to 10% | Common small-model defaults; one recipe keeps rungs comparable; no LR sweep, to save compute |
| 2026-10-14 | Global batch fixed in sequences, independent of micro-batch | The same run can continue on L4 (micro-batch 16) or A100 (32) with identical data order and maths |
| 2026-10-14 | A resume refuses changed run-defining settings (LR, batch, seed, token budget, ...) but accepts changed micro-batch, compile, logging, checkpoint cadence | A silently different experiment under the same run name would invalidate comparisons |
| 2026-10-15 | Every MoE expert processes one extra zero row with weight 0 | `torch.compile` kernels crash on zero-sized data-dependent shapes when an expert receives no token; outputs and gradients are unchanged |
| 2026-10-15 | Kill stale kernels at the top of the training notebook; checkpoint every 15 minutes | In VS Code + Colab, interrupt did not stop cells and Restart Kernel left the old process on the GPU (out-of-memory in the new kernel) |
| 2026-10-15 | Report validation in bits per byte as well as nats per token | Nats per token are not comparable across languages or tokenizers |
| 2026-10-16 | Pretraining of `s-50m` and `m-100m` complete; next phase is SFT on both | Results in README; `baseline-25m` reproduction and `l-200m` stay deferred |
| open | Fine-tuning task | To be decided before Phase 4 |

## Pretraining, measured (A100 40 GB, full training loop with gradient accumulation)

| Rung | Tokens/s | Tokens | Training hours | Final val loss (mixed) |
|---|---:|---:|---:|---:|
| `s-50m` | 166,499 | 1.06B | 1.77 | 2.948 |
| `m-100m` | 99,714 | 1.92B | 5.35 | 2.636 |

The full loop ran 1.2–1.4× faster than the Phase 0 benchmark below (8-way gradient accumulation amortises per-step overhead), so the estimates below were conservative. Pretraining both rungs used roughly 45–50 units including setup, evaluation, uploads, and one restart (estimate from session hours).

## Compute budget (Phase 0 estimate, A100 40 GB, torch.compile + TF32, commit 181c686)

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

## Session-loss resilience (as implemented)

- Checkpoint weights (`safetensors`), AdamW state, RNG states, step, and loader position every 15–25 minutes and at the end; the upload runs in the training loop (about 30–60 s for `m-100m`).
- Upload order keeps the Hub consistent if a session dies mid-way: checkpoint folder → `latest.json` → delete stale checkpoints.
- Keep the two most recent resume checkpoints; the final checkpoint is a milestone and is never deleted.
- Re-running the notebook resumes from `latest.json`; metrics logged after that checkpoint are dropped so no step appears twice.
- Exact resume is tested on CPU (12 steps uninterrupted vs 6 + interrupt + resume from step 4: identical weights) and was observed on the A100 (identical loss and gradient norm after resuming `s50m-001`).
- Not done: a per-session log in the receipt.

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