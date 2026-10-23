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
| 2026-10-16 | SFT data: `smol-smoltalk` (English) plus ~10% `InstrucTurca` (Turkish) | Made for small models and includes coding requests; the Turkish share matches pretraining. InstrucTurca is machine-translated, so its Turkish quality is limited |
| 2026-10-16 | Loss only on assistant tokens and the end-of-turn token; whole conversations packed next-fit into 1,025-token windows | The model should learn to answer and to stop, not to imitate the user; a split conversation would ask it to answer a question it cannot see |
| 2026-10-16 | Cut over-long conversations after their last complete assistant turn instead of dropping them | Dropping lost 46% of English conversations, biased towards long answers; cutting skips ~2.5% |
| 2026-10-16 | SFT reuses the pretraining loop (`init_from`, fresh AdamW) and evaluates the pretraining validation sets too | One tested, resumable loop; the extra validation measures forgetting |
| 2026-10-17 | SFT recipe: LR 1e-4, 50 warmup steps, 128 × 1024-token batches, one epoch, not tuned | Common fine-tuning defaults; a sweep is an ablation candidate |
| 2026-10-17 | Executable code eval: 40 own tasks with unit tests, completion and chat prompts, greedy + 10 samples, reference-solution loss | pass@k is near zero for base models; the reference loss still separates them |
| 2026-10-17 | Generated code runs behind a deny-list, in a subprocess, in a temp dir, with a timeout | Enough for 40 short functions from our own models; not a sandbox, so RL at scale moves to a container |
| 2026-10-17 | Scripts get names that no library module has (`talk.py`, `run_code_eval.py`) | Same-named files let the editor rewrite imports and overwrite library code twice |
| 2026-10-18 | No replay of pretraining data in SFT | Python validation loss rose ~30%, but the reference loss on the coding tasks fell on 33/40 (`s-50m`) and 37/40 (`m-100m`) tasks: the rise is a distribution shift, not lost coding ability |
| 2026-10-18 | Code eval on an A100, not an L4 | Without a KV cache, generation is launch-bound; one evaluation took ~10 min on an A100 and was too slow on an L4 |
| 2026-10-18 | Phase 3 complete on `s-50m` and `m-100m` | Results in README |
| 2026-10-18 | Phase 4 task: review rating, 1-5 stars, in Turkish (SentiTurca e-commerce, CC-BY-SA-4.0) and English (Yelp Review Full, Yelp Dataset Agreement) | Measurable without generation, unrelated to the RL code rewards, and the same task in a language the model saw 60% of and one it saw 10% of. Yelp is a different domain (businesses vs products), so absolute scores are not compared across languages, only differences between methods and sizes |
| 2026-10-18 | English sets sampled to the Turkish sizes; the same 512-token prompt limit in both | Data size and length filter are not variables |
| 2026-10-18 | Start from the SFT models; answer is one digit in the chat format; score by the most likely digit token | RL starts from the same models; one forward pass per review, exact and fast |
| 2026-10-18 | Full fine-tuning vs LoRA (all attention and expert layers, alpha 16): LR grid of 3 each on Turkish validation macro-F1, then ranks 4 / 16 / 64, then English with the Turkish LRs; test sets opened once | Matched data and steps; the selection metric is fixed before the runs; English reuses the Turkish LRs so only the language changes |
| 2026-10-18 | Packed fine-tuning windows hold several reviews, which attend to earlier reviews in the same window | Same packing as SFT; scoring sees one review alone. A train/test difference to keep in mind; per-document attention masks are not implemented |
| 2026-10-18 | English training batch is 112 windows vs 32 in Turkish | English reviews are ~3.2× longer; equal reviews per step (~525) and equal steps (~142) keep the comparison about language, not step count |
| 2026-10-18 | LR grids extended by one value (full 3e-4, LoRA 3e-3) before opening the test sets | All four best LRs were at the top edge of the first grid; with the extension the LoRA gap at `m-100m` shrank from 0.051 to 0.012 (val), so the first grid would have overstated full fine-tuning's advantage. Three choices are still at the new edge; not extended further |
| 2026-10-18 | Phase 4 comparison complete; test sets opened once for 8 selected runs plus baselines | Results in README; receipts in `artifacts/finetune/` |
| 2026-10-18 | Rule for any later fine-tuning that must keep earlier skills: select on task score **and** forgetting (SFT val loss, code reference loss), prefer the lower LR when scores are close, and mix in replay of earlier data | Selecting on F1 alone picked the LRs that erased coding ability (code reference loss 0.36 -> 7-9 nats) for gains of <0.01 F1 |
| 2026-10-21 | Joint fine-tuning per size before RL: Turkish + English reviews + SFT replay (1/3 of windows), full fine-tuning, LR 3e-5 or 1e-4 | One model must rate reviews in both languages and keep chat and code; notebook 05 showed full fine-tuning dominates LoRA at matched forgetting and that single-task fine-tuning erases earlier skills |
| 2026-10-21 | Loss averaged per window for the mix (`loss_per_window`) | A chat window has ~600 learned tokens, a review window 10-30; with the token mean the replay would be ~95% of the loss |
| 2026-10-21 | Selection for the joint model: best mean val macro-F1 among runs with SFT val loss <= +10% (each language) and code reference loss <= +25%; else the least-forgetting run | The rule from Phase 4, made concrete before the runs |
| open | Batched, KV-cached generation before RL | RL needs far more generation than the code eval |

## Pretraining, measured (A100 40 GB, full training loop with gradient accumulation)

| Rung | Tokens/s | Tokens | Training hours | Final val loss (mixed) |
|---|---:|---:|---:|---:|
| `s-50m` | 166,499 | 1.06B | 1.77 | 2.948 |
| `m-100m` | 99,714 | 1.92B | 5.35 | 2.636 |

The full loop ran 1.2–1.4× faster than the Phase 0 benchmark below (8-way gradient accumulation amortises per-step overhead), so the estimates below were conservative. Pretraining both rungs used roughly 45–50 units including setup, evaluation, uploads, and one restart (estimate from session hours).

## SFT, measured (A100 40 GB)

| Rung | Tokens | Steps | Training hours | Tokens/s |
|---|---:|---:|---:|---:|
| `s-50m` | 406.6M | 3,102 | 0.68 | 167,311 |
| `m-100m` | 406.6M | 3,102 | 1.22 | 92,296 |

About 95 of 300 units were used through Phase 3 (Colab balance), including pretraining, SFT, the code evaluation, and failed or slow sessions (the L4 evaluation attempt).

## Fine-tuning, measured (A100 40 GB, no `torch.compile`)

| Run type | Training time per epoch | Tokens/s |
|---|---:|---:|
| `s-50m`, Turkish (4.6M tokens), full / LoRA | 78 s / 111 s | ~60k / ~42k |
| `m-100m`, Turkish, full / LoRA | 108 s / 153 s | ~43k / ~30k |
| `s-50m`, English (16.1M tokens), full / LoRA | 263 s / 388 s | ~62k / ~41k |
| `m-100m`, English, full / LoRA | 369 s / 530 s | ~43k / ~30k |

31 runs plus baselines and the test took about 2.5 hours of A100 time including scoring and checkpoint saves, roughly 25–30 units (estimate from session hours).

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

- **Padding in SFT.** Packed SFT windows are 76% full; a quarter of SFT compute goes to padding. Best-fit packing or variable-length batches would recover most of it.
- **Replay and SFT learning rate.** Does mixing a little pretraining data into SFT, or a lower LR, reduce forgetting without losing the coding gains?

- **Double residual.** HybridBlock returns `x + delta` and HyperConnection adds it to the
  stream again, so the input is counted twice per layer (inherited from the original model,
  kept for parity). Ablation: blocks return only `delta`. Does it change loss or stability?
- **RoPE base.** 1,000 vs 10,000 vs 100,000 at 1024-token context.