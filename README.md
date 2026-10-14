# GLM Ladder From Scratch

A family of GLM-5.3-Flash-inspired mixture-of-experts language models at 25M, 50M, 100M, and 200M parameters, pretrained from random initialization, post-trained, fine-tuned, and improved with reinforcement learning, entirely on Google Colab GPUs.

This project extends [vukrosic/glm-5.3-flash-from-scratch](https://github.com/vukrosic/glm-5.3-flash-from-scratch), which showed that executable-reward RL can teach a 25.7M model narrow coding behaviors. The central question here is what happens to that result as the model grows.

> **Status: Phase 0 (infrastructure).** No training results yet. Every number below is a design target or an estimate, and is labeled as such. Measured results will replace them as receipts are produced.

## Research question

> With the architecture, training pipeline, and evaluation protocol held fixed, how do the gains from executable-reward reinforcement learning change from 25M to 200M parameters, and does RL interference on untrained task families shrink or grow with scale?

Secondary questions, each run as a controlled ablation:

- Does a working MoE load-balancing loss change pretraining quality? (The original's balance loss had no gradient; see below.)
- Byte-level versus BPE tokenization at equal compute.
- RLOO versus GRPO versus DPO under a matched rollout budget.
- Full fine-tuning versus LoRA on the same downstream task.

## The ladder

All rungs keep the original rhythm: three linear-attention blocks then one sparse-attention block, 8 routed experts with top-2 routing plus one shared expert, and four residual streams. Only width, depth, vocabulary, and context change.

| Rung | Width × layers | Heads | Total params | Active params/token | Tokenizer | Context |
|---|---:|---:|---:|---:|---|---:|
| `baseline-25m` | 192 × 12 | 6 | 25,730,592 | 9,805,344 | bytes (260) | 192 |
| `s-50m` | 384 × 12 | 6 | 53,229,024 | 26,686,944 | BPE (16,384) | 1024 |
| `m-100m` | 512 × 16 | 8 | 96,027,264 | 48,841,344 | BPE (16,384) | 1024 |
| `l-200m` | 640 × 18 | 10 | 199,345,424 | 93,177,104 | BPE (16,384) | 1024 |

Parameter counts come from the closed-form formula in [`glm_ladder/config.py`](glm_ladder/config.py) and are checked against the PyTorch modules by the test suite. The baseline row is identical to the original repository.

## What changed from the original model

The rewritten model keeps the original parameter names, so original checkpoints load without conversion. A parity test loads the original model's weights into the new one and requires the same logits.

| Issue in the original | Effect | Fix |
|---|---|---|
| Expert usage computed from integer counts | The router balance loss had **zero gradient** and did nothing | Switch Transformer auxiliary loss (routed fraction × mean router probability); a test shows the router now receives gradient and the original did not |
| Linear attention stored a d × d state per position | Memory grows as batch × time × heads × d², which does not fit at 1024 tokens and 200M parameters | Chunked linear attention: one state per chunk, identical output (tested against the original formula in float64) |
| Sparse attention rebuilt gather indices with Python loops every forward pass | CPU-bound at long context | Cached boolean mask with PyTorch's fused attention kernel; the mask is tested position-by-position against the original pattern |
| Linear-attention prefix sums ran in bf16 under autocast | Precision loss over long prefixes | Attention core runs in float32 |
| Uniform 0.02 initialization at every depth | Residual variance grows with depth | Optional GPT-2 style scaled init of residual projections (on for ladder rungs, off for the baseline) |

The sparse layers still compute a full masked score matrix. At 1024 tokens that is fast, but it is not a sub-quadratic kernel.

## Tokenizer

A 16,384-token byte-level BPE (Hugging Face `tokenizers`, GPT-2 pre-tokenization) trained on a
200 MB sample with the same mix as the pretraining data: 60% English (FineWeb-Edu),
30% Python (codeparrot-clean), 10% Turkish (FineWeb2-HQ). Ids fit in uint16.
Sixteen special tokens are reserved up front (document separator, padding, chat roles,
ten spares) because the vocabulary cannot change after pretraining.

Held-out compression (UTF-8 bytes per token; higher is better; the original byte tokenizer scores 1.0):

| Text | With 10% Turkish | Same English + code, no Turkish |
|---|---:|---:|
| English | 4.103 | 4.169 |
| Python | 3.468 | 3.513 |
| Turkish | **3.507** | 1.943 |

Adding 10% Turkish to the tokenizer sample makes Turkish text 1.8× shorter in tokens,
at a cost of 1.6% on English and 1.3% on Python. Both tokenizers saw identical English and
code documents, so the difference is attributable to the Turkish data.

Known limitation: GPT-2 pre-tokenization keeps digit runs together (305 number tokens,
up to 8 digits), which makes arithmetic harder than digit-split schemes.

Receipt: [`artifacts/receipts/tokenizer-16k.json`](artifacts/receipts/tokenizer-16k.json).
Reproduce: `python scripts/train_tokenizer.py --compare-without-turkish`.

## Plan

| Phase | Work | Exit criterion |
|---|---|---|
| 0. Infrastructure | Rewritten model, parity tests, throughput benchmark | Tests pass on Colab; measured tokens/s for every rung |
| 1. Data | BPE tokenizer, mixed English + Python corpus, sharded and hashed | Deterministic, resumable data stream |
| 2. Pretraining | Baseline reproduction, then S → M → L | Loss curves, held-out perplexity, receipts |
| 3. Post-training | Supervised fine-tuning on instruction data | Instruction-following eval before vs after |
| 4. Fine-tuning | Full fine-tuning vs LoRA on one task (task to be chosen) | Matched-compute comparison |
| 5. Reinforcement learning | Reproduce RLOO, then GRPO, DPO, reward and KL ablations at every rung | Locked confirmation-set results with paired statistics |
| 6. Report | Scaling analysis, negative results, limitations | `REPORT.md` |

The evaluation protocol follows the original repository: decisions are made on a dev split, a separate confirmation split is opened once, and every comparison reports paired exact tests or bootstrap intervals. Negative results are reported.

## Compute and storage

- **Compute:** Google Colab Pro (L4 and A100), 300 compute units in total. Pretraining is planned at roughly 70–80 units; measured throughput from Phase 0 will replace this estimate.
- **Resumable training:** Colab sessions can end at any time. Training state (weights, optimizer, scheduler, RNG, data position) is checkpointed every 20–30 minutes, and re-running a notebook resumes from the latest checkpoint.
- **Code and receipts:** this GitHub repository.
- **Checkpoints and tokenized data:** a Hugging Face Hub repository (too large for GitHub), saved as `safetensors` with SHA-256 hashes recorded in receipts here.

## Quick start

Notebooks are run from VS Code on a Colab GPU runtime through the Google Colab extension; nothing needs to be opened on the Colab website.

1. Open [`notebooks/00_setup_smoke_test.ipynb`](notebooks/00_setup_smoke_test.ipynb) in VS Code.
2. *Select Kernel → Colab → New Colab Server → GPU*.
3. Set `REPO_URL` in the second code cell and run all cells.

Local CPU tests:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Repository map

- [`glm_ladder/config.py`](glm_ladder/config.py): model configuration, ladder presets, closed-form parameter counts.
- [`glm_ladder/model.py`](glm_ladder/model.py): hybrid linear/sparse attention, MoE with shared expert, residual streams.
- [`glm_ladder/benchmark.py`](glm_ladder/benchmark.py): throughput, MFU, and compute-unit estimates per rung.
- [`tests/test_model.py`](tests/test_model.py): parity with the original model, kernel equivalence, causality, balance-loss gradient.
- [`tests/reference/original_model.py`](tests/reference/original_model.py): copy of the original model, used only by tests.
- [`notebooks/00_setup_smoke_test.ipynb`](notebooks/00_setup_smoke_test.ipynb): Colab setup, tests, throughput.
- [`PLAN.md`](PLAN.md): decision log and compute budget.

## Sources

- [vukrosic/glm-5.3-flash-from-scratch](https://github.com/vukrosic/glm-5.3-flash-from-scratch)
- [GLM-5.3-Flash official model card](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961), load-balancing loss
- [Transformers are RNNs (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236), linear attention

## License

MIT. Builds on vukrosic/glm-5.3-flash-from-scratch (MIT). Independent educational project, not affiliated with Z.ai.
