# GLM Ladder From Scratch

A family of GLM-5.3-Flash-inspired mixture-of-experts language models at 25M, 50M, 100M, and 200M parameters, pretrained from random initialization, post-trained, fine-tuned, and improved with reinforcement learning, entirely on Google Colab GPUs.

This project extends [vukrosic/glm-5.3-flash-from-scratch](https://github.com/vukrosic/glm-5.3-flash-from-scratch), which showed that executable-reward RL can teach a 25.7M model narrow coding behaviors. The central question here is what happens to that result as the model grows.

> **Status: Phase 3 (supervised fine-tuning) complete for `s-50m` and `m-100m`.** Both models are pretrained to 20 tokens per parameter ([results](#pretraining-results)) and then fine-tuned on chat data ([results](#post-training-sft-results)). After SFT, `m-100m` writes a function that passes its unit tests for 65% of 40 coding tasks within 10 tries. `baseline-25m` reproduction and `l-200m` are deferred until fine-tuning and RL are done on the two trained rungs. Every number in this README comes from a receipt in [`artifacts/`](artifacts/) unless it is labeled as an estimate.

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

## Pretraining corpus

2.2B tokens, tokenized once on a Colab CPU runtime and stored as uint16 shards (100M tokens each, cut on document boundaries) in a private Hugging Face dataset repository. Every shard's SHA-256 is recorded in a per-source manifest and checked before training.

| Source | Train tokens | Validation tokens | Mix weight (tokens) |
|---|---:|---:|---:|
| English (FineWeb-Edu `sample-10BT`) | 1,200,008,673 | 10,006,579 | 0.560 |
| Python (codeparrot-clean) | 700,033,862 | 10,003,441 | 0.331 |
| Turkish (FineWeb2-HQ `tur_Latn`) | 300,008,360 | 10,003,016 | 0.109 |

The mix is defined in bytes (60 / 30 / 10) and converted to token weights with the tokenizer's measured bytes per token. Each training sequence is drawn from one source with these probabilities, in a seeded order that is fully determined by the step number, so a resumed run sees exactly the data an uninterrupted run would. Neither run completes an epoch of any source.

Receipt: [`artifacts/receipts/corpus-manifest.json`](artifacts/receipts/corpus-manifest.json).

## Pretraining results

Both rungs use the same recipe; only the model and the token budget (20 tokens per parameter) change.

| Setting | Value |
|---|---|
| Batch | 256 sequences × 1024 tokens = 262,144 tokens per step |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 on matrices only, gradient clip 1.0 |
| Learning rate | 6e-4, 100 warmup steps, cosine decay to 6e-5 |
| MoE balance loss | coefficient 0.01 |
| Precision | bf16 autocast, float32 linear-attention core, TF32, `torch.compile` |
| Hardware | one A100 40 GB (Colab Pro) |

| | `s-50m` | `m-100m` |
|---|---:|---:|
| Total / active parameters | 53.2M / 26.7M | 96.0M / 48.8M |
| Training tokens | 1.06B | 1.92B |
| Steps | 4,044 | 7,325 |
| Training time (A100) | 1.77 h | 5.35 h |
| Mean throughput | 166,499 tok/s | 99,714 tok/s |
| **Validation loss, mixed** (nats/token) | 2.948 | **2.636** |
| English (nats/token) | 3.541 | 3.220 |
| Python | 1.771 | 1.488 |
| Turkish | 3.479 | 3.126 |

Losses per token are not comparable across languages, because a token covers a different amount of text in each. Dividing by ln 2 and by the tokenizer's bytes per token gives **bits per UTF-8 byte**, which is:

| Bits per byte | `s-50m` | `m-100m` | Relative gain |
|---|---:|---:|---:|
| English | 1.245 | 1.132 | 9.1% |
| Python | 0.737 | 0.619 | **16.0%** |
| Turkish | 1.431 | 1.286 | 10.2% |

![Validation curves of s-50m and m-100m](artifacts/plots/pretraining.png)

What the numbers show:

- **Python gains the most from scale**, relative to its starting point. Code has long-range structure (definitions used far later, nested blocks) that a larger model captures better.
- **Turkish is the hardest source per byte** for both models, and gains slightly more than English. It is 11% of the training tokens and morphologically rich.
- **Model size and data grew together** (1.8× each), so the gap is the effect of compute-optimal scaling, not of size alone. One matched-token point exists: at 1.05B tokens, mid-schedule and with its learning rate still high, `m-100m` was at 2.766 against `s-50m`'s final 2.948.
- **Diminishing returns:** 3× the GPU time bought 0.31 nats.
- **Initial loss** was 9.87 for `s-50m` and 10.67 for `m-100m`, above ln(16,384) = 9.70. With tied embeddings and near-zero blocks at initialization, the output logit of the *input* token is about 0.02 × width, so the model starts out predicting "the same token again" with probability 12% (`s-50m`) or 62% (`m-100m`). This accounts for both values and disappears within ~10 steps.

Receipts: [`pretrain-s50m-001.json`](artifacts/receipts/pretrain-s50m-001.json), [`pretrain-m100m-001.json`](artifacts/receipts/pretrain-m100m-001.json). Plot: `python scripts/plot_pretraining.py --runs s50m-001 m100m-001`.

### Samples

Same prompt, seed 0, temperature 0.8, top-k 50, 80 new tokens. The prompt is in `[[ ]]`. One sample per prompt is anecdotal; the validation loss above is the measurement. All samples: [`artifacts/samples/`](artifacts/samples/).

| | `s-50m` | `m-100m` |
|---|---|---|
| English | `[[Photosynthesis is the process by which]] the molecules (does and dyes begin producing their own cells) of proteins. When all in a molecule goes through a process called the cell, it gets made up of RNA (the molecule) and then it is called the cell.` | `[[Photosynthesis is the process by which]] the molecules (or the cell) receive and store energy. To keep up the work of producing energy, the cell is made to continue to grow, to provide energy for the plant.` |
| Turkish | `[[Yapay zeka, günümüzde]] yer alan üçgende yer alan buz, buz, deniz ve deniz kapakları ile diğer bir yerden yapılmış ve en çok kullanılan bir gaban.` | `[[Yapay zeka, günümüzde]] hala aynı teknolojinin en yaygın yaygın kullanımıyla ilgili olmakla birlikte, zeka bilimleri uzmanları, bilişsel ve zihinsel gelişimlerinin, özellikle de bu teknolojinin gelişmesini amaçlayan teknolojinin…` |

Both models write fluent-looking text with no reliable facts and fall into repetition loops. The visible difference is topical coherence: `s-50m` drifts away from the prompt within a sentence (artificial intelligence → ice and seas), while `m-100m` stays on topic for a paragraph. In Python both produce correctly indented, syntactically plausible code without working logic.

A practical note on prompts: ending a code prompt with `"\n"` made `s-50m` drop the indentation, because newline plus indentation is usually a single BPE token, and a prompt that ends on a bare `"\n"` forces a token boundary the model rarely saw. Without the trailing newline both models indent correctly.

### Engineering notes from the runs

- **Resume is bit-exact on the GPU.** A run killed at step ~990 and resumed from its step-942 checkpoint logged loss 3.4936 and gradient norm 0.31 at step 950, identical to the uninterrupted run; the CPU test suite checks the same property on final weights.
- **`torch.compile` and data-dependent shapes.** When the router sends no token of a micro-batch to some expert, the compiled kernel for that expert receives a zero-sized input and fails with `ZeroDivisionError`. Every expert now also processes one zero row with weight 0, which leaves outputs and gradients unchanged.
- **`forward or model` on a compiled module** calls `len()` on the wrapper and raises `TypeError`; use `model if forward is None else forward`.
- **Stale kernels.** In VS Code + Colab, interrupt did not stop a running cell, and *Restart Kernel* left the old process training on the GPU, which caused an out-of-memory error in the new kernel. The training notebook now kills stale kernels before touching CUDA.

## Post-training (SFT) results

Supervised fine-tuning turns the pretrained models into chat models. Both rungs start from their final pretraining checkpoint and get the same data and recipe.

### Data and chat template

| Source | Share of SFT tokens | Notes |
|---|---:|---|
| [`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | ~90% | English conversations, made for small models; includes coding requests |
| [`turkish-nlp-suite/InstrucTurca`](https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca) | ~10% | Turkish instructions, machine-translated from English datasets |

The 90 / 10 split matches the share of Turkish in pretraining. Every conversation is written with chat special tokens that were reserved in the tokenizer before pretraining:

```text
<|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>
```

- **Only the assistant learns.** The loss is computed on the assistant's tokens and its closing `<|end|>`, and nowhere else (targets set to -100). The model learns to answer, and to stop, but not to imitate the user.
- **Whole conversations are packed** into 1,025-token windows (1,024 inputs + 1 shifted target), next-fit, separated by the document token and padded at the end. A conversation is never split across windows, so the model never has to answer a question it cannot see.
- **Long conversations are cut at a turn boundary**, after the last complete assistant answer that fits. Dropping every conversation longer than a window would have thrown away 46% of the English data, biased towards the long, detailed answers; with the cut, about 2.5% are skipped and 43% are shortened.
- The result is 397,049 training windows (406.6M tokens), one epoch. Token ids and masks are stored as uint16 / uint8 arrays with SHA-256 hashes in the dataset repo.

Reproduce: `python scripts/prepare_sft.py`. Counts and hashes: [`artifacts/receipts/sft-data.json`](artifacts/receipts/sft-data.json).

### Recipe

Same training loop as pretraining (resumable, bit-exact), with a fresh AdamW state:

| Setting | Value |
|---|---|
| Batch | 128 sequences × 1024 tokens = 131,072 tokens per step |
| Learning rate | 1e-4 (1/6 of pretraining), 50 warmup steps, cosine decay to 1e-5 |
| Length | 1 epoch = 3,102 steps |
| Training time (A100) | 0.68 h (`s-50m`), 1.22 h (`m-100m`) |

The learning rate and epoch count were not tuned.

### What SFT costs: forgetting

The pretraining validation sets were evaluated again after SFT (nats per token, lower is better):

| | `s-50m` before → after | Change | `m-100m` before → after | Change |
|---|---:|---:|---:|---:|
| English (FineWeb-Edu) | 3.541 → 4.002 | +13.0% | 3.220 → 3.596 | +11.7% |
| Python (codeparrot) | 1.771 → 2.319 | **+30.9%** | 1.488 → 1.932 | **+29.9%** |
| Turkish (FineWeb2) | 3.479 → 3.899 | +12.1% | 3.126 → 3.493 | +11.7% |
| SFT validation, English | | 1.915 | | 1.660 |
| SFT validation, Turkish | | 2.646 | | 2.348 |

- **Relative forgetting is nearly the same at both sizes.** Doubling the model did not make it more robust to SFT at this scale; both lose 12–13% on prose and 30% on raw Python.
- **Python suffers most.** The SFT data contains very little raw source code, so the model drifts away from it the most. The next section shows that this loss rise does *not* mean the model got worse at writing code.

### Code evaluation

To measure coding ability directly, 40 small Python tasks with unit tests ([`evals/code_tasks.jsonl`](evals/code_tasks.jsonl), e.g. `is_even`, `reverse_words`, `running_sum`) were written for this project. A test of the task set checks that every reference solution passes and that `return None` fails every task. Each generated function runs in a subprocess with a timeout, after a deny-list check (no `os`, `sys`, `subprocess`, `open(`, `eval(`, ...). This is a basic guard, not a sandbox; RL at scale will run generated code in a container.

The same tasks are asked in two ways:

- **completion:** the model continues `def name(args):` and its docstring, as in pretraining data.
- **chat:** the task is asked as a user request; the code is taken from the answer's code block. Only possible after SFT.

Per task, the model generates one greedy answer and 10 samples at temperature 0.8 (top-k 50, at most 160 tokens). pass@k uses the unbiased estimator of [Chen et al. (2021)](https://arxiv.org/abs/2107.03374), the probability that at least one of k samples passes. A pass rate near zero cannot show progress, so the evaluation also measures the **reference loss**, the model's loss on the reference solution, which is informative even when nothing passes.

| Model | Prompt | Reference loss | Greedy pass@1 | Sampled pass@1 | pass@5 | pass@10 |
|---|---|---:|---:|---:|---:|---:|
| `s-50m` pretrained | completion | 1.401 | 5.0% | 0.5% | 2.5% | 5.0% |
| `s-50m` + SFT | completion | 0.895 | 12.5% | 6.5% | 16.7% | 20.0% |
| `s-50m` + SFT | chat | 0.471 | 37.5% | 14.0% | 38.3% | 50.0% |
| `m-100m` pretrained | completion | 1.142 | 5.0% | 1.5% | 6.4% | 10.0% |
| `m-100m` + SFT | completion | 0.586 | 35.0% | 21.5% | 44.8% | 55.0% |
| `m-100m` + SFT | chat | 0.362 | **47.5%** | **29.0%** | **55.3%** | **65.0%** |

Reference losses in chat mode are not comparable with completion mode: in chat mode the `def` line is part of the scored answer, and it is easy to copy from the request.

What the numbers show:

- **SFT improved coding despite the higher Python loss.** On the same completion prompts, the reference loss went down on 33 of 40 tasks for `s-50m` and 37 of 40 for `m-100m` (two-sided sign test, p ≈ 4 × 10⁻⁵ and p ≈ 2 × 10⁻⁸). The Python validation loss rose because it measures random GitHub files, a different distribution from short, documented functions. Mixing pretraining data back into SFT ("replay") was considered and rejected on this evidence.
- **The scale effect appears after SFT.** Both pretrained models pass 5% of the tasks greedily, so pass rates cannot separate them, but the reference loss already could (1.14 vs 1.40). After SFT, `m-100m` passes 2.8× more tasks than `s-50m` in completion mode (greedy 35% vs 12.5%).
- **The larger model depends less on the prompt format.** Asking in chat instead of completion triples `s-50m`'s greedy pass rate (12.5% → 37.5%) but raises `m-100m`'s by only 1.4× (35% → 47.5%).
- **There is signal for RL.** pass@10 (65%) is well above pass@1 (29%): for many tasks the model can write a passing solution but does not do so reliably. That is the gap an executable-reward RL phase can close.
- **Uncertainty.** 40 tasks are few. A greedy pass rate moves 2.5 points per task, and the difference between two greedy results is not significant on its own; the sampled pass@1 (400 samples per row) and the per-task reference losses are the stronger evidence.

Reproduce: [`notebooks/04_code_eval.ipynb`](notebooks/04_code_eval.ipynb) (all six rows, about an hour on an A100), or one row on CPU with `python scripts/run_code_eval.py --run sft-m100m-001 --mode chat`. Receipts with every generated function and test result: [`artifacts/evals/`](artifacts/evals/).

### Chat samples

Temperature 0.7, top-k 40, seed 0. From [`artifacts/samples/sft-s50m-001.md`](artifacts/samples/sft-s50m-001.md):

> **User:** Write a Python function that returns the sum of a list of numbers.
>
> **`s-50m` + SFT:**
> ````text
> ```python
> def sum_numbers(numbers):
>     return sum(numbers)
> ```
>
> You can use this function like this: ...
> ````

> **User:** What is the capital of France?
>
> **`s-50m` + SFT:** The capital of France is Paris. Paris is well known for its historical and regional charm, with a focus on the Mediterranean region. The capital, Paris, is Paris, and the capital, with a focus on the French government ...

The model has learned the format of an assistant: it answers the question that was asked, uses code blocks and lists, and ends short answers with `<|end|>` by itself. Its content is unreliable and repetitive after the first sentence, and its Turkish answers are fluent but off-topic, as expected at 50M parameters with 10% Turkish data.

Talk to a model on CPU: `python scripts/talk.py --run sft-m100m-001` (multi-turn; `/reset`, `/quit`).

### Engineering notes from post-training

- **A generator consumed twice.** The masked evaluation first passed a generator to a function that iterates its input twice; the second pass saw nothing, so masking was silently skipped. A test now checks that the evaluation averages over exactly the learned targets.
- **Script names must not shadow modules.** `scripts/chat.py` next to `glm_ladder/chat.py` let the editor's import refactoring overwrite the library module. Scripts now have unique names (`talk.py`, `run_code_eval.py`).
- **Token boundaries matter for evaluation.** A completion prompt ending in `"\n"` broke indentation (see [samples](#samples)), so code prompts end on the docstring. The reference loss encodes prompt and solution together and checks that the prompt's tokens are a prefix of the result.
- **Generation is the bottleneck.** Without a KV cache, each new token re-runs the full sequence, and the small MoE model is bound by kernel launches. One evaluation took ~10 minutes on an A100 and was too slow on an L4. RL needs much more generation than this, so batched, cached generation comes before RL.

## Plan

| Phase | Work | Exit criterion | Status |
|---|---|---|---|
| 0. Infrastructure | Rewritten model, parity tests, throughput benchmark | Tests pass on Colab; measured tokens/s for every rung | Done |
| 1. Data | BPE tokenizer, mixed English + Python + Turkish corpus, sharded and hashed | Deterministic, resumable data stream | Done |
| 2. Pretraining | S → M; baseline reproduction and L deferred | Loss curves, held-out loss, receipts | Done for S and M |
| 3. Post-training | Supervised fine-tuning on chat data | Eval before vs after | Done for S and M: forgetting, executable code eval, chat samples |
| 4. Fine-tuning | Full fine-tuning vs LoRA on one task (task to be chosen) | Matched-compute comparison | Next |
| 5. Reinforcement learning | Reproduce RLOO, then GRPO, DPO, reward and KL ablations at every rung | Locked confirmation-set results with paired statistics | |
| 6. Report | Scaling analysis, negative results, limitations | `REPORT.md` | |

The evaluation protocol follows the original repository: decisions are made on a dev split, a separate confirmation split is opened once, and every comparison reports paired exact tests or bootstrap intervals. Negative results are reported.

## Compute and storage

- **Compute:** Google Colab Pro (L4 and A100), 300 compute units in total. Pretraining both rungs took about 7 hours of training on an A100, roughly 45–50 units including setup, evaluation, and checkpoint uploads (estimate from session hours). After SFT and the GPU code evaluation, about 95 of the 300 units have been used (Colab balance).
- **Resumable training:** Colab sessions can end at any time. Training state (weights, AdamW moments, RNG, step, data position) is pushed to the Hub every 15–25 minutes, and re-running the notebook resumes from the latest checkpoint with the same data order. The two most recent resume checkpoints and the final weights are kept.
- **Code and receipts:** this GitHub repository.
- **Checkpoints and tokenized data:** a Hugging Face Hub repository (too large for GitHub), saved as `safetensors` with SHA-256 hashes recorded in receipts here.

## Quick start

Notebooks are run from VS Code on a Colab GPU runtime through the Google Colab extension; nothing needs to be opened on the Colab website.

| Notebook | Runtime | What it does |
|---|---|---|
| [`00_setup_smoke_test`](notebooks/00_setup_smoke_test.ipynb) | GPU | Tests, throughput benchmark |
| [`01_tokenize_corpus`](notebooks/01_tokenize_corpus.ipynb) | CPU | Tokenizes the corpus into shards on the Hub (resumable) |
| [`02_pretrain`](notebooks/02_pretrain.ipynb) | A100 | Pretrains one rung; set `RUN` and run all cells (resumable) |
| [`03_sft`](notebooks/03_sft.ipynb) | A100 | Supervised fine-tuning of one pretrained rung (resumable) |
| [`04_code_eval`](notebooks/04_code_eval.ipynb) | A100 | Code evaluation of every model and prompt mode; uploads receipts |

In VS Code: open a notebook, *Select Kernel → Colab → New Colab Server*, pick the runtime, run all cells. The Hugging Face token is typed into a hidden prompt and never stored.

Local CPU tests and tools (Python 3.12):

```bash
conda create -n glm-ladder python=3.12 && conda activate glm-ladder
pip install -r requirements.txt
python -m pytest -q

# Talk to a trained model on CPU (downloads only the weights), rebuild a receipt, plot runs
python scripts/sample.py --run m100m-001 --prompt "def fibonacci(n):"
python scripts/pretrain_receipt.py --run m100m-001
python scripts/plot_pretraining.py --runs s50m-001 m100m-001

# Chat with a fine-tuned model, evaluate it on the coding tasks, fetch GPU eval receipts
python scripts/talk.py --run sft-m100m-001
python scripts/run_code_eval.py --run sft-m100m-001 --mode chat
python scripts/fetch_evals.py
```

## Repository map

- [`glm_ladder/config.py`](glm_ladder/config.py): model configuration, ladder presets, closed-form parameter counts.
- [`glm_ladder/model.py`](glm_ladder/model.py): hybrid linear/sparse attention, MoE with shared expert, residual streams.
- [`glm_ladder/benchmark.py`](glm_ladder/benchmark.py): throughput, MFU, and compute-unit estimates per rung.
- [`glm_ladder/toy_bpe.py`](glm_ladder/toy_bpe.py), [`tokenizer.py`](glm_ladder/tokenizer.py): BPE from scratch for learning, and the production tokenizer.
- [`glm_ladder/data_sources.py`](glm_ladder/data_sources.py), [`shards.py`](glm_ladder/shards.py), [`corpus.py`](glm_ladder/corpus.py): streaming sources, uint16 shards, resumable upload to the Hub.
- [`glm_ladder/loader.py`](glm_ladder/loader.py): deterministic, resumable mixed-source batches.
- [`glm_ladder/train.py`](glm_ladder/train.py): one optimizer step with gradient accumulation, LR schedule, AdamW groups.
- [`glm_ladder/checkpoint.py`](glm_ladder/checkpoint.py), [`runs.py`](glm_ladder/runs.py): checkpoints with SHA-256, metrics log, checkpoint store on the Hub.
- [`glm_ladder/evaluation.py`](glm_ladder/evaluation.py): validation loss per source, bits per byte.
- [`glm_ladder/pretrain.py`](glm_ladder/pretrain.py): the pretraining loop with evaluation, checkpoints, and exact resume.
- [`glm_ladder/generate.py`](glm_ladder/generate.py): temperature and top-k sampling.
- [`glm_ladder/chat.py`](glm_ladder/chat.py): chat template and the assistant-only loss mask.
- [`glm_ladder/sft_data.py`](glm_ladder/sft_data.py): packing whole conversations into windows; the SFT data source.
- [`glm_ladder/code_eval.py`](glm_ladder/code_eval.py): code tasks, running unit tests, pass@k, batched sampling, reference loss.
- [`evals/code_tasks.jsonl`](evals/code_tasks.jsonl): the 40 coding tasks with reference solutions and tests.
- [`tests/`](tests/): parity with the original model, kernel equivalence, causality, balance-loss gradient, loader determinism, bit-exact resume of the full training loop, chat masks and packing, the code-eval task set.
- [`tests/reference/original_model.py`](tests/reference/original_model.py): copy of the original model, used only by tests.
- [`scripts/`](scripts/): tokenizer training, benchmark, sampling, receipts, plots, SFT data preparation, chat, code evaluation.
- [`artifacts/`](artifacts/): receipts, code-eval results, samples, plots.
- [`PLAN.md`](PLAN.md): decision log and compute budget.

## Sources

- [vukrosic/glm-5.3-flash-from-scratch](https://github.com/vukrosic/glm-5.3-flash-from-scratch)
- [GLM-5.3-Flash official model card](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961), load-balancing loss
- [Transformers are RNNs (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236), linear attention
- [SmolLM2 (Allal et al., 2025)](https://arxiv.org/abs/2502.02737), source of the `smol-smoltalk` SFT data
- [InstrucTurca](https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca), Turkish SFT data
- [Evaluating Large Language Models Trained on Code (Chen et al., 2021)](https://arxiv.org/abs/2107.03374), unbiased pass@k

## License

MIT. Builds on vukrosic/glm-5.3-flash-from-scratch (MIT). Independent educational project, not affiliated with Z.ai.
