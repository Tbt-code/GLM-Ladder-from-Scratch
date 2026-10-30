# GLM Ladder From Scratch

A ladder of GLM-5.3-Flash-inspired mixture-of-experts language models, pretrained from random initialization, post-trained, fine-tuned, and improved with reinforcement learning, entirely on Google Colab GPUs. Four sizes were designed (25M, 50M, 100M, 200M parameters); two of them, **`s-50m` and `m-100m`, went through every phase** and are the project's output.

This project extends [vukrosic/glm-5.3-flash-from-scratch](https://github.com/vukrosic/glm-5.3-flash-from-scratch), which showed that executable-reward RL can teach a 25.7M model narrow coding behaviors. The central question here is what happens to that result as the model grows.

> **Status: complete for `s-50m` and `m-100m`.** Both models are pretrained to 20 tokens per parameter ([results](#pretraining-results)), fine-tuned on chat data ([results](#post-training-sft-results)), compared under full fine-tuning and LoRA on review rating in Turkish and English ([results](#fine-tuning-results-full-vs-lora)), jointly fine-tuned into one model per size that keeps chat and code ([results](#joint-fine-tuning-one-model-per-size)), and finally trained with RL on executable code rewards ([results](#reinforcement-learning-on-code)). RL raised the pass rate on held-out tasks of the trained kinds from 0.9% to 40% (`s-50m`) and from 6.7% to 71% (`m-100m`), and on kinds of task never trained on the gain was 8× larger at the larger size; review rating, chat losses, and code reference loss barely moved. See [Final models](#final-models) for one table across all phases. **`baseline-25m` (reproduction of the original) and `l-200m` are deferred, not dropped: they can be done later if compute allows**; the code, configs, and notebooks already support both. Every number in this README comes from a receipt in [`artifacts/`](artifacts/) unless it is labeled as an estimate.

## Research question

> With the architecture, training pipeline, and evaluation protocol held fixed, how do the gains from executable-reward reinforcement learning change from 25M to 200M parameters, and does RL interference on untrained task families shrink or grow with scale?

What was answered: the question was measured at two rungs, 50M and 100M ([RL results](#reinforcement-learning-on-code)). Extending it to 25M and 200M is the deferred part.

Secondary questions, each planned as a controlled ablation:

| Question | Status |
|---|---|
| Full fine-tuning versus LoRA on the same downstream task | Done ([results](#fine-tuning-results-full-vs-lora)) |
| RL with versus without the KL penalty | Done on `s-50m` ([results](#kl-ablation-s-50m)) |
| Does a working MoE load-balancing loss change pretraining quality? (The original's balance loss had no gradient; see below.) | Not run: the fix is in, the ablation is deferred |
| Byte-level versus BPE tokenization at equal compute | Not run, deferred |
| RLOO versus GRPO versus DPO under a matched rollout budget | Not run: only RLOO was trained, deferred |

## The ladder

All rungs keep the original rhythm: three linear-attention blocks then one sparse-attention block, 8 routed experts with top-2 routing plus one shared expert, and four residual streams. Only width, depth, vocabulary, and context change.

| Rung | Width × layers | Heads | Total params | Active params/token | Tokenizer | Context | Status |
|---|---:|---:|---:|---:|---|---:|---|
| `baseline-25m` | 192 × 12 | 6 | 25,730,592 | 9,805,344 | bytes (260) | 192 | deferred |
| `s-50m` | 384 × 12 | 6 | 53,229,024 | 26,686,944 | BPE (16,384) | 1024 | **all phases done** |
| `m-100m` | 512 × 16 | 8 | 96,027,264 | 48,841,344 | BPE (16,384) | 1024 | **all phases done** |
| `l-200m` | 640 × 18 | 10 | 199,345,424 | 93,177,104 | BPE (16,384) | 1024 | deferred |

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
- **Long conversations are cut at a turn boundary**, after the last complete assistant answer that fits. Dropping every conversation longer than a window would have thrown away 46% of the English data, biased towards the long, detailed answers; with the cut, about 2% are skipped and 43% are shortened. Turkish conversations are single-turn, so they cannot be cut; 6% were skipped as too long.
- The result is 397,049 training windows (357,344 English, 39,705 Turkish), one epoch. Windows are 76% full: they hold 307.9M conversation tokens, of which 239.0M are learned assistant tokens; the rest is padding. Token ids and masks are stored as uint16 / uint8 arrays with SHA-256 hashes in the dataset repo.

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

To measure coding ability directly, 40 small Python tasks with unit tests ([`evals/code_tasks.jsonl`](evals/code_tasks.jsonl), e.g. `is_even`, `reverse_words`, `running_sum`) were written for this project. A test of the task set checks that every reference solution passes and that `return None` fails every task. Each generated function runs in a subprocess with a timeout, after a deny-list check (no `os`, `sys`, `subprocess`, `open(`, `eval(`, ...). This is a basic guard, not a sandbox; for RL it was hardened with resource limits and an unprivileged user (see [Running generated code](#running-generated-code)).

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
- **Generation is the bottleneck.** Without a KV cache, each new token re-runs the full sequence, and the small MoE model is bound by kernel launches. One evaluation took ~10 minutes on an A100 and was too slow on an L4. RL needs much more generation than this, so batched, cached generation comes before RL (done: see [Fast generation](#fast-generation)).

## Fine-tuning results: full vs LoRA

Phase 4 asks how full fine-tuning compares with LoRA on one downstream task, at two sizes and in two languages. Both rungs start from their SFT checkpoints.

### Task and data

Rate a customer review from 1 to 5 stars, answering in the chat format with a single digit:

```text
<|user|>Rate the following review from 1 to 5 stars. Answer with the digit only.

Review: ...<|end|><|assistant|>4<|end|>
```

| Language | Source | Train / val / test | Star distribution |
|---|---|---:|---|
| Turkish | [SentiTurca](https://huggingface.co/datasets/turkish-nlp-suite/SentiTurca) e-commerce (Trendyol, Hepsiburada; CC-BY-SA-4.0) | 73,912 / 14,997 / 14,997 | skewed: 59% are 4 or 5 stars |
| English | [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) (Yelp Dataset Agreement) | 73,912 / 14,997 / 14,997 | balanced, ~20% each |

- **Same size in both languages.** The English sets are seeded samples with exactly as many reviews as the Turkish ones; English validation comes from the Yelp training split, disjoint from English train.
- **Same length rule.** Reviews whose prompt exceeds 512 tokens are skipped: 8 Turkish and ~7% of English reviews, so the English set leans slightly towards shorter reviews.
- **Equal reviews per step.** A Turkish review is ~60 tokens, an English one ~190, so a 1,024-token window holds ~16 or ~5 reviews. The batch is 32 windows in Turkish and 112 in English: both languages see ~525 reviews per step and ~142 steps per epoch.
- **Only the digit and `<|end|>` are learned.** Several reviews are packed per window and can attend to earlier reviews in the window; scoring sees one review alone.
- The domains differ (products vs. local businesses) and so do the label distributions, so **scores are compared within a language, not across languages.**

Reproduce: `python scripts/prepare_finetune.py`. Counts and hashes are in the data manifest.

### Scoring without generation

Each review is scored with one forward pass: the prediction is the most likely of the tokens "1" … "5" after `<|assistant|>` ([`glm_ladder/classify.py`](glm_ladder/classify.py)). The metric used for every decision is **macro-F1** (each star counts equally), because accuracy rewards always answering "4" in Turkish. Mean absolute error in stars is reported too. `label_mass`, the probability the model puts on the five digits at all, shows whether it follows the format.

### Baselines

| Test set | Majority class: acc / F1 | Zero-shot SFT model: acc / F1 | Zero-shot `label_mass` |
|---|---:|---:|---:|
| Turkish | 0.303 / 0.093 | 0.128 / 0.045 | 0.06 (`s-50m`), 0.12 (`m-100m`) |
| English | 0.209 / 0.069 | 0.196 / 0.067 | 0.02 |

Both SFT models answer "1" to every review in both languages, below the majority class: at this size they do not follow a "digit only" instruction, and they want to answer in words.

### Method and learning-rate sweep

Full fine-tuning and LoRA (rank 16, alpha 16, on every attention and expert projection; the router and the tied embedding are frozen) were trained for one epoch with the same data, steps, warmup (5%), cosine decay, and no weight decay. The learning rate was chosen on **Turkish validation macro-F1** from four values per method, and the English runs reuse the Turkish choice, so only the language changes. The first grid had three values; all four best values were at its top edge, so each grid was extended by one value before the test sets were opened.

| Turkish val macro-F1 (SFT Turkish val loss after fine-tuning) | | | | |
|---|---:|---:|---:|---:|
| **`s-50m` full**, LR | 1e-5: 0.444 (2.79) | 3e-5: 0.524 (3.07) | **1e-4: 0.575** (3.93) | 3e-4: 0.570 (7.26) |
| **`s-50m` LoRA r16**, LR | 1e-4: 0.313 (2.76) | 3e-4: 0.444 (3.05) | 1e-3: 0.535 (4.01) | **3e-3: 0.556** (6.94) |
| **`m-100m` full**, LR | 1e-5: 0.418 (2.42) | 3e-5: 0.558 (2.64) | 1e-4: 0.601 (3.43) | **3e-4: 0.609** (7.60) |
| **`m-100m` LoRA r16**, LR | 1e-4: 0.255 (2.42) | 3e-4: 0.413 (2.64) | 1e-3: 0.550 (3.84) | **3e-3: 0.597** (6.71) |

SFT Turkish validation loss before fine-tuning: 2.646 (`s-50m`), 2.348 (`m-100m`). Rank 4, 16, and 64 at the selected LoRA learning rate differ by less than 0.02 macro-F1 with no consistent direction (0.567 / 0.556 / 0.566 for `s-50m`, 0.587 / 0.597 / 0.576 for `m-100m`); rank is not what limits LoRA here.

### Test results

Selected runs, test sets opened once (14,997 reviews each, macro-F1):

| Model | Method (LR) | Trained params | Turkish model on Turkish test | English model on English test |
|---|---|---:|---:|---:|
| `s-50m` | full (1e-4) | 53.2M | **0.572** | **0.586** |
| `s-50m` | LoRA r16 (3e-3) | 3.4M (6%) | 0.568 | 0.550 |
| `m-100m` | full (3e-4) | 96.0M | **0.607** | **0.617** |
| `m-100m` | LoRA r16 (3e-3) | 5.4M (6%) | 0.593 | 0.610 |

Accuracy of the same runs: 0.610 / 0.608 / 0.637 / 0.624 in Turkish and 0.588 / 0.551 / 0.616 / 0.610 in English; mean absolute error 0.44–0.56 stars. Validation and test agree within 0.004 for seven of the eight selected runs and within 0.012 for the eighth, so choosing among 31 runs did not overfit the validation sets.

What the numbers show:

- **Full fine-tuning wins all four comparisons, by 0.004 to 0.036 macro-F1** at each method's best learning rate. Only `s-50m` English (0.036) and `m-100m` Turkish (0.014) are clearly larger than the sampling noise of a 15k-review test set (roughly ±0.005 for a paired difference); differences between seeds were not measured.
- **The learning rate mattered more than the method.** With the first, too-low grid, LoRA trailed by 0.025–0.05 on validation; after one more grid step the gap at `m-100m` fell to 0.006–0.014.
- **At equal forgetting, full fine-tuning learns more.** In the sweep, full fine-tuning at LR *x* and LoRA at 10*x* forget almost exactly the same amount (SFT Turkish loss 2.79 vs 2.76, 3.07 vs 3.05, 2.42 vs 2.42, 2.64 vs 2.64), and full fine-tuning has the higher F1 in every such pair (e.g. `m-100m`: 0.558 vs 0.413 at 2.64). LoRA forgets less at a given learning rate, but only because it learns less.
- **Scale helps both methods:** `m-100m` is 0.025–0.060 macro-F1 better than `s-50m`.
- **Cross-lingual transfer is weak.** A model fine-tuned in one language learns the answer format in the other (`label_mass` rises from 0.02 to 1.00), but its macro-F1 there is only 0.13–0.26.

### What fine-tuning costs: forgetting

Fine-tuning on one narrow task damages what SFT taught, and the damage depends on the language and, above all, on the learning rate.

| After fine-tuning (selected runs) | SFT val loss, Turkish | SFT val loss, English | Code reference loss |
|---|---:|---:|---:|
| `s-50m` SFT model (before) | 2.646 | 1.915 | 0.471 |
| Turkish, full, 1e-4 | 3.930 (+49%) | 2.295 (+20%) | 1.305 |
| English, full, 1e-4 | 3.191 (+21%) | 2.841 (+48%) | 1.620 |
| Turkish, LoRA, 3e-3 | 6.938 | 5.712 | 5.213 |
| English, LoRA, 3e-3 | 4.769 | 5.655 | 7.852 |
| `m-100m` SFT model (before) | 2.348 | 1.660 | 0.362 |
| Turkish, full, 3e-4 | 7.605 | 4.907 | 8.599 |
| English, full, 3e-4 | 5.968 | 4.923 | 7.382 |
| Turkish, LoRA, 3e-3 | 6.712 | 3.495 | 7.723 |
| English, LoRA, 3e-3 | 10.077 | 3.918 | 6.667 |

- **Forgetting is language-specific at moderate learning rates.** Training in Turkish raises the Turkish SFT loss by 49% but the English one by 20%; training in English does the opposite (+48% English, +21% Turkish). The model learns "after a user message in this language, answer with one digit".
- **At the learning rates chosen for F1, the coding ability is gone.** A code reference loss of 5–9 nats per token is close to knowing nothing (ln 16,384 = 9.7). Only full fine-tuning at 1e-4 keeps it at 1.3–1.6 nats (2.8–3.4× the SFT model).
- **The selection rule caused this.** It looked only at the task's F1: for `m-100m` full, 3e-4 beat 1e-4 by 0.008 F1 and cost almost all of the model's coding and chat ability. Any fine-tuning that has to keep earlier skills needs forgetting in its selection rule, a lower learning rate, and replay of earlier data.

### Engineering notes from fine-tuning

- **LoRA does not save time here.** Without `torch.compile`, LoRA steps were 1.4× slower than full fine-tuning: the backward pass still runs through every frozen matrix, and each of the 240–320 adapted layers adds two small matrix multiplications, many of them inside separately launched experts. LoRA saves optimizer memory, which these models do not need.
- **LoRA checkpoints load as plain models.** Parameter names are unchanged (`...qkv.weight` plus `...lora_A`, `...lora_B`), so pretrained weights load before adapters are added, and `scripts/sample.py` folds the adapters into the weights on load. A test checks that merged and unmerged models give the same logits.
- **Right-padding is safe for scoring.** Prompts of different lengths share a batch; a test checks against scoring each alone, because the model is causal and the MoE routes each token separately.

Reproduce: [`notebooks/05_finetune.ipynb`](notebooks/05_finetune.ipynb) (about 2.5 hours on an A100 for all 31 runs and the test). Receipts for every run and the test summary: [`artifacts/finetune/`](artifacts/finetune/).

## Joint fine-tuning: one model per size

The Phase 4 specialists rate reviews in one language and have lost much of their chat and coding ability. RL needs a model that keeps them, so each size gets **one** more fine-tuning run that learns both languages at once and replays SFT data. These two models are the starting point for RL.

### Recipe

| Setting | Value |
|---|---|
| Start | the SFT checkpoints |
| Data, by share of windows | Turkish reviews 14.9%, English reviews 51.8%, SFT replay 33.3% (English 30.0%, Turkish 3.3%) |
| Length | one epoch of both review sets: 30,358 windows, 31.1M tokens, 238 steps of 128 windows |
| Method | full fine-tuning (better than LoRA at every matched level of forgetting in Phase 4), no weight decay, warmup 5%, cosine decay, `torch.compile` |
| Loss | **averaged per window** (`loss_per_window`) |
| Learning rate | 3e-5 or 1e-4, chosen by the rule below |

**Why the loss is averaged per window.** A chat window has ~600 learned tokens; a review window has 10–32 (one digit and `<|end|>` per review). With the usual mean over all learned tokens, the replay would make up ~95% of the loss and the reviews would barely be learned. Averaging inside each window first makes every window count equally, so each source contributes its share of windows.

**Selection rule, fixed before the runs.** Among the runs whose SFT validation loss rises by at most 10% in each language and whose code reference loss rises by at most 25%, pick the highest mean validation macro-F1 over both languages.

### Results

| | LR | Val F1 TR / EN | SFT val loss EN / TR | Code reference loss |
|---|---|---:|---:|---:|
| `s-50m` | 3e-5 | 0.489 / 0.560 | +0.8% / +0.9% | +1.8% |
| `s-50m` | **1e-4** (selected) | **0.555 / 0.583** | +1.8% / +2.4% | +4.2% |
| `m-100m` | 3e-5 | 0.522 / 0.590 | +0.6% / +0.8% | +7.1% |
| `m-100m` | **1e-4** (selected) | **0.585 / 0.621** | +1.9% / +2.4% | +4.6% |

All four runs stayed well within the limits, so the rule picked the higher learning rate at both sizes. Test sets:

| Test macro-F1 (accuracy) | Turkish | English | Code reference loss |
|---|---:|---:|---:|
| `s-50m` SFT model | 0.045 (0.128) | 0.067 (0.196) | 0.471 |
| **`s-50m` joint** | **0.561** (0.599) | **0.584** (0.591) | **0.491** |
| `s-50m` specialists (Phase 4, full) | 0.572 | 0.586 | 1.305 / 1.620 |
| `m-100m` SFT model | 0.045 (0.128) | 0.067 (0.196) | 0.362 |
| **`m-100m` joint** | **0.582** (0.617) | **0.611** (0.615) | **0.379** |
| `m-100m` specialists (Phase 4, full) | 0.607 | 0.617 | 8.599 / 7.382 |

- **One model now does what two specialists did, without paying for it in chat or code.** In English the joint models are within 0.006 macro-F1 of the specialists; the SFT validation losses rise by at most 2.4% instead of 20–224%, and the code reference loss by ~5% instead of 2.8–24×.
- **Turkish is 0.011–0.025 behind the specialists, and the mix is the likely reason.** The window mean equalises windows, not reviews: a Turkish window holds ~16 reviews and an English one ~5, so each Turkish review weighs about 3.4× less than an English one. Weighting by reviews instead of windows is the obvious fix and was not tried.
- The specialists' code losses above are from their own learning rates (1e-4 for `s-50m`, 3e-4 for `m-100m`), so the `m-100m` comparison is partly a learning-rate effect. The joint runs use 1e-4 at both sizes.

### Format leakage into Turkish chat

One greedy sample showed the problem that the validation losses could not: asked `Sağlıklı beslenme için üç öneri ver.` ("Give three tips for healthy eating."), the joint `m-100m` answered `5`. Almost every Turkish user message in the mix is a review to rate (Turkish SFT replay is only 3.3% of windows), so the model partly learned "Turkish request → digit". The SFT loss barely moves, because only the first token of the answer changes.

Measured on 20 ordinary requests per language: the probability that the first answer token is a star digit.

| | Turkish: mean | Turkish: requests above 0.5 | English: mean | English: requests above 0.5 |
|---|---:|---:|---:|---:|
| `s-50m` SFT → joint | 0.029 → 0.065 | 0 → 1 of 20 | 0.013 → 0.004 | 0 → 0 |
| `m-100m` SFT → joint | 0.035 → 0.089 | 0 → 1 of 20 | 0.037 → 0.038 | 0 → 0 |

The leak is real but small and only in Turkish, which fits the explanation: English has about nine times more replay. The models were accepted as the RL starting point; the measurement was repeated after RL ([results](#what-rl-cost-interference)). More Turkish replay would be the fix.

Other samples (greedy, [`artifacts/finetune-mix/summary.json`](artifacts/finetune-mix/summary.json)): both models still answer English chat requests in the SFT style, rate the two example reviews correctly (5 and 1), and `m-100m` still writes a correct `is_even`.

Reproduce: [`notebooks/06_finetune_mix.ipynb`](notebooks/06_finetune_mix.ipynb) (about 45 minutes on an A100). Receipts: [`artifacts/finetune-mix/`](artifacts/finetune-mix/). Weights: `runs/mix-s50m-lr1e-04` and `runs/mix-m100m-lr1e-04` in the runs repo.

## Fast generation

RL learns from the model's own samples, so it needs far more generation than any evaluation so far. The code evaluation already took 10–17 minutes per model on an A100. Four changes, each found by measuring the previous one, made generation 13–19× faster at every batch size measured. Every row is measured on an A100 40 GB: a coding prompt of ~60 tokens, 160 new tokens per sequence, no early stop.

| `s-50m`, tokens/s | n = 1 | n = 8 | n = 32 | n = 128 | n = 256 |
|---|---:|---:|---:|---:|---:|
| No cache (every earlier evaluation) | 14 | 106 | 353 | — | — |
| 1. Generation cache | 16 | 118 | 394 | 961 | — |
| 2. + dense MoE in decoding | 25 | 175 | 533 | 1,134 | — |
| 3. + CUDA graph (run log; this table was overwritten by row 4, not kept as a receipt) | 192 | 814 | 1,360 | 1,740 | 1,746 |
| 4. + sampling on the GPU | **235** | **1,487** | **5,467** | **18,719** | **32,431** |
| `m-100m`, all four | 173 | 1,152 | 4,108 | 13,477 | 21,196 |

1. **Cache** ([`glm_ladder/model.py`](glm_ladder/model.py)). The prompt is processed once, then one position per step. Linear-attention layers keep a fixed-size running sum per head (`state = Σ kᵀv`, `normalizer = Σ k`, the recurrent form of the chunked kernel); sparse-attention layers keep every earlier key and value. A test checks that prefill plus token-by-token decoding gives the full-forward logits at every position. **Gain: 1.1×**, because the arithmetic was never the bottleneck.
2. **Dense MoE for small batches.** Sending each token to its two experts uses `torch.where` and `bincount`, whose output sizes depend on the data, so the CPU must wait for the GPU about 9 times per layer per token. In eval mode with at most 256 tokens, every expert now runs on every token, weighted by its gate (0 when not selected): same result, no waiting. **Gain: 1.2–1.6×**; ~40 ms per step remained at n = 1.
3. **CUDA graph** ([`glm_ladder/graph_decode.py`](glm_ladder/graph_decode.py)). The remaining 40 ms were the CPU launching ~1,600 small operations per step. One decode step is recorded once and replayed with a single call. This needs fixed shapes and fixed memory: a preallocated cache over the full context, the position as a GPU tensor, and **in-place** cache updates, because the graph replays the recorded addresses and a new tensor would never be read. **Gain: 7.7× at n = 1** (5 ms per step), but only 1.2–1.5× at n ≥ 128.
4. **Sampling on the GPU.** At large batches the step time grew linearly with n: every step copied `[n, 16,384]` logits to the CPU and sampled there. Now only the n chosen ids are copied. **Gain: up to 19×**, which uncovered the graph's real benefit at large batches (5–6× over the cached step with GPU sampling).

The code evaluation of one model now takes ~100 s instead of 825–1,002 s. Greedy pass rates were unchanged by the graph (up to one task whose two best tokens are nearly tied), but random numbers now come from the GPU, so sampled results differ from earlier runs by chance.

### Coding baseline for RL

The two RL starting models, evaluated with the final decoding (40 tasks, chat prompts, 1 greedy + 10 sampled answers each):

| | Greedy pass@1 | Sampled pass@1 | pass@5 | pass@10 | Reference loss |
|---|---:|---:|---:|---:|---:|
| `mix-s50m-lr1e-04` | 0.375 | 0.130 | 0.393 | 0.525 | 0.491 |
| `mix-m100m-lr1e-04` | 0.450 | 0.253 | 0.544 | 0.700 | 0.379 |

**How noisy is this?** The same `m-100m` model scored pass@10 = 0.550 with CPU sampling and 0.700 with GPU sampling: only the random numbers changed. With 10 samples per task, a task whose solution appears 1 time in 10 passes or fails by chance, and each task moves pass@10 by 0.025. An RL gain of this size could not be told apart from luck, so the RL evaluation draws 64 samples per task on 229 tasks and compares models task by task.

Receipts: [`artifacts/evals/generation-speed*.json`](artifacts/evals/), [`code-mix-*-chat.json`](artifacts/evals/). Reproduce: [`notebooks/07_generation.ipynb`](notebooks/07_generation.ipynb).

## Reinforcement learning on code

The last phase trains each joint model on its own answers to coding tasks, rewarded only by whether the code passes the tests. Three runs: one per size with a KL penalty (the final models) and `s-50m` without it (the ablation).

### Training tasks

The 40 hand-written tasks are too few to train on and must stay untouched for evaluation, so tasks are generated ([`glm_ladder/task_gen.py`](glm_ladder/task_gen.py), [`evals/rl_tasks/`](evals/rl_tasks/)). A *family* is one kind of function ("count the numbers greater than k") and a *variant* fixes its parameter (k = 3). Tests are random inputs run through the reference solution, so the expected outputs are always right, and a task is kept only if its tests reject `return None` and constant answers.

| Split | Tasks | What it measures |
|---|---:|---|
| `train` | 301 | 24 families (arithmetic, comparisons, list filters and maps, string edits) |
| `test_id` | 82 | **new variants of the trained families** (a quarter of each family's variants, held out) |
| `test_ood` | 107 | **9 families never trained on** (e.g. `last_word`, `adjacent_differences`, `round_down_to_multiple`) |
| hand-written | 40 | the tasks from Phase 3: **near transfer**, written separately, with some close relatives in `train` (a fixed-range `clamp` vs `clamp(x, low, high)`) |

Every generated reference solution is a single `return` line. This turned out to matter (see [below](#what-rl-did-to-the-hand-written-tasks)).

**Difficulty filter.** RLOO learns only from a task whose answers differ: if all of them fail, or all pass, every advantage is zero. Each starting model wrote 32 answers to every training task, and the tasks it solved at least once but not always form its training set. A first filter with 8 answers kept only 19 tasks for `s-50m`, which 300 steps of 32 tasks would have repeated ~500 times each.

| Training tasks solved by the starting model | `s-50m` | `m-100m` |
|---|---:|---:|
| Never in 32 answers | 235 | 116 |
| Always | 0 | 0 |
| **Sometimes (the training set)** | **66** | **185** |

The larger model can learn from 2.8× more of the same 301 tasks. That is part of the scale effect below: RL can only sharpen what the model already does sometimes.

### Recipe

| Setting | Value |
|---|---|
| Algorithm | RLOO: 16 answers per task, each answer's advantage is its reward minus the mean reward of the other 15 |
| Reward | 1 if the code block passes every test, else 0 |
| Update | on-policy: one AdamW step per batch of fresh answers, so no importance ratio or clipping; loss averaged over answer tokens |
| KL penalty | 0.02 × k3 estimate of KL(model ‖ starting model), per answer token; 0 for the ablation |
| Batch | 16 tasks × 16 answers = 256 answers per step, 200 steps, at most 192 new tokens |
| Sampling | temperature 1.0, no top-k (the distribution the policy gradient assumes) |
| Learning rate | 5e-6, 10 warmup steps, MoE balance loss 0.01, gradient clip 1.0 |
| Time per step (A100) | ~6 s (`s-50m`), ~8 s (`m-100m`): generation 1.6–2.4 s, tests ~1.8 s, update 2.1–3.8 s |

The learning rate is from a 20-step trial on `s-50m`: training reward rose from 0.03 to 0.20 with KL 0.02 and steady gradient norms, so it was kept. The trial also showed that in its first steps only 12–28% of the tasks in a step gave any signal (answers that differ), which led to the larger filter and to 16 answers per task instead of 8; in the final runs, 55–100% of the tasks in a step had signal after the first few steps.

Training reward (mean over the 256 answers of a step) went from 0.02 to ~0.75 for `s-50m` and from 0.09 to ~0.85 for `m-100m`; the KL to the starting model leveled off at ~0.06 per token after ~60 steps in both runs.

### Evaluation

Every model is measured the same way before and after ([`notebooks/08_rl.ipynb`](notebooks/08_rl.ipynb), section 7): 64 samples per task at temperature 1.0 plus one greedy answer, on 229 tasks. pass@1 is the mean per-task pass rate over the 64 samples. Before/after differences are **paired by task**, with a bootstrap 95% interval over tasks; an interval that excludes 0 is a change the sampling noise cannot explain. The same measurement also covers review rating (both test sets), SFT validation loss, code reference loss, the Turkish format leak, and chat samples.

### Results

| pass@1 (64 samples), before → after | `s-50m` | `m-100m` |
|---|---:|---:|
| `test_id` (trained families, new variants) | 0.009 → **0.402** | 0.067 → **0.712** |
| paired difference [95% interval] | +0.393 [+0.317, +0.468] | **+0.646** [+0.582, +0.705] |
| `test_ood` (families never trained on) | 0.001 → 0.010 | 0.007 → **0.082** |
| paired difference | +0.009 [+0.006, +0.013] | **+0.075** [+0.044, +0.110] |
| hand-written (near transfer) | 0.048 → 0.109 | 0.122 → 0.193 |
| paired difference | +0.062 [+0.025, +0.107] | +0.070 [+0.014, +0.133] |

| Other metrics, before → after | `s-50m` | `m-100m` |
|---|---:|---:|
| `test_id` pass@64 / greedy | 0.390 → 0.756 / 0.098 → 0.659 | 0.659 → 0.915 / 0.427 → 0.890 |
| `test_ood` pass@64 / greedy | 0.056 → 0.280 / 0.000 → 0.000 | 0.280 → 0.383 / 0.075 → 0.150 |
| hand-written pass@64 / greedy | 0.500 → 0.600 / 0.400 → **0.175** | 0.850 → 0.725 / 0.450 → **0.275** |

What the numbers show:

- **RL works at both sizes, and the gain grows with scale.** On new variants of the trained families, the paired gain is 1.6× larger at `m-100m`; on families never trained on, it is **8× larger** (+0.075 vs +0.009). `s-50m` learns the trained kinds of task and almost nothing beyond them; `m-100m` carries some of it to new kinds. For the research question: from 50M to 100M, RL gains grow, and transfer to untrained task families grows more.
- **RL mostly makes rare solutions likely.** pass@1 grew 45× (`s-50m`) and 11× (`m-100m`) on `test_id`, pass@64 only 1.9× and 1.4×: most of the gain is on tasks the model could already solve once in 64 tries, now solved most of the time. The larger model starts with more of those (185 vs 66 training tasks in the filter).
- **Greedy got worse on the hand-written tasks** at both sizes, while sampled pass@1 improved; and `m-100m`'s hand-written pass@64 fell from 0.850 to 0.725. The next section explains why.

### What RL did to the hand-written tasks

The 40 greedy answers of `m-100m` before and after RL were compared task by task (`scripts/run_code_eval.py --run rl-m100m-001 --mode chat --verbose`): 10 tasks pass both times, 1 is gained (`max_of_two`), 8 are lost, 21 fail both times.

**After RL, all 40 greedy answers are a single `return` line.** None has a loop, an `if`, a helper variable, or an `import`. Every training reference was a single line, so the reward only ever reinforced that shape, and the model's most likely answer became one line even where one line cannot work:

| Task | Before RL (passes) | After RL (fails) |
|---|---|---|
| `factorial` | base case, then `n * factorial(n - 1)` | `return n * factorial(n)` |
| `is_prime` | loop up to √n | `return n % 2 == 0` |
| `index_of` | `enumerate` loop, `-1` if absent | `return items[value]` |
| `gcd` | `import math`, `math.gcd(a, b)` | `return gcd(a, b)` |

8 of the 40 hand-written reference solutions need more than one line. Before RL, `m-100m` passed 5 of them greedily; after RL, none. On the 32 one-line tasks it went from 13 to 11. Sampled pass@1 still rose, because RL made correct one-line answers much more likely on the many tasks where one line is enough. Function names were not the problem: every greedy answer used the requested name. This was checked only for `m-100m`; `s-50m` shows the same greedy drop and was not analysed.

The lesson is about the task distribution, not the algorithm: an executable reward teaches exactly the shape of solution that the training tasks admit. Families whose solutions need control flow (loops, early returns, accumulators) are the obvious next step.

### KL ablation (`s-50m`)

| `s-50m` after RL | KL 0.02 | no KL |
|---|---:|---:|
| `test_id` pass@1 | 0.402 | **0.488** |
| paired gain vs start [95% interval] | +0.393 [+0.317, +0.468] | +0.479 [+0.393, +0.562] |
| `test_ood` pass@1 / pass@64 | **0.010 / 0.280** | 0.006 / 0.196 |
| hand-written pass@1 | **0.109** | 0.097 |
| SFT val loss, English (start 1.949) | **1.956** (+0.4%) | 1.993 (+2.3%) |
| Code reference loss (start 0.491) | **0.443** | 0.591 (worse than start) |
| Answer length, last 50 steps (tokens) | ~95–135 | ~185–190 (limit 192) |

**Without the KL penalty the model gains more on the trained kinds of task and loses elsewhere:** less transfer to new families, more forgetting, a worse loss on the reference solutions, and answers that grow until they hit the length limit. The KL penalty trades some in-distribution gain for generality. One run per setting; the ablation was not repeated with other seeds.

The length growth was not explained further. One possibility: the loss averages over answer tokens in the batch, so a long answer with a positive advantage gets more total weight than a short one, and nothing but the KL term pushes back.

### What RL cost: interference

| Before → after RL (KL runs) | `s-50m` | `m-100m` |
|---|---:|---:|
| Review rating, test macro-F1 Turkish | 0.561 → 0.550 | 0.582 → 0.572 |
| Review rating, test macro-F1 English | 0.584 → 0.580 | 0.611 → 0.610 |
| SFT val loss English / Turkish | 1.949 → 1.956 / 2.709 → 2.712 | 1.692 → 1.699 / 2.404 → 2.411 |
| Code reference loss | 0.491 → 0.443 | 0.379 → 0.353 |
| Turkish format leak (digit probability on 20 ordinary requests) | 0.065 → 0.139 | 0.089 → 0.065 |

- **Interference is small.** Review rating drops by at most 0.011 macro-F1 and the SFT losses rise by at most 0.4%; the code reference loss even improves.
- **The Turkish leak doubled at `s-50m`** (mean 0.065 → 0.139) and fell at `m-100m`. 20 prompts are few, and the change is not explained.
- **Chat got no better and in places worse.** Samples (greedy, in the receipts) are repetitive at both sizes; asked `Sağlıklı beslenme için üç öneri ver.`, `m-100m` repeated the question and `s-50m` repeated `Sağlıklı beslenme` until the limit. Chat was not rewarded and was measured only by losses and four samples.

### Running generated code

Each RL run executes ~51,000 model-written programs (200 steps × 256 answers), and each evaluation another ~15,000. Every one of them:

- is checked against a deny-list (no `os`, `sys`, `subprocess`, `open(`, `eval(`, `importlib`, `ctypes`, `threading`, `multiprocessing`, `signal`, network modules, `pickle`, `/proc/`, ...);
- runs in its own subprocess, in an empty temporary directory, with an empty environment except `PATH`, and a 2 s timeout;
- has resource limits set before it runs: 1 GB address space, CPU time, 1 MB per written file;
- runs as the unprivileged user `nobody` when the runtime is root (as on Colab), so it cannot change the runtime's files or reach the notebook process that holds the Hugging Face token. A test checks the privilege drop when run as root.

This is defense in depth for small functions written by our own models, not a sandbox against a deliberate attacker; a container or gVisor would be the next step.

### Losing a session, and resumability

The first RL session died during the third run, after the two final models had been trained and measured. Nothing was saved: the notebook kept everything in memory until the end, so all of it had to be recomputed. RL was then made resumable like pretraining:

- every 25 steps the full state (weights, AdamW moments, RNG states, history, task order) is pushed to the runs repo; a re-run continues from the last checkpoint, at most 25 steps lost;
- every result (baselines, filtered training sets, final measurements, comparison) is uploaded as a receipt the moment it exists; a re-run downloads it instead of recomputing;
- the reference model for the KL penalty is always reloaded from the starting checkpoint, never from the resumed one.

A test checks that a run interrupted and resumed ends with exactly the weights of an uninterrupted run. The rerun repeated the lost session's training logs exactly (reward, KL, and length at every logged step); its evaluation numbers matched too, except hand-written pass@64 of `m-100m`, which differed by one task.

Reproduce: [`notebooks/08_rl.ipynb`](notebooks/08_rl.ipynb) (about 2 hours on an A100, resumable). Tasks: `python scripts/make_rl_tasks.py`. Receipts: [`artifacts/rl/`](artifacts/rl/) (measurements before and after, training sets, paired comparison). Weights: `runs/rl-s50m-001`, `runs/rl-m100m-001`, `runs/rl-s50m-nokl-001` in the runs repo.

## Final models

The project's output: one model per size, each through every phase. Each row is measured on that phase's model.

| Phase | Measurement | `rl-s50m-001` | `rl-m100m-001` |
|---|---|---:|---:|
| Pretraining (`s50m-001`, `m100m-001`) | Validation loss, mixed (nats/token) | 2.948 | 2.636 |
| SFT (`sft-*-001`) | 40 coding tasks, chat: greedy / pass@10 | 37.5% / 50.0% | 47.5% / 65.0% |
| Joint fine-tuning (`mix-*-lr1e-04`) | Review rating test macro-F1, Turkish / English | 0.561 / 0.584 | 0.582 / 0.611 |
| RL (`rl-*-001`) | `test_id` / `test_ood` pass@1 | 0.402 / 0.010 | 0.712 / 0.082 |
| RL | Hand-written tasks: pass@1 / greedy | 0.109 / 0.175 | 0.193 / 0.275 |
| **Final model** | Review rating test macro-F1, Turkish / English | 0.550 / 0.580 | 0.572 / 0.610 |
| **Final model** | SFT val loss, English / Turkish | 1.956 / 2.712 | 1.699 / 2.411 |

What each model can do, and what it cannot:

- It writes short Python functions in the chat format, and is reliable on one-line functions of the kinds it was trained on (`m-100m`: 89% greedy on new variants).
- It rates English and Turkish reviews from 1 to 5 stars about as well as a single-task fine-tuned model.
- It answers chat requests in the right format, but its content is unreliable and repetitive, and its Turkish chat is weak.
- It does not write multi-line algorithms reliably; RL made this worse under greedy decoding.

The weights stay in the private runs repository. Every model can be rebuilt from this repository: the notebooks in order, with the configs and seeds in the receipts.

Talk to a final model on CPU: `python scripts/talk.py --run rl-m100m-001`.

## Plan

| Phase | Work | Exit criterion | Status |
|---|---|---|---|
| 0. Infrastructure | Rewritten model, parity tests, throughput benchmark | Tests pass on Colab; measured tokens/s for every rung | Done |
| 1. Data | BPE tokenizer, mixed English + Python + Turkish corpus, sharded and hashed | Deterministic, resumable data stream | Done |
| 2. Pretraining | S → M; baseline reproduction and L deferred | Loss curves, held-out loss, receipts | Done for S and M; `baseline-25m` and `l-200m` deferred |
| 3. Post-training | Supervised fine-tuning on chat data | Eval before vs after | Done for S and M: forgetting, executable code eval, chat samples |
| 4. Fine-tuning | Full fine-tuning vs LoRA on review rating (1-5 stars) in Turkish and English | Matched-data comparison on held-out test sets | Done for S and M: LR sweep, ranks, test, forgetting; joint model per size for RL |
| 5. Reinforcement learning | RLOO with executable rewards on generated tasks; KL ablation | Held-out results with paired statistics | Done for S and M: three runs, paired bootstrap on 229 tasks, interference; GRPO, DPO, and reward ablations deferred |
| 6. Report | Scaling analysis, negative results, limitations | This README ([RL](#reinforcement-learning-on-code), [final models](#final-models), [limitations](#limitations-and-deferred-work)) | Done |

The evaluation protocol follows the original repository: decisions are made on a dev split, a separate confirmation split is opened once, and every comparison reports paired exact tests or bootstrap intervals. Negative results are reported. In RL, the only choices made from results (learning rate, filter size, answers per task) used the training tasks; the test splits were measured once before and once after.

## Limitations and deferred work

**Deferred, not dropped.** These can be done later if compute allows; the code, configs, and notebooks already support them:

- `l-200m`: every phase at 200M parameters (pretraining alone is ~118 units at 20 tokens per parameter, ~59 at 10), which would give the research question a third point.
- `baseline-25m`: reproducing the original model and its RL result with this pipeline, the fourth point.
- The planned ablations: balance loss on/off in pretraining, byte vs BPE tokenizer, GRPO and DPO against RLOO, reward shaping.

**Limitations of what was done:**

- **Two sizes, one seed each.** A scaling trend from two points is a direction, not a law, and no result was repeated with another seed.
- **RL tasks are narrow.** Every generated reference solution is one line, and RL taught that style; tasks that need control flow got worse under greedy decoding ([details](#what-rl-did-to-the-hand-written-tasks)).
- **Joint fine-tuning underweights Turkish reviews** (window mean, not review mean), and a small Turkish format leak remains.
- **Chat quality is low** at both sizes and was measured only by losses and a few samples.
- **Running generated code** relies on a deny-list, resource limits, and an unprivileged user, not a real sandbox.
- **Compute was measured in session hours**, so unit counts are estimates.

## Compute and storage

- **Compute:** Google Colab Pro (L4 and A100), 300 compute units in total. Pretraining both rungs took about 7 hours of training on an A100, roughly 45–50 units including setup, evaluation, and checkpoint uploads (estimate from session hours). After SFT and the GPU code evaluation, about 95 of the 300 units had been used (Colab balance); the fine-tuning study added roughly 25–30, the joint fine-tuning about 6, the generation work about 6, and RL about 25 including the lost session (estimates from session hours). In total **151 of the 300 units** (Colab balance at the end: 149 left), with pretraining about 30% of it.
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
| [`05_finetune`](notebooks/05_finetune.ipynb) | A100 | Full fine-tuning vs LoRA on review rating: baselines, LR sweep, ranks, English, test |
| [`06_finetune_mix`](notebooks/06_finetune_mix.ipynb) | A100 | Joint fine-tuning (both languages + SFT replay): the RL starting models |
| [`07_generation`](notebooks/07_generation.ipynb) | A100 | Generation speed (uncached, cached, CUDA graph) and the coding baseline of the RL starting models |
| [`08_rl`](notebooks/08_rl.ipynb) | A100 | RL on code: baselines, difficulty filter, three runs, before vs after (resumable) |

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

# Build the review-rating data; fetch the fine-tuning receipts after notebook 05
python scripts/prepare_finetune.py
python scripts/fetch_evals.py --pattern "finetune/*.json" --to artifacts/finetune
python scripts/fetch_evals.py --pattern "finetune-mix/*.json" --to artifacts/finetune-mix

# Generate the RL tasks; fetch the RL receipts after notebook 08; talk to a final model
python scripts/make_rl_tasks.py
python scripts/fetch_evals.py --pattern "rl/*.json" --to artifacts/rl
python scripts/talk.py --run rl-m100m-001
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
- [`glm_ladder/lora.py`](glm_ladder/lora.py): LoRA layers, applying adapters by layer type, merging them into the weights.
- [`glm_ladder/graph_decode.py`](glm_ladder/graph_decode.py): decoding from a CUDA graph with a static cache.
- [`glm_ladder/classify.py`](glm_ladder/classify.py): review-rating prompts, scoring by label-token probability, accuracy / macro-F1 / MAE.
- [`glm_ladder/task_gen.py`](glm_ladder/task_gen.py): generated coding tasks (families, variants, tests from the reference), train / `test_id` / `test_ood` splits.
- [`glm_ladder/rl.py`](glm_ladder/rl.py): RLOO advantages, policy loss with the k3 KL penalty, rewards, batched rollouts, paired bootstrap, the resumable trainer.
- [`evals/code_tasks.jsonl`](evals/code_tasks.jsonl): the 40 coding tasks with reference solutions and tests.
- [`evals/rl_tasks/`](evals/rl_tasks/): the 490 generated tasks and their manifest.
- [`tests/`](tests/): parity with the original model, kernel equivalence, causality, balance-loss gradient, loader determinism, bit-exact resume of the full training loop, chat masks and packing, the code-eval task set, LoRA (formula, gradients, merge, resume), padding-invariant scoring, per-window loss, cached and static-cache decoding against the full forward, dense vs sparse MoE, CUDA-graph replay and batches of different prompts, generated tasks and the privilege drop, RLOO (a toy reward is learned, exact resume, checkpoint cadence).
- [`tests/reference/original_model.py`](tests/reference/original_model.py): copy of the original model, used only by tests.
- [`scripts/`](scripts/): tokenizer training, benchmark, sampling, receipts, plots, SFT and fine-tuning data preparation, RL task generation, chat, code evaluation.
- [`artifacts/`](artifacts/): receipts, code-eval, fine-tuning, and RL results, samples, plots.
- [`PLAN.md`](PLAN.md): decision log and compute budget.

## Sources

- [vukrosic/glm-5.3-flash-from-scratch](https://github.com/vukrosic/glm-5.3-flash-from-scratch)
- [GLM-5.3-Flash official model card](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961), load-balancing loss
- [Transformers are RNNs (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236), linear attention
- [SmolLM2 (Allal et al., 2025)](https://arxiv.org/abs/2502.02737), source of the `smol-smoltalk` SFT data
- [InstrucTurca](https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca), Turkish SFT data
- [Evaluating Large Language Models Trained on Code (Chen et al., 2021)](https://arxiv.org/abs/2107.03374), unbiased pass@k
- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Buy 4 REINFORCE Samples, Get a Baseline for Free! (Kool et al., 2019)](https://openreview.net/forum?id=r1lgTGL5DE), the leave-one-out baseline
- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (Ahmadian et al., 2024)](https://arxiv.org/abs/2402.14740), RLOO for language models
- [Approximating KL Divergence (Schulman, 2020)](http://joschu.net/blog/kl-approx.html), the k3 estimator
- [SentiTurca / TrGLUE (Altinok, 2025)](https://huggingface.co/datasets/turkish-nlp-suite/SentiTurca), Turkish review data
- [Character-level Convolutional Networks for Text Classification (Zhang et al., 2015)](https://arxiv.org/abs/1509.01626), Yelp Review Full

## License

MIT. Builds on vukrosic/glm-5.3-flash-from-scratch (MIT). Independent educational project, not affiliated with Z.ai.
