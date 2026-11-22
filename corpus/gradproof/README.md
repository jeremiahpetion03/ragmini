# gradproof

A GPT-style transformer written in pure NumPy, with every backward pass derived and written by hand and then **proven correct**: checked entry by entry against finite differences, cross-checked against PyTorch autograd, and stress-tested with planted bugs the checks must catch.

```
pip install numpy pytest          # torch optional (enables the PyTorch parity test)
pytest                            # 26 tests in ~11 s (+3 PyTorch parity tests if torch is installed)
python -m gradproof.train --task sort --seed 0 --out runs/sort-seed0
python -m gradproof.bench
```

## Results

| What | Result |
|---|---|
| Gradient check, **every entry of every parameter** (7,083 entries, 30 tensors, float64) | max relative error **3.1e-7**, max absolute error **3.2e-10** |
| Planted backprop bugs caught | **6 / 6**, each with relative error **1.8 to 2.0**, about **6 orders of magnitude** above a correct pass |
| Sort 8 digits, held-out exact match (1,000 unseen sequences) | **100%** on seeds 0, 1, 2 |
| Reverse 8 digits, held-out exact match | **100%** (1,500 steps, 29 s) |
| Same seed run twice | metrics log and all 30 weight tensors **bitwise identical** (SHA-256 match) |
| Profiling fix (one line in GELU) | training step **47.4 ms → 18.7 ms, 2.53x faster** |
| float32 vs float64 | **1.80x faster**, **2.00x less** activation memory |

Training: 102,539-parameter model (2 layers, 4 heads, d=64), 3,000 steps, ~57 s per run at ~54K tokens/s on 2 CPU cores. All numbers are from `runs/*/final.json` and `python -m gradproof.bench` on that machine.

## What's implemented

Every layer has a hand-written `forward` and `backward` (`gradproof/nn.py`):

- **Linear**, **Embedding** (with `np.add.at` so repeated tokens accumulate), learned **positional embedding**
- **LayerNorm** with the full three-term backward
- **GELU** (tanh approximation, as in GPT-2)
- **Causal multi-head self-attention**, including the softmax Jacobian-vector product
- **Pre-norm transformer block** with residual gradients
- **Cross-entropy** via log-sum-exp with a position mask, so loss is computed only on answer tokens
- **AdamW** with decoupled weight decay (matrices only), **global gradient-norm clipping**, **warmup + cosine** learning-rate schedule

The tasks (`gradproof/data.py`) are synthetic on purpose: sorting and reversing digit sequences have exact answers, so accuracy is measured by **greedy generation from the prompt alone**, not teacher-forced token guesses.

## How correctness is verified

**1. Finite differences on every entry.** `gradcheck` perturbs each parameter by ±ε and compares `(L(p+ε) − L(p−ε)) / 2ε` with the analytic gradient, in float64. ε = 1e-5 was chosen by measurement: at 1e-4, O(ε²) truncation error dominates; at 1e-6, rounding error does.

**2. Error is judged where it is measurable.** On the first run, the worst "errors" (up to 5e-3) were on gradients around 1e-8. Sweeping ε showed that error followed rounding noise exactly (about machine-epsilon × |L| / ε ≈ 5e-11), not a backprop mistake. Relative error is now taken only on gradients ≥ 1e-4, and absolute error on all entries.

**3. A mathematical invariant as a test.** The gradient with respect to the **key bias** must be exactly zero: adding a constant to every key shifts each row of attention scores equally, and softmax ignores uniform shifts. The test confirms it is below 1e-12, while the query bias gradient is not.

**4. Mutation testing.** A checker is only trustworthy if it catches real mistakes. Each of these is planted into the model, and the test requires the checker to flag it:

| Planted bug | Max relative error |
|---|---|
| LayerNorm backward drops the mean term | 2.0 |
| GELU ignores the tanh derivative | 1.9 |
| Embedding uses `+=` (drops repeated tokens) | 2.0 |
| Transformer block drops the residual gradient | 2.0 |
| Attention forgets the 1/√d scale in backward | 1.8 |
| Softmax backward missing its subtraction term | 1.8 |

**5. Independent reference.** `test_torch_parity.py` rebuilds the network from PyTorch primitives, loads identical weights, and requires the loss and every gradient to match autograd (rtol 1e-8). It runs in CI and is skipped automatically where PyTorch is not installed.

**6. Numerical stability.** Tests show the naive softmax overflows to NaN (at logits ≈ 100 in float32), while the stable version stays finite and exact. Cross-entropy gradients sum to zero per row, masked positions get exactly zero gradient, and attention stays finite with scores above 1e4.

**7. Behavior checks.** Causality (changing token *t* never changes logits before *t*), dtype preservation (float32 stays float32 end to end), state-dict round trip, and bitwise reproducibility of training.

## Profiling

`cProfile` showed **GELU's forward taking 62% of the entire training step**. The cause was `x**3`: NumPy fast-paths `x**2` but sends `x**3` through its generic `pow` loop, which measured **14.3 ms vs 0.15 ms** for `x*x*x` on the same array. Changing that one line:

| float32, batch 64 | Before | After |
|---|---|---|
| Training step (median) | 47.4 ms | **18.7 ms** |
| Throughput | 21.6K tokens/s | **54.7K tokens/s** |

After the fix, matrix multiplication dominates the profile, as it should. The full gradient check passes unchanged.

**Memory.** At batch 64, cached activations take **11.0 MB** against **0.39 MB** of parameters, about 28x. That's why techniques like gradient checkpointing (trading recomputation for memory) and lower precision matter at scale. float32 halves activation memory exactly (2.00x) and runs 1.80x faster than float64. A test guards against NumPy 2's type promotion silently upcasting float32 to float64.

## Reproducibility

Each run writes `config.json` (every hyperparameter, plus NumPy and Python versions), `metrics.jsonl`, `final.json`, and `model.npz`. Model initialization and data sampling use separate seeded generators. Running the same command twice produces a byte-identical metrics log and identical weights. CI runs the whole test suite plus a smoke training run that must reach over 90% held-out accuracy.

## Layout

```
gradproof/
  nn.py         layers with hand-written forward/backward, stable softmax and cross-entropy
  model.py      GPT model, greedy generation, activation-memory accounting
  optim.py      AdamW, gradient clipping, LR schedule
  data.py       sort / reverse tasks with exact answers
  gradcheck.py  finite-difference checker
  train.py      reproducible training CLI
  bench.py      throughput and memory benchmark
tests/          gradient checks, planted bugs, numerics, model behavior, reproducibility, PyTorch parity
```
