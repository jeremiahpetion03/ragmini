# Kestrel

A PyTorch training and inference pipeline for a GPT-style language model, built around one rule: **every efficiency technique must be proven equivalent before it is measured.** Gradient checkpointing must produce the same gradients, the KV cache must produce the same tokens, and bf16 and int8 must stay within a measured error bound. Only then do their speed and memory numbers count.

**Measured results are in [RESULTS.md](RESULTS.md).** That file is generated from the run's output files, so no number in it is typed by hand.

## Quick start (CPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest

pytest                                            # full test suite
python -m kestrel.train --out runs/shakespeare    # downloads TinyShakespeare, trains, evaluates
python -m kestrel.bench --run runs/shakespeare    # speed, memory, KV cache, int8
python -m kestrel.report --run runs/shakespeare   # writes RESULTS.md
```

The default model has 826K parameters (4 layers, 4 heads, d=128, context 128) and trains for 3,000 steps on character-level TinyShakespeare. On the 2-thread laptop CPU used for [RESULTS.md](RESULTS.md), training took about 43 minutes and the benchmarks about 15. Faster CPUs will finish sooner.

## Running on an NVIDIA GPU

```bash
# 1. Replace the CPU build of PyTorch with the CUDA build
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Confirm PyTorch sees the GPU (should print True and the card's name)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. Train, benchmark, report
python -m kestrel.train --device cuda --out runs/gpu
python -m kestrel.bench --run runs/gpu --device cuda
python -m kestrel.report --run runs/gpu --out RESULTS_GPU.md

# 4. Optional: KV-cache speedup by generation batch size
python -m kestrel.bench --run runs/gpu --device cuda --only generation --gen-batch 1 8 32
```

On CUDA the benchmark adds an fp16 row (with loss scaling) and reports peak GPU memory from `torch.cuda.max_memory_allocated`. Every timing synchronizes the GPU before reading the clock, because CUDA kernels run asynchronously and an unsynchronized timer measures only how long the work took to queue. The GPU model, memory, compute capability, CUDA version, and native bf16 support are recorded with every run.

## Headline results

**The central finding: whether an optimization helps depends on hardware and scale, so every one is measured, not assumed.** The same code was benchmarked on a laptop CPU and on an NVIDIA RTX 3060 at two model sizes (826K and 25.4M parameters), and the answers changed sign:

| Technique | Laptop CPU, 826K params | RTX 3060, 826K params | RTX 3060, 25.4M params |
|---|---|---|---|
| bf16 vs fp32 (training step) | 16.5x **slower** | 1.25x **slower** | **2.45x faster** |
| fp16 vs fp32 (training step) | n/a | 1.31x **slower** | **2.33x faster** |
| Gradient checkpointing | 6.3x less activation memory, +14% time | 2.3x less peak GPU memory, **+70%** time | 3.0x less peak GPU memory, **+22%** time |
| KV cache (batch 1) | **2.8x faster** | 0.94x\* | 0.93x |
| KV cache (batch 32) | | 0.93x\* | **3.09x faster** |
| Int8 weight-only | 3.5x smaller, +0.0004 val loss | 3.5x smaller, +0.0004 val loss | 3.9x smaller, −0.00004 val loss |

\*Re-measured after the attention-mask fix described below ([`runs/gpu-recheck`](runs/gpu-recheck)). The first 826K run, made before the fix, measured 0.8x; under the same conditions (16-token prompt, 112 new tokens) cached decoding rose from 124 to 150 tokens/s (+21%) while uncached moved 155 to 160 (+3%), so the gain came from the fix. The same-seed retrain reproduced the validation loss exactly (1.58168).

Every row passed its equivalence check first: identical tokens with and without the cache, identical loss and gradients with checkpointing, and bf16/fp16 loss within 2% of fp32. Full tables: [RESULTS.md](RESULTS.md) (CPU), [RESULTS_GPU.md](RESULTS_GPU.md), [RESULTS_GPU_LARGE.md](RESULTS_GPU_LARGE.md).

### Finding: on a GPU, a small model is launch-bound, not compute-bound

My first GPU prediction was wrong. I expected bf16 to turn the CPU's 16.5x slowdown into a speedup on the RTX 3060, which has native bf16 tensor cores. At 826K parameters it was still 1.25x *slower*, and the KV cache was slower too. The hypothesis: with 128-wide layers, each matrix multiply takes microseconds, so step time is dominated by launching kernels, not by arithmetic. Mixed precision adds cast kernels and saves arithmetic that was never the bottleneck.

I tested that by changing one variable at a time:

1. **Scale the model 31x** (d=512, 8 layers, 25.4M parameters). bf16 went from 1.25x slower to **2.45x faster**, and checkpointing's time cost fell from +70% to +22%: once arithmetic dominates, both behave as they do at production scale.
2. **Scale the generation batch** for the KV cache. At 25.4M parameters, speedup grew from **0.93x (batch 1) to 1.05x (batch 8) to 3.09x (batch 32)**, with identical output at every size. At batch 1 each decode step is a few tiny kernels either way, so the arithmetic the cache saves is hidden behind launch overhead. Larger batches make each step compute-bound and expose it. Batch size alone isn't enough, though: at 826K parameters the cache stays below 1x even at batch 32 (0.94x, 0.96x, 0.93x at batch 1, 8, 32), because with 128-wide layers and a 128-token context each step is still too small to be compute-bound. The cache pays off only when both the model and the batch are large enough.
3. **Profiling the cache path** turned up a real inefficiency: every single-token decode step built an all-True attention mask, two extra kernel launches per layer per token. Removing it made batch-1 cached generation **23% faster** (66 → 81 tokens/s).

On the CPU the cache was 2.8x faster even at batch 1, because the CPU is compute-bound at this size. Same code, opposite conclusions. That's why the benchmark records the hardware with every run.

### Cross-hardware reproducibility

The same seed trained on the laptop CPU (43 minutes) and on the RTX 3060 (**87 seconds, 29.5x faster**) reached a validation loss of **1.58168 on both**, matching to all five reported decimals, with step-by-step losses equal to within floating-point rounding.

### CPU results in detail

Measured on a 2-thread laptop CPU.

| Technique | Effect | Equivalence check |
|---|---|---|
| Training | **1.58** validation loss (perplexity 4.86) on held-out text | Bitwise-reproducible from the seed |
| KV cache | **2.8x** faster generation | Token-for-token identical output |
| Gradient checkpointing | **6.3x** less activation memory (133.6 → 21.1 MB), 14% slower step | Identical loss and gradients |
| Int8 quantization | **3.5x** smaller model (3.15 → 0.90 MB) | Validation loss +0.0004 |
| bf16 autocast | 42% less activation memory, but **16.5x slower** step | Loss within 2% of fp32 |

### Why bf16 was 16.5x slower on the CPU

On the laptop CPU, bf16 made each training step **16.5x slower** (719 ms → 11,879 ms) while saving 42% of activation memory. The CPU has no native bfloat16 arithmetic, so the low-precision math is emulated, and converting values back and forth costs far more than the memory it saves. On the RTX 3060's native bf16 tensor cores, the same code is 2.45x *faster* once the model is large enough (see above).

The same benchmark shows the opposite trade for int8: weight-only quantization makes the model 3.5x smaller with negligible quality loss, but the forward pass is 7% *slower* (57 → 61 ms), because weights are dequantized on every call. Int8 here saves storage and memory bandwidth, not compute. A real speedup would need int8 matrix-multiply kernels.

## What's inside

| Component | What it does | How it's verified |
|---|---|---|
| **GPT model** (`model.py`) | Pre-norm decoder, causal attention via `scaled_dot_product_attention`, GPT-2 residual init scaling | Causality test: changing token *t* never changes logits before *t* |
| **KV cache** | Each decoding step processes 1 new token and reuses cached keys and values, instead of re-running the whole prefix | Logits match a full forward pass at every position; generated tokens are **identical** with and without the cache; multi-token chunks use a correctly offset causal mask |
| **Gradient checkpointing** | Frees each block's activations after forward and recomputes them during backward, trading compute for memory | Loss and **every parameter gradient** match the non-checkpointed model |
| **bf16 autocast** | Mixed precision on CPU; the loss stays in float32 | Loss within 2% of fp32, all gradients finite |
| **Int8 quantization** (`quant.py`) | Per-output-channel symmetric weight quantization, implemented directly (not a framework call) | Every weight's error ≤ half a quantization step (the theoretical bound); validation-loss change is measured |
| **Activation memory** (`memory.py`) | Counts the exact bytes autograd saves for backward, using `saved_tensors_hooks` | Exact for a single linear layer (equals input bytes); confirms checkpointing more than halves activation memory |
| **Training loop** (`train.py`) | AdamW (no decay on biases or LayerNorm), warmup + cosine schedule, gradient clipping, a fixed evaluation set | Same seed gives **bitwise-identical** losses and weights; a different seed gives a different run; the model memorizes periodic text |

## Why measure memory this way

On CPU there's no `torch.cuda.max_memory_allocated`, and process RSS is noisy. Autograd's saved-tensor hooks see every tensor kept for the backward pass, which is exactly the activation memory that checkpointing and lower precision are meant to reduce. Storages are deduplicated and parameters excluded. For non-reentrant checkpointing, PyTorch applies only the innermost hooks, so tensors inside a checkpointed block are correctly *not* counted. The block inputs the checkpoint keeps alive are added back explicitly, so the comparison stays fair.

## Reproducibility

Every run writes `config.json` (all hyperparameters plus the PyTorch, Python, and thread-count versions), `metrics.jsonl`, `final.json`, `ckpt.pt`, `tokenizer.json`, and `sample.txt`. Model initialization and data order use separate seeds. Validation uses the last 10% of the text, which is never seen in training, and is evaluated on the same fixed batches every time, so numbers are comparable across steps and runs.

## CI

Every push runs the full test suite on Python 3.11 and 3.12, then does an end-to-end smoke run on real TinyShakespeare (train → bench → report). The smoke run's `RESULTS.md` is uploaded as a build artifact.

## Layout

```
kestrel/
  model.py    GPT, KV cache, gradient checkpointing, generation
  data.py     TinyShakespeare download, char tokenizer, seeded batching
  train.py    reproducible training CLI (fp32 / bf16, checkpointing)
  quant.py    int8 weight-only quantization
  memory.py   exact activation-memory measurement
  bench.py    training-step, generation, and quantization benchmarks
  report.py   RESULTS.md generator
tests/        equivalence, numerics, quantization bounds, reproducibility, end-to-end
```
