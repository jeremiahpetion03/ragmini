# kvdiff

A differential tester for KV-cache correctness in [Hugging Face Transformers](https://github.com/huggingface/transformers). It checks that batching, padding, caching, and `generate()` don't change what a model computes, and it separates real failures from ordinary floating-point rounding before calling anything a bug.

## What it checks

For every model × attention implementation × precision, each path is compared against a reference that does **not** share the code under test:

| Check | Path under test | Reference |
|---|---|---|
| **padding** | padded batch, full forward | each row alone, unpadded |
| **cache** | padded batch, token-by-token decoding through the KV cache | padded batch, full forward |
| **generate** | `model.generate(use_cache=True)`, batched with left padding | each row generated alone, no cache |

Prompts are deliberately uneven (one is 3 tokens long next to 37-token rows), so the batch needs heavy left padding. Position ids come from the attention mask, so padding never shifts positions. fp32 runs with TF32 disabled, because Ampere GPUs otherwise run fp32 matmuls at about 10 mantissa bits, which would hide real discrepancies.

## Results

Run on an NVIDIA RTX 3060 with Transformers 5.23. The first run used PyTorch 2.15 (CUDA 13.4); the committed `results.json` and `triage.json` are from a re-run on PyTorch 2.11 (CUDA 12.8), with the same 54 checks and the same 8 flags. Models: GPT-2, SmolLM2-135M (Llama architecture), Qwen2.5-0.5B (RoPE, grouped-query attention). Attention: eager and sdpa. Precision: fp32, bf16, fp16.

**54 checks: 46 passed, 8 flagged.** All 18 fp32 checks passed, with relative errors around 2×10⁻⁶. Triage of the 8 flags:

| Flags | Configuration | Verdict | Evidence |
|---|---|---|---|
| 6 | GPT-2, bf16 (eager and sdpa) | **False positive** (my threshold) | Every flagged token had a reference margin of exactly 1.000. GPT-2's logits sit near ±100 to ±200, where adjacent bf16 values are 1.0 apart, so the "preferred" token was a tie. Against an fp32 ground truth, cached and full-forward errors match (ratio 0.60–1.25). |
| 1 | Qwen2.5, eager, fp16 (`cache`) | **Real, deterministic NaN** | See below |
| 1 | Qwen2.5, eager, fp16 (`generate`) | Same root cause | First-step divergence on the same heavily padded row |

### The Qwen2.5 eager fp16 NaN

Re-running the exact inputs 30 times ([`repro2.py`](repro2.py)):

- **NaN logits in 30/30 runs**, bit-identical every time, always on the 3-token prompt with 34 padding tokens.
- A forward hook on every module reports the first **module** whose output is non-finite: `model.layers.1.self_attn.o_proj`, during prefill. Direct instrumentation (below) showed the NaN is created earlier, inside the attention math, which is not a module.
- Controls: the same row **without padding** is clean. **sdpa** is clean. **bf16** is clean. **fp32** is clean.

**Mechanism, measured.** [`snapshot.py`](snapshot.py) copies every module's inputs before it runs and stops at the first non-finite output, with logits bit-identical with hooks on and off. [`confirm_mask_overflow.py`](confirm_mask_overflow.py) then recomputes the eager attention scores. Results: [`snapshot.json`](snapshot.json), [`snapshot_sdpa.json`](snapshot_sdpa.json), [`confirm_mask_overflow.json`](confirm_mask_overflow.json).

1. Eager attention adds an additive mask whose masked value is fp16's minimum, **−65,504** (not −∞).
2. Near 65,504, adjacent fp16 values are 32 apart, so `score + (−65,504)` rounds to **−∞** once the score is below about −16.
3. In **layer 1, head 9**, for all **34 padding query positions** of the 3-token row, every key's score was below that threshold (the highest was **−17.375**). Every masked score became −∞. The same sum in fp32 produced no −∞.
4. Softmax over a row that is entirely −∞ is NaN. The NaN softmax rows were **exactly** the all-−∞ rows (34 of 34). Layer 0 had none.
5. The NaN flows through `o_proj` (its input already held 2,176 NaN = 34 positions × 64 head dimensions, and no infinities) into the residual stream, and into the KV cache at all 34 padding positions from layer 2 on.
6. In layer 2, the padding positions' keys are NaN. Masking can't remove them, because NaN + (−65,504) is still NaN, and one NaN score makes a whole softmax row NaN: all **518** of that sequence's attention rows in layer 2 (37 query positions × 14 heads) were NaN. From layer 3 on, the row's 3 real tokens are non-finite in the KV cache too, and its logits are NaN right after prefill.

**Why only the cached path fails.** The padded full forward pass (prompt and continuation in one pass, no cache) is clean on the same row, though its 34 padding positions are also fully masked: [`confirm_mask_overflow.py --path full`](confirm_mask_overflow_full.json) shows 34 rows with no allowed key in both paths. The difference is the raw scores. Head 9's best score in those rows was **−17.375** in the prompt-only prefill, past the −16 threshold, but **−11.414** in the full pass, which is safe. A fully masked row gets the same −65,504 added to every key, so the offset cancels inside the softmax, and the row attends to every key in the pass, weighted by raw score. The prefill has 37 keys and the full pass has 52 (it includes the continuation), so the padding positions' hidden states differ, and by layer 1 head 9 lands on opposite sides of the threshold. Both paths are numerically fragile; one happened to cross.

sdpa receives a boolean mask instead, outputs exact zeros at padding positions, and stays finite. bf16 and fp32 have exponent range to spare, so adding their minimum to a score of −17 does not overflow.

**My first hypothesis was wrong.** I had attributed the NaN to an activation overflowing fp16 in layer 1's `o_proj`, because that was the first module the hooks flagged. Instrumenting inside the module showed `o_proj`'s input was already NaN, and recomputing `o_proj` in fp32 never exceeds 65,504 at real positions. Module-level hooks can only blame modules; this failure happens in functional code between them.

**Two fixes, tested** ([`test_fixes.py`](test_fixes.py), results in [`test_fixes.json`](test_fixes.json)). Each replaces the eager attention function on the same failing inputs, compared with an fp32 eager reference over prefill and 15 decode steps:

| Variant | NaN rows | Rows 1–5 vs. original | Row 0 max \|Δ\| vs fp32 | Row 0 argmax agreement |
|---|---|---|---|---|
| Original (Transformers) | row 0 | — | NaN | — |
| My copy, no fix (control) | row 0 | bit-identical | NaN | — |
| **Add the mask in fp32** | none | **bit-identical** | 0.0545 | 16/16 |
| **Zero rows with no allowed key** | none | **bit-identical** | 0.0545 | 16/16 |
| sdpa fp16 (baseline) | none | different kernel | 0.0336 | 16/16 |

- The control reproduces the library's output bit for bit, including the NaN, so the comparison is valid.
- Both fixes change nothing for the rows the bug never touched, and leave row 0 as accurate as the other eager rows (0.041–0.071).
- Both give row 0 identical numbers: real tokens give padding keys a weight of exactly 0, so what a padding row holds doesn't matter, as long as it is finite.
- sdpa is slightly more accurate on every row, a kernel difference that predates either fix.

**Prior report.** The symptom (Qwen2, eager attention, fp16 → NaN) was reported in Transformers PR [#33317](https://github.com/huggingface/transformers/pull/33317) (September 2024), which was closed. That PR attributed the NaN to q·k overflow on long inputs and proposed computing q·k and the mask addition in fp32. This project rediscovered the symptom independently through differential testing and measured a different trigger: the mask addition overflows with modest negative scores on fully masked padding rows. The fp32 mask addition from that PR would also prevent it. It still reproduces in Transformers 5.23, with a standalone script: [`minimal_repro.py`](minimal_repro.py).

**Reported upstream:** [huggingface/transformers#51079](https://github.com/huggingface/transformers/issues/51079), with the standalone reproduction, the measured cause, and both tested fixes.

### Hypotheses this project ruled out

- **"Padding activations overflow fp16."** The first reproduction ([`repro.py`](repro.py)) used different prompts: all 100 trials were clean, and padding-position activations peaked at 1,281, about 50x below fp16's limit.
- **"Layer 1's `o_proj` overflows."** Ruled out by direct instrumentation, as described above: the NaN comes from the attention mask overflowing to −∞, not from a large activation.

### Resolved: why the first run reported token flips instead of NaN

The first `kvdiff.py` run reported token flips for the Qwen2.5 eager fp16 cache check, while `repro2.py` produced NaN on the same inputs. The cause was a bug in kvdiff's own scoring: when combining rows into one result, a NaN worst error was overwritten by the next row's finite value. Row 0 was NaN and rows 1–5 were finite, so the NaN vanished from the summary and only row 0's broken argmaxes showed up, as flips. The check was still flagged, but mislabeled. After the one-line fix, the re-run reports `rel=nan` for that check; every other check is unchanged.

## Methodology notes

- **Every threshold is per precision.** Relative-error tolerance is 1e-4 for fp32, 1e-2 for fp16, and 5e-2 for bf16.
- **Token flips count only with a clear margin.** A flip counts only when the reference preferred its token by more than a per-precision margin. The GPT-2 false positives showed that the margin also has to account for the precision's step size at the logits' magnitude.
- **Triage against fp32 ground truth.** [`triage.py`](triage.py) measures every low-precision path against the same fp32 reference. If the cached path's error is comparable to the full forward's, the disagreement is rounding; if it is several times larger, the cache path is adding error. NaN is reported as `NON-FINITE` and never as rounding (an earlier version mislabeled it, because comparisons with NaN are always false).

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu134
pip install transformers accelerate

python kvdiff.py      # 54 checks; writes results.json
python triage.py      # per-row error vs fp32 ground truth; writes triage.json
python repro.py       # overflow hypothesis, 5 controlled configurations
python repro2.py      # exact failing inputs x 30, first non-finite module
```

All scripts also run on CPU (`--device cpu` for kvdiff and triage), much more slowly. Python 3.13 is recommended: some dependencies don't yet publish Windows wheels for 3.15.

## Files

```
kvdiff.py     differential tester: padding, cache, and generate checks
triage.py     error vs fp32 ground truth, per row, with NaN handling
repro.py      controlled test of the overflow hypothesis
repro2.py     deterministic reproduction and first-bad-module localization
results.json  kvdiff output from the run above
triage.json   triage output from the run above
snapshot.py   first-non-finite snapshot + attention probe (inputs copied before each module runs)
confirm_mask_overflow.py  measures whether fp16 score + mask rounds to -inf
snapshot*.json, confirm_mask_overflow.json  outputs of the two scripts above (RTX 3060)
confirm_mask_overflow_full.json  the same check on the padded full forward pass (no overflow)
minimal_repro.py  standalone reproduction (torch + transformers only): one prompt, controls, pad-count sweep
test_fixes.py  tests the two fixes against an fp32 reference, with a bit-identical control
test_fixes.json  its results (RTX 3060)
```

