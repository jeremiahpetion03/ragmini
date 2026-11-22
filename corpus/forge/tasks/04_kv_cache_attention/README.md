# KV-cache attention for incremental decoding

`starter.py` produces correct attention outputs during generation, but it gets slower with every token. On each step it re-projects keys and values for the **entire** history.

A single attention head has projection matrices `Wq, Wk, Wv` (each `d x d`). For token embeddings `x_0 ... x_t`:

```
q_t = x_t @ Wq,  k_i = x_i @ Wk,  v_i = x_i @ Wv
out_t = softmax( q_t · [k_0 .. k_t] / sqrt(d) ) @ [v_0 .. v_t]
```

Each token attends to itself and every earlier token.

**Implement** class `CachedAttention(Wq, Wk, Wv)` with method `step(x) -> out` (shapes `(d,)` → `(d,)`) that:

- Produces outputs matching the formula above to floating-point rounding, at every step.
- Projects each token **once** and reuses cached keys and values. It must be at least **3x faster** than the starter over 800 steps with `d = 64`.
- Stays finite and accurate when attention scores are large.
- Keeps state per instance: two caches used at the same time must not interfere.
