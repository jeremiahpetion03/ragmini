# Tiled attention with online softmax

`starter.py` computes attention correctly, but it materializes the full `n x m` score matrix. At long sequence lengths that matrix dominates memory, and on a GPU, writing it to and reading it from slow global memory dominates time. FlashAttention avoids this by processing keys in tiles and keeping only running statistics. This task is that algorithm, in NumPy.

**Implement** `tiled_attention(Q, K, V, key_mask, block)`:

- `Q`: `(n, d)`, `K`: `(m, d)`, `V`: `(m, dv)`, float64.
- `key_mask`: `(m,)` bool, `True` = key may be attended to (for example, `False` for padding).
- `block`: positive integer tile size. Process keys in tiles of `block` rows; the last tile may be smaller.
- Returns `(n, dv)`: `softmax(Q K^T / sqrt(d))` over the allowed keys, times `V`.

Requirements:

- Matches full attention to floating-point rounding, for any `block` (1, not dividing `m`, larger than `m`).
- Stable when scores are in the thousands.
- If **every** key is masked, return zeros, never NaN.
- A tile in which every key is masked must not produce NaN (`-inf - (-inf)` is NaN).
- Peak extra memory at most **50% of the full `n x m` score matrix** for `n = m = 2000`, `block = 128`.

Hint: keep, for every query, a running max `m_i`, a running sum of exponentials `l_i`, and an output accumulator. When a tile raises the max, rescale the old sum and accumulator by `exp(old_max - new_max)` before adding the tile.
