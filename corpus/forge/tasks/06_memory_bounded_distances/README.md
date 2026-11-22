# Pairwise distances under a memory budget

A nearest-neighbor evaluation step runs out of memory. `starter.py` computes squared Euclidean distances by broadcasting, which builds an `N x M x D` temporary array: for 2,000 x 500 points in 32 dimensions, that's 256 MB just to produce an 8 MB answer.

**Implement** `pairwise_sq_dists(A, B)` for `A` of shape `(N, D)` and `B` of shape `(M, D)` (float64), returning `out[i, j] = ||A[i] - B[j]||^2` with shape `(N, M)`, such that:

- Values match an exact reference (relative error ≤ 1e-9 on well-conditioned data).
- **Peak extra memory stays under 3x the size of the output** (measured with `tracemalloc`).
- No distance is ever **negative**, and a point's distance to itself is ~0, even when all points sit far from the origin (e.g. every coordinate near 1e5).
- The inputs are not modified.

*Hint:* `||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b` avoids the 3-D temporary, but it has its own numerical trap.
