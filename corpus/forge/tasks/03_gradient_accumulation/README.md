# Gradient accumulation over uneven micro-batches

A batch is too large for memory, so it is split into micro-batches and gradients are accumulated. Results drift from single-batch training whenever the dataset size is not a multiple of the micro-batch size. The accumulation in `starter.py` is to blame.

`grad_fn(X, y, w)` returns the **mean** gradient over the rows it is given.

**Implement** `accumulated_gradient(batches, w, grad_fn)`:

- `batches` is a list of `(X, y)` micro-batches. They may have **different sizes**, and some may be **empty** (0 rows).
- Return a gradient **exactly equal** (to floating-point rounding) to `grad_fn` applied to all rows at once.
- **Never** pass more rows to `grad_fn` than the largest single micro-batch: the whole point is that the full batch doesn't fit in memory.
- Never call `grad_fn` on an empty micro-batch, since a mean over zero rows is undefined.
