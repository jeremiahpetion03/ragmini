# Numerically stable log-softmax

Training a language model produces NaN loss once logits grow large. The culprit is `log_softmax` in `starter.py`.

**Implement** `log_softmax(x, axis=-1)` in NumPy so that:

- It returns `log(softmax(x))` along `axis`, with the same shape and dtype as `x` (float32 in → float32 out).
- It is finite and accurate for **very large** logits (e.g. 1e4) and **very negative** logits (e.g. −1e4), in both float32 and float64.
- Entries equal to `-inf` (masked positions) return exactly `-inf` and never NaN. Every slice along `axis` has at least one finite entry.
- It works for any `axis` on arrays of any rank.

Hidden checks compare against an independent reference and cover each requirement above.
