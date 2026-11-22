# nDCG@k for ranking and retrieval evaluation

A retrieval system scores `N` candidate documents for each of `Q` queries. Offline evaluation needs **normalized discounted cumulative gain at k** (nDCG@k) for every query. `starter.py` is correct but loops over queries in Python, which is too slow for evaluating thousands of queries on every training checkpoint.

**Implement** `ndcg_at_k(scores, relevance, k)`:

- `scores`: `(Q, N)` float array of model scores. Higher means ranked earlier. Scores are finite or `-inf` (filtered out, ranked last).
- `relevance`: `(Q, N)` array of graded relevance labels, non-negative (for example 0 to 3).
- `k`: a positive integer.
- Returns a `(Q,)` float array.

Definitions, per query:

```
rank documents by score, descending; ties go to the LOWER document index first
DCG@k  = sum over the top min(k, N) positions i = 1, 2, ... of (2**rel_i - 1) / log2(i + 1)
IDCG@k = the same sum with documents sorted by relevance, descending (the best possible ranking)
nDCG@k = DCG@k / IDCG@k
```

Requirements:

- Matches the definition above to floating-point rounding, including graded (not just binary) relevance.
- A query with **no relevant documents** (IDCG = 0) scores exactly `0.0`, never NaN.
- Works when `k` is larger than `N`.
- Does not modify its inputs.
- At least **5x faster** than a per-query Python loop for `Q = 2000`, `N = 100`, `k = 10`.
