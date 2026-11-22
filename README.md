# ragmini

A small retrieval pipeline in pure NumPy, evaluated the way I evaluate everything else: labeled queries, controls, and planted bad variants.

**Pipeline:** chunk → BM25 (sparse) and LSA (dense: TF-IDF + truncated SVD) → reciprocal rank fusion → phrase-overlap rerank.
The dense stage is LSA, not a neural embedding model, and the reranker counts shared query bigrams; it is not a cross-encoder.

**Corpus:** a frozen snapshot of the markdown docs of four of my other projects (Forge, Kestrel, gradproof, kvdiff: 18 files, 117 chunks), committed in `corpus/` with the source commits in `corpus/SNAPSHOT.txt`. The snapshot is frozen so results don't change when those projects' docs do. **Labels:** 26 natural-language queries; a chunk is relevant if it contains the gold answer string, so labels are checkable by anyone.

## Results (`python run_eval.py`, saved in `results.txt`)

| Config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| bm25 | 0.38 | 0.77 | 0.51 |
| dense (LSA) | 0.27 | 0.73 | 0.44 |
| hybrid | 0.38 | 0.77 | 0.52 |
| hybrid + rerank | 0.38 | 0.73 | 0.52 |

Random ranking scores about 0.05 at recall@5 (analytic chance), so the pipeline is far above chance. The bootstrap 95% intervals (for example 0.62 to 0.92 for hybrid recall@5) overlap across every config, so **this set cannot say which config is best**. I do not claim hybrid or rerank helps.

**Changelog:** an earlier version read the docs of whatever sibling folders existed, so the corpus changed under it and the evaluation stopped running (a gold answer string had been edited out of a source doc). It now reads the frozen `corpus/`. Four gold strings were updated to the current wording of the same answers (the queries are unchanged), and the numbers above were regenerated, not carried over.

## What failed, and why
Most misses are vocabulary mismatch: the query says "activation function" and the doc says "GELU"; "8-bit" versus "int8". Lexical and LSA methods both miss these; a neural embedding model is the natural next step.

## Checks (`pytest tests`)
All 9 pass. Passing: chunker keeps every word, BM25 matches a hand computation, fusion prefers agreement, every gold string exists, pipeline far above chance, reversed ranking is caught, shuffled labels collapse, runs are deterministic.
**Planted bad variant:** removing IDF from BM25 lowers recall@5 (0.69 vs 0.77) on this corpus. On the earlier, smaller corpus the same change was invisible, so a mutant check depends on the evaluation set being large enough to see it.

## Limits
Small corpus, small query set, queries written by me, no neural embeddings, no production index (exact search, not approximate nearest neighbor).
