import glob, sys, os
from ragmini import Retriever, chunk, evaluate, summarize
from queries import QUERIES

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
files = sorted(f for f in glob.glob(os.path.join(root, "*", "**", "*.md"), recursive=True))
chunks = [c for f in files for c in chunk(open(f, encoding="utf-8").read())]
print(f"{len(files)} files, {len(chunks)} chunks, {len(QUERIES)} queries")
R = Retriever(chunks)
def cell(m):
    return f"{m[0]:.2f} [{m[1]:.2f}-{m[2]:.2f}]".rjust(18)

print(f"{'config':<16}" + "".join(k.rjust(18) for k in ("recall@1", "recall@3", "recall@5", "mrr")))
for mode in ("bm25", "dense", "hybrid"):
    for rerank in (False, True):
        s = summarize(evaluate(R, QUERIES, mode, rerank))
        name = f"{mode}{'+rerank' if rerank else ''}"
        print(f"{name:<16}" + "".join(cell(s[k]) for k in ("recall@1", "recall@3", "recall@5", "mrr")))

import numpy as np
from ragmini import relevant_set
rng = np.random.default_rng(0)
rand = []
for q, g in QUERIES:
    rel = relevant_set(chunks, g)
    r = list(rng.permutation(len(chunks)))
    rand.append({"recall@5": float(any(d in rel for d in r[:5]))})
print("random-ranking control recall@5 (one draw per query):", np.mean([x["recall@5"] for x in rand]).round(2),
      "| analytic chance:", round(float(np.mean([1 - np.prod([(len(chunks)-len(relevant_set(chunks,g))-j)/(len(chunks)-j) for j in range(5)]) for _, g in QUERIES])), 2))
print("\nhybrid+rerank misses at recall@5:")
for q, g in QUERIES:
    rel = relevant_set(chunks, g)
    r = R.search(q, "hybrid")
    if not any(d in rel for d in r[:5]):
        rank = next((i + 1 for i, d in enumerate(r) if d in rel), None)
        print(f"  rank={rank}  {q}")
