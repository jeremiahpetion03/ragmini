"""ragmini: a small, fully measured retrieval pipeline in pure NumPy.

Stages: chunk -> BM25 (sparse) + LSA (dense, TF-IDF + truncated SVD) -> reciprocal
rank fusion -> phrase-overlap rerank. Evaluated with recall@k, MRR and a bootstrap CI.
The dense stage is LSA, NOT a neural embedding model. Labels are verifiable: a chunk
is relevant to a query if it contains the gold answer string.
"""
import math
import re
from collections import Counter

import numpy as np

STOP = set("a an the of to in on for and or is are was were it its this that with as by at be "
           "do does did how what why which who when from not no than then so can".split())
TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def norm(text):
    return re.sub(r"\s+", " ", text.lower().replace("*", "").replace("`", "")).strip()


def tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def chunk(text, min_words=60, max_words=120):
    """Split on blank lines, merge small paragraphs, hard-split long ones."""
    out, cur = [], []
    for para in re.split(r"\n\s*\n", text):
        words = para.split()
        if not words:
            continue
        for i in range(0, len(words), max_words):
            cur.extend(words[i:i + max_words])
            if len(cur) >= min_words:
                out.append(" ".join(cur))
                cur = []
    if cur:
        out.append(" ".join(cur))
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean()
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.tf))
        for t in set(tokens(query)):
            if t not in self.idf:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                    s[i] += self.idf[t] * f * (self.k1 + 1) / denom
        return s


class LSA:
    """Dense retrieval stand-in: TF-IDF -> truncated SVD -> cosine."""

    def __init__(self, docs, dim=48):
        toks = [tokens(d) for d in docs]
        df = Counter(t for ts in toks for t in set(ts))
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
        X = np.zeros((n, len(self.vocab)))
        for i, ts in enumerate(toks):
            for t, f in Counter(ts).items():
                X[i, self.vocab[t]] = (1 + math.log(f)) * self.idf[self.vocab[t]]
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(dim, len(S))
        self.V = Vt[:k].T
        D = X @ self.V
        self.D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)

    def scores(self, query):
        x = np.zeros(len(self.vocab))
        for t, f in Counter(tokens(query)).items():
            if t in self.vocab:
                x[self.vocab[t]] = (1 + math.log(f)) * self.idf[self.vocab[t]]
        q = x @ self.V
        nq = np.linalg.norm(q)
        return self.D @ (q / nq) if nq > 0 else np.zeros(len(self.D))


def rrf(rankings, k=60):
    """Reciprocal rank fusion over lists of doc indices (best first)."""
    s = Counter()
    for r in rankings:
        for rank, d in enumerate(r):
            s[d] += 1.0 / (k + rank + 1)
    return [d for d, _ in s.most_common()]


def bigrams(ts):
    return {(a, b) for a, b in zip(ts, ts[1:])}


class Retriever:
    def __init__(self, docs, dim=48):
        self.docs = docs
        self.bm25 = BM25(docs)
        self.lsa = LSA(docs, dim)
        self.dtoks = [tokens(d) for d in docs]

    @staticmethod
    def order(s):
        return list(np.argsort(-s, kind="stable"))

    def search(self, q, mode="hybrid", top=30, rerank=True):
        if mode == "bm25":
            r = self.order(self.bm25.scores(q))
        elif mode == "dense":
            r = self.order(self.lsa.scores(q))
        else:
            r = rrf([self.order(self.bm25.scores(q)), self.order(self.lsa.scores(q))])
        r = r[:top]
        if not rerank:
            return r
        qb = bigrams(tokens(q))
        bonus = {d: len(qb & bigrams(self.dtoks[d])) for d in r}
        # stable: fused rank breaks ties; each shared query bigram moves a doc up
        pos = {d: i for i, d in enumerate(r)}
        return sorted(r, key=lambda d: (pos[d] - 3 * bonus[d]))


def relevant_set(chunks, gold):
    g = norm(gold)
    return {i for i, c in enumerate(chunks) if g in norm(c)}


def metrics(ranked, rel, ks=(1, 3, 5)):
    out = {f"recall@{k}": float(any(d in rel for d in ranked[:k])) for k in ks}
    rr = next((1 / (i + 1) for i, d in enumerate(ranked) if d in rel), 0.0)
    out["mrr"] = rr
    return out


def evaluate(retriever, queries, mode, rerank=True, shuffle_docs=None):
    rows = []
    for q, gold in queries:
        rel = relevant_set(retriever.docs, gold)
        assert rel, f"gold string not found in corpus: {gold!r}"
        rows.append(metrics(retriever.search(q, mode, rerank=rerank), rel))
    return rows


def summarize(rows, boots=2000, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for key in rows[0]:
        v = np.array([r[key] for r in rows])
        bs = rng.choice(v, size=(boots, len(v))).mean(axis=1)
        res[key] = (v.mean(), *np.percentile(bs, [2.5, 97.5]))
    return res
