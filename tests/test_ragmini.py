import glob, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ragmini as rm
from queries import QUERIES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "corpus"))
FILES = sorted(f for f in glob.glob(os.path.join(ROOT, "*", "**", "*.md"), recursive=True))
CHUNKS = [c for f in FILES for c in rm.chunk(open(f, encoding="utf-8").read())]


def r5(retr, mode="hybrid", rerank=True):
    rows = rm.evaluate(retr, QUERIES, mode, rerank)
    return np.mean([r["recall@5"] for r in rows])


def test_chunker_keeps_every_word():
    text = "alpha beta\n\ngamma " + "w " * 300 + "\n\ndelta"
    out = rm.chunk(text)
    assert " ".join(out).split() == text.split()
    assert all(len(c.split()) <= 240 for c in out)


def test_bm25_matches_hand_computation():
    docs = ["cat sat mat", "dog sat log", "cat cat cat"]
    b = rm.BM25(docs, k1=1.5, b=0.75)
    s = b.scores("cat")
    n, df = 3, 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    avg = 3.0
    expect = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 3 / avg))
    assert abs(s[0] - expect) < 1e-12
    assert s[1] == 0 and s[2] > s[0]


def test_rrf_prefers_agreement():
    assert rm.rrf([[1, 2, 3], [1, 3, 2]])[0] == 1
    assert set(rm.rrf([[0], [1]])) == {0, 1}


def test_gold_strings_exist_and_are_not_trivial():
    for _, gold in QUERIES:
        rel = rm.relevant_set(CHUNKS, gold)
        assert 0 < len(rel) <= 6, (gold, len(rel))


def test_pipeline_far_above_chance():
    assert r5(rm.Retriever(CHUNKS)) >= 0.6   # chance is about 0.06


def test_bad_variant_reversed_ranking_is_caught():
    R = rm.Retriever(CHUNKS)
    rows = []
    for q, g in QUERIES:
        rel = rm.relevant_set(CHUNKS, g)
        rev = list(reversed(R.order(R.bm25.scores(q))))
        rows.append(rm.metrics(rev, rel)["recall@5"])
    assert np.mean(rows) <= 0.1


import pytest


def test_bad_variant_no_idf_is_worse():
    good = rm.Retriever(CHUNKS)
    bad = rm.Retriever(CHUNKS)
    bad.bm25.idf = {t: 1.0 for t in bad.bm25.idf}
    assert r5(bad, "bm25", False) < r5(good, "bm25", False)


def test_bad_variant_shuffled_labels_collapse():
    R = rm.Retriever(CHUNKS)
    rng = np.random.default_rng(0)
    gold = [g for _, g in QUERIES]
    rng.shuffle(gold)
    shuffled = [(q, g) for (q, _), g in zip(QUERIES, gold)]
    rows = rm.evaluate(R, shuffled, "hybrid")
    assert np.mean([r["recall@5"] for r in rows]) < 0.3


def test_deterministic():
    a = rm.Retriever(CHUNKS).search(QUERIES[0][0])
    b = rm.Retriever(CHUNKS).search(QUERIES[0][0])
    assert a == b
