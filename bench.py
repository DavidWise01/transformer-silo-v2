#!/usr/bin/env python3
"""transformer-silo v2 — the honest head-to-head.

v1 built the silo and asked the fair question out loud: a clustering pre-pass
saves compute, but does the compression COST you? These are UNTRAINED toy
transformers, so "accuracy on a benchmark" would be meaningless -- there is
nothing trained to be accurate. So v2 uses the honest, industry-standard tests
for exactly this question:

  * THE ELBOW METHOD  — variance explained by K clusters (the textbook way to
    evaluate k-means: explained variance = 1 - WCSS/TSS).
  * RATE vs DISTORTION — what you SAVE (attention pairs fall from N^2 to K^2)
    against what you LOSE (variance, and output fidelity vs the plain model).

Plain transformer = the K = N end of the sweep (no clustering, attends over all
N tokens, 100% variance, output agreement 1.0 by construction). The silo is any
K < N. The comparison is a trade-off, reported honestly -- never "the silo wins."
"""
from __future__ import annotations
import math
from silo import embed, centrifuge, transformer, D, dot


# ---------- the two standard axes ----------
def attention_pairs(n: int) -> int:
    """Self-attention compares every position to every position: N^2 pairs."""
    return n * n


def _mean(vs):
    return [sum(v[d] for v in vs) / len(vs) for d in range(D)]


def _sq(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def tss(vs):
    """Total sum of squares -- the spread of the context around its mean."""
    m = _mean(vs)
    return sum(_sq(v, m) for v in vs)


def wcss(vs, centroids, assign):
    """Within-cluster sum of squares -- k-means inertia = the centrifuge energy."""
    return sum(_sq(vs[i], centroids[assign[i]]) for i in range(len(vs)))


def explained_variance(vs, centroids, assign):
    """The elbow-method metric: fraction of context variance the K intents keep."""
    t = tss(vs)
    if t <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, 1.0 - wcss(vs, centroids, assign) / t))


def _pooled(seq):
    out = transformer(seq)
    return [sum(o[d] for o in out) / len(out) for d in range(D)]


def cosine(a, b):
    na, nb = math.sqrt(dot(a, a)), math.sqrt(dot(b, b))
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


# ---------- plain vs silo ----------
def plain(context):
    vs = [embed(t) for t in context]
    return {"n": len(vs), "pairs": attention_pairs(len(vs)), "pooled": _pooled(vs)}


def silo(context, k):
    vs = [embed(t) for t in context]
    k = max(1, min(k, len(vs)))
    intents, assign, hist = centrifuge(vs, k)
    return {"k": k, "pairs": attention_pairs(k),
            "clustering_ops": len(vs) * k * len(hist),   # the one-time overhead, disclosed
            "evr": explained_variance(vs, intents, assign),
            "pooled": _pooled(intents), "assign": assign}


def compare(context, k):
    p, s = plain(context), silo(context, k)
    return {
        "n": p["n"], "k": s["k"], "compression": round(p["n"] / s["k"], 2),
        "plain_pairs": p["pairs"], "silo_pairs": s["pairs"],
        "attn_speedup": round(p["pairs"] / s["pairs"], 2),
        "clustering_ops": s["clustering_ops"],
        "variance_explained": round(s["evr"], 4),
        "variance_lost": round(1.0 - s["evr"], 4),
        "output_agreement": round(cosine(p["pooled"], s["pooled"]), 4),
    }


def sweep(context, threshold=0.9):
    """Sweep K = 1..N: the elbow curve + a recommended K (smallest K that keeps
    >= `threshold` of the variance)."""
    n = len(context)
    rows = [compare(context, k) for k in range(1, n + 1)]
    rec = next((r["k"] for r in rows if r["variance_explained"] >= threshold), n)
    return {"rows": rows, "recommended_k": rec, "threshold": threshold, "n": n}


def verdict(context, threshold=0.9):
    sw = sweep(context, threshold)
    rec = sw["recommended_k"]
    r = next(row for row in sw["rows"] if row["k"] == rec)
    return (f"On this context (N={sw['n']}), keeping {int(threshold*100)}% of the "
            f"variance needs K={rec} intents: a {r['attn_speedup']}x cut in attention "
            f"pairs ({r['plain_pairs']}->{r['silo_pairs']}) for {r['variance_lost']*100:.1f}% "
            f"variance lost and output agreement {r['output_agreement']}. A trade-off, not a "
            f"free win -- whether it is worth it depends on your data.")


if __name__ == "__main__":
    ctx = "cat cat sat sky sky sky run mat mat on".split()
    sw = sweep(ctx)
    print(f"{'K':>2} {'comp':>6} {'attn':>7} {'var%':>7} {'agree':>7}")
    for r in sw["rows"]:
        print(f"{r['k']:>2} {r['compression']:>5}x {r['attn_speedup']:>6}x "
              f"{r['variance_explained']*100:>6.1f}% {r['output_agreement']:>7}")
    print("\n" + verdict(ctx))
