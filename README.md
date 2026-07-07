# transformer-silo v2 — the honest head-to-head

v1 built the silo (a centrifuge that clusters the preloaded context, then a
transformer over the K intents) and asked the fair question out loud: *the
clustering pre-pass saves compute, but does the compression cost you?* v2
answers it — honestly.

## Why not an "accuracy" benchmark

These are **untrained** toy transformers with fixed weights. There is nothing
trained to be accurate, so an accuracy score would be theatre. Instead v2 uses
the **industry-standard, simple** tests for exactly this question:

- **The elbow method** — *variance explained* by K clusters (`1 − WCSS/TSS`, the
  textbook k-means metric). How much of the context do the K intents keep?
- **Rate vs distortion** — what you **save** (self-attention drops from `N²` to
  `K²` pairs) against what you **lose** (variance, and output fidelity vs the
  plain model). A Pareto trade-off, not a winner.

The **plain transformer is the `K = N` end** of the sweep: no clustering, attends
over all N tokens, 100% variance, output agreement 1.0 **by construction** (a
self-test anchor). Everything at `K < N` is the silo trading fidelity for compute.

## What it shows (default context, N=10)

| K | compression | attention cut | variance kept | output agreement |
|---|-------------|---------------|---------------|------------------|
| 1 | 10× | **100×** | 0% | 0.03 |
| **3** | **3.3×** | **11×** | **90%** | **0.99** |
| 5 | 2× | 4× | 100% | 0.99 |
| 10 (= plain) | 1× | 1× | 100% | 1.00 |

The **elbow is at K≈3**: an 11× cut in attention pairs for ~10% variance lost and
near-identical output. Whether that trade is worth it depends on your data — on
this toy it just shows the mechanism, measured with the right tools.

## Verify first

```bash
python selftest.py    # 14 checks: EVR anchors, monotonic elbow, K=N == plain, N²/K², honest verdict
python bench.py       # print the sweep + the verdict for a sample context
```

The self-test enforces the honesty, not just the math: the verdict is asserted to
be a **trade-off** and to **never** claim the silo "beats" the plain transformer.

## Files

| File | Role |
|------|------|
| `silo.py` | the v1 engine (centrifuge + toy transformer), vendored verbatim |
| `bench.py` | the comparison: `explained_variance`, `attention_pairs`, `compare`, `sweep`, `verdict` |
| `selftest.py` | proves the metrics + the honest framing |
| `index.html` | the head-to-head — sweep K, read the elbow, see rate vs distortion |

Sibling of [transformer-silo v1](https://davidwise01.github.io/transformer-silo/) and
[the-forward-pass](https://davidwise01.github.io/the-forward-pass/).

## The honest bottom line

The silo buys a `K²/N²` attention reduction at the cost of `(1 − variance
explained)`. That is a real, classic **rate–distortion trade-off** — the same
shape you get compressing anything. v2 does **not** claim the silo is better; it
gives you the curve and lets the elbow speak. A trained, benchmarked comparison
on real data would be v3 — and it would need real training, honestly reported.

---
David Lee Wise / ROOT0 / TriPod LLC · CC-BY-ND-4.0
