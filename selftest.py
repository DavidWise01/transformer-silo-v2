#!/usr/bin/env python3
"""Verify-first self-test for the v2 head-to-head. Proves, with no network, that
the comparison is computed correctly AND framed honestly:
(1) explained variance is a real [0,1] elbow metric with the right anchors
(EVR(K=1)=0, EVR(K=N)=1 for distinct tokens); (2) it is monotonically
non-decreasing in K; (3) at K=N the silo IS the plain transformer (output
agreement 1.0) -- the honest baseline; (4) the attention-pair speedup is exactly
N^2/K^2; (5) output agreement is a valid cosine; (6) the recommended K is the
smallest K clearing the variance threshold; (7) deterministic; (8) the verdict is
framed as a TRADE-OFF and never claims the silo beats the plain model.
"""
from __future__ import annotations
from bench import (compare, sweep, verdict, plain, silo,
                   explained_variance, attention_pairs)
from silo import embed, centrifuge

fails = 0
def check(cond, msg):
    global fails
    print(("ok  · " if cond else "FAIL· ") + msg)
    fails += 0 if cond else 1


DISTINCT = ["the", "cat", "sat", "on", "mat", "run", "sky"]   # 7 distinct-salt words
N = len(DISTINCT)

# 1. Explained variance is a real [0,1] metric with the right anchors.
sw = sweep(DISTINCT)
evr = [r["variance_explained"] for r in sw["rows"]]
check(all(0.0 <= e <= 1.0 for e in evr), "explained variance stays in [0,1]")
check(evr[0] == 0.0, "EVR(K=1) = 0 (one centroid = the global mean, no variance explained)")
check(abs(evr[-1] - 1.0) < 1e-9, "EVR(K=N) = 1 for distinct tokens (each its own intent)")

# 2. Non-decreasing in K (the elbow curve only rises). NOTE: this is an EMPIRICAL
#    invariant of this deterministic seeding, not a theorem -- Lloyd's from a fixed
#    seed does not guarantee EVR is monotonic in K in general.
check(all(evr[i] <= evr[i + 1] + 1e-9 for i in range(len(evr) - 1)),
      f"explained variance is non-decreasing in K here ({evr})")

# 3. The honest baseline: at K = N the silo IS the plain transformer -- and this
#    now holds by construction (K==N short-circuits to the raw embeddings), so it
#    is exact for DUPLICATE-token contexts too, not just distinct ones.
cN = compare(DISTINCT, N)
check(abs(cN["output_agreement"] - 1.0) < 1e-9,
      "at K=N the silo output equals the plain transformer (agreement 1.0)")
check(cN["attn_speedup"] == 1.0, "at K=N there is no attention speedup (1.0x)")
DUP = ["cat", "cat", "sky", "sky", "sky", "mat"]              # duplicate tokens
cDup = compare(DUP, len(DUP))
check(abs(cDup["output_agreement"] - 1.0) < 1e-9 and abs(cDup["variance_explained"] - 1.0) < 1e-9,
      "at K=N the silo == plain EVEN WITH DUPLICATE TOKENS (agreement 1.0, EVR 1.0)")

# 4. The rate axis: attention pairs are N^2 (plain) vs K^2 (silo); speedup = N^2/K^2.
c3 = compare(DISTINCT, 3)
check(c3["plain_pairs"] == N * N and c3["silo_pairs"] == 3 * 3,
      f"attention pairs: plain N^2={N*N}, silo K^2=9")
check(abs(c3["attn_speedup"] - round((N * N) / 9.0, 2)) < 1e-9, "attention speedup is N^2/K^2 (reported to 2 dp)")
c1 = compare(DISTINCT, 1)
check(c1["attn_speedup"] == float(N * N), "K=1 gives the maximum N^2x attention cut")

# 5. Output agreement is a valid cosine in [-1, 1].
check(all(-1.0 - 1e-9 <= r["output_agreement"] <= 1.0 + 1e-9 for r in sw["rows"]),
      "output agreement is a valid cosine in [-1,1]")

# 6. Recommended K is the smallest K clearing the variance threshold.
rec = sw["recommended_k"]
below = [r for r in sw["rows"] if r["k"] < rec]
check(all(r["variance_explained"] < sw["threshold"] for r in below)
      and next(r for r in sw["rows"] if r["k"] == rec)["variance_explained"] >= sw["threshold"],
      f"recommended K={rec} is the smallest K reaching {int(sw['threshold']*100)}% variance")

# 7. Deterministic.
check(sweep(DISTINCT)["rows"] == sw["rows"], "the sweep is deterministic")

# 8. HONESTY: the verdict is a trade-off, never a 'silo wins' claim.
v = verdict(DISTINCT).lower()
check("trade-off" in v and "free win" in v, "the verdict frames it as a trade-off, not a free win")
check(not any(w in v for w in ("beats", "outperforms", "better than", "wins", "superior")),
      "the verdict never claims the silo beats the plain transformer")

print("\n" + ("SOME CHECKS FAILED" if fails else "all transformer-silo-v2 checks passed"))
raise SystemExit(1 if fails else 0)
