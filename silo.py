#!/usr/bin/env python3
"""transformer-silo v1 — a two-floor "silo": an intent-engine centrifuge that
clusters preloaded user context like-to-like, then a normal transformer over the
result.

  FLOOR 1 · the INTENT ENGINE (the centrifuge)
     ()\\<   intake: the user's preloaded context (a bag of token vectors)
     <\\o/>  spin: assign each token to its nearest centroid, recompute centroids,
             repeat. "Centrifugal, like-to-like" = convergent similarity
             clustering (Lloyd's algorithm): similar vectors settle into the same
             bin, and the total intra-cluster energy is non-increasing every spin
             until it settles. Output: K INTENT summaries (the centroids) -- a
             genuine compression of N context vectors down to K.
     >>>>   spit the K intents upstairs.

  FLOOR 2 · a NORMAL TRANSFORMER
     a real (toy) forward pass over the K intents -> a prediction.

Honest by construction: this is a runnable TOY that really does each step --
it converges, it compresses N->K, and it runs a real forward pass on the result.
It is not a trained model and makes no claim to beat a standard transformer; it
is an honest, see-through sketch of the architecture David described.
"""
from __future__ import annotations
import math

D = 4                      # vector width (shared with the transformer floor)


# ---------- small linear algebra (stdlib only, deterministic) ----------
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def add(a, b): return [x + y for x, y in zip(a, b)]
def sub(a, b): return [x - y for x, y in zip(a, b)]
def scale(v, s): return [x * s for x in v]
def dist2(a, b): return sum((x - y) ** 2 for x, y in zip(a, b))


def _tok_salt(t): return sum(ord(c) for c in t) % 11


def embed(tok):
    """Fixed deterministic embedding -- shared with the transformer floor."""
    s = _tok_salt(tok)
    return [round(math.sin((k + 1) * 1.3 + s * 0.7), 4) for k in range(D)]


# ---------- FLOOR 1: the intent engine (the centrifuge) ----------
def _seed_centroids(vectors, k):
    """Deterministic farthest-point seeding: start at vector 0, then repeatedly
    take the vector farthest from all chosen centroids. No RNG."""
    chosen = [list(vectors[0])]
    while len(chosen) < k:
        best_i, best_d = None, -1.0
        for i, v in enumerate(vectors):
            d = min(dist2(v, c) for c in chosen)
            if d > best_d:
                best_d, best_i = d, i
        chosen.append(list(vectors[best_i]))
    return chosen


def energy(vectors, centroids, assign):
    return sum(dist2(vectors[i], centroids[assign[i]]) for i in range(len(vectors)))


def _nearest(v, centroids):
    best_j, best_d = 0, float("inf")
    for j, c in enumerate(centroids):
        d = dist2(v, c)
        if d < best_d:
            best_d, best_j = d, j
    return best_j


def centrifuge(vectors, k, max_spins=25):
    """Spin the centrifuge: assign -> recenter -> repeat until it settles.
    Returns (intents, assignments, energy_history). history[0] is the energy of
    the initial (seed) clustering; each later entry is the energy after a
    recenter+reassign spin. It is monotonically non-increasing and settles -- so
    the drop from history[0] to history[-1] is the centrifuge's actual work."""
    centroids = _seed_centroids(vectors, k)
    assign = [_nearest(v, centroids) for v in vectors]      # seed assignment
    history = [round(energy(vectors, centroids, assign), 6)]  # energy at the seeds
    for _ in range(max_spins):
        # recompute centroids as the mean of their members (empty -> keep)
        for j in range(k):
            members = [vectors[i] for i in range(len(vectors)) if assign[i] == j]
            if members:
                centroids[j] = [sum(m[d] for m in members) / len(members) for d in range(D)]
        new_assign = [_nearest(v, centroids) for v in vectors]  # like-to-like
        history.append(round(energy(vectors, centroids, new_assign), 6))
        if new_assign == assign:          # settled -- no token changed bin
            break
        assign = new_assign
    intents = [[round(x, 4) for x in c] for c in centroids]
    return intents, assign, history


# ---------- FLOOR 2: a normal (toy) transformer over the intents ----------
def _grid(rows, cols, salt):
    return [[round(math.sin((i + 1) * 2.3 + (j + 1) * 1.7 + salt * 0.9), 4)
             for j in range(cols)] for i in range(rows)]


def matvec(M, v): return [dot(row, v) for row in M]
def relu(v): return [x if x > 0 else 0.0 for x in v]


def rmsnorm(v):
    ms = sum(x * x for x in v) / len(v)
    return [x / math.sqrt(ms + 1e-6) for x in v]


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e)
    return [x / s for x in e]


def _layer(li):
    return {"Wq": _grid(D, D, li * 3 + 1), "Wk": _grid(D, D, li * 3 + 2),
            "Wv": _grid(D, D, li * 3 + 3), "Wo": _grid(D, D, li * 3 + 4),
            "W1": _grid(D * 2, D, li * 3 + 5), "W2": _grid(D, D * 2, li * 3 + 6)}


def transformer(seq, layers=2):
    """A real toy transformer over a sequence of D-vectors (no causal mask here:
    the intents are a set, so every intent may attend to every intent)."""
    x = [list(v) for v in seq]
    n = len(x)
    for li in range(layers):
        W = _layer(li)
        nx = [rmsnorm(t) for t in x]
        Q = [matvec(W["Wq"], t) for t in nx]
        K = [matvec(W["Wk"], t) for t in nx]
        V = [matvec(W["Wv"], t) for t in nx]
        for i in range(n):
            scores = [dot(Q[i], K[j]) / math.sqrt(D) for j in range(n)]
            w = softmax(scores)
            ctx = [sum(w[j] * V[j][d] for j in range(n)) for d in range(D)]
            x[i] = add(x[i], matvec(W["Wo"], ctx))
        x = [add(t, matvec(W["W2"], relu(matvec(W["W1"], rmsnorm(t))))) for t in x]
    return x


VOCAB = ["the", "cat", "sat", "on", "mat", "run", "sky", "map"]


def readout(pooled):
    logits = [(tok, round(dot(pooled, embed(tok)), 4)) for tok in VOCAB]
    logits.sort(key=lambda kv: -kv[1])
    return logits


# ---------- the whole silo ----------
def run_silo(context_tokens, k=3):
    """context_tokens: the user's preloaded context (list of words)."""
    vectors = [embed(t) for t in context_tokens]
    k = max(1, min(k, len(vectors)))                      # can't have more intents than tokens
    intents, assign, history = centrifuge(vectors, k)     # FLOOR 1
    out = transformer(intents)                            # FLOOR 2
    pooled = [sum(o[d] for o in out) / len(out) for d in range(D)]  # mean-pool the intents
    logits = readout(pooled)
    return {
        "n_context": len(vectors), "k_intents": k,
        "assignments": assign, "intents": intents,
        "converged_energy": history[-1] if history else None,
        "spins": len(history), "energy_history": history,
        "prediction": logits[0][0], "logits": logits,
        "compression": round(len(vectors) / k, 2),
    }


if __name__ == "__main__":
    ctx = ["cat", "cat", "sat", "sky", "sky", "sky", "run", "mat", "mat"]
    r = run_silo(ctx, k=3)
    print("context:", ctx)
    print(f"centrifuge: {r['n_context']} tokens -> {r['k_intents']} intents "
          f"in {r['spins']} spins (energy {r['energy_history']})")
    print("assignments:", r["assignments"])
    print("compression:", r["compression"], "x")
    print("prediction:", r["prediction"])
