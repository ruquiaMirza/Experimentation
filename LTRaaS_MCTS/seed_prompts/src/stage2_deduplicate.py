"""
Two-stage deduplication pipeline.

Stage A — MinHash + LSH  (datasketch)
    Collapses near-exact and templated variants cheaply on character shingles.
    Handles the bulk of the 21k corpus (expected 21k → ~8-12k).

Stage B — Sentence-embedding clustering
    Embeds Stage A survivors with all-MiniLM-L6-v2, mines paraphrase pairs by
    cosine similarity, then clusters with union-find + complete-linkage validation.
    Representative is the medoid (prompt closest to cluster centroid) — the
    canonical attack form for MCTS root expansion.

Each output SeedCandidate carries cluster_size = total originals it absorbed,
which signals how productive that attack family is for later expansion budgeting.
"""

import time
from collections import defaultdict

import numpy as np
import tqdm
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer, util as st_util
from sklearn.cluster import AgglomerativeClustering

from models import SeedCandidate


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_minhash(text: str, num_perm: int, shingle_size: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    t = text.lower()
    for i in range(max(1, len(t) - shingle_size + 1)):
        m.update(t[i : i + shingle_size].encode("utf-8"))
    return m


def _union_find_clusters(n: int, pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    parent = list(range(n))
    size   = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px == py:
            return
        if size[px] < size[py]:
            px, py = py, px
        parent[py]  = px
        size[px]   += size[py]

    for i, j in pairs:
        union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)
    return clusters


# ── Stage A ───────────────────────────────────────────────────────────────────

def _stage_a_minhash(
    candidates: list[SeedCandidate],
    threshold: float,
    num_perm: int,
    shingle_size: int,
) -> list[SeedCandidate]:
    n = len(candidates)
    print(f"  [dedup/A] building {n} MinHashes (shingle={shingle_size}, perm={num_perm}) ...")

    minhashes = [
        _make_minhash(c.prompt, num_perm, shingle_size)
        for c in tqdm.tqdm(candidates, desc="  MinHash", unit="prompt", dynamic_ncols=True)
    ]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(str(i), m)

    print(f"  [dedup/A] querying LSH (Jaccard ≥ {threshold}) ...")
    pairs: list[tuple[int, int]] = []
    for i, m in enumerate(minhashes):
        for key in lsh.query(m):
            j = int(key)
            if j > i:
                pairs.append((i, j))

    print(f"  [dedup/A] {len(pairs):,} near-duplicate pairs found")
    clusters = _union_find_clusters(n, pairs)

    survivors: list[SeedCandidate] = []
    for members in clusters.values():
        # All members are near-identical templates; pick any (longest is fine here)
        rep           = max(members, key=lambda idx: len(candidates[idx].prompt))
        c             = candidates[rep]
        c.cluster_size = sum(candidates[m].cluster_size for m in members)
        survivors.append(c)

    print(f"  [dedup/A] {n} → {len(survivors)} ({n - len(survivors):,} removed)")
    return survivors


# ── Stage B ───────────────────────────────────────────────────────────────────

def _stage_b_semantic(
    candidates: list[SeedCandidate],
    sim_threshold: float,
    chunk_size: int,
) -> list[SeedCandidate]:
    n     = len(candidates)
    texts = [c.prompt for c in candidates]
    t0    = time.time()

    print(f"  [dedup/B] encoding {n} prompts with all-MiniLM-L6-v2 ...")
    encoder    = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"  [dedup/B] mining paraphrase pairs (cosine ≥ {sim_threshold}) ...")
    all_pairs = st_util.paraphrase_mining(
        encoder,
        texts,
        batch_size=64,
        corpus_chunk_size=chunk_size,
        show_progress_bar=True,
    )
    pairs = [(i, j) for score, i, j in all_pairs if score >= sim_threshold]
    print(f"  [dedup/B] {len(pairs):,} semantic pairs above threshold")

    raw_clusters = _union_find_clusters(n, pairs)

    # Complete-linkage cohesion check: splits transitive over-merges
    print("  [dedup/B] complete-linkage cohesion validation ...")
    validated: list[list[int]] = []
    splits = 0
    for members in raw_clusters.values():
        if len(members) <= 2:
            validated.append(members)
            continue
        sub  = embeddings[np.array(members)]
        sim  = sub @ sub.T
        dist = np.clip(1.0 - sim, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        clust  = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - sim_threshold,
            metric="precomputed",
            linkage="complete",
        )
        labels = clust.fit_predict(dist)
        n_sub  = len(set(labels))
        if n_sub > 1:
            splits += n_sub - 1
        for label in set(labels):
            validated.append([members[i] for i, lbl in enumerate(labels) if lbl == label])

    if splits:
        print(f"  [dedup/B] {splits} over-merged cluster(s) split → {len(validated)} clusters")

    # Pick medoid: prompt with highest cosine similarity to the (normalised) centroid
    survivors: list[SeedCandidate] = []
    for members in validated:
        if len(members) == 1:
            survivors.append(candidates[members[0]])
            continue
        idxs     = np.array(members)
        sub      = embeddings[idxs]
        centroid = sub.mean(axis=0)
        norm     = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid /= norm
        sims   = sub @ centroid
        medoid = members[int(np.argmax(sims))]
        c      = candidates[medoid]
        c.cluster_size = sum(candidates[m].cluster_size for m in members)
        survivors.append(c)

    print(f"  [dedup/B] {n} → {len(survivors)} "
          f"({n - len(survivors):,} removed, {time.time()-t0:.1f}s)")
    return survivors


# ── public entry point ────────────────────────────────────────────────────────

def deduplicate(
    candidates: list[SeedCandidate],
    jaccard_threshold: float = 0.8,
    sim_threshold: float = 0.85,
    num_perm: int = 128,
    shingle_size: int = 3,
    chunk_size: int = 1_000,
) -> list[SeedCandidate]:
    """Two-stage deduplication: MinHash-LSH then sentence-embedding clustering.

    Parameters
    ----------
    jaccard_threshold:
        Jaccard threshold for Stage A MinHash-LSH (~0.8).
    sim_threshold:
        Cosine similarity threshold for Stage B semantic clustering (~0.85).
    num_perm:
        MinHash permutations. 128 is accurate at 21k scale.
    shingle_size:
        Character n-gram size for MinHash (3 works well for natural language).
    chunk_size:
        Batch size for paraphrase_mining in Stage B.
    """
    t_total = time.time()
    n_start = len(candidates)
    print(f"  [dedup] {n_start} candidates entering two-stage pipeline")

    survivors = _stage_a_minhash(candidates, jaccard_threshold, num_perm, shingle_size)
    survivors = _stage_b_semantic(survivors, sim_threshold, chunk_size)

    print(f"Deduplicated: {n_start} → {len(survivors)} "
          f"({n_start - len(survivors):,} removed total, {time.time()-t_total:.1f}s)")
    return survivors
