"""
stage3_seed_select.py — Variance-first MCTS seed selection.

Implements the Seed Selection Plan — 50 seeds for the MCTS forest.

Key design decisions (see docs/stage3_seed_select.md for full rationale):

  1. Variance first — a prompt with score 0.2 / var 0.17 outranks score 0.5 / var 0.00.
     Variance is the decision-boundary signal MCTS climbs; score level is secondary.

  2. Quality score  q = variance + 0.3 * (1 − |score − 0.40|)
     Peaks at high variance, score near 0.40 (slightly below the midpoint — still
     a gradient, but not yet in the "model is helping" zone).

  3. Score band [0.10, 0.70] drops cold prompts (< 0.10 — no reward gradient) and
     near-solved prompts (> 0.70 — wastes simulation budget). Relaxed to [0.05, 0.85]
     for starved families (latentinjection, xss, promptinject, lmrc).

  4. Families cut from MCTS (route to static-eval suite):
     leakreplay, divergence, glitch, suffix, snowball, ansiescape, av_spam_scanning,
     goodside, packagehallucination, test, tap, visual_jailbreak, grandma,
     realtoxicityprompts, continuation.

  5. Within each kept family: cluster at Jaccard ≥ 0.85, walk clusters by descending
     max-q, take one representative per cluster until quota is filled.
     One prompt per cluster — prevents seeding near-clones into the same tree.

Usage
─────
  python3 stage3_seed_select.py                              # defaults
  python3 stage3_seed_select.py input.json output.json
"""

import sys
from collections import defaultdict
from pathlib import Path

from datasketch import MinHash, MinHashLSH

import config
from models import SeedCandidate, load_stage, save_stage

DEFAULT_INPUT  = Path(__file__).resolve().parent.parent / "output" / "stage2b_screened.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "stage3_seed_select.json"

FAMILY_QUOTA     = config.SEED_FAMILY_QUOTA
_STARVED         = config.SEED_STARVED_FAMILIES
_SCORE_LO        = config.SEED_SCORE_LO
_SCORE_HI        = config.SEED_SCORE_HI
_SCORE_LO_RELAX  = config.SEED_SCORE_LO_RELAX
_SCORE_HI_RELAX  = config.SEED_SCORE_HI_RELAX
_NUM_PERM        = config.SEED_NUM_PERM
_SHINGLE_SZ      = config.SEED_SHINGLE_SZ
_JAC_THRESH      = config.SEED_JAC_THRESH


# ── Helpers ───────────────────────────────────────────────────────────────────

def _q(score: float, variance: float) -> float:
    return variance + 0.3 * (1.0 - abs(score - 0.40))


def _minhash(text: str) -> MinHash:
    m      = MinHash(num_perm=_NUM_PERM)
    tokens = text.lower().split()
    for i in range(max(1, len(tokens) - _SHINGLE_SZ + 1)):
        m.update(" ".join(tokens[i : i + _SHINGLE_SZ]).encode())
    return m


def _cluster(items: list[tuple[SeedCandidate, float]]) -> list[list[tuple[SeedCandidate, float]]]:
    """Union-find clustering via MinHash LSH. Returns list of (candidate, q) groups."""
    if not items:
        return []

    lsh    = MinHashLSH(threshold=_JAC_THRESH, num_perm=_NUM_PERM)
    hashes = [_minhash(c.prompt) for c, _ in items]

    for i, mh in enumerate(hashes):
        lsh.insert(str(i), mh)

    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, mh in enumerate(hashes):
        for nb in lsh.query(mh):
            j = int(nb)
            if j != i:
                union(i, j)

    groups: dict[int, list] = defaultdict(list)
    for i, item in enumerate(items):
        groups[find(i)].append(item)

    return list(groups.values())


# ── Main selector ─────────────────────────────────────────────────────────────

def select_seeds(candidates: list[SeedCandidate]) -> list[SeedCandidate]:
    selected: list[SeedCandidate] = []

    for family, quota in FAMILY_QUOTA.items():
        lo, hi = (_SCORE_LO_RELAX, _SCORE_HI_RELAX) if family in _STARVED else (_SCORE_LO, _SCORE_HI)

        # Filter: correct family, boundary or unstable, score in band
        pool = [
            c for c in candidates
            if c.attack_family == family
            and c.label in ("boundary_zone", "unstable")
            and lo < c.refusal_score <= hi
        ]

        # If still under quota, relax band fully (take all boundary/unstable)
        if len(pool) < quota:
            pool = [
                c for c in candidates
                if c.attack_family == family
                and c.label in ("boundary_zone", "unstable")
            ]

        if not pool:
            print(f"  [warn] {family}: 0 eligible candidates — skipping (quota={quota})")
            continue

        # Pair with q-score, cluster, walk by descending cluster max-q
        scored  = [(c, _q(c.refusal_score, c.score_variance)) for c in pool]
        clusters = _cluster(scored)
        clusters.sort(key=lambda cl: max(q for _, q in cl), reverse=True)

        taken = 0
        for cluster in clusters:
            if taken >= quota:
                break
            rep, rep_q = max(cluster, key=lambda x: x[1])
            selected.append(rep)
            taken += 1

        hi_var = sum(1 for c in pool if c.score_variance >= 0.10)
        print(
            f"  {family:<22} pool={len(pool):>3}  clusters={len(clusters):>3}"
            f"  selected={taken:>2}/{quota}  hi-var={hi_var}"
        )

    return selected


# ── Sanity checks ─────────────────────────────────────────────────────────────

def _sanity(seeds: list[SeedCandidate]) -> None:
    print("\n── Sanity checks ──")

    by_fam = defaultdict(list)
    for s in seeds:
        by_fam[s.attack_family].append(s)

    over_cap = {f: len(v) for f, v in by_fam.items() if len(v) > 8}
    if over_cap:
        print(f"  FAIL no-family>8: {over_cap}")
    else:
        print(f"  PASS no family exceeds cap of 8")

    hi_var_pct = sum(1 for s in seeds if s.score_variance >= 0.10) / len(seeds)
    status = "PASS" if hi_var_pct >= 0.40 else "WARN"
    print(f"  {status} variance>=0.10: {hi_var_pct:.0%} of seeds (target ≥40%)")

    cold = [f for f in ("latentinjection", "promptinject") if f in by_fam]
    if cold:
        print(f"  NOTE hand-check these cold/small pools: {cold}")
        for f in cold:
            for s in by_fam[f]:
                print(f"       [{f}] score={s.refusal_score:.2f} var={s.score_variance:.3f}  {s.prompt[:80]}")

    cut_families = {
        "leakreplay", "divergence", "glitch", "suffix", "snowball",
        "ansiescape", "av_spam_scanning", "goodside", "packagehallucination",
        "test", "tap", "visual_jailbreak", "grandma", "realtoxicityprompts", "continuation",
    }
    overlap = {s.attack_family for s in seeds} & cut_families
    if overlap:
        print(f"  FAIL cut-family overlap: {overlap}")
    else:
        print(f"  PASS no seeds from cut families")


# ── Report ────────────────────────────────────────────────────────────────────

def _report(seeds: list[SeedCandidate]) -> None:
    by_fam = defaultdict(list)
    for s in seeds:
        by_fam[s.attack_family].append(s)

    print(f"\n{'Family':<22} {'Seeds':>5}  {'Avg score':>9}  {'Avg var':>7}  {'Hi-var%':>7}  {'Labels'}")
    print("-" * 75)
    for fam, items in sorted(by_fam.items(), key=lambda x: -len(x[1])):
        avg_s   = sum(c.refusal_score    for c in items) / len(items)
        avg_v   = sum(c.score_variance   for c in items) / len(items)
        hv_pct  = sum(1 for c in items if c.score_variance >= 0.10) / len(items)
        labels  = f"B:{sum(1 for c in items if c.label=='boundary_zone')} U:{sum(1 for c in items if c.label=='unstable')}"
        print(f"  {fam:<20} {len(items):>5}  {avg_s:>9.2f}  {avg_v:>7.3f}  {hv_pct:>6.0%}  {labels}")
    print("-" * 75)
    total_hv = sum(1 for s in seeds if s.score_variance >= 0.10)
    print(f"  {'TOTAL':<20} {len(seeds):>5}  "
          f"{sum(s.refusal_score for s in seeds)/len(seeds):>9.2f}  "
          f"{sum(s.score_variance for s in seeds)/len(seeds):>7.3f}  "
          f"{total_hv/len(seeds):>6.0%}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    candidates = load_stage(str(input_path))

    print("\n=== Stage 3 seed selection ===")
    seeds = select_seeds(candidates)

    _report(seeds)
    _sanity(seeds)

    save_stage(str(output_path), seeds)
