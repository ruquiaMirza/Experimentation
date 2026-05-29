# Low-Level Design — Adversarial Seed Corpus Builder

## 1. Overview

`build_seed_corpus.py` is a seven-stage data pipeline that transforms garak's raw probe library into a scored, filtered, and clustered set of adversarial seed prompts. These seeds are starting points for MCTS-based jailbreak search: rather than exploring from random text, MCTS begins from prompts that are already near the model's refusal boundary.

The pipeline is intentionally staged. Each function reads a JSON file written by the previous stage, so individual stages can be re-run in isolation, interrupted and resumed, or replaced with different implementations without touching the rest of the pipeline.

---

## 2. Data model

### `SeedCandidate`

The central record flowing through every stage. It is a flat dataclass that serializes cleanly to JSON.

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | The raw adversarial prompt text |
| `probe_module` | `str` | Fully-qualified module name (e.g. `garak.probes.dan`) |
| `probe_class` | `str` | Class name within the module |
| `tags` | `list[str]` | Metadata tags from the probe |
| `goal` | `str` | Plain-English description of the attack's intended outcome |
| `attack_family` | `str` | Module leaf name used for grouping (e.g. `"dan"`, `"knownbadsignatures"`) |
| `refusal_score` | `float` | 0.0 = always refused → 1.0 = always bypassed. Populated by stage 6. |
| `buff_origin` | `str` | Set to a buff name by external augmentation stages; `"original"` otherwise |

### `_PathEncoder`

Some garak probes store file paths in their prompt lists. `_PathEncoder` extends `json.JSONEncoder` to serialize `pathlib.Path` objects as plain strings, preventing `TypeError` during JSON serialization.

---

## 3. Stage-by-stage design

### Stage 1 — `extract_all_prompts`

**Input:** garak installed package (no file input)
**Output:** `stage1_extracted.json`

Uses `pkgutil.walk_packages` to discover every module under `garak.probes`, then calls `dir(module)` to find subclasses of `Probe`. Each probe class is instantiated and its `.prompts` attribute is harvested.

At this stage, deduplication is exact — SHA-256 of the raw prompt text. This catches identical strings but not paraphrases; semantic deduplication happens in stage 2.

Dict-type prompts (some probes use structured objects rather than plain strings) are JSON-serialized before hashing to produce stable, comparable strings.

**Design note:** Probe instantiation is wrapped in a broad `try/except` because some probes have non-trivial `__init__` bodies that fail when optional dependencies are missing. Failing silently is correct here — a broken probe shouldn't abort the entire extraction.

---

### Stage 2 — `deduplicate`

**Input:** `stage1_extracted.json`
**Output:** `stage2_deduplicated.json`

Near-duplicate detection runs in three passes:

**Pass 1 — TF-IDF vectorization**
Produces a sparse float representation of each prompt (`max_features=10,000`, unigrams through trigrams, English stop words removed). This is the foundation for the similarity comparison.

**Pass 2 — Chunked cosine similarity + union-find**
The full n×n similarity matrix is never materialized. Instead, `chunk_size` rows are multiplied against the full TF-IDF matrix at a time. Any pair whose cosine similarity exceeds `sim_threshold` is merged via union-find.

Union-find with path compression is chosen over `AgglomerativeClustering` at this scale because:
- It has O(n α(n)) time complexity and constant memory overhead per merged pair
- `AgglomerativeClustering` requires materializing the full distance matrix — O(n²) in memory, which is fatal at 72k+ prompts

**Pass 3 — Representative selection**
One prompt is kept per union-find cluster: whichever has the most characters. Longer prompts tend to carry more context and make better MCTS starting points.

**Memory characteristics:**

| n prompts | TF-IDF (sparse) | Peak per chunk (chunk=1000) |
|-----------|-----------------|------------------------------|
| 10,000    | ~40 MB          | ~80 MB |
| 72,000    | ~280 MB         | ~576 MB |
| 100,000   | ~400 MB         | ~800 MB |

Reduce `chunk_size` to lower peak memory at the cost of more iterations.

---

### Stage 3a — `length_filter`

**Input:** `stage2_deduplicated.json`
**Output:** `stage3a_filtered.json`

Word count (whitespace split) is used as a token-count proxy. This is intentionally imprecise — the goal is to cut clear outliers (single-word prompts, huge document dumps) rather than enforce a precise token budget. The default bounds [15, 800] were chosen empirically across garak's probe library.

---

### Stage 3b — `quality_filter`

**Input:** `stage3a_filtered.json`
**Output:** `stage3b_quality_filtered.json`

Two signal types are rejected:

**High non-alpha ratio**
GCG (Greedy Coordinate Gradient) attacks and similar optimization-based attacks produce token soups with very few alphabetic characters. These prompts are model-specific and don't transfer well, so they're poor MCTS seeds. Threshold: fewer than 60% of characters being alphabetic or whitespace.

**High word repetition**
Suffix attacks frequently repeat a phrase many times. A prompt where fewer than 30% of words are unique is likely an optimization artifact rather than a crafted adversarial argument.

---

### Stage 3c — `cap_per_family`

**Input:** `stage3b_quality_filtered.json`
**Output:** `stage3c_capped.json`

Some attack families (especially `dan`, `knownbadsignatures`) can have hundreds of prompts after stage 3b; others have fewer than five. Without capping, the larger families dominate the seed corpus.

For families above the cap, `AgglomerativeClustering` with a precomputed cosine-distance matrix clusters the family into `max_per_family` groups, and the longest prompt is selected from each cluster. This is affordable at family level because even the largest families have at most a few hundred prompts — the O(n²) memory cost of the distance matrix is manageable at that scale.

---

### Stages 4–5 — External augmentation

These stages apply garak's buff system (paraphrase, back-translation, template expansion, etc.) to generate additional candidate prompts from the filtered set. They are implemented in separate modules and write their output into files compatible with what stage 6 expects. The artifact `stage3_expanded.json` is their typical output.

---

### Stage 6 — `screen_with_generator`

**Input:** `stage3c_capped.json` (or buff-augmented equivalent)
**Output:** `stage6_screened.json`, `stage6_checkpoint.json`

The most expensive stage. For each prompt:

1. Send the prompt to `claude-haiku-4-5` `generations_per_prompt` times, independently.
2. Feed all responses through `garak.detectors.mitigation.MitigationBypass`.
3. `refusal_score = mean(detector outputs)`.

**Concurrency**
`ThreadPoolExecutor` with `max_workers=15`. The Anthropic Python SDK is thread-safe; 15 workers stays comfortably within Haiku's rate limits while keeping wall-clock time manageable.

**Checkpointing**
After each prompt completes, the score is written to `stage6_checkpoint.json` as a `{sha256_hash: score}` dict. On restart, already-scored prompts are skipped. This makes long runs safe to interrupt — no work is lost.

**Authentication priority**
`CLAUDE_CODE_OAUTH_TOKEN` (Max plan, higher rate limits) is tried first. If absent, `ANTHROPIC_API_KEY` is used. If neither is set, the stage is skipped and all candidates pass through with `refusal_score=0.0`.

**Scoring semantics**

| Score | Meaning |
|-------|---------|
| 0.0 | Haiku refused on every call |
| 0.33 | Haiku bypassed on 1 of 3 calls |
| 0.67 | Haiku bypassed on 2 of 3 calls |
| 1.0 | Haiku complied every time |

---

### Stage 7 — `select_seeds`

**Input:** `stage6_screened.json`
**Output:** `stage7_seeds.json`

Per attack family, the stage runs the following logic:

**Zone partition**
Candidates are split into three buckets:
- `boundary`: `target_zone[0]` ≤ score ≤ `target_zone[1]` (partial bypass — most useful)
- `high`: score > upper bound (always bypasses — useful as anchors)
- `refused`: score < lower bound (always refused — least useful)

**Pool selection**
The boundary zone is the preferred pool. If the boundary zone is empty, the full family is used as a fallback.

**Diversity clustering**
If the pool exceeds the budget, `AgglomerativeClustering` partitions it and the highest-scoring prompt from each cluster is selected. This ensures the selected seeds cover different parts of the attack surface rather than clustering around one phrasing style.

**Anchor injection**
If a slot remains after the diversity selection, the single highest-scoring prompt from the high zone is added. This gives MCTS a reference point for evaluating how perturbations affect bypass rate.

**Why boundary prompts?**
MCTS gets the most signal from prompts where small changes flip the model's decision. A prompt that always gets refused gives no gradient information; one that always bypasses is already a solution. The boundary zone (partial bypass) is where the search space is richest and the most can be learned from each iteration.

---

## 4. Cross-cutting concerns

### Determinism

Stage 2 and stages 3c/7 use `AgglomerativeClustering`, which is deterministic for a fixed distance matrix. Stage 1 ordering depends on `pkgutil.walk_packages` traversal (filesystem-dependent but stable between runs on the same machine). Stage 6 is non-deterministic (stochastic model outputs), but checkpointing ensures scores don't change once written.

### Error handling philosophy

Each stage function logs errors and continues rather than aborting. With thousands of probe classes and tens of thousands of API calls, a handful of failures should not abort the entire pipeline. The progress bars and milestone logs make it easy to spot whether the failure rate is abnormally high.

### Memory budget

The largest in-memory object is the TF-IDF matrix in stage 2. At 72k prompts with `max_features=10,000` and typical sparsity ~99%, this is around 280 MB. The chunked similarity pass adds one dense slice at a time; peak overhead is bounded by `chunk_size`.

---

## 5. File layout

```
build_seed_corpus.py                    Main pipeline script
stage7_report.py                        Reporting / analysis script for stage 7 output
output/
  stage1_extracted.json                 Stage 1 output
  stage2_deduplicated.json              Stage 2 output
  stage3a_filtered.json                 Stage 3a output
  stage3b_quality_filtered.json         Stage 3b output
  stage3c_capped.json                   Stage 3c output
  stage3_expanded.json                  Stages 4–5 output (external augmentation)
  stage6_checkpoint.json                Live scoring checkpoint for stage 6
  stage6_screened.json                  Stage 6 output
  stage7_seeds.json                     Final seed corpus (pipeline output)
```

`OUTPUT_DIR` (`output/`) is created automatically by the `__main__` block on first run via `Path.mkdir(exist_ok=True)`, so no manual setup is needed.

---

## 6. Dependencies

| Package | Used for |
|---------|----------|
| `garak` | Probe library (prompt source) and `MitigationBypass` detector |
| `anthropic` | Claude Haiku API client for stage 6 |
| `scikit-learn` | TF-IDF vectorization, cosine similarity, agglomerative clustering |
| `numpy` | Dense matrix operations in the similarity pass |
| `tqdm` | Progress bars for long-running stages |
