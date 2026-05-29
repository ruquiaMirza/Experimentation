# Adversarial Seed Corpus Builder

This tool pulls every adversarial prompt out of [garak](https://github.com/leondz/garak)'s probe library, runs it through a multi-stage cleaning and scoring pipeline, and produces a compact JSON file of the best seed prompts to feed into MCTS-based red-teaming.

The idea is pretty simple: instead of starting MCTS from scratch or from random text, you start from prompts that are *already near the model's refusal boundary* — the ones that sometimes get refused and sometimes don't. That ambiguous zone is where a tree search actually has something to explore.

---

## What it does

The pipeline runs in seven stages:

| Stage | Function | What it does |
|-------|----------|--------------|
| 1 | `extract_all_prompts` | Loads every garak probe class and collects unique prompts |
| 2 | `deduplicate` | Removes near-duplicates using TF-IDF + cosine similarity |
| 3a | `length_filter` | Drops prompts that are too short or too long |
| 3b | `quality_filter` | Drops symbol-heavy or repetitively worded prompts |
| 3c | `cap_per_family` | Limits each attack family to 50 diverse prompts |
| 4–5 | *(external)* | Augmentation and buff passes — see `stage3_expanded.json` |
| 6 | `screen_with_generator` | Sends each prompt to Claude Haiku; scores the bypass rate |
| 7 | `select_seeds` | Picks the most promising seeds per attack family |

Each stage reads from the previous stage's JSON output and writes its own, so you can run one stage at a time and resume safely if anything breaks or you need to swap parameters.

---

## Setup

You'll need Python 3.10+ and garak installed in the same environment.

```bash
pip install anthropic scikit-learn numpy tqdm
```

For stage 6 (the screening step that actually calls the model), set one of these:

```bash
export CLAUDE_CODE_OAUTH_TOKEN="..."   # Claude Max plan — preferred, higher rate limits
# or
export ANTHROPIC_API_KEY="..."         # Standard API key
```

If neither is set, stage 6 is skipped and candidates pass through with a `refusal_score` of 0.0.

---

## Running a stage

Each stage in `build_seed_corpus.py` is commented out in the `__main__` block. Uncomment the one you want to run:

```python
# Stage 1 — extract
print("\n=== Stage 1: extract ===")
candidates = extract_all_prompts()
save_stage(STAGE1, candidates)
```

Then just:

```bash
python build_seed_corpus.py
```

You can also import individual functions directly if you're building a custom pipeline.

---

## Output files

All generated files land in the `output/` folder, which is created automatically on first run.

| File | What's in it |
|------|--------------|
| `output/stage1_extracted.json` | All unique prompts from garak probes |
| `output/stage2_deduplicated.json` | After near-duplicate removal |
| `output/stage3a_filtered.json` | After length filtering |
| `output/stage3b_quality_filtered.json` | After quality filtering |
| `output/stage3c_capped.json` | After per-family cap |
| `output/stage3_expanded.json` | After external augmentation passes (stages 4–5) |
| `output/stage6_checkpoint.json` | Live checkpoint — safe to kill and resume |
| `output/stage6_screened.json` | With `refusal_score` on each prompt |
| `output/stage7_seeds.json` | **Final output** — seeds ready for MCTS |

---

## A note on stage 6 costs

Stage 6 calls Claude Haiku 3 times per prompt. With ~5,000 prompts after filtering, that's roughly 15,000 Haiku calls. At current pricing it's not expensive, but it's not free either — worth keeping in mind if you're on a pay-as-you-go key rather than a Max plan.

The checkpoint file (`stage6_checkpoint.json`) means you can kill the process mid-run and resume from where you left off. Nothing gets re-scored.

---

## Understanding `refusal_score`

Every `SeedCandidate` gets a `refusal_score` between 0.0 and 1.0 after stage 6:

- **0.0** — Haiku refused on every call
- **0.33** — Haiku bypassed on 1 of 3 calls
- **0.67** — Haiku bypassed on 2 of 3 calls
- **1.0** — Haiku complied every time

Stage 7 preferentially selects prompts in the `(0.1, 0.8)` range — the "interesting" zone where the model is genuinely uncertain. MCTS can then explore what nudges a prompt toward full bypass.

---

## Tuning the pipeline

The most impactful knobs, roughly in order:

| Parameter | Stage | Default | What changing it does |
|-----------|-------|---------|----------------------|
| `sim_threshold` | 2 | 0.90 | Higher = less aggressive dedup |
| `max_per_family` | 3c | 50 | Controls how many prompts survive per attack type |
| `generations_per_prompt` | 6 | 3 | More samples = better signal, higher cost |
| `seeds_per_family` | 7 | 5 | Controls the final seed count per family |
| `target_zone` | 7 | (0.1, 0.8) | Boundary zone for "interesting" prompts |
