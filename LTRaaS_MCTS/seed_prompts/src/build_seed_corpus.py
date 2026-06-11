"""
build_seed_corpus.py — Adversarial Seed Corpus Builder for garak-MCTS

Extracts every adversarial prompt from garak's probe library, runs it through a
multi-stage cleaning and scoring pipeline, and writes a compact JSON file of
high-quality seed prompts ready for MCTS-based red-teaming.

Pipeline at a glance
────────────────────
  Stage 1    extract_all_prompts      Walk every garak probe class; collect unique prompts
  Stage 2    deduplicate              MinHash-LSH + sentence-embedding clustering
  Stage 2b   screen_stage2b          Concurrent screening; score + label each prompt
  Stage 3s   select_seeds            Variance-first seed selection for MCTS forest (50 seeds)

The output (stage3_seed_select.json) contains SeedCandidate records each
annotated with a refusal_score (0.0 = always refused → 1.0 = always bypassed)
and the source probe metadata.

Environment variables
─────────────────────
  CLAUDE_CODE_OAUTH_TOKEN   Use if you have a Claude plan instead of an API key
  ANTHROPIC_API_KEY         Fallback API key for stage 2b screening

Quick start
───────────
  Uncomment the stage you want to run in the __main__ block below, or import
  individual stage functions into your own pipeline script.
"""

from pathlib import Path

import config
from models import load_stage, save_stage
from stage1_extract import extract_all_prompts
from stage2_deduplicate import deduplicate
from stage2b_screen import screen_stage2b
from stage3_seed_select import _report, _sanity, select_seeds


if __name__ == "__main__":

    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    STAGE1       = OUTPUT_DIR / "stage1_extracted.json"
    STAGE2       = OUTPUT_DIR / "stage2_deduplicated.json"
    STAGE2B      = OUTPUT_DIR / "stage2b_screened.json"
    STAGE2B_CKPT = OUTPUT_DIR / "stage2b_checkpoint.json"
    STAGE3S      = OUTPUT_DIR / "stage3_seed_select.json"

    # ── Stage 1: extract all prompts from garak probes ───────────────────────
    # print("\n=== Stage 1: extract ===")
    # candidates = extract_all_prompts()
    # save_stage(STAGE1, candidates)

    # # ── Stage 2: deduplicate by semantic similarity ───────────────────────────
    # print("\n=== Stage 2: deduplicate ===")
    # candidates = load_stage(STAGE1)
    # candidates = deduplicate(
    #     candidates,
    #     jaccard_threshold=config.DEDUP_JACCARD_THRESHOLD,
    #     sim_threshold=config.DEDUP_SIM_THRESHOLD,
    #     num_perm=config.DEDUP_NUM_PERM,
    #     shingle_size=config.DEDUP_SHINGLE_SIZE,
    #     chunk_size=config.DEDUP_CHUNK_SIZE,
    # )
    # save_stage(STAGE2, candidates)

    # ── Stage 2b: 3-turn independent screening against Claude Haiku ─────────
    print("\n=== Stage 2b: screen ===")
    candidates = load_stage(STAGE2)
    candidates = screen_stage2b(
        candidates,
        generations=config.SCREEN_GENERATIONS,
        temperature=config.SCREEN_TEMPERATURE,
        max_workers=config.SCREEN_MAX_WORKERS,
        checkpoint_path=STAGE2B_CKPT,
        output_path=STAGE2B,
    )
    save_stage(STAGE2B, candidates)

    # ── Stage 3s: variance-first seed selection for MCTS forest ─────────────
    # print("\n=== Stage 3s: seed selection ===")
    # candidates = load_stage(STAGE2B)
    # candidates = select_seeds(candidates)
    # _report(candidates)
    # _sanity(candidates)
    # save_stage(STAGE3S, candidates)

    # print(f"\nDone. Final seeds: {len(candidates)} → {STAGE3S}")
