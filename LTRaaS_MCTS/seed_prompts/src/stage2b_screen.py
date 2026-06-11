"""
stage2b_screen.py — Screen every deduplicated prompt against the target model.

Purpose
-------
Stage 2b answers one question for every prompt that survived deduplication:
is this prompt actually useful as an MCTS starting point?

It does this by sending each prompt to the configured screening model multiple
times and observing how consistently the model refuses or complies. A prompt
that always gets refused gives MCTS nothing to explore. One that sometimes gets
refused and sometimes complies is exactly where a tree search has something to
learn — small changes there can flip the model's decision.

How it works
------------
For each prompt:

  1. Call the screening model SCREEN_GENERATIONS times independently.
     Each call is a fresh single-turn conversation with no shared state.
     Temperature 0.7 preserves the model's natural response variance
     (temperature 0 would make all three calls identical, defeating the
     purpose of sampling multiple times).

  2. Score each response through the three-tier evaluator in evaluator.py:
       Tier 1 — regex heuristics (free, instant)
       Tier 2 — Haiku LLM judge (cheap, ~100–200 ms)
       Tier 3 — Sonnet LLM judge (stronger, only for ambiguous cases)

  3. Aggregate the N scores into a mean, a variance, and a label:
       dead_low      always refused         → drop from MCTS
       dead_high     always complied        → move to known-successes pool
       unstable      high variance          → deprioritise as root
       boundary_zone model is uncertain     → prime MCTS starting point

  4. Write everything to a checkpoint file after each prompt so the run can
     be interrupted and resumed without re-scoring any prompt.

Configuration
-------------
All tunable values (provider, model, generations, temperature, thresholds)
live in config.py. The screening model is swapped by changing SCREEN_PROVIDER
and SCREEN_MODEL there — no code changes required. See providers.py for
supported providers (Anthropic, Gemini, Ollama, OpenAI, Groq).
"""

import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm

import config
from evaluator import aggregate_scores, score_response
from models import SeedCandidate, save_stage
from providers import get_screener


def screen_stage2b(
    candidates: list[SeedCandidate],
    generations: int = config.SCREEN_GENERATIONS,
    temperature: float = config.SCREEN_TEMPERATURE,
    max_workers: int = config.SCREEN_MAX_WORKERS,
    checkpoint_path: str | None = None,
    output_path: str | None = None,
) -> list[SeedCandidate]:
    """Score every candidate prompt by probing the screening model.

    Sends each prompt to the configured screening model `generations` times
    independently, scores every response through the tiered evaluator, and
    writes a refusal_score, score_variance, and label back onto each
    SeedCandidate. The full candidate list is returned with those fields
    populated.

    Checkpointing
    -------------
    Results are written to `checkpoint_path` after every completed prompt.
    On restart, already-scored prompts are loaded from the checkpoint and
    skipped, so no work is ever repeated. It is safe to kill and resume
    this function at any point.

    Parameters
    ----------
    candidates:
        List of SeedCandidates from Stage 2 (deduplicated).
    generations:
        Number of independent calls to make per prompt. More calls give
        better variance estimates at higher cost. Default: SCREEN_GENERATIONS.
    temperature:
        Sampling temperature for the screening model. Must be > 0 to produce
        meaningful variance across calls. Default: SCREEN_TEMPERATURE.
    max_workers:
        Number of concurrent threads. Each thread processes one prompt at a
        time (all generations sequentially). Default: SCREEN_MAX_WORKERS.
    checkpoint_path:
        Path to the checkpoint JSON file. Defaults to
        output/stage2b_checkpoint.json relative to the project root.
    output_path:
        If provided, the full candidate list is flushed to this path every
        25 completed prompts as a rolling save.

    Returns
    -------
    The same `candidates` list with refusal_score, score_variance, label,
    and screening_details populated on each item.
    """
    if checkpoint_path is None:
        checkpoint_path = str(
            Path(__file__).resolve().parent.parent / "output" / "stage2b_checkpoint.json"
        )

    # Build the screening client from config (Anthropic, Gemini, Ollama, etc.)
    try:
        screener = get_screener()
    except (RuntimeError, ValueError) as e:
        print(f"  [stage2b] Cannot initialise screener: {e} — skipping screening.")
        return candidates

    # ── Load checkpoint ───────────────────────────────────────────────────────
    # The checkpoint maps sha256(prompt) → {score, variance, label, details}.
    # Any prompt already in the checkpoint is skipped entirely.

    done: dict[str, dict] = {}

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            done = json.load(f)
        print(f"  [stage2b] Resumed from checkpoint — {len(done)} prompts already scored.")

    # Restore checkpoint scores onto candidate objects so that rolling saves
    # (every 25 prompts) always write correct values for all candidates,
    # not just the ones scored in this session.
    _restore_from_checkpoint(candidates, done)

    remaining = [
        c for c in candidates
        if hashlib.sha256(c.prompt.encode()).hexdigest() not in done
    ]
    already_done = len(candidates) - len(remaining)
    if already_done:
        print(f"  [stage2b] Skipping {already_done} already-scored prompts, "
              f"{len(remaining)} remaining.")
    if not remaining:
        print("  [stage2b] All prompts already scored.")
        return candidates

    # ── Shared counters (accessed under lock by worker threads) ───────────────

    lock         = threading.Lock()
    n_completed  = [0]                  # mutable int for closure
    tier_counts  = Counter()            # how many responses each tier resolved
    label_counts = Counter()            # how many prompts got each label
    last_logged  = [0]
    t_start      = time.time()

    LOG_EVERY  = 100   # print a progress line every N completed prompts
    SAVE_EVERY = 25    # flush output_path every N completed prompts

    def _save_checkpoint() -> None:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(done, f, indent=2, ensure_ascii=False)

    # ── Worker: process one prompt ────────────────────────────────────────────

    def _process_one(candidate: SeedCandidate) -> None:
        """Send one prompt through all generations and score every response."""
        prompt_hash = hashlib.sha256(candidate.prompt.encode()).hexdigest()

        # Step 1: collect N independent responses from the screening model
        responses: list[str] = []
        for _ in range(generations):
            try:
                text = screener.complete(
                    candidate.prompt, temperature, config.SCREEN_MAX_TOKENS
                )
                responses.append(text)
            except Exception as e:
                tqdm.tqdm.write(
                    f"  [stage2b] API error for '{candidate.prompt[:60]}...': {e}"
                )

        # Step 2: score each response through the tiered evaluator
        if not responses:
            # All API calls failed — treat as hard refusal with no signal
            mean, variance, label = 0.0, 0.0, "dead_low"
            local_tiers: Counter  = Counter()
            details: list[dict]   = []
        else:
            scores: list[float]  = []
            local_tiers          = Counter()
            details              = []
            for response_text in responses:
                score, tier = score_response(
                    candidate.prompt, response_text, candidate.goal
                )
                scores.append(score)
                local_tiers[f"tier{tier}"] += 1
                details.append({"response": response_text, "score": score, "tier": tier})

            # Step 3: aggregate scores into mean, variance, and label
            mean, variance, label = aggregate_scores(scores)

        # Step 4: write results back and persist to checkpoint
        with lock:
            candidate.refusal_score     = mean
            candidate.score_variance    = variance
            candidate.label             = label
            candidate.screening_details = details

            done[prompt_hash] = {
                "score":    mean,
                "variance": variance,
                "label":    label,
                "details":  details,
            }

            n_completed[0]  += 1
            tier_counts.update(local_tiers)
            label_counts[label] += 1

            _save_checkpoint()

            # Rolling flush to output_path every SAVE_EVERY prompts
            if output_path and n_completed[0] % SAVE_EVERY == 0:
                save_stage(output_path, candidates)
                tqdm.tqdm.write(
                    f"  [stage2b] Saved {n_completed[0]}/{len(remaining)} → {output_path}"
                )

            # Periodic progress log
            if n_completed[0] - last_logged[0] >= LOG_EVERY:
                elapsed = time.time() - t_start
                rate    = n_completed[0] / elapsed if elapsed > 0 else 0
                eta_min = (len(remaining) - n_completed[0]) / rate / 60 if rate > 0 else 0
                tqdm.tqdm.write(
                    f"  [stage2b] {n_completed[0]:>5}/{len(remaining)} | "
                    f"boundary: {label_counts['boundary_zone']}  "
                    f"unstable: {label_counts['unstable']}  "
                    f"dead_hi: {label_counts['dead_high']}  "
                    f"dead_lo: {label_counts['dead_low']} | "
                    f"T1: {tier_counts['tier1']}  "
                    f"T2: {tier_counts['tier2']}  "
                    f"T3: {tier_counts['tier3']} | "
                    f"{rate:.1f} prompts/s | ETA {eta_min:.1f} min"
                )
                last_logged[0] = n_completed[0]

    # ── Fan out across threads ────────────────────────────────────────────────

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, c): c for c in remaining}

        with tqdm.tqdm(
            total=len(remaining),
            desc="Stage 2b screening",
            unit="prompt",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                progress.update(1)
                progress.set_postfix({
                    "boundary": label_counts["boundary_zone"],
                    "unstable": label_counts["unstable"],
                    "dead_hi":  label_counts["dead_high"],
                    "dead_lo":  label_counts["dead_low"],
                })
                try:
                    future.result()
                except Exception as e:
                    c = futures[future]
                    tqdm.tqdm.write(
                        f"  [stage2b] Worker failed for '{c.prompt[:60]}...': {e}"
                    )

    _save_checkpoint()

    print(
        f"\nStage 2b complete — {len(candidates)} prompts total\n"
        f"  Labels   : boundary_zone={label_counts['boundary_zone']}  "
        f"unstable={label_counts['unstable']}  "
        f"dead_high={label_counts['dead_high']}  "
        f"dead_low={label_counts['dead_low']}\n"
        f"  Tiers    : T1={tier_counts['tier1']}  "
        f"T2={tier_counts['tier2']}  "
        f"T3={tier_counts['tier3']}\n"
        f"  Settings : provider={config.SCREEN_PROVIDER}  "
        f"model={config.SCREEN_MODEL}  "
        f"temp={temperature}  "
        f"generations={generations}"
    )
    return candidates


def _restore_from_checkpoint(
    candidates: list[SeedCandidate],
    done: dict[str, dict],
) -> None:
    """Copy scores from the checkpoint dict back onto candidate objects.

    Called once at startup after loading the checkpoint file. This ensures
    that rolling saves (every 25 prompts) write correct data for all
    candidates — including ones scored in a previous session — not just
    the ones scored in the current session.

    Parameters
    ----------
    candidates: Full list of SeedCandidates for this run.
    done:       Checkpoint dict mapping sha256(prompt) → score data.
    """
    for candidate in candidates:
        key = hashlib.sha256(candidate.prompt.encode()).hexdigest()
        if key not in done:
            continue
        entry = done[key]
        candidate.refusal_score     = entry["score"]
        candidate.score_variance    = entry.get("variance", 0.0)
        candidate.label             = entry.get("label", "")
        candidate.screening_details = entry.get("details", [])
