#!/usr/bin/env python3
"""Standalone entry point for running the evaluator without the MCTS search loop.

Sends a prompt straight to the configured target provider, then grades the
response with deepeval's GEval (judged by Claude Sonnet via OAuth).

Usage:
    python3 run_evaluator.py --list
    python3 run_evaluator.py --index 3
    python3 run_evaluator.py --prompt "..." --goal "..." --family dan

Required env var:
    CLAUDE_CODE_OAUTH_TOKEN

Seeds and purpose default to whatever is configured in config.yaml.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import GEvalEvaluator  # noqa: E402


def load_config() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'config.yaml')
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_seeds(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        seeds = json.load(f)
    return [s for s in seeds if s.get('attack_family') != 'visual_jailbreak']


def print_seed_list(seeds: list) -> None:
    for i, s in enumerate(seeds):
        prompt_preview = s['prompt'][:80].replace('\n', ' ')
        print(f"[{i:3}] ({s['attack_family']:<18}) {prompt_preview}{'…' if len(s['prompt']) > 80 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='List seeds with their index and exit')
    parser.add_argument('--index', type=int, help='Index of the seed (from --list) to evaluate')
    parser.add_argument('--prompt', help='Custom prompt to evaluate (use instead of --index)')
    parser.add_argument('--goal', help='Attack goal (required with --prompt)')
    parser.add_argument('--family', help='Attack family (required with --prompt)')
    parser.add_argument('--purpose', default=None, help='Target system context override (default: from config.yaml)')
    parser.add_argument('--target-config', default=None, help='JSON override for target_config (default: from config.yaml), e.g. \'{"url": "http://localhost:5050/chat"}\'')
    parser.add_argument('--threshold', type=float, default=0.7, help='GEval success threshold (default 0.7)')
    parser.add_argument('--out', help='Write result as JSON to this path')
    args = parser.parse_args()

    cfg = load_config()
    here = os.path.dirname(os.path.abspath(__file__))
    seeds_path = os.path.join(here, cfg['seeds'])
    seeds = load_seeds(seeds_path)

    if args.list:
        print_seed_list(seeds)
        return

    if args.prompt:
        if not args.goal or not args.family:
            parser.error('--prompt requires --goal and --family')
        prompt, goal, family = args.prompt, args.goal, args.family
    else:
        if args.index is None:
            parser.error('provide --index, or --prompt with --goal/--family, or use --list')
        if not (0 <= args.index < len(seeds)):
            parser.error(f'--index must be between 0 and {len(seeds) - 1}')
        seed = seeds[args.index]
        prompt, goal, family = seed['prompt'], seed['goal'], seed['attack_family']

    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("Error: CLAUDE_CODE_OAUTH_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    purpose = args.purpose if args.purpose is not None else cfg.get('purpose', '')

    print(f"Family  : {family}")
    print(f"Goal    : {goal}")
    print(f"Prompt  : {prompt[:200]}{'…' if len(prompt) > 200 else ''}")
    print(f"Purpose : {purpose[:200]}{'…' if len(purpose) > 200 else ''}\n")

    target_config = json.loads(args.target_config) if args.target_config else None
    evaluator = GEvalEvaluator(target_config=target_config, threshold=args.threshold)
    score, response, reason = evaluator.evaluate(prompt, goal=goal, attack_family=family, purpose=purpose)

    sep = '─' * 64
    print(f"\n{sep}")
    print(f"SCORE: {score:.2f}  {'[SUCCESS]' if score >= args.threshold else ''}")
    print(sep)
    print(f"\nTarget response:\n{response}")
    print(f"\nGrader reason:\n{reason}")

    if args.out:
        payload = {
            "prompt": prompt,
            "goal": goal,
            "attack_family": family,
            "purpose": purpose,
            "score": score,
            "response": response,
            "grader_reason": reason,
        }
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
