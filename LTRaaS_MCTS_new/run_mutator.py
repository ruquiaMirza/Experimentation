#!/usr/bin/env python3
"""Standalone entry point for running PromptMutator without the MCTS search loop.

Usage:
    python run_mutator.py --list
    python run_mutator.py --index 3 --n 3
    python run_mutator.py --index 3 --operator persona_injection fictional_wrapper
    python run_mutator.py --index 3 --all-operators
    python run_mutator.py --prompt "..." --goal "..." --family dan --operator paraphrase

Required env var:
    CLAUDE_CODE_OAUTH_TOKEN
Optional env var (fallback when Claude refuses/errors):
    GROQ_API_KEY

Seeds and purpose default to whatever is configured in config.yaml.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mutator import PromptMutator, OPERATORS, OPERATOR_INSTRUCTIONS, DEFAULT_OPERATORS  # noqa: E402


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
    parser.add_argument('--index', type=int, help='Index of the seed (from --list) to mutate')
    parser.add_argument('--prompt', help='Custom seed prompt (use instead of --index)')
    parser.add_argument('--goal', help='Attack goal (required with --prompt)')
    parser.add_argument('--family', help='Attack family, selects the operator pool (required with --prompt)')
    parser.add_argument('--operator', nargs='+', choices=sorted(OPERATOR_INSTRUCTIONS), help='Specific operator(s) to apply')
    parser.add_argument('--all-operators', action='store_true', help="Apply every operator in the family's pool")
    parser.add_argument('-n', type=int, default=3, help='Number of random operators to sample (default 3, ignored if --operator/--all-operators set)')
    parser.add_argument('--model', default=None, help='Mutator model override (default: from config.yaml)')
    parser.add_argument('--purpose', default=None, help='Target system context override (default: from config.yaml)')
    parser.add_argument('--out', help='Write results as JSON to this path')
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

    model = args.model or cfg.get('mutator_model', 'claude-haiku-4-5-20251001')
    purpose = args.purpose if args.purpose is not None else cfg.get('purpose', '')

    print(f"Seed family : {family}")
    print(f"Goal        : {goal}")
    print(f"Prompt      : {prompt[:200]}{'…' if len(prompt) > 200 else ''}")
    print(f"Mutator     : {model}\n")

    mutator = PromptMutator(model=model)

    if args.all_operators:
        pool = OPERATORS.get(family, DEFAULT_OPERATORS)
    elif args.operator:
        pool = args.operator
    else:
        pool = None  # let mutator.mutate() randomly sample -n from the family pool

    results = []
    if pool is not None:
        for operator in pool:
            ret = mutator._apply_operator(prompt, goal, operator, purpose)
            if ret is None:
                print(f"  [{operator}] refused / failed — skipped\n")
                continue
            mutated, used_model = ret
            results.append((operator, mutated, used_model))
    else:
        results = mutator.mutate(prompt, goal, family, n=args.n, purpose=purpose)

    print(f"\n{'─' * 64}")
    print(f"Generated {len(results)} mutation(s)")
    print('─' * 64)
    for operator, mutated, used_model in results:
        print(f"\n[{operator}]  (via {used_model})")
        print(f"  {OPERATOR_INSTRUCTIONS[operator]}")
        print(f"  → {mutated}")

    if args.out:
        payload = [
            {
                "seed_prompt": prompt,
                "goal": goal,
                "attack_family": family,
                "operator": operator,
                "mutated_prompt": mutated,
                "model": used_model,
            }
            for operator, mutated, used_model in results
        ]
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
