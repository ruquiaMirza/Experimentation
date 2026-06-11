"""
report_stage2b.py — Print a family-level label distribution from stage2b_screened.json.

Usage:
    python report_stage2b.py [path/to/stage2b_screened.json]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "output" / "stage2b_screened.json"

path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH

with open(path, encoding="utf-8") as f:
    data = json.load(f)

scored = [d for d in data if d.get("label")]

families: dict = defaultdict(lambda: {"boundary_zone": 0, "dead_high": 0, "dead_low": 0, "unstable": 0, "total": 0})

for d in scored:
    fam = d.get("attack_family") or "(unknown)"
    families[fam][d["label"]] += 1
    families[fam]["total"] += 1

print(f"Scored prompts: {len(scored)} / {len(data)} total\n")
print(f"{'Family':<30} {'Total':>5}  {'Boundary':>8}  {'Dead-Hi':>7}  {'Dead-Lo':>7}  {'Unstable':>8}")
print("-" * 75)
for fam, c in sorted(families.items(), key=lambda x: -x[1]["total"]):
    print(f"{fam:<30} {c['total']:>5}  {c['boundary_zone']:>8}  {c['dead_high']:>7}  {c['dead_low']:>7}  {c['unstable']:>8}")
print("-" * 75)
t = {k: sum(f[k] for f in families.values()) for k in ["total", "boundary_zone", "dead_high", "dead_low", "unstable"]}
print(f"{'TOTAL':<30} {t['total']:>5}  {t['boundary_zone']:>8}  {t['dead_high']:>7}  {t['dead_low']:>7}  {t['unstable']:>8}")
