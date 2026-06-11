import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


class _PathEncoder(json.JSONEncoder):
    """Serialize pathlib.Path objects as plain strings in JSON output."""
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


@dataclass
class SeedCandidate:
    """A single adversarial prompt extracted from a garak probe, with scoring metadata."""
    prompt: str
    probe_module: str
    probe_class: str
    tags: list = field(default_factory=list)
    goal: str = ""
    attack_family: str = ""
    refusal_score: float = 0.0   # 0.0 = hard refusal, 1.0 = full bypass
    buff_origin: str = "original"
    response: str = ""
    cluster_size: int = 1        # number of original prompts this candidate represents
    score_variance: float = 0.0  # stdev of per-response scores from stage 2b
    label: str = ""              # dead_low | dead_high | unstable | boundary_zone
    screening_details: list = field(default_factory=list)
    # per-response breakdown from stage 2b, each entry:
    # {"response": str, "score": float, "tier": int}


def save_stage(path: str, candidates: list[SeedCandidate]) -> None:
    """Write candidates to a JSON file so the stage can be resumed later."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in candidates], f, indent=2,
                  ensure_ascii=False, cls=_PathEncoder)
    print(f"  → saved {len(candidates)} records to {path}")


def load_stage(path: str) -> list[SeedCandidate]:
    """Read candidates written by save_stage and reconstruct SeedCandidate objects."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    candidates = [SeedCandidate(**r) for r in records]
    print(f"  ← loaded {len(candidates)} records from {path}")
    return candidates
