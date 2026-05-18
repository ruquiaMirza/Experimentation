"""
Section 0 of the design doc: the data contracts between components.

These dataclasses are the "spine" of the platform. Every component reads
and writes these shapes. Change them carefully — they are the API between
all the other modules.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Target:
    """A customer's LLM application registered for testing."""
    id: str
    name: str
    endpoint_url: str
    system_type: str          # "chatbot" | "rag" | "tool_agent"
    purpose: str              # free-text app description, fed to promptfoo
    auth_ref: str             # pointer into the vault, NEVER the raw key


@dataclass
class ScanDefinition:
    """What the customer asked us to test."""
    id: str
    target_id: str
    name: str
    categories: list[str]     # unified vulnerability categories
    repetitions: int = 3      # N runs per probe — outputs are stochastic
    severity_threshold: str = "high"


@dataclass
class Job:
    """One unit of work for a single engine. A ScanDefinition produces many Jobs."""
    id: str
    scan_id: str
    target_id: str
    engine: str               # "promptfoo" | "garak" | "mock"
    category_id: str          # which unified category this job covers
    config_blob: dict         # the YAML/JSON the engine will read
    repetitions: int


@dataclass
class ScanPlan:
    """A ScanDefinition translated into engine-specific Jobs."""
    scan_id: str
    jobs: list[Job]


@dataclass
class RawResult:
    """What a worker brings back from running an engine. Verbatim engine output."""
    job_id: str
    engine: str
    status: str               # "succeeded" | "failed"
    raw_output: list[dict]    # the engine's native output, untouched
    error: str | None = None


@dataclass
class Finding:
    """The unified, engine-agnostic finding. This is what reports are built from."""
    id: str
    scan_id: str
    target_id: str
    category_id: str          # unified taxonomy
    severity: str             # "low" | "medium" | "high" | "critical"
    engine_source: str        # breadcrumb: which engine caught this
    probe_text: str           # the adversarial prompt
    target_response: str
    judge_verdict: str        # "pass" | "fail"
    judge_reasoning: str
    runs: int                 # how many times we ran this probe
    successes: int            # how many of those runs broke the target
    success_rate: float       # successes / runs
    cluster_id: str | None = None    # set by the deduper
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)
