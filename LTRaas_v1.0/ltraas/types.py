"""
Data contracts shared across the entire pipeline.

Every module reads and writes these shapes — they are the only thing
connecting scan ingestion to engine execution to reporting.
Change them carefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Target:
    """An LLM application registered for red-teaming."""
    id: str
    name: str
    endpoint_url: str
    system_type: str          # "chatbot" | "rag" | "tool_agent"
    purpose: str              # free-text description used to craft probes
    auth_ref: str             # vault reference — never the raw API key


@dataclass
class ScanDefinition:
    """What to test and how many times to test it."""
    id: str
    target_id: str
    name: str
    categories: list[str]          # unified vulnerability categories to run
    llm_repetitions: int = 3       # runs per probe (mock / llm engines)
    promptfoo_num_tests: int = 10  # test cases per plugin (promptfoo engine)
    promptfoo_concurrency: int = 4 # max parallel requests to the target (promptfoo engine)


@dataclass
class Job:
    """One engine run covering one vulnerability category.

    A ScanDefinition produces one Job per category. Each Job carries a
    config_blob that the target engine reads directly.
    """
    id: str
    scan_id: str
    target_id: str
    engine: str               # "promptfoo" | "llm" | "mock"
    category_id: str          # which unified category this job covers
    config_blob: dict         # engine-specific config (promptfoo YAML shape)
    repetitions: int


@dataclass
class ScanPlan:
    """A ScanDefinition broken into engine-specific Jobs."""
    scan_id: str
    jobs: list[Job]


@dataclass
class RawResult:
    """Verbatim output from one engine run — nothing normalised yet."""
    job_id: str
    engine: str
    status: str               # "succeeded" | "failed"
    raw_output: list[dict]    # the engine's native output, untouched
    error: str | None = None


@dataclass
class Finding:
    """A single vulnerability result, normalised from raw engine output.

    Multiple runs of the same probe are collapsed into one Finding with
    a success_rate so you can see how reliably the target is vulnerable.
    """
    id: str
    scan_id: str
    target_id: str
    category_id: str          # unified taxonomy
    severity: str             # "low" | "medium" | "high" | "critical"
    engine_source: str        # which engine produced this finding
    probe_text: str           # the adversarial prompt that was sent
    target_response: str
    judge_verdict: str        # "pass" | "fail"
    judge_reasoning: str
    runs: int                 # how many times this probe was sent
    successes: int            # how many of those runs broke the target
    success_rate: float       # successes / runs
    plugin_id: str | None = None        # engine plugin that generated this probe
    cluster_id: str | None = None       # dedup key; same vuln across probes shares one ID
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)
