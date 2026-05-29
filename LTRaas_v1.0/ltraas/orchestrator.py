"""
Runs a complete scan from start to finish.

Coordinates the full pipeline: translate → dispatch → normalize → persist.
This is the right place to add per-tenant fairness, rate limiting, cost
budgets, or retry logic — none of that needs to touch the engines or
the translator.

submit() blocks until the scan is done. A background queue would be a
drop-in replacement: same interface, asynchronous execution.
"""

from __future__ import annotations

import logging
from datetime import datetime
from . import storage, scan_translator, target_registry, worker, normalizer
from .types import ScanDefinition, ScanPlan

logger = logging.getLogger(__name__)


def submit(
    scan_def: ScanDefinition,
    engine_impl=None,
    models: dict = None,
    engine_type: str = None,
) -> str:
    """Run a scan end-to-end. Returns the scan_id when done.

    In the design doc this returns immediately and runs in the background.
    For the MVP it's blocking so you can see the whole flow in one trace.

    engine_impl  — engine instance to use (LLMEngine, PromptfooEngine, etc.).
                   Defaults to MockEngine when None.
    models       — provider/model config for target, judge, and attack_generator.
                   Required for llm and promptfoo engines.
    engine_type  — "mock" | "llm" | "promptfoo". Tells the scan translator which
                   config blob shape to produce for each job.
    """
    if models is None:
        raise ValueError("orchestrator.submit: models is required")
    if engine_type is None:
        raise ValueError("orchestrator.submit: engine_type is required")
    logger.info("Starting scan %s on target %s", scan_def.id, scan_def.target_id)

    target = target_registry.get_target(scan_def.target_id)
    storage.save_scan(
        scan_def.id, scan_def.target_id, scan_def.name,
        status="running", created_at=datetime.utcnow().isoformat(),
    )

    plan: ScanPlan = scan_translator.build_plan(
        scan_def, target, models=models, engine_type=engine_type,
    )
    logger.info("Plan built — %d job(s)", len(plan.jobs))

    all_findings = []
    for job in plan.jobs:
        raw = worker.process_job(job, engine_impl=engine_impl)
        if raw.status != "succeeded":
            logger.error("Job %s FAILED: %s", raw.job_id, raw.error)

        findings = normalizer.normalize(job, raw)
        all_findings.extend(findings)

    storage.update_scan_status(scan_def.id, "complete")
    logger.info("Scan %s complete — %d finding(s) persisted",
                scan_def.id, len(all_findings))
    return scan_def.id
