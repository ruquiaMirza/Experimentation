"""
Component 4: Orchestrator.

In production this is a distributed job queue (Celery, Temporal, Step
Functions). For the MVP it's a synchronous loop — same shape, no queue.

The point of having an orchestrator at all (vs. inlining the work) is
that this is the chokepoint for:
  - per-tenant fairness
  - rate limits to the target
  - cost budgets
  - retries
You can add each of those independently without touching workers or
the translator.
"""

from datetime import datetime
from . import storage, scan_translator, target_registry, worker, normalizer
from .types import ScanDefinition, ScanPlan


def submit(scan_def: ScanDefinition) -> str:
    """Run a scan end-to-end. Returns the scan_id when done.

    In the design doc this returns immediately and runs in the background.
    For the MVP it's blocking so you can see the whole flow in one trace.
    """
    print(f"\n[orchestrator] starting scan {scan_def.id} on target {scan_def.target_id}")

    target = target_registry.get_target(scan_def.target_id)
    storage.save_scan(
        scan_def.id, scan_def.target_id, scan_def.name,
        status="running", created_at=datetime.utcnow().isoformat(),
    )

    plan: ScanPlan = scan_translator.build_plan(scan_def, target)
    print(f"[orchestrator] plan has {len(plan.jobs)} jobs")

    all_findings = []
    for job in plan.jobs:
        raw = worker.process_job(job)
        findings = normalizer.normalize(job, raw)
        all_findings.extend(findings)

    storage.update_scan_status(scan_def.id, "complete")
    print(f"[orchestrator] scan {scan_def.id} complete — "
          f"{len(all_findings)} findings persisted")
    return scan_def.id
