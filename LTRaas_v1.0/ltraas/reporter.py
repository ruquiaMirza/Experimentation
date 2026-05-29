"""
Generates a scan report from the findings in the database.

Two outputs per scan:
  Console  — summary table with counts, break rates, and top offenders.
  JSON     — full findings export at outputs/report_<scan_id>.json,
             suitable for dashboards, CI checks, or compliance tooling.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from . import storage

logger = logging.getLogger(__name__)


def generate(scan_id: str) -> dict:
    """Build a report from the findings table. Prints summary, returns dict."""
    findings = storage.findings_for_scan(scan_id)

    by_category = Counter(f.category_id for f in findings)
    by_severity = Counter(f.severity for f in findings)
    failed = [f for f in findings if f.judge_verdict == "fail"]
    clusters = {f.cluster_id for f in failed}

    by_plugin = Counter(f.plugin_id for f in findings if f.plugin_id)

    lines = [
        "",
        "=" * 60,
        f"  SCAN REPORT — {scan_id}",
        "=" * 60,
        f"  Total probes tested:      {len(findings)}",
        f"  Probes that broke target: {len(failed)}",
        f"  Unique vulnerabilities:   {len(clusters)}",
        f"  Break rate:               "
        + (f"{len(failed)/len(findings)*100:.1f}%" if findings else "n/a"),
        "",
        "  Findings by category:",
    ]
    for cat, count in by_category.most_common():
        cat_findings = [f for f in findings if f.category_id == cat]
        breaks = sum(1 for f in cat_findings if f.judge_verdict == "fail")
        lines.append(f"    {cat:20s} {breaks:2d}/{count:2d} broke "
                     f"({breaks/count*100:.0f}%)")

    if by_plugin:
        lines.append("")
        lines.append("  Findings by plugin:")
        for plugin_id, count in by_plugin.most_common():
            plugin_findings = [f for f in findings if f.plugin_id == plugin_id]
            breaks = sum(1 for f in plugin_findings if f.judge_verdict == "fail")
            lines.append(f"    {plugin_id:20s} {breaks:2d}/{count:2d} broke "
                         f"({breaks/count*100:.0f}%)")

    lines.append("")
    lines.append("  Findings by severity:")
    for sev in ["critical", "high", "medium", "low"]:
        if sev in by_severity:
            lines.append(f"    {sev:10s} {by_severity[sev]}")
    lines.append("")
    lines.append("  Top broken probes (engineer view):")
    top = sorted(failed, key=lambda f: -f.success_rate)[:3]
    for i, f in enumerate(top, 1):
        plugin_label = f" [{f.plugin_id}]" if f.plugin_id else ""
        lines.append(f"    [{i}] {f.category_id}{plugin_label} ({f.severity}, "
                     f"broke {f.successes}/{f.runs} runs)")
        lines.append(f"        probe:    {f.probe_text[:80]}")
        lines.append(f"        response: {f.target_response[:80]}")
        lines.append(f"        reason:   {f.judge_reasoning}")
    lines.append("=" * 60)

    for line in lines:
        logger.info(line)

    report = {
        "scan_id": scan_id,
        "summary": {
            "total_probes": len(findings),
            "broken": len(failed),
            "unique_vulnerabilities": len(clusters),
            "by_category": dict(by_category),
            "by_plugin": dict(by_plugin),
            "by_severity": dict(by_severity),
        },
        "findings": [f.to_dict() for f in findings],
    }

    out_path = Path(f"outputs/report_{scan_id}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Full report written to: %s", out_path)
    return report
