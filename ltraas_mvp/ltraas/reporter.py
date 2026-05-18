"""
Component 7: Reporter.

Reads Findings from the database and produces three views:
  1. Executive summary (counts and rates)
  2. Engineer view (per-finding detail with repro info)
  3. JSON export (for the API, dashboards, compliance tools)

In production these get rendered as a web dashboard + PDF + JSON API.
For the MVP they print to console and write a JSON file.
"""

import json
from collections import Counter
from pathlib import Path
from . import storage


def generate(scan_id: str) -> dict:
    """Build a report from the findings table. Prints summary, returns dict."""
    findings = storage.findings_for_scan(scan_id)

    by_category = Counter(f.category_id for f in findings)
    by_severity = Counter(f.severity for f in findings)
    failed = [f for f in findings if f.judge_verdict == "fail"]
    clusters = {f.cluster_id for f in failed}

    print("\n" + "=" * 60)
    print(f"  SCAN REPORT — {scan_id}")
    print("=" * 60)
    print(f"  Total probes tested:      {len(findings)}")
    print(f"  Probes that broke target: {len(failed)}")
    print(f"  Unique vulnerabilities:   {len(clusters)}")
    print(f"  Break rate:               "
          f"{len(failed)/len(findings)*100:.1f}%" if findings else "n/a")
    print()
    print("  Findings by category:")
    for cat, count in by_category.most_common():
        cat_findings = [f for f in findings if f.category_id == cat]
        breaks = sum(1 for f in cat_findings if f.judge_verdict == "fail")
        print(f"    {cat:20s} {breaks:2d}/{count:2d} broke "
              f"({breaks/count*100:.0f}%)")
    print()
    print("  Findings by severity:")
    for sev in ["critical", "high", "medium", "low"]:
        if sev in by_severity:
            print(f"    {sev:10s} {by_severity[sev]}")
    print()
    print("  Top broken probes (engineer view):")
    top = sorted(failed, key=lambda f: -f.success_rate)[:3]
    for i, f in enumerate(top, 1):
        print(f"    [{i}] {f.category_id} ({f.severity}, "
              f"broke {f.successes}/{f.runs} runs)")
        print(f"        probe:    {f.probe_text[:80]}")
        print(f"        response: {f.target_response[:80]}")
        print(f"        reason:   {f.judge_reasoning}")
    print("=" * 60 + "\n")

    report = {
        "scan_id": scan_id,
        "summary": {
            "total_probes": len(findings),
            "broken": len(failed),
            "unique_vulnerabilities": len(clusters),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
        },
        "findings": [f.to_dict() for f in findings],
    }

    out_path = Path(f"outputs/report_{scan_id}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"  Full report written to: {out_path}\n")
    return report
