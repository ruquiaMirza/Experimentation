"""
generate_report.py — Build an interactive HTML report from stage2b_screened.json.

Usage:
    python generate_report.py [input.json] [output.html]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT  = Path(__file__).resolve().parent.parent / "output" / "stage2b_screened.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "stage2b_report.html"

input_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

with open(input_path, encoding="utf-8") as f:
    data = json.load(f)

scored = [d for d in data if d.get("label")]
total  = len(data)

# Group by family → label → list of candidates
families: dict = defaultdict(lambda: {"boundary_zone": [], "dead_high": [], "dead_low": [], "unstable": []})
for d in scored:
    fam   = d.get("attack_family") or "(unknown)"
    label = d.get("label", "")
    if label in families[fam]:
        families[fam][label].append(d)

# Sort families by boundary_zone count descending
sorted_families = sorted(families.items(), key=lambda x: -len(x[1]["boundary_zone"]))

LABEL_META = {
    "boundary_zone": {"color": "#2563eb", "bg": "#eff6ff", "badge": "#dbeafe", "text": "Boundary"},
    "unstable":      {"color": "#d97706", "bg": "#fffbeb", "badge": "#fef3c7", "text": "Unstable"},
    "dead_high":     {"color": "#16a34a", "bg": "#f0fdf4", "badge": "#dcfce7", "text": "Dead-High"},
    "dead_low":      {"color": "#dc2626", "bg": "#fef2f2", "badge": "#fee2e2", "text": "Dead-Low"},
}


def esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def tier_badge(tier: int) -> str:
    colors = {1: ("#64748b", "#f1f5f9"), 2: ("#7c3aed", "#f5f3ff"), 3: ("#0369a1", "#e0f2fe")}
    c, bg = colors.get(tier, ("#64748b", "#f1f5f9"))
    return f'<span style="font-size:10px;font-weight:600;background:{bg};color:{c};border-radius:4px;padding:1px 5px">T{tier}</span>'


def score_bar(score: float) -> str:
    pct   = int(score * 100)
    color = "#dc2626" if score < 0.15 else "#16a34a" if score > 0.80 else "#d97706" if score < 0.45 else "#2563eb"
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="flex:1;height:6px;background:#e5e7eb;border-radius:3px">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:3px"></div></div>'
        f'<span style="font-size:11px;color:#374151;min-width:30px">{score:.2f}</span></div>'
    )


def render_prompt_card(d: dict, idx: int, fam_id: str) -> str:
    label     = d.get("label", "")
    meta      = LABEL_META.get(label, {"color": "#6b7280", "bg": "#f9fafb", "badge": "#f3f4f6", "text": label})
    mean      = d.get("refusal_score", 0.0)
    variance  = d.get("score_variance", 0.0)
    details   = d.get("screening_details") or []
    prompt    = esc(d.get("prompt", ""))
    goal      = esc(d.get("goal", ""))
    card_id   = f"{fam_id}_p{idx}"

    responses_html = ""
    for i, det in enumerate(details):
        resp  = esc(det.get("response", ""))
        score = det.get("score", 0.0)
        tier  = det.get("tier", 1)
        responses_html += f"""
        <div style="margin-bottom:10px;padding:10px;background:#f8fafc;border-radius:6px;border:1px solid #e2e8f0">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="font-size:11px;font-weight:600;color:#64748b">Response {i+1}</span>
            {tier_badge(tier)}
            <div style="flex:1">{score_bar(score)}</div>
          </div>
          <div style="font-size:12px;color:#374151;white-space:pre-wrap;line-height:1.5;max-height:200px;overflow-y:auto">{resp}</div>
        </div>"""

    return f"""
    <div style="border:1px solid {meta['color']}33;border-radius:8px;margin-bottom:8px;overflow:hidden">
      <button onclick="toggleCard('{card_id}')"
              style="width:100%;text-align:left;padding:10px 14px;background:{meta['bg']};
                     border:none;cursor:pointer;display:flex;align-items:flex-start;gap:10px">
        <span style="margin-top:2px;color:{meta['color']};font-size:14px" id="{card_id}_icon">▶</span>
        <div style="flex:1;min-width:0">
          <div style="font-size:12px;color:#374151;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%">{prompt}</div>
          {f'<div style="font-size:11px;color:#6b7280;margin-top:2px">Goal: {goal}</div>' if goal else ''}
        </div>
        <div style="display:flex;align-items:center;gap:8px;flex-shrink:0">
          <div style="width:80px">{score_bar(mean)}</div>
          <span style="font-size:10px;color:#6b7280;white-space:nowrap">var {variance:.3f}</span>
          <span style="font-size:10px;font-weight:600;background:{meta['badge']};color:{meta['color']};
                       border-radius:4px;padding:2px 6px">{meta['text']}</span>
        </div>
      </button>
      <div id="{card_id}" style="display:none;padding:12px 14px;background:#fff;border-top:1px solid #e5e7eb">
        <div style="margin-bottom:10px">
          <div style="font-size:11px;font-weight:600;color:#374151;margin-bottom:4px">Full prompt</div>
          <div style="font-size:12px;color:#374151;white-space:pre-wrap;background:#f8fafc;padding:8px;
                      border-radius:6px;border:1px solid #e2e8f0;max-height:150px;overflow-y:auto">{prompt}</div>
        </div>
        <div style="font-size:11px;font-weight:600;color:#374151;margin-bottom:6px">
          {len(details)} independent response{'' if len(details)==1 else 's'}
        </div>
        {responses_html}
      </div>
    </div>"""


def render_label_section(fam_id: str, label: str, items: list) -> str:
    if not items:
        return ""
    meta    = LABEL_META[label]
    sec_id  = f"{fam_id}_{label}"
    cards   = "".join(render_prompt_card(d, i, sec_id) for i, d in enumerate(items))
    return f"""
    <div style="margin-bottom:12px">
      <button onclick="toggleSection('{sec_id}')"
              style="display:flex;align-items:center;gap:8px;width:100%;text-align:left;
                     padding:8px 12px;background:{meta['badge']};border:none;border-radius:6px;
                     cursor:pointer;margin-bottom:4px">
        <span id="{sec_id}_icon" style="color:{meta['color']};font-size:13px">▶</span>
        <span style="font-size:13px;font-weight:600;color:{meta['color']}">{meta['text']}</span>
        <span style="font-size:12px;color:#6b7280">{len(items)} prompt{'s' if len(items)!=1 else ''}</span>
      </button>
      <div id="{sec_id}" style="display:none;padding-left:8px">{cards}</div>
    </div>"""


# ── Build family blocks ───────────────────────────────────────────────────────
family_blocks = ""
for fam, labels in sorted_families:
    fam_id  = fam.replace(" ", "_").replace(".", "_")
    counts  = {k: len(v) for k, v in labels.items()}
    total_f = sum(counts.values())
    variance_vals = [
        d.get("score_variance", 0.0)
        for v in labels.values() for d in v
    ]
    avg_var = sum(variance_vals) / len(variance_vals) if variance_vals else 0.0

    badge_row = ""
    for lbl in ("boundary_zone", "unstable", "dead_high", "dead_low"):
        n = counts[lbl]
        if n == 0:
            continue
        m = LABEL_META[lbl]
        badge_row += f'<span style="font-size:11px;font-weight:600;background:{m["badge"]};color:{m["color"]};border-radius:4px;padding:2px 7px">{m["text"]}: {n}</span> '

    label_sections = "".join(
        render_label_section(fam_id, lbl, labels[lbl])
        for lbl in ("boundary_zone", "unstable", "dead_high", "dead_low")
    )

    family_blocks += f"""
    <div style="border:1px solid #e5e7eb;border-radius:10px;margin-bottom:12px;overflow:hidden">
      <button onclick="toggleFamily('{fam_id}')"
              style="width:100%;text-align:left;padding:14px 16px;background:#f9fafb;
                     border:none;cursor:pointer;display:flex;align-items:center;gap:12px">
        <span id="{fam_id}_icon" style="color:#374151;font-size:15px">▶</span>
        <div style="flex:1">
          <span style="font-size:15px;font-weight:700;color:#111827">{esc(fam)}</span>
          <span style="font-size:12px;color:#6b7280;margin-left:8px">{total_f} prompts · avg variance {avg_var:.3f}</span>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">{badge_row}</div>
      </button>
      <div id="{fam_id}" style="display:none;padding:14px 16px">
        {label_sections}
      </div>
    </div>"""

# ── Summary stats ─────────────────────────────────────────────────────────────
total_counts = {k: sum(len(labels[k]) for _, labels in sorted_families) for k in LABEL_META}

summary_cards = ""
for lbl, meta in LABEL_META.items():
    n = total_counts[lbl]
    pct = n / len(scored) * 100 if scored else 0
    summary_cards += f"""
    <div style="background:{meta['bg']};border:1px solid {meta['color']}33;border-radius:8px;padding:14px 18px;text-align:center">
      <div style="font-size:24px;font-weight:700;color:{meta['color']}">{n}</div>
      <div style="font-size:12px;font-weight:600;color:{meta['color']};margin-top:2px">{meta['text']}</div>
      <div style="font-size:11px;color:#6b7280;margin-top:2px">{pct:.1f}%</div>
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage 2b Screening Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f3f4f6; color: #111827; }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 24px 16px; }}
  button {{ font-family: inherit; }}
</style>
</head>
<body>
<div class="container">
  <h1 style="font-size:22px;font-weight:700;margin-bottom:4px">Stage 2b Screening Report</h1>
  <p style="font-size:13px;color:#6b7280;margin-bottom:20px">
    {len(scored)} scored / {total} total &nbsp;·&nbsp; {len(sorted_families)} attack families
  </p>

  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px">
    {summary_cards}
  </div>

  <div id="families">
    {family_blocks}
  </div>
</div>

<script>
function toggleFamily(id) {{
  var el   = document.getElementById(id);
  var icon = document.getElementById(id + '_icon');
  var open = el.style.display === 'none';
  el.style.display   = open ? 'block' : 'none';
  icon.textContent   = open ? '▼' : '▶';
}}
function toggleSection(id) {{
  var el   = document.getElementById(id);
  var icon = document.getElementById(id + '_icon');
  var open = el.style.display === 'none';
  el.style.display   = open ? 'block' : 'none';
  icon.textContent   = open ? '▼' : '▶';
}}
function toggleCard(id) {{
  var el   = document.getElementById(id);
  var icon = document.getElementById(id + '_icon');
  var open = el.style.display === 'none';
  el.style.display   = open ? 'block' : 'none';
  icon.textContent   = open ? '▼' : '▶';
}}
</script>
</body>
</html>"""

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Report written to {output_path}")
print(f"Scored: {len(scored)} / {total}  |  Families: {len(sorted_families)}")
for lbl, meta in LABEL_META.items():
    print(f"  {meta['text']:12s}: {total_counts[lbl]}")
