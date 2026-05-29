"""Generate a local HTML report from stage7_seeds.json."""

import json
import html
from collections import defaultdict
from pathlib import Path

_ROOT  = Path(__file__).resolve().parent.parent
INPUT  = _ROOT / "output" / "stage7_seeds.json"
OUTPUT = _ROOT / "output" / "stage7_report.html"

with open(INPUT, encoding="utf-8") as f:
    seeds = json.load(f)

# ── Categorise ────────────────────────────────────────────────────────────────
for s in seeds:
    score = s["refusal_score"]
    if score >= 0.9:
        s["_category"] = "anchor"
    elif 0.1 <= score <= 0.8:
        s["_category"] = "mcts"
    else:
        s["_category"] = "refused"

by_family = defaultdict(list)
for s in seeds:
    by_family[s["attack_family"]].append(s)

mcts_seeds    = [s for s in seeds if s["_category"] == "mcts"]
anchor_seeds  = [s for s in seeds if s["_category"] == "anchor"]
refused_seeds = [s for s in seeds if s["_category"] == "refused"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def score_bar(score: float) -> str:
    pct   = int(score * 100)
    if score >= 0.9:
        color = "#e74c3c"   # red  — anchor / full bypass
    elif score >= 0.5:
        color = "#e67e22"   # orange — upper boundary
    elif score >= 0.1:
        color = "#f1c40f"   # yellow — lower boundary
    else:
        color = "#2ecc71"   # green  — refused
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar" style="width:{pct}%;background:{color}"></div>'
        f'<span class="bar-label">{score:.2f}</span>'
        f'</div>'
    )

def category_badge(cat: str) -> str:
    styles = {
        "mcts":    "background:#2980b9;color:#fff",
        "anchor":  "background:#e74c3c;color:#fff",
        "refused": "background:#27ae60;color:#fff",
    }
    labels = {"mcts": "MCTS seed", "anchor": "Anchor", "refused": "Refused"}
    return f'<span class="badge" style="{styles[cat]}">{labels[cat]}</span>'

def prompt_card(s: dict, idx: int) -> str:
    tags_html = " ".join(
        f'<span class="tag">{html.escape(t)}</span>' for t in s.get("tags", [])
    )
    prompt_text = html.escape(s["prompt"]).replace("\\n", "<br>")
    return f"""
    <div class="card {s['_category']}">
      <div class="card-header">
        <span class="card-num">#{idx}</span>
        {category_badge(s['_category'])}
        <span class="probe">{html.escape(s['probe_class'])}</span>
        <span class="goal">{html.escape(s.get('goal',''))}</span>
        <span class="buff">origin: {html.escape(s.get('buff_origin','original'))}</span>
        {score_bar(s['refusal_score'])}
      </div>
      <pre class="prompt-text">{prompt_text}</pre>
      <div class="tags">{tags_html}</div>
    </div>"""

# ── Family summary rows ───────────────────────────────────────────────────────
family_rows = ""
for fam, members in sorted(by_family.items()):
    mcts_n   = sum(1 for m in members if m["_category"] == "mcts")
    anchor_n = sum(1 for m in members if m["_category"] == "anchor")
    ref_n    = sum(1 for m in members if m["_category"] == "refused")
    avg      = sum(m["refusal_score"] for m in members) / len(members)
    family_rows += f"""
    <tr>
      <td><a href="#fam-{html.escape(fam)}">{html.escape(fam)}</a></td>
      <td>{len(members)}</td>
      <td><span class="badge" style="background:#2980b9;color:#fff">{mcts_n}</span></td>
      <td><span class="badge" style="background:#e74c3c;color:#fff">{anchor_n}</span></td>
      <td><span class="badge" style="background:#27ae60;color:#fff">{ref_n}</span></td>
      <td>{score_bar(avg)}</td>
    </tr>"""

# ── Per-family sections ───────────────────────────────────────────────────────
family_sections = ""
for fam, members in sorted(by_family.items()):
    cards = "".join(prompt_card(s, i+1) for i, s in enumerate(members))
    family_sections += f"""
    <section id="fam-{html.escape(fam)}">
      <h2>{html.escape(fam)}
        <span class="fam-count">{len(members)} seed{"s" if len(members)>1 else ""}</span>
      </h2>
      {cards}
    </section>"""

# ── Full HTML ─────────────────────────────────────────────────────────────────
html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage 7 Seeds Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e0e0e0; line-height: 1.5; }}
  a {{ color: #5dade2; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* ── Layout ── */
  .sidebar {{ position: fixed; top: 0; left: 0; width: 220px; height: 100vh;
              background: #1a1d27; overflow-y: auto; padding: 20px 12px;
              border-right: 1px solid #2a2d3a; }}
  .main {{ margin-left: 220px; padding: 32px 40px; max-width: 1100px; }}

  /* ── Sidebar ── */
  .sidebar h3 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
                color: #888; margin: 16px 0 6px; }}
  .sidebar a {{ display: block; padding: 4px 8px; border-radius: 4px;
               font-size: 13px; color: #ccc; }}
  .sidebar a:hover {{ background: #2a2d3a; color: #fff; text-decoration: none; }}

  /* ── Hero stats ── */
  .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }}
  .stat-box {{ background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px;
              padding: 16px 24px; min-width: 130px; }}
  .stat-box .num {{ font-size: 32px; font-weight: 700; }}
  .stat-box .lbl {{ font-size: 12px; color: #888; margin-top: 2px; }}
  .stat-box.blue  .num {{ color: #2980b9; }}
  .stat-box.red   .num {{ color: #e74c3c; }}
  .stat-box.green .num {{ color: #27ae60; }}
  .stat-box.total .num {{ color: #e0e0e0; }}

  /* ── Summary table ── */
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #2a2d3a;
            font-size: 13px; }}
  th {{ background: #1a1d27; color: #888; font-weight: 600;
        text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
  tr:hover td {{ background: #1e2130; }}

  /* ── Section headings ── */
  section {{ margin-bottom: 48px; }}
  h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
  h2 {{ font-size: 18px; font-weight: 600; margin: 0 0 16px;
        padding-bottom: 8px; border-bottom: 1px solid #2a2d3a;
        display: flex; align-items: center; gap: 10px; }}
  .fam-count {{ font-size: 13px; color: #888; font-weight: 400; }}
  .subtitle {{ color: #888; font-size: 14px; margin-bottom: 32px; }}

  /* ── Cards ── */
  .card {{ background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px;
           margin-bottom: 14px; overflow: hidden; }}
  .card.mcts    {{ border-left: 3px solid #2980b9; }}
  .card.anchor  {{ border-left: 3px solid #e74c3c; }}
  .card.refused {{ border-left: 3px solid #27ae60; }}

  .card-header {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
                  padding: 10px 16px; background: #1e2130; }}
  .card-num {{ font-size: 12px; color: #555; min-width: 24px; }}
  .probe {{ font-size: 12px; color: #7fb3d3; font-family: monospace; }}
  .goal  {{ font-size: 12px; color: #aaa; flex: 1; }}
  .buff  {{ font-size: 11px; color: #666; }}

  .prompt-text {{ padding: 14px 16px; font-size: 13px; font-family: monospace;
                  white-space: pre-wrap; word-break: break-word;
                  color: #d4d4d4; max-height: 200px; overflow-y: auto;
                  background: #13151f; border-top: 1px solid #2a2d3a; }}

  .tags {{ padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 6px;
           border-top: 1px solid #2a2d3a; }}
  .tag {{ font-size: 10px; background: #252836; color: #888;
          padding: 2px 7px; border-radius: 10px; }}

  /* ── Score bar ── */
  .bar-wrap {{ display: flex; align-items: center; gap: 6px; min-width: 110px; }}
  .bar {{ height: 6px; border-radius: 3px; min-width: 2px; transition: width .3s; }}
  .bar-label {{ font-size: 12px; color: #aaa; white-space: nowrap; }}

  /* ── Badge ── */
  .badge {{ font-size: 11px; padding: 2px 8px; border-radius: 10px;
            font-weight: 600; white-space: nowrap; }}

  /* ── Filter buttons ── */
  .filters {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
  .filter-btn {{ padding: 6px 16px; border-radius: 20px; border: 1px solid #2a2d3a;
                 background: #1a1d27; color: #ccc; cursor: pointer; font-size: 13px; }}
  .filter-btn:hover, .filter-btn.active {{ background: #2980b9; color: #fff; border-color: #2980b9; }}

  @media (max-width: 768px) {{
    .sidebar {{ display: none; }}
    .main {{ margin-left: 0; padding: 20px; }}
  }}
</style>
</head>
<body>

<!-- Sidebar nav -->
<nav class="sidebar">
  <a href="#summary" style="font-weight:700;font-size:15px;color:#fff">Stage 7 Seeds</a>
  <h3>Jump to family</h3>
  {"".join(f'<a href="#fam-{html.escape(f)}">{html.escape(f)}</a>' for f in sorted(by_family))}
</nav>

<!-- Main content -->
<div class="main">

  <h1>Stage 7 — Seed Selection Report</h1>
  <p class="subtitle">{len(seeds)} seeds across {len(by_family)} attack families &nbsp;·&nbsp;
     Generated from <code>stage7_seeds.json</code></p>

  <!-- Hero stats -->
  <div class="stats" id="summary">
    <div class="stat-box total">
      <div class="num">{len(seeds)}</div>
      <div class="lbl">Total seeds</div>
    </div>
    <div class="stat-box blue">
      <div class="num">{len(mcts_seeds)}</div>
      <div class="lbl">MCTS seeds (score 0.1–0.8)</div>
    </div>
    <div class="stat-box red">
      <div class="num">{len(anchor_seeds)}</div>
      <div class="lbl">Anchor prompts (score &gt; 0.8)</div>
    </div>
    <div class="stat-box green">
      <div class="num">{len(refused_seeds)}</div>
      <div class="lbl">Refused (score &lt; 0.1)</div>
    </div>
    <div class="stat-box total">
      <div class="num">{len(by_family)}</div>
      <div class="lbl">Attack families</div>
    </div>
  </div>

  <!-- Family summary table -->
  <h2>Family Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Attack Family</th>
        <th>Seeds</th>
        <th>MCTS</th>
        <th>Anchor</th>
        <th>Refused</th>
        <th>Avg Score</th>
      </tr>
    </thead>
    <tbody>{family_rows}</tbody>
  </table>

  <!-- Per-family seed cards -->
  <h2>All Seeds by Family</h2>
  <div class="filters">
    <button class="filter-btn active" onclick="filterCards('all')">All ({len(seeds)})</button>
    <button class="filter-btn" onclick="filterCards('mcts')">MCTS seeds ({len(mcts_seeds)})</button>
    <button class="filter-btn" onclick="filterCards('anchor')">Anchors ({len(anchor_seeds)})</button>
    <button class="filter-btn" onclick="filterCards('refused')">Refused ({len(refused_seeds)})</button>
  </div>

  {family_sections}

</div>

<script>
function filterCards(type) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.card').forEach(card => {{
    if (type === 'all' || card.classList.contains(type)) {{
      card.style.display = '';
    }} else {{
      card.style.display = 'none';
    }}
  }});
  // hide empty sections
  document.querySelectorAll('section[id^="fam-"]').forEach(sec => {{
    const visible = [...sec.querySelectorAll('.card')].some(c => c.style.display !== 'none');
    sec.style.display = visible ? '' : 'none';
  }});
}}
</script>

</body>
</html>"""

Path(OUTPUT).write_text(html_out, encoding="utf-8")
print(f"Report written to {OUTPUT}")
print(f"  {len(seeds)} total seeds | {len(mcts_seeds)} MCTS | {len(anchor_seeds)} anchors | {len(refused_seeds)} refused")
