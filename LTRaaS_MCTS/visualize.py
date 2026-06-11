from __future__ import annotations
import json
from datetime import datetime

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MCTS Tree Visualization</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111827; color: #e5e7eb; font-family: 'Segoe UI', sans-serif; overflow: hidden; }

    #header {
      position: fixed; top: 0; left: 0; right: 0; z-index: 10;
      background: #1f2937; border-bottom: 1px solid #374151;
      padding: 10px 18px; display: flex; align-items: center; gap: 16px;
    }
    #header h1 { font-size: 15px; font-weight: 600; color: #f9fafb; }
    #header .hint { font-size: 12px; color: #9ca3af; }

    #legend {
      display: flex; gap: 14px; margin-left: auto; align-items: center; font-size: 12px;
    }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .legend-dot { width: 12px; height: 12px; border-radius: 50%; }

    svg { width: 100vw; height: 100vh; margin-top: 0; }

    .link { fill: none; stroke: #374151; stroke-width: 1.5px; }

    .node circle {
      cursor: pointer;
      stroke-width: 1.8px;
      transition: filter 0.15s;
    }
    .node circle:hover { filter: brightness(1.3); }
    .node text { font-size: 11px; fill: #d1d5db; pointer-events: none; }
    .node.has-children circle { stroke: #60a5fa; }
    .node.leaf circle { stroke: #4b5563; }

    #tooltip {
      position: fixed;
      background: #1f2937;
      border: 1px solid #374151;
      color: #e5e7eb;
      padding: 12px 15px;
      border-radius: 8px;
      font-size: 12px;
      max-width: 420px;
      pointer-events: none;
      display: none;
      line-height: 1.6;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
      z-index: 100;
    }
    #tooltip .tt-family { font-weight: 600; color: #60a5fa; font-size: 13px; }
    #tooltip .tt-goal { color: #9ca3af; font-size: 11px; margin-bottom: 6px; }
    #tooltip .tt-stats { display: flex; gap: 16px; margin-bottom: 8px; }
    #tooltip .tt-stat label { color: #6b7280; font-size: 10px; display: block; }
    #tooltip .tt-stat span { font-weight: 600; color: #f9fafb; }
    #tooltip .tt-divider { border: none; border-top: 1px solid #374151; margin: 8px 0; }
    #tooltip .tt-prompt { color: #9ca3af; font-size: 10px; margin-bottom: 3px; }
    #tooltip .tt-prompt-text { color: #d1d5db; font-size: 11px; word-break: break-word; }

    #stats {
      position: fixed; bottom: 16px; left: 16px; z-index: 10;
      background: #1f2937; border: 1px solid #374151;
      border-radius: 8px; padding: 10px 14px; font-size: 12px;
      line-height: 1.8;
    }
    #stats .stat-label { color: #6b7280; }
    #stats .stat-val { color: #f9fafb; font-weight: 600; }
  </style>
</head>
<body>

<div id="header">
  <h1>MCTS Tree</h1>
  <span class="hint">Click a node to expand/collapse &nbsp;|&nbsp; Scroll to zoom &nbsp;|&nbsp; Drag to pan</span>
  <div id="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#6b7280"></div> Unvisited</div>
    <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div> Visited / No success</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div> Partial success</div>
    <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div> Full success</div>
  </div>
</div>

<div id="tooltip"></div>
<div id="stats"></div>
<svg id="tree-svg"><g id="tree-g"></g></svg>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const RAW_DATA = __TREE_DATA__;

// ── helpers ──────────────────────────────────────────────────────────────────

function nodeColor(d) {
  if (d.visits === 0) return '#6b7280';
  if (d.win_rate === 0) return '#3b82f6';
  // 0→1 mapped through amber→green
  return d3.interpolateRdYlGn(0.4 + d.win_rate * 0.6);
}

function nodeRadius(d) {
  if (d.depth < 0) return 14;            // virtual root
  if (d.visits === 0) return 6;
  return Math.max(6, Math.min(16, 6 + d.visits * 2));
}

function truncate(str, n) {
  if (!str) return '—';
  return str.length <= n ? str : str.slice(0, n) + '…';
}

// ── layout ───────────────────────────────────────────────────────────────────

const NODE_SEP  = 26;   // vertical gap between sibling nodes
const LEVEL_SEP = 340;  // horizontal gap between depth levels

const svg = d3.select('#tree-svg');
const g   = d3.select('#tree-g');

const zoom = d3.zoom()
  .scaleExtent([0.05, 4])
  .on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);

const tree     = d3.tree().nodeSize([NODE_SEP, LEVEL_SEP]);
const diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x);

let idSeq = 0;
const root = d3.hierarchy(RAW_DATA);
root.x0 = 0;
root.y0 = 0;

root.descendants().forEach(d => {
  d.id = idSeq++;
  d._children = d.children;
  // start collapsed: show virtual root + its seed children only
  if (d.depth >= 1) d.children = null;
});

// ── tooltip ───────────────────────────────────────────────────────────────────

const tt = document.getElementById('tooltip');

function showTip(event, d) {
  const nd = d.data;
  const wr = nd.visits > 0 ? (nd.win_rate * 100).toFixed(0) + '%' : '—';
  tt.innerHTML = `
    <div class="tt-family">${nd.attack_family || 'root'}</div>
    <div class="tt-goal">${truncate(nd.goal, 100)}</div>
    <div class="tt-stats">
      <div class="tt-stat"><label>depth</label><span>${nd.depth}</span></div>
      <div class="tt-stat"><label>visits</label><span>${nd.visits}</span></div>
      <div class="tt-stat"><label>win rate</label><span>${wr}</span></div>
      <div class="tt-stat"><label>value</label><span>${nd.value}</span></div>
    </div>
    <hr class="tt-divider">
    <div class="tt-prompt">prompt</div>
    <div class="tt-prompt-text">${truncate(nd.prompt, 300)}</div>
  `;
  const tipW = 440, tipH = 200;
  const left = event.clientX + 14 + tipW > window.innerWidth
    ? event.clientX - tipW - 8
    : event.clientX + 14;
  const top = Math.min(event.clientY - 10, window.innerHeight - tipH - 10);
  tt.style.left = left + 'px';
  tt.style.top  = top  + 'px';
  tt.style.display = 'block';
}

function moveTip(event) {
  const tipW = 440, tipH = 200;
  const left = event.clientX + 14 + tipW > window.innerWidth
    ? event.clientX - tipW - 8
    : event.clientX + 14;
  const top = Math.min(event.clientY - 10, window.innerHeight - tipH - 10);
  tt.style.left = left + 'px';
  tt.style.top  = top  + 'px';
}

function hideTip() { tt.style.display = 'none'; }

// ── update ────────────────────────────────────────────────────────────────────

function update(source) {
  const dur = 300;
  tree(root);

  const nodes = root.descendants().reverse();
  const links = root.links();

  // links
  const link = g.selectAll('path.link').data(links, d => d.target.id);

  const linkEnter = link.enter().insert('path', 'g')
    .attr('class', 'link')
    .attr('d', () => {
      const o = { x: source.x0 ?? 0, y: source.y0 ?? 0 };
      return diagonal({ source: o, target: o });
    });

  link.merge(linkEnter).transition().duration(dur).attr('d', diagonal);

  link.exit().transition().duration(dur)
    .attr('d', () => {
      const o = { x: source.x, y: source.y };
      return diagonal({ source: o, target: o });
    })
    .remove();

  // nodes
  const node = g.selectAll('g.node').data(nodes, d => d.id);

  const nodeEnter = node.enter().append('g')
    .attr('class', d => 'node ' + (d._children ? 'has-children' : 'leaf'))
    .attr('transform', () => `translate(${source.y0 ?? 0},${source.x0 ?? 0})`)
    .attr('opacity', 0)
    .on('click', (event, d) => {
      d.children = d.children ? null : d._children;
      update(d);
    })
    .on('mouseover', showTip)
    .on('mousemove', moveTip)
    .on('mouseout', hideTip);

  nodeEnter.append('circle')
    .attr('r', d => nodeRadius(d.data))
    .attr('fill', d => nodeColor(d.data));

  nodeEnter.append('text')
    .attr('dy', '0.32em')
    .attr('x', d => (d._children || d.children) ? -(nodeRadius(d.data) + 5) : (nodeRadius(d.data) + 5))
    .attr('text-anchor', d => (d._children || d.children) ? 'end' : 'start')
    .text(d => {
      if (d.data.depth < 0) return 'MCTS Forest';
      const fam = d.data.attack_family;
      const wr  = d.data.visits > 0 ? ` · ${(d.data.win_rate * 100).toFixed(0)}%` : '';
      const v   = d.data.visits > 0 ? ` (${d.data.visits}v)` : '';
      return `${fam}${wr}${v}`;
    });

  const nodeMerge = node.merge(nodeEnter);

  nodeMerge.transition().duration(dur)
    .attr('class', d => 'node ' + (d._children ? 'has-children' : 'leaf'))
    .attr('transform', d => `translate(${d.y},${d.x})`)
    .attr('opacity', 1);

  nodeMerge.select('circle')
    .attr('r', d => nodeRadius(d.data))
    .attr('fill', d => nodeColor(d.data));

  nodeMerge.select('text')
    .attr('x', d => (d._children || d.children) ? -(nodeRadius(d.data) + 5) : (nodeRadius(d.data) + 5))
    .attr('text-anchor', d => (d._children || d.children) ? 'end' : 'start');

  node.exit().transition().duration(dur)
    .attr('transform', () => `translate(${source.y},${source.x})`)
    .attr('opacity', 0)
    .remove();

  root.each(d => { d.x0 = d.x; d.y0 = d.y; });
}

// ── stats panel ───────────────────────────────────────────────────────────────

function renderStats() {
  const all  = root.descendants().filter(d => d.data.depth >= 0);
  const vis  = all.filter(d => d.data.visits > 0);
  const succ = all.filter(d => d.data.win_rate >= 0.7);
  document.getElementById('stats').innerHTML = `
    <div><span class="stat-label">Total nodes: </span><span class="stat-val">${all.length}</span></div>
    <div><span class="stat-label">Visited:     </span><span class="stat-val">${vis.length}</span></div>
    <div><span class="stat-label">Successes:   </span><span class="stat-val">${succ.length}</span></div>
  `;
}

// ── init ──────────────────────────────────────────────────────────────────────

update(root);
renderStats();

// Center the virtual root node in the viewport
const initT = d3.zoomIdentity.translate(180, window.innerHeight / 2);
svg.call(zoom.transform, initT);
</script>
</body>
</html>
"""


_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MCTS Red-Team Report</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111827; color: #e5e7eb; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; line-height: 1.5; }

    a { color: #60a5fa; }
    hr { border: none; border-top: 1px solid #1f2937; margin: 0; }

    /* ── layout ── */
    header { background: #1f2937; border-bottom: 1px solid #374151; padding: 18px 28px; display: flex; align-items: baseline; gap: 14px; }
    header h1 { font-size: 18px; font-weight: 700; color: #f9fafb; }
    header .meta { font-size: 12px; color: #6b7280; }
    main { max-width: 1400px; margin: 0 auto; padding: 24px 28px; display: flex; flex-direction: column; gap: 28px; }

    /* ── cards ── */
    .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    .card { background: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 18px 20px; }
    .card .label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }
    .card .value { font-size: 28px; font-weight: 700; color: #f9fafb; }
    .card .value.green { color: #22c55e; }
    .card .value.red   { color: #ef4444; }
    .card .value.amber { color: #f59e0b; }

    /* ── section ── */
    section { background: #1f2937; border: 1px solid #374151; border-radius: 10px; overflow: hidden; }
    section .sec-header { padding: 14px 20px; border-bottom: 1px solid #374151; display: flex; align-items: center; justify-content: space-between; }
    section .sec-header h2 { font-size: 14px; font-weight: 600; color: #f9fafb; }
    section .sec-body { padding: 20px; }

    /* ── family chart ── */
    .family-rows { display: flex; flex-direction: column; gap: 10px; }
    .family-row { display: grid; grid-template-columns: 160px 1fr 90px; align-items: center; gap: 12px; }
    .family-row .fam-name { font-size: 12px; color: #d1d5db; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bar-track { background: #111827; border-radius: 4px; height: 16px; overflow: hidden; }
    .bar-fill  { height: 100%; border-radius: 4px; transition: width .4s ease; }
    .family-row .fam-stat { font-size: 11px; color: #9ca3af; }

    /* ── filters ── */
    .filters { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; padding: 14px 20px; border-bottom: 1px solid #374151; }
    .filters input[type=search] {
      background: #111827; border: 1px solid #374151; border-radius: 6px;
      color: #e5e7eb; padding: 6px 10px; font-size: 13px; outline: none; width: 220px;
    }
    .filters input[type=search]:focus { border-color: #60a5fa; }
    .filters select {
      background: #111827; border: 1px solid #374151; border-radius: 6px;
      color: #e5e7eb; padding: 6px 10px; font-size: 13px; outline: none; cursor: pointer;
    }
    .filters select:focus { border-color: #60a5fa; }
    .filter-btns { display: flex; gap: 6px; }
    .filter-btn {
      background: #111827; border: 1px solid #374151; border-radius: 6px;
      color: #9ca3af; padding: 5px 12px; font-size: 12px; cursor: pointer;
    }
    .filter-btn.active { background: #1d4ed8; border-color: #3b82f6; color: #fff; }
    .filters .count-badge { font-size: 12px; color: #6b7280; margin-left: auto; }

    /* ── table ── */
    .tbl-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    thead th {
      padding: 10px 14px; text-align: left; font-size: 11px; font-weight: 600;
      color: #6b7280; text-transform: uppercase; letter-spacing: .05em;
      background: #1f2937; border-bottom: 1px solid #374151;
      white-space: nowrap; cursor: pointer; user-select: none;
    }
    thead th:hover { color: #d1d5db; }
    thead th .sort-arrow { margin-left: 4px; opacity: 0.4; }
    thead th.sorted .sort-arrow { opacity: 1; color: #60a5fa; }

    tbody tr { border-bottom: 1px solid #1f2937; cursor: pointer; transition: background .1s; }
    tbody tr:hover { background: #1a2336; }
    tbody tr.expanded { background: #1a2336; }
    tbody td { padding: 10px 14px; vertical-align: top; color: #d1d5db; }

    .score-badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 600; }
    .score-badge.pass { background: #14532d; color: #4ade80; }
    .score-badge.fail { background: #1f2937; color: #6b7280; border: 1px solid #374151; }

    .fam-badge { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 11px; background: #1e3a5f; color: #93c5fd; }
    .depth-badge { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 11px; background: #1c1917; color: #a8a29e; border: 1px solid #292524; }

    .cell-prompt { max-width: 320px; }
    .cell-response { max-width: 360px; }
    .text-clip { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #9ca3af; font-size: 12px; }

    /* ── expanded detail panel ── */
    .detail-row td { padding: 0; }
    .detail-panel { padding: 16px 24px 20px; background: #0f172a; border-top: 1px solid #1e3a5f; display: none; }
    .detail-panel.open { display: block; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .detail-box h4 { font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 8px; }
    .detail-box pre { background: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 12px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 12px; color: #d1d5db; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; }

    /* ── empty state ── */
    .empty { text-align: center; padding: 40px; color: #4b5563; }

    footer { text-align: center; font-size: 11px; color: #4b5563; padding: 20px; }
  </style>
</head>
<body>

<header>
  <h1>MCTS Red-Team Report</h1>
  <div class="meta">Generated __TIMESTAMP__</div>
</header>

<main>

  <!-- summary cards -->
  <div class="cards">
    <div class="card">
      <div class="label">Total Evaluations</div>
      <div class="value" id="c-total">—</div>
    </div>
    <div class="card">
      <div class="label">Successful Attacks</div>
      <div class="value green" id="c-success">—</div>
    </div>
    <div class="card">
      <div class="label">Overall Success Rate</div>
      <div class="value amber" id="c-rate">—</div>
    </div>
    <div class="card">
      <div class="label">Families Tested</div>
      <div class="value" id="c-families">—</div>
    </div>
  </div>

  <!-- family chart -->
  <section>
    <div class="sec-header"><h2>Success Rate by Attack Family</h2></div>
    <div class="sec-body">
      <div class="family-rows" id="family-chart"></div>
    </div>
  </section>

  <!-- results table -->
  <section>
    <div class="filters">
      <input type="search" id="search" placeholder="Search prompt or goal…">
      <select id="fam-filter"><option value="">All families</option></select>
      <div class="filter-btns">
        <button class="filter-btn active" data-score="all">All</button>
        <button class="filter-btn" data-score="pass">Success</button>
        <button class="filter-btn" data-score="fail">Failure</button>
      </div>
      <select id="depth-filter"><option value="">All depths</option></select>
      <span class="count-badge" id="row-count"></span>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th data-col="score">Score <span class="sort-arrow">↕</span></th>
            <th data-col="attack_family">Family <span class="sort-arrow">↕</span></th>
            <th data-col="goal">Goal <span class="sort-arrow">↕</span></th>
            <th data-col="depth">Depth <span class="sort-arrow">↕</span></th>
            <th>Prompt</th>
            <th>Response</th>
          </tr>
        </thead>
        <tbody id="tbl-body"></tbody>
      </table>
      <div class="empty" id="empty-msg" style="display:none">No results match the current filters.</div>
    </div>
  </section>

</main>

<footer>MCTS Adversarial Prompt Search · mcts_results.json</footer>

<script>
const HISTORY = __HISTORY_JSON__;
const STATS   = __STATS_JSON__;

// ── summary cards ──────────────────────────────────────────────────────────
const total    = HISTORY.length;
const successes = HISTORY.filter(r => r.score >= 0.7).length;
const rate     = total ? (successes / total * 100).toFixed(1) : 0;
const families = Object.keys(STATS).length;

document.getElementById('c-total').textContent    = total;
document.getElementById('c-success').textContent  = successes;
document.getElementById('c-rate').textContent     = rate + '%';
document.getElementById('c-families').textContent = families;

// ── family chart ───────────────────────────────────────────────────────────
function rateColor(r) {
  // 0 → red, 0.5 → amber, 1 → green  via HSL
  const h = Math.round(r * 120);
  return `hsl(${h},70%,45%)`;
}

const sorted = Object.entries(STATS).sort((a, b) => b[1].success_rate - a[1].success_rate);
const chartEl = document.getElementById('family-chart');
sorted.forEach(([fam, s]) => {
  const pct = (s.success_rate * 100).toFixed(0);
  const row = document.createElement('div');
  row.className = 'family-row';
  row.innerHTML = `
    <div class="fam-name" title="${fam}">${fam}</div>
    <div class="bar-track">
      <div class="bar-fill" style="width:${s.success_rate*100}%;background:${rateColor(s.success_rate)}"></div>
    </div>
    <div class="fam-stat">${pct}% &nbsp;<span style="color:#4b5563">(${s.successes}/${s.trials})</span></div>
  `;
  chartEl.appendChild(row);
});

// ── family & depth filter dropdowns ───────────────────────────────────────
const famSel = document.getElementById('fam-filter');
sorted.forEach(([fam]) => {
  const o = document.createElement('option');
  o.value = fam; o.textContent = fam;
  famSel.appendChild(o);
});

const depths = [...new Set(HISTORY.map(r => r.depth))].sort((a,b)=>a-b);
const depthSel = document.getElementById('depth-filter');
depths.forEach(d => {
  const o = document.createElement('option');
  o.value = d; o.textContent = `Depth ${d}`;
  depthSel.appendChild(o);
});

// ── table state ───────────────────────────────────────────────────────────
let sortCol = 'score', sortDir = -1;   // -1 = desc
let scoreFilter = 'all';
let expandedIdx = null;

function applyFilters() {
  const q     = document.getElementById('search').value.toLowerCase();
  const fam   = famSel.value;
  const depth = depthSel.value;

  return HISTORY.map((r, i) => ({ ...r, _i: i })).filter(r => {
    if (scoreFilter === 'pass' && r.score < 0.7)  return false;
    if (scoreFilter === 'fail' && r.score >= 0.7) return false;
    if (fam   && r.attack_family !== fam) return false;
    if (depth !== '' && r.depth !== +depth) return false;
    if (q && !r.prompt.toLowerCase().includes(q) && !r.goal.toLowerCase().includes(q)) return false;
    return true;
  });
}

function sortData(data) {
  return [...data].sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va < vb) return sortDir;
    if (va > vb) return -sortDir;
    return 0;
  });
}

function trunc(s, n) { return s && s.length > n ? s.slice(0, n) + '…' : (s || '—'); }

function render() {
  const filtered = applyFilters();
  const sorted   = sortData(filtered);

  document.getElementById('row-count').textContent = `${sorted.length} of ${total}`;
  document.getElementById('empty-msg').style.display = sorted.length ? 'none' : 'block';

  const tbody = document.getElementById('tbl-body');
  tbody.innerHTML = '';

  sorted.forEach((r, visIdx) => {
    const pass = r.score >= 0.7;

    // main row
    const tr = document.createElement('tr');
    tr.dataset.idx = r._i;
    tr.className = expandedIdx === r._i ? 'expanded' : '';
    tr.innerHTML = `
      <td><span class="score-badge ${pass ? 'pass' : 'fail'}">${pass ? '✓ Pass' : '✗ Fail'}</span></td>
      <td><span class="fam-badge">${r.attack_family}</span></td>
      <td style="color:#9ca3af;font-size:12px;max-width:220px">${trunc(r.goal, 80)}</td>
      <td><span class="depth-badge">${r.depth}</span></td>
      <td class="cell-prompt"><div class="text-clip" style="max-width:300px">${trunc(r.prompt, 120)}</div></td>
      <td class="cell-response"><div class="text-clip" style="max-width:340px">${trunc(r.response, 120)}</div></td>
    `;
    tr.addEventListener('click', () => toggleDetail(r._i, detailTr));
    tbody.appendChild(tr);

    // detail row (hidden until expanded)
    const detailTr = document.createElement('tr');
    detailTr.className = 'detail-row';
    detailTr.innerHTML = `<td colspan="6">
      <div class="detail-panel ${expandedIdx === r._i ? 'open' : ''}" id="detail-${r._i}">
        <div class="detail-grid">
          <div class="detail-box">
            <h4>Prompt</h4>
            <pre>${escHtml(r.prompt)}</pre>
          </div>
          <div class="detail-box">
            <h4>Response</h4>
            <pre>${escHtml(r.response || '(no response recorded)')}</pre>
          </div>
        </div>
      </div>
    </td>`;
    tbody.appendChild(detailTr);
  });

  // update sort arrows
  document.querySelectorAll('thead th').forEach(th => {
    th.classList.toggle('sorted', th.dataset.col === sortCol);
    const arrow = th.querySelector('.sort-arrow');
    if (arrow && th.dataset.col === sortCol) arrow.textContent = sortDir === -1 ? '↓' : '↑';
    else if (arrow) arrow.textContent = '↕';
  });
}

function toggleDetail(idx, detailTr) {
  if (expandedIdx === idx) {
    expandedIdx = null;
  } else {
    expandedIdx = idx;
  }
  render();
}

function escHtml(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── event wiring ──────────────────────────────────────────────────────────
document.getElementById('search').addEventListener('input', render);
famSel.addEventListener('change', render);
depthSel.addEventListener('change', render);

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    scoreFilter = btn.dataset.score;
    render();
  });
});

document.querySelectorAll('thead th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    if (sortCol === th.dataset.col) sortDir *= -1;
    else { sortCol = th.dataset.col; sortDir = -1; }
    render();
  });
});

render();
</script>
</body>
</html>
"""


def generate_results_html(history: list, stats: dict, out_path: str) -> None:
    """Write a self-contained HTML report for MCTS evaluation results."""
    history_json = json.dumps(history, ensure_ascii=False).replace('</', '<\\/')
    stats_json   = json.dumps(stats,   ensure_ascii=False).replace('</', '<\\/')
    timestamp    = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = (
        _REPORT_TEMPLATE
        .replace('__HISTORY_JSON__', history_json)
        .replace('__STATS_JSON__',   stats_json)
        .replace('__TIMESTAMP__',    timestamp)
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)


def generate_tree_html(roots_data: list, out_path: str) -> None:
    """Write a self-contained D3 collapsible tree HTML file for the MCTS forest."""
    total_visits = sum(r.get('visits', 0) for r in roots_data)
    total_value  = round(sum(r.get('value',  0) for r in roots_data), 3)
    forest_root = {
        'name': 'MCTS Forest',
        'prompt': '',
        'goal': '',
        'attack_family': 'root',
        'depth': -1,
        'visits': total_visits,
        'value': total_value,
        'win_rate': round(total_value / total_visits, 3) if total_visits else 0.0,
        'children': roots_data,
    }

    # Embed JSON safely — escape </script> to prevent early tag close
    json_str = json.dumps(forest_root, ensure_ascii=False).replace('</', '<\\/')

    html = _HTML_TEMPLATE.replace('__TREE_DATA__', json_str)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
