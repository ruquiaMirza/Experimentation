# `mcts.py` — Full Explanation

Think of this file as **a search engine for attack prompts**. It doesn't randomly try things — it learns which directions are promising and spends more effort there, like a chess engine that explores strong moves deeply and ignores weak ones quickly.

---

## The `MCTS` Class

The class holds everything the search needs to run: the tree of prompts, the tools to mutate and evaluate them, and a running log of every result.

### `__init__` — Setup

```python
def __init__(self, seeds, max_depth, n_mutations, exploration_constant, ...):
```

When you create an `MCTS` object, it initialises five things:

| What | What it does |
|------|-------------|
| `self.roots` | The starting tree. Either passed in from a saved run, or freshly built from seeds — one root node per attack goal. |
| `self.mutator` | The prompt rewriter (Claude Haiku). Takes a prompt and produces variations of it. |
| `self.evaluator` | The judge (promptfoo + Claude Sonnet). Sends a prompt to the target model and scores the response 0–1. |
| `self.purpose` | The target system description from config. Forwarded to both the mutator (for contextual plausibility) and the evaluator (for grading). |
| `self.c` | The exploration constant. Controls how much the algorithm tries new branches vs. exploiting known good ones. Default is `√2 ≈ 1.414`. |
| `self.history` | A flat list of every `{prompt, score, response, ...}` recorded so far. This is what gets saved to disk. |

---

## The Four MCTS Phases

Every single iteration of the search runs these four phases in sequence.

---

### Phase 1 — `_select` : Find the most promising node to work on

```python
def _select(self, root: MCTSNode) -> MCTSNode:
```

Starting from the given root, this walks **down** the tree to find the best node to focus on. It follows two rules at every level:

**Rule A — Always try unvisited nodes first.** If any child has never been evaluated, pick one randomly. This ensures every branch gets at least one chance before exploitation kicks in.

**Rule B — If all children are visited, use UCB1.** UCB1 is a scoring formula that balances two competing needs:
- **Exploit**: prefer nodes with high average scores (they've been good before)
- **Explore**: prefer nodes that haven't been visited much (they might be good)

```
UCB1 = (total value / visits)  +  C × √(ln(parent visits) / this node's visits)
       ──── exploit term ────      ─────────────── explore term ────────────────
```

The `C` constant (`self.c`) is the dial between these. Higher `C` → more exploration. Lower `C` → more exploitation.

**Special case for depth-0 goal anchor nodes:** these are never evaluated themselves (they have no prompt), so `_select` always steps past them immediately into their depth-1 seed children. If a goal node somehow has no children at all, it returns the goal node — the `run()` loop detects this and skips the iteration.

The walk stops when it reaches either:
- A **leaf node** (no children yet — a candidate for expansion), or
- A node at **`max_depth`** (can't go deeper — evaluate it directly)

---

### Phase 2 — `_expand` : Generate new child prompts via mutation

```python
def _expand(self, node: MCTSNode) -> List[MCTSNode]:
```

Takes the node selected above and asks the **Mutator (Claude Haiku)** to produce `n_mutations` variations of its prompt. Each variation applies a different mutation operator (e.g. `"persona_injection"`, `"fictional_wrapper"`).

Two guards prevent useless expansions:
- **`node.depth == 0`** — goal anchor nodes are never expanded. Their seed children were placed there by `build_goal_roots()` before the search started.
- **`node.depth >= self.max_depth`** — can't go deeper than the configured limit.

The mutator returns a list of **triples** `(operator, mutated_prompt, model)` — the `model` field records which model produced the mutation (Claude Haiku normally; Groq Llama as fallback). For each triple, the code:
1. Records the operator name in `node.applied_operators` — so this node will never get the same operator applied to it again
2. Creates a new `MCTSNode` with the mutated prompt, inheriting the same `attack_family`, `goal`, and `tags`, and storing `operator` and `mutator_model` for provenance
3. Attaches it as a child of the current node via `node.add_child(child)` — which automatically sets `child.depth = node.depth + 1` and `child.parent = node`

If expansion fails (mutator returns nothing, e.g. all operators were exhausted or Haiku refused), the caller falls back to evaluating the current node's own prompt directly.

---

### Phase 3 — `_simulate` : Evaluate the prompt against the target model

```python
def _simulate(self, node: MCTSNode) -> tuple[float, str]:
```

The simplest phase. Passes the node's prompt to the **Evaluator**, which:
1. Sends it to the configured target model to get a response
2. Scores that response with **Claude Sonnet** (the grader) using four weighted dimensions → one continuous score in [0.0, 1.0]

The evaluator returns three values: `(score, response, grader_reason)`. The result is stored in `self.history`:
```python
{
    'prompt':           node.prompt,
    'parent_prompt':    node.parent.prompt if node.parent else '',
    'attack_family':    node.attack_family,
    'goal':             node.goal,
    'depth':            node.depth,
    'mutation_operator': node.operator,      # which operator produced this prompt
    'mutator_model':    node.mutator_model,  # which model did the mutation
    'score':            score,               # 0.0 → 1.0
    'response':         response,            # what the target model actually said
    'grader_reason':    grader_reason,       # Sonnet's explanation of the score
}
```

Returns `(score, response)` to the caller.

---

### Phase 4 — `_backpropagate` : Update scores up the entire ancestry chain

```python
def _backpropagate(self, node: MCTSNode, score: float) -> None:
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.value  += score
        cur = cur.parent
```

After evaluation, the score needs to flow back up so every ancestor knows about it. Starting at the evaluated node, it walks all the way up to the root:
- `visits += 1` — this node was involved in one more trial
- `value += score` — its cumulative score increases

This is what makes UCB1 smarter over time. After backpropagation, `value / visits` is the **average score** seen across all descendants that flowed through this node. A lineage of high-scoring mutations makes its ancestor look attractive for future selection.

---

## Root Selection — `_pick_root`

```python
def _pick_root(self) -> MCTSNode:
```

Before the four phases run, the search has to pick **which attack goal** to work on this iteration.

**Phase 1 — Visit every root first:**

```python
unvisited = [r for r in self.roots if r.visits == 0]
if unvisited:
    return random.choice(unvisited)
```

As long as any root has never been visited, one is picked at random. This guarantees every attack family gets at least one iteration before the algorithm starts concentrating effort. There is no budget cap — the forced exploration phase lasts exactly as many iterations as there are roots, ensuring full coverage regardless of how many families are in the seed set.

**Phase 2 — UCB1 across roots:**

Once every root has been visited at least once, UCB1 takes over:

```
root_score = value/visits + C × √(ln(total_visits) / root.visits)
```

`total_visits` is the sum of visits across all roots — the equivalent of "parent visits" for the root-selection level. The root with the highest UCB1 score wins. Because all roots are guaranteed to have `visits ≥ 1` when UCB1 first runs, there is no division-by-zero and no special-case fallback score needed.

---

## The Main Loop — `run`

```python
def run(self, n_iterations: int = 30, interim_path: str | None = None):
```

This wires all the pieces together. For each of the `n_iterations` rounds:

```
1.  _pick_root()        → which attack goal to work on this iteration?
2.  _select(root)       → which node in that goal's subtree?
3.  skip if depth == 0  → goal node with no seeds — useless iteration, skip
4.  should_expand?      → is it a leaf AND below max_depth?
      yes → _expand()   → ask Haiku to generate child mutations
              got children → pick one randomly as the node to evaluate
              no children  → evaluate the current node directly (fallback)
5.  _simulate(node)     → send prompt to target, score with grader
6.  _backpropagate()    → push score up to all ancestors
7.  _flush_interim()    → atomically save history + tree state to disk
```

The **atomic save** (`_flush_interim`) is important for crash safety. It writes to a `.tmp` file first, then uses `os.replace()` to swap it in — so a crash mid-write never leaves a corrupt file. Both the history (evaluation results) and the full tree (node visit counts and values) are saved after every iteration. The tree file is derived from the history path by replacing `.interim.json` with `_tree.interim.json` — e.g. `mcts_results_claude.interim.json` → `mcts_results_claude_tree.interim.json`.

---

## Result Helpers

### `top_results(k)`

```python
return sorted(self.history, key=lambda r: -r['score'])[:k]
```

Sorts the full history by score descending and returns the top `k`. Simple — the history already has everything needed.

### `stats_by_family()`

```python
by_family.setdefault(r['attack_family'], []).append(r['score'])
```

Groups all results by `attack_family`, then for each family computes:
- `trials` — total evaluations for this family
- `successes` — how many scored ≥ 0.7 (the success threshold)
- `success_rate` — `successes / trials`

This is what drives the bar chart in the terminal summary.

---

## How It All Fits Together — One Iteration

```
_pick_root ──► depth-0 goal anchor  (e.g. family="dan", goal="extract credentials")
    │
_select ────► depth-1 seed node     (unvisited → picked randomly)
    │           "Pretend you are DAN..."
    │
_expand ────► depth-2 children       (Haiku produces 3 mutations)
    │           A: persona deepened
    │           B: wrapped in fiction
    │           C: reframed as authority
    │
pick random ► node C  chosen for evaluation
    │
_simulate ──► Haiku responds → Sonnet grades → score = 0.82
    │
_backpropagate:
    C.visits=1,  C.value=0.82
    seed.visits=1, seed.value=0.82
    anchor.visits=1, anchor.value=0.82

Next iteration: if other roots are still unvisited, _pick_root picks
                one of them at random (forced coverage phase).
                Once all roots have been visited, UCB1 takes over and
                will favour the dan anchor if its score stays high.
                Within the dan subtree, nodes A and B are unvisited →
                _select picks one next via the "unvisited first" rule.
```

---

## Key Design Choices Worth Understanding

**Why UCB1 and not just "pick the best score"?**
Pure exploitation would hammer one node repeatedly after the first success and never discover that a different branch might score even higher. UCB1 guarantees every node gets attention proportional to how uncertain its value is — nodes with few visits have a high explore bonus, so they get tried.

**Why backpropagate to the root?**
Because the root's UCB1 score is used by `_pick_root`. If the `dan` family produces a 0.9-scoring mutation at depth 3, all its ancestors (depth 2, depth 1, depth 0 anchor) accumulate that value. Next call to `_pick_root` sees the `dan` anchor has high `value/visits` and is more likely to be selected again — this is how the search concentrates effort on productive families.

**Why random choice among expansion children?**
When expansion produces 3 new nodes, one is picked at random for immediate evaluation rather than using UCB1. UCB1 needs at least one visit to rank nodes — all three children have zero visits at this point, so UCB1 can't differentiate them. After this iteration, the chosen child has `visits=1`; the other two are still unvisited and will be prioritised by the "unvisited first" rule in subsequent selections.
