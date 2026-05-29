# LTRaaS MVP

A minimal, runnable Python implementation of the LLM Red Teaming as a Service platform from the design doc. ~600 lines total, no external dependencies (uses only the Python standard library).

## What this is

A working skeleton that demonstrates the full pipeline:

```
customer → API → translator → orchestrator → worker(mock engine) → normalizer → reporter
```

It uses a `MockEngine` that fakes promptfoo's output so the whole thing runs on a laptop with zero setup. Swap in real promptfoo by editing one class.

## Run it

```bash
cd ltraas_mvp
python main.py
```

You'll see a trace of every step, then a printed report and a JSON file in `outputs/`.

## File map (which component is which)

Each file corresponds to one component from the HLD diagram:

| File | HLD component | What it does |
|------|---------------|--------------|
| `ltraas/types.py` | Section 0 (contracts) | The dataclasses every other module uses |
| `ltraas/storage.py` | Storage backend | SQLite + JSON files + in-memory vault |
| `ltraas/target_registry.py` | Component 2 | Register LLM targets, vault API keys |
| `ltraas/scan_translator.py` | Component 3 | Map unified categories → engine configs |
| `ltraas/orchestrator.py` | Component 4 | Run jobs in sequence (sync MVP) |
| `ltraas/worker.py` | Component 5 | Run the engine, includes `MockEngine` |
| `ltraas/normalizer.py` | Component 6 | Raw output → unified `Finding` rows |
| `ltraas/reporter.py` | Component 7 | Build the executive + engineer reports |
| `main.py` | End-to-end driver | Wires it all together |

## What's deliberately simplified vs. production

| Production | MVP |
|------------|-----|
| Postgres + S3 + Vault | SQLite + local files + dict |
| Async queue (Celery/Temporal) | Sync for-loop in orchestrator |
| Containerized workers | Function calls |
| Real promptfoo + garak | `MockEngine` returning fake data |
| Embedding-based deduper | Hash of first-5-words |
| HTTPS + RBAC + audit logs | None |
| Trend analysis across scans | Single-scan only |

Each of these can be added later without changing the contracts in `types.py` — that's the design's promise paying off.

## Where to go next

Pick one and the system grows in that direction:

1. **Real promptfoo.** Implement `PromptfooEngine.run()` in `worker.py`. Install promptfoo via npm, shell out to it, parse the JSON.
2. **Real LLM target.** Make `MockEngine` actually call an LLM API (OpenAI, Anthropic) and have a judge LLM grade responses.
3. **Async orchestration.** Replace the for-loop in `orchestrator.py` with a queue (start with `concurrent.futures`, graduate to Celery).
4. **A second engine.** Add a `GarakEngine` class. The translator picks the right engine per category. The normalizer parses both.
5. **An HTTP API.** Wrap `target_registry`, `orchestrator`, `reporter` in FastAPI endpoints. Add auth.
6. **A web dashboard.** Read from the findings table, render charts.

Each addition is shippable on its own. The architecture's spine — the types in `types.py` and the component boundaries — does not change.
