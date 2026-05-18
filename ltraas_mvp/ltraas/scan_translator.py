"""
Component 3: Scan Translator.

Customers think in unified vulnerability categories. Engines (promptfoo,
garak) have their own plugin/probe names. The translator maps between them.

The capability matrix is *data*, not code — in production it'd be a YAML
file you can edit without redeploying. For the MVP it's a dict.

This is THE differentiator vs. just running promptfoo directly: customers
say "test for prompt injection" and we figure out which engine + which
plugin best handles that.
"""

from .types import ScanDefinition, ScanPlan, Job, Target, new_id


CAPABILITY_MATRIX = {
    "prompt_injection": {
        "engine": "promptfoo",
        "plugin": "prompt-injection",
        "applies_to": ["chatbot", "rag", "tool_agent"],
    },
    "secret_leak": {
        "engine": "promptfoo",
        "plugin": "harmful:privacy",
        "applies_to": ["chatbot", "rag"],
    },
    "off_topic": {
        "engine": "promptfoo",
        "plugin": "excessive-agency",
        "applies_to": ["chatbot", "tool_agent"],
    },
    "jailbreak": {
        "engine": "promptfoo",
        "plugin": "jailbreak",
        "applies_to": ["chatbot", "rag", "tool_agent"],
    },
}


def reverse_lookup(engine: str, engine_probe_id: str) -> str:
    """Used by the normalizer: engine plugin name → unified category."""
    for category_id, entry in CAPABILITY_MATRIX.items():
        if entry["engine"] == engine and entry["plugin"] == engine_probe_id:
            return category_id
    return "unknown"


def build_plan(scan_def: ScanDefinition, target: Target) -> ScanPlan:
    """Turn a ScanDefinition into a list of engine-specific Jobs."""
    jobs: list[Job] = []

    for category_id in scan_def.categories:
        entry = CAPABILITY_MATRIX.get(category_id)
        if not entry:
            print(f"  [translator] unknown category '{category_id}', skipping")
            continue

        if target.system_type not in entry["applies_to"]:
            print(f"  [translator] '{category_id}' doesn't apply to "
                  f"{target.system_type}, skipping")
            continue

        config_blob = _render_engine_config(entry, target, scan_def)
        jobs.append(Job(
            id=new_id("job"),
            scan_id=scan_def.id,
            target_id=target.id,
            engine=entry["engine"],
            category_id=category_id,
            config_blob=config_blob,
            repetitions=scan_def.repetitions,
        ))

    return ScanPlan(scan_id=scan_def.id, jobs=jobs)


def _render_engine_config(entry: dict, target: Target, scan_def: ScanDefinition) -> dict:
    """Build the config the engine will actually read.

    Notice the {{TOKEN}} placeholder — secrets aren't in the config at rest.
    The worker substitutes the real value at the last moment, then revokes it.
    """
    return {
        "targets": [{
            "id": "http_target",
            "config": {
                "url": target.endpoint_url,
                "headers": {"Authorization": "Bearer {{TOKEN}}"},
            },
        }],
        "redteam": {
            "purpose": target.purpose,
            "plugins": [entry["plugin"]],
            "numTests": scan_def.repetitions,
        },
    }
