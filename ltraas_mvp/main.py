"""
LTRaaS MVP — end-to-end demo.

Runs the full pipeline:
  customer registers target → defines a scan → orchestrator translates it →
  worker runs the (mock) engine → normalizer produces findings → reporter
  prints summary.

Each step prints what it's doing so you can follow the trace.

Run this with:  python main.py
"""

import random
from ltraas import (
    storage,
    target_registry,
    orchestrator,
    reporter,
)
from ltraas.types import ScanDefinition, new_id


def main():
    random.seed(42)

    print("[main] initializing database...")
    storage.init_db()

    print("\n[main] step 1: customer registers their LLM target")
    target = target_registry.register_target(
        name="PizzaBot Production",
        endpoint_url="https://api.pizzabot.example.com/chat",
        api_key="sk-secret-pizzabot-key-do-not-leak",
        system_type="chatbot",
        purpose=(
            "A customer-service chatbot for a pizza chain. "
            "Must never reveal the secret sauce recipe, "
            "never discuss topics other than pizza orders, "
            "and never share internal pricing rules."
        ),
    )
    print(f"  → registered target {target.id} ({target.name})")
    print(f"  → api key stored in vault under {target.auth_ref}")
    print("  → notice: target row in DB has NO api_key, only auth_ref pointer")

    print("\n[main] step 2: customer defines a scan")
    scan_def = ScanDefinition(
        id=new_id("scn"),
        target_id=target.id,
        name="weekly redteam",
        categories=[
            "prompt_injection",
            "secret_leak",
            "off_topic",
            "jailbreak",
        ],
        repetitions=5,
    )
    print(f"  → scan {scan_def.id} requests 4 categories, 5 runs each")

    print("\n[main] step 3: orchestrator runs the scan end-to-end")
    orchestrator.submit(scan_def)

    print("\n[main] step 4: reporter generates the report")
    reporter.generate(scan_def.id)

    print("[main] done. inspect outputs/ for the JSON report,")
    print("       and data/artifacts/ for the raw engine outputs.")


if __name__ == "__main__":
    main()
