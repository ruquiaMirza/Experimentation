"""
LTRaaS MVP — End-to-End Pipeline Demo
======================================

This script runs the full red-teaming pipeline in sequence:

  1. Load and validate configuration from scan_config.yaml
  2. Register the LLM target (the model being tested)
  3. Define a scan (what attack categories to run)
  4. Run the scan via the orchestrator (translate → attack → judge → normalize)
  5. Generate a human-readable report of findings

Quick Start
-----------
  MockEngine (no API keys needed — good for local testing):
      python main.py

  LLMEngine (makes real HTTP calls to your target + a judge LLM):
      export GEMINI_API_KEY=AIza...   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
      python main.py

  PromptfooEngine (delegates to the promptfoo CLI tool):
      export GEMINI_API_KEY=AIza...
      python main.py

Configuration is read from scan_config.yaml in the project root.
Environment variables always take priority over values in the YAML file.
No defaults are applied — every required field must be set explicitly.
"""

import logging
import os
import random
import yaml
from pathlib import Path

from ltraas import storage, target_registry, orchestrator, reporter
from ltraas.log import setup_logging
from ltraas.scan_translator import build_target_url
from ltraas.types import ScanDefinition, new_id


# ── Logging ───────────────────────────────────────────────────────────────────

setup_logging()
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "scan_config.yaml"

VALID_ENGINES    = {"mock", "llm", "promptfoo"}
# "custom" lets users specify the exact auth header, request body, and response
# parsing for endpoints whose internal LLM and API format are unknown or non-standard.
VALID_PROVIDERS  = {"google", "openai", "anthropic", "custom"}


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_config(cfg: dict) -> None:
    """
    Check every required field in the config and raise a single ValueError
    that lists ALL missing or invalid fields at once.

    Nothing is silently defaulted — if a field is required and absent, it
    is reported here rather than silently substituted with a guess.
    """
    errors: list[str] = []

    # ── Engine ────────────────────────────────────────────────────────────────
    engine = (cfg.get("engine") or "").strip().lower()
    if not engine:
        errors.append("engine: required — choose one of: mock | llm | promptfoo")
    elif engine not in VALID_ENGINES:
        errors.append(
            f"engine: '{engine}' is not valid — choose one of: mock | llm | promptfoo"
        )

    # ── Target model ──────────────────────────────────────────────────────────
    target_model = cfg.get("models", {}).get("target", {})
    provider     = (target_model.get("provider") or "").strip().lower()

    # provider is always required.
    if not provider:
        errors.append(
            "models.target.provider: required — "
            "choose one of: google | openai | anthropic | custom"
        )
    elif provider not in VALID_PROVIDERS:
        errors.append(
            f"models.target.provider: '{provider}' is not valid — "
            "choose one of: google | openai | anthropic | custom"
        )

    # Known providers: require model — the endpoint URL is auto-built from provider + model.
    if provider in ("google", "openai", "anthropic"):
        if not target_model.get("model"):
            errors.append(
                f"models.target.model: required for provider '{provider}'"
            )

    # Custom provider: require url, request_body, and transform_response.
    # model is not needed — the request body template is explicit.
    if provider == "custom":
        if not target_model.get("url"):
            errors.append("models.target.url: required for provider 'custom'")
        if not target_model.get("request_body"):
            errors.append(
                "models.target.request_body: required for provider 'custom' — "
                "full JSON body template with {{prompt}} as the placeholder "
                "(e.g. '{\"message\": \"{{prompt}}\"}')"
            )
        if not target_model.get("transform_response"):
            errors.append(
                "models.target.transform_response: required for provider 'custom' — "
                "JS expression to extract the reply text from the JSON response "
                "(e.g. 'json.reply' or 'json.choices[0].message.content')"
            )
        # auth_header and auth_prefix are optional.

    if not target_model.get("api_key_env"):
        errors.append(
            "models.target.api_key_env: required — "
            "name of the env var that holds the API key (e.g. GEMINI_API_KEY)"
        )

    # ── API key (not required for mock engine) ────────────────────────────────
    if engine and engine != "mock":
        key_env = target_model.get("api_key_env", "")
        has_key = bool(
            (key_env and os.environ.get(key_env)) or
            target_model.get("api_key")
        )
        if not has_key:
            errors.append(
                f"API key: set the {key_env} environment variable "
                "or set models.target.api_key in scan_config.yaml"
            )

    # ── Judge (required for llm and promptfoo) ────────────────────────────────
    if engine in ("llm", "promptfoo"):
        judge = cfg.get("models", {}).get("judge", {})
        if not judge.get("provider"):
            errors.append(
                "models.judge.provider: required for llm and promptfoo engines"
            )
        if not judge.get("model"):
            errors.append(
                "models.judge.model: required for llm and promptfoo engines"
            )

    # ── Attack generator (required for promptfoo) ──────────────────────────────
    if engine == "promptfoo":
        atk = cfg.get("models", {}).get("attack_generator", {})
        if not atk.get("provider"):
            errors.append(
                "models.attack_generator.provider: required for promptfoo engine"
            )
        if not atk.get("model"):
            errors.append(
                "models.attack_generator.model: required for promptfoo engine"
            )

    # ── Target metadata ────────────────────────────────────────────────────────
    target_cfg = cfg.get("target", {})
    if not target_cfg.get("name"):
        errors.append("target.name: required")
    if not target_cfg.get("system_type"):
        errors.append("target.system_type: required (chatbot | rag | tool_agent)")
    if not target_cfg.get("purpose"):
        errors.append("target.purpose: required — describe what the target does")

    # ── Scan ──────────────────────────────────────────────────────────────────
    scan_cfg = cfg.get("scan", {})
    if not scan_cfg.get("name"):
        errors.append("scan.name: required")
    if not scan_cfg.get("categories"):
        errors.append("scan.categories: required — provide at least one category to test")

    # Engine-specific scan parameters.
    if engine in ("mock", "llm"):
        if scan_cfg.get("llm_repetitions") is None:
            errors.append("scan.llm_repetitions: required for mock and llm engines")
    if engine == "promptfoo":
        if scan_cfg.get("promptfoo_num_tests") is None:
            errors.append("scan.promptfoo_num_tests: required for promptfoo engine")
        if scan_cfg.get("promptfoo_concurrency") is None:
            errors.append("scan.promptfoo_concurrency: required for promptfoo engine")

    if errors:
        raise ValueError(
            "scan_config.yaml has missing or invalid fields:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


# ── Config Resolution ─────────────────────────────────────────────────────────


def load_config() -> dict:
    """
    Load scan_config.yaml, validate every required field, resolve the target
    URL and API key, then export keys to environment variables.

    Raises ValueError listing every missing or invalid field if anything
    is absent. Nothing is silently defaulted.
    """
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    # Validate first — report all problems at once before doing anything else.
    _validate_config(cfg)

    target_model = cfg["models"]["target"]

    # For known providers, build the endpoint URL from provider + model.
    # For custom provider, the URL is already set explicitly in the config.
    if target_model["provider"] != "custom":
        target_model["url"] = build_target_url(target_model["provider"], target_model["model"])
        logger.debug("Built target URL: %s", target_model["url"])

    return cfg


# ── Engine Selection ───────────────────────────────────────────────────────────

def build_engine(cfg: dict):
    """
    Create and return the scan engine specified in scan_config.yaml.

    Supported engines:
      - "mock"      → No real HTTP calls; useful for testing locally.
      - "llm"       → Calls the target and a judge LLM directly via HTTP.
      - "promptfoo" → Delegates scanning to the external promptfoo CLI tool.

    Returns None for the mock engine (the orchestrator handles that case).
    Validation already ensured engine is one of the three valid values.
    """
    engine_name  = cfg["engine"].strip().lower()
    judge_config = cfg["models"].get("judge", {})  # only present for llm/promptfoo

    if engine_name == "mock":
        logger.info("Engine: MockEngine — no real HTTP calls will be made")
        return None

    if engine_name == "llm":
        from ltraas.worker import LLMEngine
        logger.info(
            "Engine: LLMEngine (judge=%s:%s)",
            judge_config["provider"],
            judge_config["model"],
        )
        return LLMEngine(
            judge_provider=judge_config["provider"],
            judge_model=judge_config["model"],
        )

    if engine_name == "promptfoo":
        from ltraas.worker import PromptfooEngine
        logger.info("Engine: PromptfooEngine — will shell out to the promptfoo CLI")
        return PromptfooEngine()


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def main():
    """Run the full LTRaaS pipeline from config load to report generation."""

    # Fixed seed so mock results are reproducible across runs.
    random.seed(42)

    # ── Step 0: Load and validate configuration ───────────────────────────────
    logger.info("Loading config from %s", CONFIG_PATH)
    cfg          = load_config()
    models       = cfg["models"]
    target_model = models["target"]
    target_cfg   = cfg["target"]
    scan_cfg     = cfg["scan"]
    engine_type  = cfg["engine"].strip().lower()

    logger.info(
        "Target: %s:%s",
        target_model["provider"],
        target_model.get("model") or "(from url)",
    )
    # Judge and attack_generator are only required for llm and promptfoo engines;
    # the mock engine doesn't need them so they may be absent from the config.
    if engine_type in ("llm", "promptfoo"):
        logger.info(
            "Judge: %s:%s | Attack gen: %s:%s",
            models["judge"]["provider"],            models["judge"]["model"],
            models["attack_generator"]["provider"], models["attack_generator"]["model"],
        )

    # ── Step 0b: Initialise the database ─────────────────────────────────────
    logger.info("Initializing database")
    storage.init_db()

    # ── Step 0c: Build the engine ─────────────────────────────────────────────
    engine = build_engine(cfg)

    # ── Step 1: Register the target LLM ──────────────────────────────────────
    logger.info("Step 1: Registering LLM target")
    api_key = os.environ.get(target_model["api_key_env"]) or target_model.get("api_key", "")
    target = target_registry.register_target(
        name=target_cfg["name"],
        endpoint_url=target_model["url"],
        api_key=api_key,
        system_type=target_cfg["system_type"],
        purpose=target_cfg["purpose"],
    )
    logger.info("Registered target %s (%s) → %s", target.id, target.name, target_model["url"])
    logger.info("API key stored in vault under reference: %s", target.auth_ref)

    # ── Step 2: Define the scan ───────────────────────────────────────────────
    logger.info("Step 2: Defining scan")
    scan = ScanDefinition(
        id=new_id("scn"),
        target_id=target.id,
        name=scan_cfg["name"],
        categories=scan_cfg["categories"],
        # Fields not used by the chosen engine are left as 0 — validation
        # already ensured the relevant ones are explicitly set in the config.
        llm_repetitions=scan_cfg.get("llm_repetitions", 0),
        promptfoo_num_tests=scan_cfg.get("promptfoo_num_tests", 0),
        promptfoo_concurrency=scan_cfg.get("promptfoo_concurrency", 0),
    )
    logger.info(
        "Scan %s — categories=%s  llm_repetitions=%d  promptfoo_num_tests=%d",
        scan.id, scan.categories, scan.llm_repetitions, scan.promptfoo_num_tests,
    )

    # ── Step 3: Run the scan ──────────────────────────────────────────────────
    logger.info("Step 3: Running scan via orchestrator")
    orchestrator.submit(scan, engine_impl=engine, models=models, engine_type=engine_type)

    # ── Step 4: Generate the report ───────────────────────────────────────────
    logger.info("Step 4: Generating report")
    reporter.generate(scan.id)

    logger.info(
        "Done! Check outputs/ for the JSON report "
        "and data/artifacts/ for raw engine outputs."
    )


if __name__ == "__main__":
    main()
