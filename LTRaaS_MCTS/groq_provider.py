import json
import os

import yaml
from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Read purpose from config.yaml once at import time so it is available as
# a fallback system prompt when the JSON envelope doesn't carry one.
def _load_purpose() -> str:
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f).get("purpose", "").strip()
    except Exception:
        return ""

_PURPOSE = _load_purpose()


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    return Groq(api_key=api_key)


def _parse_prompt(prompt) -> tuple[str, str]:
    """Return (system_text, user_text).

    Accepts the same JSON-envelope format as claude_provider.py:
      {"system": "<purpose>", "user": "<attack prompt>"}
    Falls back to (BASE_SYSTEM, raw_text) for plain strings.
    """
    if isinstance(prompt, list):
        raw = " ".join(
            p.get("text", str(p)) if isinstance(p, dict) else str(p)
            for p in prompt
        )
    elif isinstance(prompt, dict):
        raw = prompt.get("text") or prompt.get("content") or str(prompt)
    else:
        raw = str(prompt)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "user" in parsed:
            return parsed.get("system", ""), parsed["user"]
    except (json.JSONDecodeError, TypeError):
        pass

    return "", raw


def call_api(prompt, options, context):
    config = options.get("config", {}) if isinstance(options, dict) else {}
    model  = config.get("model") or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL

    system_text, user_text = _parse_prompt(prompt)
    system = system_text or _PURPOSE

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})

    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        content = resp.choices[0].message.content or ""
        return {"output": content}
    except Exception as e:
        import sys
        print(f"[groq_provider] {type(e).__name__}: {e}", file=sys.stderr)
        return {"error": str(e)}
