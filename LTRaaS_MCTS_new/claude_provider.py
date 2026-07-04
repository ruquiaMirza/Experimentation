import json
import os
from openai import OpenAI

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
BASE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"


def _client():
    return OpenAI(
        api_key=os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"),
        base_url=ANTHROPIC_BASE_URL,
        default_headers={
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            "user-agent": "claude-cli/2.1.85 (external, cli)",
            "x-app": "cli",
        },
    )


def _parse_prompt(prompt) -> tuple[str, str]:
    """Return (purpose_text, user_text) from whatever shape promptfoo passes.

    The target prompt template is a JSON object:
      {"system": "<purpose>", "user": "<adversarial prompt>"}

    The grader prompt is a plain string — falls back to ("", raw).
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
    model = config.get("model") or os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL

    purpose_text, user_text = _parse_prompt(prompt)

    system = purpose_text if purpose_text else BASE_SYSTEM

    try:
        response = _client().chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
        )
        return {"output": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
