import json
import os

import requests


def _parse_prompt(prompt) -> tuple[str, str]:
    """Return (purpose_text, user_text) from whatever shape evaluator.py passes.

    The target prompt template is a JSON object:
      {"system": "<purpose>", "user": "<adversarial prompt>"}

    `purpose` is only used if `system_field` is set in config — most deployed
    apps have a fixed persona server-side and can't take a system prompt
    override per-request, same as attacking a real production system.
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


def _extract(data, path: str):
    """Walk a dotted path (e.g. "choices.0.message.content") into a parsed
    JSON response and return the value found there."""
    value = data
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def call_api(prompt, options, context):
    """Generic HTTP target provider — points at any locally- or remotely-
    hosted app's chat endpoint, entirely configured via config.yaml's
    `target_config`:

    target_config:
      url: http://127.0.0.1:5050/chat   # required
      method: POST                       # optional, default POST
      request_field: message             # JSON field to put the prompt in
      response_field: reply              # dotted path to the reply text
      system_field: null                 # JSON field for the system/purpose text, if the app accepts one
      extra_fields: {}                   # extra static fields merged into the request body
      headers: {}                        # extra request headers
      timeout: 30                        # seconds
    """
    config = options.get("config", {}) if isinstance(options, dict) else {}
    url = config.get("url")
    if not url:
        return {"error": "target_config.url is not set in config.yaml"}

    method = (config.get("method") or "POST").upper()
    request_field = config.get("request_field", "message")
    response_field = config.get("response_field", "reply")
    system_field = config.get("system_field")
    extra_fields = config.get("extra_fields") or {}
    headers = config.get("headers") or {}
    timeout = config.get("timeout", 30)

    purpose_text, user_text = _parse_prompt(prompt)
    body = {request_field: user_text, **extra_fields}
    if system_field and purpose_text:
        body[system_field] = purpose_text

    try:
        if method == "GET":
            resp = requests.get(url, params=body, headers=headers, timeout=timeout)
        else:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        output = _extract(data, response_field)
    except Exception as e:
        return {"error": str(e)}

    return {"output": output}
