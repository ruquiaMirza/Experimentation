#!/usr/bin/env python3
"""Standalone test: call Claude Haiku via the OpenAI-compatible chat completions
endpoint, authenticating with the Claude Code CLI OAuth token instead of a
standard ANTHROPIC_API_KEY.

Anthropic exposes an OpenAI SDK-compatible endpoint at /v1/chat/completions.
It's built for API-key auth, so this script exists to check whether the
CLI's OAuth bearer token + beta headers combo is accepted there too.

Usage:
    python3 call_haiku_chat_completions.py "your prompt here"

Required env var:
    CLAUDE_CODE_OAUTH_TOKEN
"""
from __future__ import annotations
import os
import sys

from openai import OpenAI

MODEL = "claude-haiku-4-5-20251001"
BASE_URL = "https://api.anthropic.com/v1/"


def main() -> None:
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        print("Error: CLAUDE_CODE_OAUTH_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one short sentence."

    client = OpenAI(
        api_key=token,
        base_url=BASE_URL,
        default_headers={
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            "user-agent": "claude-cli/2.1.85 (external, cli)",
            "x-app": "cli",
        },
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
