"""
providers.py — Screening model abstraction.

Swap the target model for stage 2b by changing SCREEN_PROVIDER / SCREEN_MODEL
in config.py. No changes to stage2b_screen.py are needed.

Supported providers
───────────────────
  anthropic         Uses the anthropic SDK. Auth via CLAUDE_CODE_OAUTH_TOKEN
                    (preferred) or ANTHROPIC_API_KEY.

  openai_compatible Any OpenAI-spec endpoint. Requires the openai package.
                    Set SCREEN_BASE_URL and SCREEN_API_KEY_ENV in config.py.

                    Examples (config.py values):
                      Gemini  SCREEN_MODEL="gemini-2.0-flash"
                              SCREEN_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
                              SCREEN_API_KEY_ENV="GEMINI_API_KEY"

                      Ollama  SCREEN_MODEL="llama3.2"
                              SCREEN_BASE_URL="http://localhost:11434/v1"
                              SCREEN_API_KEY_ENV=""   (no key needed)

                      OpenAI  SCREEN_MODEL="gpt-4o-mini"
                              SCREEN_BASE_URL=None
                              SCREEN_API_KEY_ENV="OPENAI_API_KEY"

                      Groq    SCREEN_MODEL="llama-3.3-70b-versatile"
                              SCREEN_BASE_URL="https://api.groq.com/openai/v1"
                              SCREEN_API_KEY_ENV="GROQ_API_KEY"
"""

import os

import config


class AnthropicScreener:
    def __init__(self, model: str) -> None:
        import anthropic

        oauth   = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if oauth:
            self._client = anthropic.Anthropic(
                auth_token=oauth,
                max_retries=6,
                default_headers={
                    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                    "user-agent":     "claude-cli/2.1.85 (external, cli)",
                    "x-app":          "cli",
                },
            )
            print("  [providers] auth=CLAUDE_CODE_OAUTH_TOKEN")
        elif api_key:
            self._client = anthropic.Anthropic(api_key=api_key, max_retries=6)
            print("  [providers] auth=ANTHROPIC_API_KEY")
        else:
            raise RuntimeError(
                "No credentials found. Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY."
            )
        self._model = model

    def complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


class OpenAICompatibleScreener:
    """Works with any OpenAI-compatible endpoint: Gemini, Ollama, Groq, OpenAI, etc."""

    def __init__(self, model: str, base_url: str | None, api_key_env: str) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package required for openai_compatible provider: pip install openai"
            ) from exc

        api_key = (os.getenv(api_key_env) if api_key_env else None) or "no-key"
        self._client  = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model   = model
        endpoint_hint = base_url or "OpenAI default"
        print(f"  [providers] openai_compatible  model={model}  endpoint={endpoint_hint}")

    def complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


def get_screener() -> AnthropicScreener | OpenAICompatibleScreener:
    """Build and return the screening client configured in config.py."""
    print(f"  [providers] provider={config.SCREEN_PROVIDER}  model={config.SCREEN_MODEL}")

    if config.SCREEN_PROVIDER == "anthropic":
        return AnthropicScreener(config.SCREEN_MODEL)

    if config.SCREEN_PROVIDER == "openai_compatible":
        return OpenAICompatibleScreener(
            config.SCREEN_MODEL,
            config.SCREEN_BASE_URL,
            config.SCREEN_API_KEY_ENV,
        )

    raise ValueError(
        f"Unknown SCREEN_PROVIDER {config.SCREEN_PROVIDER!r}. "
        "Valid values: 'anthropic', 'openai_compatible'."
    )
