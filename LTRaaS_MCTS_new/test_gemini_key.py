import os
import sys

import google.generativeai as genai

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")


def main():
    if not API_KEY:
        print("No API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY).", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=API_KEY)

    try:
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content("what is your name")
        print(f"Key works. Model: {MODEL}")
        print(f"Response: {resp.text.strip()}")
    except Exception as e:
        print(f"Key check failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()