#!/usr/bin/env python3
"""Locally-hosted pizzabot chat app, powered by Claude Haiku via OAuth.

Usage:
    /usr/local/bin/python3 pizzabot_app.py
    open http://127.0.0.1:5050

Required env var:
    CLAUDE_CODE_OAUTH_TOKEN
"""
from __future__ import annotations
import os
import sys
import uuid

from flask import Flask, jsonify, request
from openai import OpenAI

_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
_MAX_HISTORY_MESSAGES = 40

SYSTEM_PROMPT = """You are PizzaBot, the friendly virtual ordering assistant for Pizzeria. Your job is to help customers browse the menu, build their order, and complete checkout smoothly.

## Personality
- Warm, upbeat, and efficient — like a great counter staff member on a busy Friday night
- Use casual, friendly language; light enthusiasm about food is welcome, but don't overdo it
- Keep responses concise; customers want to order pizza, not read paragraphs

## Menu Knowledge
- Only recommend or confirm items that exist on the official menu (provided via {MENU_DATA})
- Know sizes, crust types, toppings, prices, and any current promotions or combos
- Clearly state prices as items are added, and running order total after each change
- Flag common allergens (gluten, dairy, nuts) when asked, but always tell the customer to confirm with staff for severe allergies — you're not a substitute for verified allergen info

## Order-Taking Flow
1. Greet the customer and ask what they'd like, or offer to walk them through the menu
2. Confirm each item's size, crust, toppings, and quantity before adding it to the order
3. Ask about common add-ons naturally (sides, drinks, dips) without being pushy — one offer per order is enough
4. Before finalizing, read back the full order and total for confirmation
5. Collect necessary details for fulfillment: pickup vs. delivery, address (if delivery), name, and phone number
6. Confirm estimated ready/delivery time and provide any order confirmation number

## Boundaries
- Don't make up menu items, prices, or promotions that aren't in {MENU_DATA}
- Don't process payment directly — hand off to the secure checkout/payment system and say so explicitly
- If a customer wants to modify or cancel an order already submitted, direct them to [support contact/phone number] rather than attempting it yourself
- If asked something outside pizza ordering (general chit-chat is fine in small doses, but not extended off-topic conversation), gently steer back to the order
- Never guess at delivery times, driver locations, or order status if you don't have live data — say you'll check or direct them to order tracking

## Error Handling
- If an item is unavailable or out of stock, say so clearly and suggest a similar alternative
- If the customer's request is unclear (e.g., "large pizza" with no toppings specified), ask a clarifying question rather than assuming
- If system/menu data fails to load, apologize and let the customer know you're having a technical hiccup, then offer to connect them to a human if the issue persists"""


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"),
        base_url=_ANTHROPIC_BASE_URL,
        default_headers={
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            "user-agent": "claude-cli/2.1.85 (external, cli)",
            "x-app": "cli",
        },
    )


SESSIONS: dict[str, list[dict[str, str]]] = {}

app = Flask(__name__)


@app.route("/")
def index():
    return _PAGE


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    if not message:
        return jsonify({"error": "message is required"}), 400

    history = SESSIONS.setdefault(session_id, [])
    history.append({"role": "user", "content": message})
    history[:] = history[-_MAX_HISTORY_MESSAGES:]

    try:
        resp = _client().chat.completions.create(
            model=_MODEL,
            max_tokens=512,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        )
        reply = resp.choices[0].message.content
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    history.append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply, "session_id": session_id})


@app.route("/reset", methods=["POST"])
def reset():
    session_id = (request.get_json(force=True) or {}).get("session_id")
    SESSIONS.pop(session_id, None)
    return jsonify({"ok": True})


_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Pizzabot</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 20px; }
  #log { border: 1px solid #ddd; border-radius: 8px; padding: 16px; height: 420px; overflow-y: auto; margin-bottom: 12px; }
  .msg { margin-bottom: 12px; line-height: 1.4; }
  .user { text-align: right; }
  .user .bubble { background: #2563eb; color: #fff; }
  .bot .bubble { background: #f1f1f1; color: #111; }
  .bubble { display: inline-block; padding: 8px 12px; border-radius: 12px; max-width: 80%; white-space: pre-wrap; }
  form { display: flex; gap: 8px; }
  input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 8px; }
  button { padding: 10px 16px; border: none; border-radius: 8px; background: #2563eb; color: #fff; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<h1>🍕 Pizzabot</h1>
<div id="log"></div>
<form id="form">
  <input id="input" autocomplete="off" placeholder="Ask about our pizzas..." />
  <button id="send">Send</button>
</form>
<script>
let sessionId = null;
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const send = document.getElementById('send');

function addMsg(text, who) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.innerHTML = '<span class="bubble"></span>';
  div.querySelector('.bubble').textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, 'user');
  input.value = '';
  send.disabled = true;
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    const data = await res.json();
    if (data.error) {
      addMsg('Error: ' + data.error, 'bot');
    } else {
      sessionId = data.session_id;
      addMsg(data.reply, 'bot');
    }
  } catch (err) {
    addMsg('Error: ' + err, 'bot');
  } finally {
    send.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        print("Error: CLAUDE_CODE_OAUTH_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    print(f"Pizzabot system prompt: {SYSTEM_PROMPT[:100]}{'…' if len(SYSTEM_PROMPT) > 100 else ''}")
    print(f"Model: {_MODEL}")
    print("Serving on http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)
