---
title: Noah PayTo Assistant API
emoji: 🛒
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Intent routing and action planning for the PayTo shopping app
---

# Noah — PayTo Assistant API

Conversational intent routing and action planning for the PayTo shopping app.
Understands English and German, plans in-app actions (loyalty cards, product and
offer search, navigation, basket optimisation), and asks a clarifying question
instead of guessing when it is unsure.

Source: https://github.com/schlechtswapnanil/Noah

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/api/chat` | Route an instruction to an action plan |

## Example

```bash
curl -s https://HF_SPACE_URL/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"Zeig mir meine Payback-Karte"}'
```

```json
{
  "domain": "WALLET",
  "intent": "OPEN_CARD",
  "sub_intent": "OPEN_CARD",
  "response_mode": "functional",
  "planner_actions": ["OPEN_WALLET_CARD"],
  "tool_sequence": ["wallet_tool"],
  "entity_loyalty_card": "PAYBACK",
  "response": "I'm opening your Payback card."
}
```

The response always carries the same 24 keys. `planner_actions: []` means Noah is
answering conversationally or asking you to rephrase — the client should render
`response` and dispatch nothing.

## Configuration

| Variable | Effect |
|---|---|
| `LLM_API_KEY` | Groq key for LLM-phrased replies. **Set it as a Space secret, not a variable.** Without it, replies come from the deterministic templates — routing and grounding are identical either way. |
| `LLM_MODEL` | Defaults to `qwen/qwen3.6-27b`. |

> This Space is public and CORS is open, so anyone can call `/api/chat`. If you
> set `LLM_API_KEY`, every public request spends your Groq quota. For an
> unattended demo, leaving it unset is the safer choice.
