# Noah AI — Intelligent Shopping & PayTo Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classification-F7931E.svg?logo=scikit-learn&logoColor=white)
![Bilingual](https://img.shields.io/badge/Languages-EN%20%2B%20DE-yellowgreen.svg)
![MCP](https://img.shields.io/badge/MCP-Offerhopper%20Protocol-8A2BE2.svg)
![Flutter](https://img.shields.io/badge/Frontend-Flutter%20Ready-02569B.svg?logo=flutter&logoColor=white)

**Noah** is an embedded conversational intelligence and action engine built for the **PayTo** mobile ecosystem. It combines calibrated route classification, deterministic action planning, live shopping route optimization via Offerhopper MCP, document-grounded retrieval, and LLM synthesis to deliver structured in-app interactions.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [Model Training & NLP Pipeline](#-model-training--nlp-pipeline)
- [API Reference](#-api-reference)
- [Offerhopper MCP Integration](#-offerhopper-mcp-integration)
- [Flutter App Integration](#-flutter-app-integration)
- [Deployment](#-deployment)
- [Testing & Verification](#-testing--verification)

---

## 🌟 Overview

Noah bridges user intent with actionable PayTo application logic and real-world grocery intelligence. Rather than acting merely as a conversational chatbot, Noah:
1. **Understands Intent**: Predicts a single *route* — `(sub_intent, planner_actions)` — with a calibrated scikit-learn classifier, then derives every other taxonomy field (domain, intent, tool, response mode, workflow, tool sequence) from that route. Below a calibrated confidence floor it does not act: the request is answered in conversation by the LLM under a constrained prompt, with any matching documentation as context, and a fixed clarifying question as the floor.
2. **Plans In-App Actions**: Translates intents into structured UI actions (e.g., barcode modals, card open triggers, offer carousels).
3. **Optimizes Grocery Basket & Routes**: Connects to the **Offerhopper MCP server** (`https://mcp.offerhopper.ai/mcp`) to split shopping lists across German supermarkets (Aldi, Lidl, Rewe, Edeka, etc.) for maximum savings and minimal travel time.
4. **Retrieves Grounded Knowledge**: Augments LLM answers with TF-IDF retrieval over an allowlist of user-facing documents (privacy policy, published FAQ). When nothing clears the relevance floor Noah says it does not have the answer rather than improvising.
5. **Emits Structured JSON**: Returns machine-readable payloads directly consumable by the Flutter frontend.

---

## 🚀 Key Features

* **Single-Route Intent Classification**: Fast CPU inference using word + character TF-IDF and a calibrated logistic-regression pipeline (`route.joblib`). Character n-grams carry typo and German-compound robustness; deriving the remaining fields from `route_registry.json` makes a self-contradictory response unrepresentable.
* **Abstention**: Requests below the calibrated confidence threshold return zero planner actions - never a guessed action. The reply is conversational: the LLM answers under a constrained general prompt that spells out what Noah can and cannot do, with any documentation that clears the retrieval floor passed as context. If the LLM is switched off (`NOAH_GENERAL_CHAT=0`) or fails, the fixed clarifying question is returned.
* **Bilingual**: English and German input, with replies in the language the user wrote in.
* **Action Planner & Tool Sequencer**: Dynamically generates execution plans (`planner_actions`, `tool_sequence`, `response_mode`).
* **Live Offerhopper MCP Integration**: Fetches real-time store splits, product prices, total savings, and interactive map URLs via JSON-RPC / Streamable HTTP.
* **Document-Grounded RAG**: TF-IDF retrieval over an explicit allowlist of *user-facing* documents (privacy policy, published FAQ) plus a relevance floor. Internal engineering documents are never indexed, so they cannot be quoted back to a user.
* **LLM provider chain**: each reply tries the primary Groq model, then a second Groq model (Groq's daily token budget is per model, so it is a separate 200K), then Gemini Flash-Lite. A step that answers 429 is skipped for its `retry-after`; if every step fails the deterministic reply is used, so a quota day-end never surfaces as an error.
* **Flutter-First Response Modes**:
  * `functional`: High-contrast barcode modal triggers with automatic screen brightness boosting.
  * `hybrid`: Product comparison tiles, store-split savings cards, and offer carousels.
  * `navigation`: Integrated directions and store locator intents.
  * `text`: Grounded conversational explanations.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User Prompt / Speech]) --> Flutter[PayTo Flutter App]
    Flutter -->|POST /api/chat| API[FastAPI Backend /chat]
    
    subgraph NLP & Planning Engine
        API --> Classifier[NLP Classifier & Entity Extractor]
        Classifier --> Planner[Action Planner & Tool Sequencer]
    end

    Planner -->|Needs MCP| MCP[Offerhopper MCP Client]
    MCP -->|plan_optimal_shopping_route| OH_Server[(Offerhopper Server\nAldi, Lidl, Rewe, Edeka...)]
    OH_Server -->|Store Split & Savings Map| MCP
    
    Planner -->|requires_rag = true| RAG[TF-IDF Retriever\nuser-facing docs only]
    RAG -->|Context Documents| LLM[LLM Response Generator\nGroq / Gemini / Ollama]

    MCP --> LLM
    Planner --> LLM

    LLM --> Formatter[Response Builder]
    Formatter -->|Structured JSON Payload| Flutter

    subgraph Dynamic Flutter UI
        Flutter --> BarcodeUI[High-Contrast Barcode Modal]
        Flutter --> CarouselUI[Offer & Store Route Cards]
        Flutter --> ChatUI[Conversational Chat Bubble]
    end
```

---

## 📂 Directory Structure

```text
Noah/
├── noah_backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── chat.py               # Main POST /api/chat router
│   │   ├── llm/
│   │   │   ├── provider.py           # Provider chain: Groq -> second Groq model -> Gemini, with cooldowns
│   │   │   ├── response_generator.py # Prompt template & context grounding
│   │   │   └── gemini.py             # Gemini adapter (REST; AI Studio or Vertex express keys)
│   │   ├── models/
│   │   │   └── noah_response.py      # Pydantic schemas & response models
│   │   ├── nlp/
│   │   │   ├── classifier.py         # Route prediction, abstention, plan expansion
│   │   │   ├── routing.py            # Shared route scoring (train + serve)
│   │   │   ├── entity_extractor.py   # Pattern & keyword entity extraction
│   │   │   ├── model_loader.py       # Route model + registry loading
│   │   │   └── train.py              # Trainer + leakage-free evaluation
│   │   ├── planner/
│   │   │   ├── action_registry.py    # Available app actions & tool maps
│   │   │   └── planner.py            # Rule & model-based action sequencer
│   │   ├── rag/
│   │   │   └── retriever.py          # TF-IDF retrieval over allowlisted documents
│   │   ├── tools/
│   │   │   └── offerhopper.py        # Offerhopper MCP client (HTTP JSON-RPC)
│   │   └── main.py                   # FastAPI initialization & health checks
│   ├── dataset/                      # Corpora, plus noah_holdout_probes.csv
│   ├── scripts/
│   │   └── build_dataset.py          # Rebuilds noah_dataset_v3.csv from the legacy export
│   ├── rag/                          # Reference PDFs & documents for RAG
│   ├── tests/
│   │   ├── test_api_chat.py          # Pytest API & integration test suite
│   │   └── test_response_contract.py # Frozen /api/chat wire contract
│   ├── trained_models/               # route.joblib, route_registry.json, evaluation.json
│   ├── Dockerfile                    # Containerization specification
│   ├── requirements.txt              # Python dependencies
│   └── verify_live.py                # Live verification & interactive test script
├── noah_flutter_implementation_plan.md # Comprehensive Flutter integration blueprint
├── pyproject.toml
└── README.md
```

---

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/schlechtswapnanil/Noah.git
cd Noah/noah_backend
```

### 2. Set Up Virtual Environment
```bash
# Using Python 3.10+
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file inside `noah_backend/` (or set environment variables):

```env
# LLM chain: primary Groq model -> second Groq model -> Gemini (app/llm/provider.py)
LLM_PROVIDER=groq                     # groq (default) or gemini: which goes first
LLM_API_KEY=gsk_...                   # Groq key (GROQ_API_KEY is also accepted)
LLM_MODEL=qwen/qwen3.8-27b            # primary Groq model
LLM_FALLBACK_MODEL=openai/gpt-oss-20b # second Groq model, own daily budget; leave empty to disable
GEMINI_API_KEY=                       # AI Studio key ("AIza...") or Vertex AI express key ("AQ...."); unset = no Gemini step
GEMINI_MODEL=gemini-2.5-flash-lite
NOAH_GENERAL_CHAT=1                   # 0: unrecognised requests get the fixed clarifying question, no LLM call

# Offerhopper MCP Server Endpoint
OFFERHOPPER_MCP_URL=https://mcp.offerhopper.ai/mcp

# Server Settings
HOST=0.0.0.0
PORT=8000
```

---

## 🧠 Model Training & NLP Pipeline

Noah predicts one label — the **route**, `(sub_intent, planner_actions)` — and derives everything else from it. In the corpus that pair determines `domain`, `intent`, `tool`, `response_mode`, the `requires_*` flags and `tool_sequence` with no ambiguity, so a single model replaces the thirteen independent heads that used to disagree with one another.

### Rebuilding the corpus
```bash
cd noah_backend
python -m scripts.build_dataset      # -> dataset/noah_dataset_v3.csv
```
The legacy export (`noah_dataset_20k_final.csv`) is 21,000 rows built from only 4,377 unique utterances, so a random row split placed 86% of the test set into training verbatim and scored 1.00. The builder de-duplicates, resolves conflicting label tuples, repairs the `single`/`single_step` taxonomy overlap, and adds German, out-of-scope negatives, cross-tool multi-intent utterances and typo/ASR noise. Every label value stays inside the legacy vocabulary — the wire contract does not move.

### Training
```bash
python -m app.nlp.train
```
Writes `trained_models/route.joblib` (pipeline + calibrated confidence threshold), `trained_models/route_registry.json` (route → all other fields) and `trained_models/evaluation.json`.

Evaluation splits by **surface template**, not by row, so a paraphrase of a training utterance cannot land in the test set. It also scores `dataset/noah_holdout_probes.csv` — hand-written utterances that never enter the corpus, half used to calibrate the confidence threshold and half held back to report generalisation. The trainer fails if any probe leaks into training.

The model that ships is refit on the whole corpus after evaluation. The split metrics describe a model that never saw 30% of the templates, and which templates those are changes whenever a row is added — a phrasing stranded in the holdout ("Zeig mir {Markt} auf der Karte") would otherwise be missing from production. The threshold stays the one calibrated on the split model; the probes are re-scored on the shipped model as `shipped_model.holdout_probes` in `evaluation.json`.

To add rows without reshuffling the corpus (which turns a few dozen new rows into a 10,000-line diff), add them to `scripts/build_dataset.py` and run `python -m scripts.append_follow_up_rows`, then retrain.

---

## 📡 API Reference

### Health Check
`GET /health`
```json
{
  "status": "ok"
}
```

### Chat & Action Endpoint
`POST /api/chat`

#### Request Body
```json
{
  "instruction": "Find the cheapest basket for milk, eggs and bread in Hamburg",
  "location": { "latitude": 52.5219, "longitude": 13.4132 }
}
```

`location` is optional and additive — a request with only `instruction` behaves exactly as before. It carries the device position (`latitude` + `longitude`, or a 5-digit `postal_code`) and is used only when the instruction does not name a place: "near me", "nearby", or no location at all. A city or postcode typed by the user always wins. Without it, "near me" falls back to a fixed default area (Munich city centre) and the reply says so.

`history` is optional and additive too — see [Conversation context](#conversation-context-history) below. It carries the previous turns so that "take me there", "add them to my list" and "which one is cheapest?" mean what the user meant.

#### Response Body

The key set is frozen and identical for every request — see `noah_backend/tests/test_response_contract.py`.

```json
{
  "instruction": "Find the cheapest basket for milk, eggs and bread in Hamburg",
  "domain": "SHOPPING",
  "intent": "FIND_CHEAPEST_BASKET",
  "sub_intent": "FIND_CHEAPEST_BASKET",
  "tool": "offerhopper_mcp",
  "response_mode": "hybrid",
  "requires_memory": false,
  "requires_rag": false,
  "requires_recommendation": true,
  "workflow_type": "single_step",
  "planner_actions": ["FIND_CHEAPEST_BASKET"],
  "planner_action_count": 1,
  "tool_sequence": ["offerhopper_mcp"],
  "entity_product": "Milk, Eggs, Bread",
  "entity_merchant": null,
  "entity_brand": null,
  "entity_category": null,
  "entity_price_min": null,
  "entity_price_max": null,
  "entity_location": "Hamburg",
  "entity_radius": null,
  "entity_loyalty_card": null,
  "offerhopperData": {
    "success": true,
    "optimized_route": {
      "stores": ["..."],
      "route_segments": [
        {
          "to_name": "Edeka",
          "to_store": { "name": "Edeka" },
          "distance_km": 1.4,
          "duration_minutes": 6,
          "estimated_cost_at_store": 5.09,
          "products_to_buy": [
            { "selected_product": "Frische Milch", "price": 1.11, "regular_price": 1.29, "discount_pct": 14, "quantity": 1 }
          ]
        }
      ],
      "estimated_total_cost": 10.44,
      "estimated_total_savings": 0.48,
      "total_distance_km": 3.2,
      "total_duration_minutes": 14
    },
    "cost_analysis": { "product_cost": 5.09, "travel_cost": 1.86, "in_store_time_cost": 3.0, "verdict": "..." },
    "total_estimated_cost": 10.44,
    "total_estimated_savings": 0.48,
    "share_url": "https://offerhopper.ai/s/OPfn_NeP",
    "ai_description": "..."
  },
  "response": "Found Vollkorn at Edeka for €1.19, Frische Milch at Edeka for €1.11, Herzstücke Eier at Edeka for €2.79. (Estimated savings: €0.48)"
}
```

**Value vocabularies.** `domain` ∈ `ACCOUNT, CHAT, KNOWLEDGE, MERCHANT, NAVIGATION, OFFER, PRODUCT, RECOMMENDATION, SHOPPING, SYSTEM, WALLET`. `response_mode` ∈ `functional, hybrid, navigation, text`. `workflow_type` ∈ `single_step, multi_step, sequential`. `planner_actions` are keys of `app/planner/action_registry.py`, and `tool_sequence[i]` is always the tool that action `i` maps to.

#### When Noah is unsure

Below the calibrated confidence threshold, or for a request outside PayTo's scope, the response keeps the same shape but carries no actions — the client should render `response` as a chat bubble and dispatch nothing:

```json
{
  "instruction": "transfer 50 euros to my brother",
  "domain": "CHAT",
  "intent": "UNKNOWN",
  "sub_intent": "UNKNOWN",
  "tool": "none",
  "response_mode": "text",
  "workflow_type": "single_step",
  "planner_actions": [],
  "planner_action_count": 0,
  "tool_sequence": [],
  "offerhopperData": null,
  "response": "I'm not sure what you need there. I can find products and offers, open your loyalty cards, or plan a cheaper shopping trip - which would you like?"
}
```

If the OfferHopper call fails, `offerhopperData` is `null` and `response` says the price service was unreachable — it never reports prices that were not returned.

#### Conversation context (`history`)

The server keeps nothing between calls. The client sends the previous turns, oldest first, newest last, built from the responses it stored:

```json
{
  "instruction": "Take me there",
  "location": { "latitude": 52.5219, "longitude": 13.4132 },
  "history": [
    {
      "instruction": "Find the cheapest basket for milk, eggs and bread in Hamburg",
      "response": "REWE on Ballindamm has the cheapest basket … https://offerhopper.ai/s/VcEZiHHo",
      "planner_actions": ["FIND_CHEAPEST_BASKET"],
      "entities": { "product": "Milk, Eggs, Bread", "location": "Hamburg" },
      "results": [
        { "name": "Weihenstephan Barista Milch 1l", "store": "REWE", "price": 0.99 },
        { "name": "REWE Beste Wahl Eier Freilandhaltung 4 Stück", "store": "REWE", "price": 1.49 },
        { "name": "Harry Kürbiskernbrot 750g", "store": "REWE", "price": 1.99 }
      ],
      "share_url": "https://offerhopper.ai/s/VcEZiHHo",
      "stores": [ { "name": "REWE", "address": "Ballindamm, 40, 20095, Hamburg", "latitude": 53.55127, "longitude": 9.99681 } ]
    }
  ]
}
```

Every field in a turn except `instruction` is optional. `entities` are the response's `entity_*` fields without the prefix; `results` / `stores` / `share_url` are a compact summary of that turn's `offerhopperData` (name, store, price; name, address, position) — the payload itself is never sent. The server keeps the last 6 turns, 10 results per turn and 2 000 characters per text field, and truncates rather than rejects; wrong types are a `422` like a malformed `location`.

`app/nlp/follow_up.py` folds the history into a context (merchant, products, typed place, loyalty card, last results — newest value wins) and decides what the new message is:

| message | resolved as | what happens |
|---|---|---|
| names a store, card, place, product or price ("Take me to the nearest Lidl") | **fresh** | classified exactly as before |
| "take me there", "wann öffnet es?", "add them to my list", "und Butter?", "show the barcode" | **rewritten** to the sentence it stands for ("Take me to REWE in Hamburg", "Setze Milch, Eier und Brot auf meine Einkaufsliste") | the rewritten sentence is classified, so `planner_actions` and the `entity_*` fields are right; `instruction` still echoes what was typed |
| "which one is cheapest?", "what's the total?", "lohnt sich die Fahrt?" | **result question** | no classifier, no OfferHopper: `domain CHAT`, `intent FOLLOW_UP`, `sub_intent RESULT_QUESTION`, `planner_actions []`, `offerhopperData null`, and `response` answers from the carried results, map link last |
| a follow-up with nothing to resolve against ("Take me there" as the first message) | **unresolved** | the empty-plan shape with a clarifying question ("which store do you mean?") instead of a navigation action with no destination |

The rewrite phrasings are the ones the route model is known to accept; the rows in `dataset/noah_dataset_v3.csv` that back them are generated by `scripts/build_dataset.py` (`follow_up_rewrite_rows`). Bare follow-ups are deliberately not in the corpus — without history they are ambiguous — and live in `dataset/noah_holdout_probes.csv` (split `follow_up`) so the trainer reports how often the model alone would have acted on one.

Set `FOLLOW_UP_LLM_REWRITE=1` to let the Groq model rewrite anaphoric messages the rules do not cover (one short sentence, temperature 0). Off by default; measure the added latency before turning it on.

---

## 🛒 Offerhopper MCP Integration

Noah implements the **Model Context Protocol (MCP)** specification connecting directly to Offerhopper:

* **Endpoint**: `https://mcp.offerhopper.ai/mcp`
* **Transport**: Streamable HTTP / JSON-RPC 2.0
* **Covered Retailers**: Aldi Süd/Nord, Lidl, Rewe, Edeka, Penny, Netto, DM, Rossmann, Müller.
* **Tool Invocation**: `plan_optimal_shopping_route(items, location)`

---

## 📱 Flutter App Integration

Noah is designed to power the Flutter conversational interface:
1. **`NoahChatProvider`**: Manages state, session history, and network calls.
2. **`NoahActionDispatcher`**: Parses `planner_actions` and `response_mode` to pop up relevant modals:
   * **Loyalty Barcodes**: Instant full-screen barcode presentation.
   * **Route & Split Cards**: Tappable store cards with interactive navigation links.
   * **Offer Carousels**: Horizontal scrollable product deal cards.

> **Client gap to close:** `NoahActionDispatcher` switches on `PLAN_ROUTE` and `OPEN_GOOGLE_MAPS`, but not on `GET_DIRECTIONS`, which is a valid `maps_tool` action in the registry. "Take me to / drive me to / navigate to X" is trained onto `PLAN_ROUTE` so the common phrasings work today, but "How do I get to Rewe?" still returns `GET_DIRECTIONS` and the app will silently do nothing. Add it to the same `case` group.

> Requests with no actions (`planner_actions: []`) are Noah asking a clarifying question or answering conversationally — render `response` and dispatch nothing.

*For complete implementation details and Dart widget architecture, refer to [`noah_flutter_implementation_plan.md`](./noah_flutter_implementation_plan.md).*

---

## 🚀 Deployment

### Render (free, currently the recommended host)

[`render.yaml`](./render.yaml) defines the `noah-backend` service (Docker, Frankfurt,
`plan: free`) and redeploys automatically on every push to `main`. Free web services
spin down after ~15 minutes idle, so the first request after a quiet spell takes
roughly 50 seconds; set the client HTTP timeout to 60s or more.

First-time setup: Render dashboard → **New** → **Blueprint** → pick this repo. Render
reads `render.yaml` and creates the service. Then set `LLM_API_KEY` under the service's
**Environment** tab (it is `sync: false` in the blueprint, so it is never committed).

**Keeping it awake.** [`.github/workflows/keep-warm.yml`](./.github/workflows/keep-warm.yml)
requests `/health` every 10 minutes from GitHub Actions, which is free for a public
repository, so the service never idles long enough to spin down. Render's free plan
allows 750 instance-hours a month; one always-on service uses about 720. Disable the
workflow under **Actions** when the demo is over. If the service moves, set the
repository variable `NOAH_BACKEND_URL` instead of editing the workflow.

### Hugging Face Spaces (needs PRO as of 2026-09)

> **Docker Spaces are no longer free.** `create_repo` returns
> `402 Payment Required`: "Static Spaces are free for everyone, but hosting Gradio
> and Docker Spaces on free cpu-basic requires a PRO subscription." A static Space
> cannot run this API. The scripts below still work on a PRO account.

```bash
# 1. Create the Space first: https://huggingface.co/new-space -> SDK: Docker -> Blank
# 2. Get a write token:      https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxx deploy/huggingface/publish.sh <your-hf-username> noah-payto-api
```

`publish.sh` assembles a Space containing only the request path — `app/`, the two
trained-model artifacts, the FAQ cache and `requirements.txt` (~9 MB). Training
corpora, the legacy `.joblib` heads and the test suite are left out. Run it with
`DRY_RUN=1` to inspect the assembled tree without pushing.

The Space serves on port 7860 as uid 1000, per
[`deploy/huggingface/Dockerfile`](./deploy/huggingface/Dockerfile). Once built:

```bash
curl -s https://<username>-noah-payto-api.hf.space/health
```

> The Space is public and CORS is open. If you set `LLM_API_KEY` as a Space
> secret, every public request spends your Groq quota — for an unattended demo,
> leaving it unset is safer. Routing and grounding are identical either way;
> only the phrasing of `response` changes.

---

## 🧪 Testing & Verification

### Run Automated Unit & Integration Tests

The suite is offline by default: `tests/conftest.py` stubs OfferHopper (which rate-limits hard) with a realistic payload and the LLM with the deterministic reply generator. `NOAH_LIVE_TESTS=1 pytest` runs against the real services.

```bash
cd noah_backend
pytest tests/ -v
```
`tests/test_response_contract.py` pins the `/api/chat` request shape, response key set, value types and label vocabularies. It must stay green across any retraining or refactor — the Flutter client dispatches on `planner_actions`, `entities.*`, `response_mode` and `offerhopperData`.

With `NOAH_LIVE_TESTS=1`, `tests/test_api_chat.py` calls the live Offerhopper MCP server and will fail with `429 Too Many Requests` if run repeatedly in quick succession.

### Run Live Interactive CLI Verification
```bash
cd noah_backend
python verify_live.py
```

### Run Development Server
```bash
cd noah_backend
uvicorn app.main:app --reload --port 8000
```

---

## 👥 Authors & Acknowledgments

* **Noah AI Team** — Built for PayTo Smart Assistant Ecosystem.
* **Offerhopper.ai** — Supermarket price data and routing MCP server.
