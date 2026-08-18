# Noah

### AI-Powered Conversational Assistant for PayTo

Noah is the AI intelligence layer behind **PayTo**, designed to understand natural-language requests, extract structured intent and entities, plan multi-step actions, retrieve information when necessary, and interact with PayTo's application services.

Instead of forcing users to navigate through multiple screens, Noah allows them to simply **tell PayTo what they want to do**.

> **"Find ice cream offers near me under €3 and navigate me to the cheapest one."**

Noah turns this into a structured plan that PayTo can execute.

---

## ✨ What Noah Does

Noah combines supervised NLP, entity extraction, planning, RAG and an LLM response layer.

```text
                    ┌──────────────────┐
                    │   User Request   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  NLP / SVM Layer │
                    │ Intent + Entities│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Noah Planner    │
                    │ Multi-step Tasks │
                    └────────┬─────────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
          PayTo APIs       RAG         App Tools
               │             │             │
               └─────────────┼─────────────┘
                             ▼
                    ┌──────────────────┐
                    │ Open-weight LLM  │
                    │ Response Layer   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Natural Response │
                    └──────────────────┘
```

---

# 🧠 Core Capabilities

### Natural-language understanding

Noah understands requests such as:

* "Find coffee near me."
* "Show me something under €5."
* "Does Netto have any ice cream offers?"
* "Take me to the nearest Lidl."

The NLP layer extracts the relevant intent, sub-intent and entities.

---

### Entity extraction

Noah identifies information such as:

* Products
* Merchants
* Brands
* Categories
* Prices
* Locations
* Search radius
* Loyalty programmes/cards

For example:

```json
{
  "entity_product": "ice cream",
  "entity_merchant": "Netto",
  "entity_price_max": 3,
  "entity_location": "CURRENT_LOCATION"
}
```

---

# 🔗 Multi-step Planning

Noah is designed to understand that one instruction can contain several actions.

For example:

> "I'm at Netto. Find ice cream offers under €3 and then show my Netto Plus card."

Noah can produce an ordered plan:

```json
{
  "workflow_type": "sequential",
  "planner_actions": [
    "SEARCH_OFFERS",
    "FILTER_PRICE",
    "DISPLAY_LOYALTY_CARD"
  ],
  "planner_action_count": 3,
  "tool_sequence": [
    "offers_tool",
    "offers_tool",
    "wallet_tool"
  ]
}
```

### Important

The field is always:

```text
planner_actions
```

**not** `planner_action`.

The order of actions is significant.

---

# 💳 Loyalty Cards

Noah explicitly distinguishes merchants from loyalty programmes.

Examples:

| User says       | Interpretation    |
| --------------- | ----------------- |
| Netto           | Merchant          |
| Lidl            | Merchant          |
| REWE            | Merchant          |
| Netto Plus      | Loyalty Card      |
| Lidl Plus       | Loyalty Card      |
| REWE Bonus      | Loyalty Card      |
| EDEKA Card      | Loyalty Card      |
| PAYBACK         | Loyalty Programme |
| DeutschlandCard | Loyalty Programme |

For example:

> "Show my Netto Plus card."

should not be interpreted simply as:

```text
entity_merchant = Netto
```

Instead, Noah understands the loyalty-card context.

---

# 📄 RAG

Noah can retrieve information from documents stored in:

```text
rag/
```

This is used when a request requires information that should come from the provided documentation rather than being generated from general model knowledge.

The intended flow is:

```text
User
 ↓
Noah
 ↓
requires_rag = true
 ↓
Retrieve relevant PDF information
 ↓
LLM
 ↓
Grounded response
```

Noah should not invent information when the required information is unavailable in the RAG sources.

---

# 🤖 LLM Response Layer

The LLM is primarily responsible for the **conversational response**, not for replacing the structured classifier.

The architecture is:

```text
Instruction
     ↓
Classifier
     ↓
Entities
     ↓
Planner
     ↓
Structured Noah JSON
     ↓
LLM
     ↓
Natural-language response
```

This separation allows Noah to retain predictable structured behaviour while still providing natural conversational responses.

The LLM provider/model is configurable and should be supplied through environment variables rather than hard-coded credentials.

---

# 💬 Conversational Context

Noah is designed to support contextual conversations.

For example:

**User**

> Find coffee near me.

**Noah**

> I found several coffee shops nearby.

**User**

> Which one is cheapest?

Noah should understand that *"which one"* refers to the previously retrieved coffee options.

Likewise:

> Find Netto near me.

> Which one is closest?

> Navigate me there.

These requests form a related conversational workflow rather than three unrelated queries.

---

# 📦 Structured Output

Noah's structured output follows the project's defined schema:

```json
{
  "instruction": "...",

  "domain": "...",
  "intent": "...",
  "sub_intent": "...",

  "tool": "...",
  "response_mode": "...",

  "entity_product": null,
  "entity_merchant": null,
  "entity_brand": null,
  "entity_category": null,
  "entity_price_min": null,
  "entity_price_max": null,
  "entity_location": null,
  "entity_radius": null,

  "requires_memory": false,
  "requires_rag": false,
  "requires_recommendation": false,

  "workflow_type": "single",

  "planner_actions": [],
  "planner_action_count": 0,

  "tool_sequence": [],

  "response": "..."
}
```

The structured fields are used by PayTo to determine what should happen next.

---

# 🏗️ Architecture

A simplified view of Noah:

```text
                         NOAH
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
       NLP/SVM        Entity Layer      Context
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                       Planner
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
          PayTo API      RAG         Tools
             │            │            │
             └────────────┼────────────┘
                          ▼
                         LLM
                          │
                          ▼
                    User Response
```

---

# 📊 Training Dataset

Noah's supervised NLP models are trained using the project's labelled dataset.

The dataset contains:

```text
instruction
domain
intent
sub_intent
tool
response_mode

entity_product
entity_merchant
entity_brand
entity_category
entity_price_min
entity_price_max
entity_location
entity_radius

requires_memory
requires_rag
requires_recommendation

workflow_type
planner_actions
planner_action_count
tool_sequence
```

The `instruction` is the natural-language input.

The remaining fields provide the structured labels used to train and evaluate the system.

---

# 🧪 Model Training

The current NLP approach uses classical supervised machine learning, including TF-IDF-based text representations and SVM-style classifiers.

This keeps the structured understanding layer:

* Fast
* Lightweight
* Inexpensive
* Controllable
* Easy to retrain

The LLM is used as a conversational layer rather than as the sole source of truth.

---

# 📁 Project Structure

The exact structure may evolve, but Noah is broadly organised around:

```text
noah/
│
├── datasets/
│   └── noah_dataset_20k_final.csv
│
├── models/
│
├── rag/
│   └── *.pdf
│
├── training/
│
├── noah/
│   ├── classifier
│   ├── entity extraction
│   ├── planner
│   ├── RAG
│   └── LLM integration
│
├── api/
│
├── tests/
│
├── requirements.txt
└── README.md
```

Use the actual project structure when adding new components rather than duplicating existing functionality.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd noah
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file for API credentials/configuration.

Example:

```env
LLM_API_KEY=your_api_key_here
LLM_MODEL=your_selected_model
```

**Never commit `.env` or API keys to GitHub.**

Add:

```text
.env
```

to `.gitignore`.

---

# 🚀 Running Noah

The exact command depends on the current API entry point.

For a FastAPI deployment, for example:

```bash
uvicorn main:app --reload
```

The API can then be consumed by the PayTo application.

---

# 🔌 PayTo Integration

Noah is designed to operate as the AI/orchestration layer for PayTo.

The PayTo application communicates with Noah through its API.

Conceptually:

```text
PayTo App
    │
    │ User instruction
    ▼
Noah API
    │
    ├── Understand
    ├── Extract
    ├── Plan
    ├── Retrieve
    └── Respond
    │
    ▼
PayTo App
```

Noah should not duplicate PayTo's core business logic.

Instead, Noah determines **what the user wants and what actions need to happen**, while PayTo executes the relevant application functionality.

---

# 🛣️ Roadmap

### Current

* [x] Intent classification
* [x] Sub-intent classification
* [x] Entity extraction
* [x] Multi-step planner representation
* [x] Structured JSON output
* [x] Loyalty-card handling
* [x] RAG architecture
* [x] LLM response layer

### Next

* [ ] Improve classification accuracy
* [ ] Expand paraphrase dataset
* [ ] Improve multi-step planning
* [ ] Improve contextual conversation handling
* [ ] Better tool execution feedback
* [ ] Production-grade RAG retrieval
* [ ] Continuous evaluation
* [ ] Latency optimisation
* [ ] Production deployment

---

# 🎯 Vision

Noah's goal is simple:

> **Let users talk to PayTo instead of learning how to use PayTo.**

Whether the user wants to find a product, compare offers, locate a merchant, use a loyalty card, get recommendations, navigate somewhere or perform several actions together, Noah should understand the request and orchestrate the appropriate PayTo functionality.

**PayTo is the application.
Noah is the intelligence layer.**
