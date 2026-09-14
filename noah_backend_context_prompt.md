# Make the Noah backend context-aware

You are working in the Noah backend (`noah_backend/`, FastAPI, deployed to
Render from this repo at `https://noah-z7qr.onrender.com`). Add conversation
context to `POST /api/chat` so that follow-ups like "take me there", "add them
to my list" and "which one is cheapest?" work, without breaking the frozen wire
contract in `tests/test_response_contract.py` or the Flutter client that
already talks to this service.

Work in the order below and verify each step with `pytest` and `curl` against a
local `uvicorn` before moving on. Make sensible decisions and note them in your
summary; ask only if something below is contradictory.

---

## 1. Why this is needed

Today every call is independent. `app/api/chat.py` classifies the *whole*
`instruction` with the sklearn route model (`app/nlp/classifier.py`, calibrated
confidence floor ≈ 0.4), extracts entities by regex from the same text
(`app/nlp/entity_extractor.py`), plans, calls OfferHopper, and only then hands
the instruction to the LLM generator (`app/llm/response_generator.py`).

Two things were measured against the live deployment, and both need fixing
server-side:

- **Replaying the chat inside `instruction` misroutes.** With the previous
  exchange folded in, "Show my Payback card" came back as
  `FIND_CHEAPEST_BASKET` with `entity_product = "Milk, Eggs, Bread"`, and
  "Take me there" re-ran the basket search instead of navigating. The model
  sees the historical words and follows them.
- **Appending a short context note drops confidence below the floor.**
  "Take me there (REWE, Hamburg)" and "Show my Payback card\n(Context: …)"
  both returned the `CHAT` / `UNKNOWN` clarifying question.

What *does* route correctly is the self-contained sentence the user meant.
These rewrites were checked one by one against the deployed classifier:

| follow-up (after a basket search in Hamburg at REWE) | send instead | routes to |
|---|---|---|
| take me there / navigate there / directions | `Take me to REWE in Hamburg` (no city: `Take me to the nearest REWE`) | `PLAN_ROUTE`, merchant REWE, location Hamburg |
| bring mich dorthin | `Navigiere mich zu REWE in Hamburg` (no city: `Bring mich zum nächsten REWE`) | `PLAN_ROUTE` |
| when does it open? | `When does REWE in Hamburg open?` | `GET_OPENING_HOURS` |
| wann öffnet es? | `Wann hat REWE in Hamburg geöffnet?` | `GET_OPENING_HOURS` |
| add them to my shopping list | `Add milk, eggs and bread to my shopping list` | `BUILD_SHOPPING_LIST`, product "Milk, Eggs, Bread" |
| setz sie auf meine Liste | `Setze Milch, Eier und Brot auf meine Einkaufsliste` | `BUILD_SHOPPING_LIST` |
| what about butter? (after a basket search) | `Find the cheapest basket for butter in Hamburg` | `FIND_CHEAPEST_BASKET`, product Butter |
| und Eis? (after a "< 2 €" search) | `Wo gibt es Eis unter 2 Euro in meiner Nähe?` | `SEARCH_PRODUCT_BY_PRICE`, max 2 |
| what about milk? (after a plain product search) | `Find the cheapest milk near me` | `FIND_CHEAPEST_BASKET` |
| show the barcode (after a wallet turn) | `Show my Payback card` | `OPEN_WALLET_CARD` |

Phrasings that do **not** route today and must not be produced (or must be
added to the training data, see §5): `Bring mich zu REWE in Hamburg`,
`Wie komme ich zu REWE?`, `Öffnungszeiten von REWE`, `Wann öffnet REWE?`,
`How far is REWE?`, `Where can I buy butter in Hamburg?`,
`Find butter offers in Hamburg`, `Search for butter near me`.

The Flutter app currently does exactly these rewrites on the client
(`lib/noah/services/noah_follow_up_resolver.dart` in the PayTo app). Moving
them here makes every client context-aware and lets result questions be
answered without re-running OfferHopper.

---

## 2. Request contract: add `history`, additive only

```json
{
  "instruction": "Take me there",
  "location": { "latitude": 52.5219, "longitude": 13.4132 },
  "history": [
    {
      "instruction": "Find the cheapest basket for milk, eggs and bread in Hamburg",
      "response": "REWE on Ballindamm has the cheapest basket … https://offerhopper.ai/s/VcEZiHHo",
      "planner_actions": ["FIND_CHEAPEST_BASKET"],
      "entities": {
        "product": "Milk, Eggs, Bread",
        "merchant": null,
        "brand": null,
        "category": null,
        "price_min": null,
        "price_max": null,
        "location": "Hamburg",
        "radius": null,
        "loyalty_card": null
      },
      "results": [
        { "name": "Weihenstephan Barista Milch 1l", "store": "REWE", "price": 0.99 },
        { "name": "REWE Beste Wahl Eier Freilandhaltung 4 Stück", "store": "REWE", "price": 1.49 },
        { "name": "Harry Kürbiskernbrot 750g", "store": "REWE", "price": 1.99 }
      ],
      "share_url": "https://offerhopper.ai/s/VcEZiHHo",
      "stores": [
        { "name": "REWE", "address": "Ballindamm, 40, 20095, Hamburg",
          "latitude": 53.55127, "longitude": 9.99681 }
      ]
    }
  ]
}
```

Rules:

- `history` is optional. A request without it must behave exactly as today
  (mirror `test_device_location_is_optional_and_additive`).
- Oldest first, newest last. Accept up to 6 turns; ignore older ones. Cap each
  `results` list at 10 entries and each text field at 2 000 characters; reject
  nothing, just truncate.
- Every field inside a turn is optional except `instruction`. `entities` uses
  the same keys as the response's `entity_*` fields, without the prefix,
  because the client copies them straight from the previous response.
- `results` / `stores` / `share_url` are a compact summary of that turn's
  `offerhopperData` (name, store, price; name, address, lat/lng). Never accept
  or echo the full OfferHopper payload.
- The **response key set stays exactly the 24 keys.** `instruction` keeps
  echoing what the user typed, not the resolved sentence. Log the resolved
  sentence instead.

Add `HistoryTurn`, `HistoryEntities`, `HistoryResult`, `HistoryStore` Pydantic
models next to `DeviceLocation` in `app/api/chat.py`. A malformed `history`
(wrong types) returns 422 like a malformed `location` does.

---

## 3. Pipeline change

In `chat()`:

```
instruction = request.instruction.strip()
context     = build_context(request.history)          # app/nlp/follow_up.py
resolved    = resolve_follow_up(instruction, context)  # app/nlp/follow_up.py
```

`resolved` is one of:

- `Fresh(text)` — the message stands alone: it names a merchant, card, PLZ,
  "near me", a capitalised place after in/near/nach/bei/zu, or a known
  product (`KNOWN_PRODUCTS` in the extractor), or it is longer than ten words
  with no anaphor. Classify `text` as today.
- `Rewritten(text)` — an anaphoric or bare follow-up turned into a
  self-contained sentence from the carried context, using only the phrasings
  in the §1 table (or ones you add to the training data in §5). Classify
  `text` as today; entities come out of the rewritten sentence, so
  `entity_merchant`, `entity_location`, `entity_product` are right without any
  extra plumbing.
- `ResultQuestion(text)` — a question about what was just shown ("which one
  is cheapest?", "is it worth the trip?", "what's the total?", "show me
  alternatives", "cheapest?", "lohnt sich die Fahrt?"): short, anaphoric or a
  bare question, and the carried context has `results`. Do **not** run the
  classifier or OfferHopper. Return the empty-plan shape (`domain CHAT`,
  `tool none`, `response_mode text`, `planner_actions []`,
  `offerhopperData null`, entities from the context) and generate the reply
  from the previous results (§4). If the contract test freezes the `intent` /
  `sub_intent` vocabulary, add one pair for this — `FOLLOW_UP` /
  `RESULT_QUESTION` — and extend the frozen set in the same commit; the client
  renders any empty plan as plain text, so nothing else changes.

Carried context (newest value wins, walking history from the end): merchant
(`entities.merchant`, else the first `stores[].name`), products
(`entities.product` from the newest turn whose `planner_actions` contain a
product/basket/recommendation action, together with that turn's `price_max`
and whether it was a basket action), typed location (`entities.location`
unless it is `CURRENT_LOCATION`), loyalty card (`entities.loyalty_card` or
`entities.brand`; use `merchant` only when that turn's actions were wallet
actions), and the newest turn that has `results`.

Anaphors (EN): there, it, its, that, those, these, this, them, one, ones,
here, same. (DE): dort, dorthin, dahin, da, hin, es, sie, das, davon, dazu,
dieselbe, gleiche, selbe, ihn, ihm, denen. Result words: alternatives, others,
another, more, options, else, total, cheaper, cheapest, worth, alternativen,
andere, weitere, mehr, insgesamt, gesamt, günstiger, günstigste, billiger,
lohnt. "What about X" cues: what about, how about, and, also, und, was ist
mit, wie wär's mit, auch. Note that Python's `\b` is Unicode-aware, unlike
Dart's, so `\böffnet\b` works here.

Language of the rewrite follows the message (`_language()` in the generator
already does this; reuse it, and fall back to the language of the last
history turn when the message has no markers).

Location handling is unchanged: a rewritten sentence that says "near me"
resolves through `_resolve_location()` to the device position when
`location` is present, exactly as a typed "near me" does.

Optional, behind `FOLLOW_UP_LLM_REWRITE=1`: when the rules produce nothing
for an anaphoric message, ask the configured Groq model for a single
self-contained sentence (system prompt: rewrite only, same language, no new
facts, return the original if unsure; `max_completion_tokens` ≈ 60,
temperature 0), then classify that. Off by default; measure latency before
turning it on.

---

## 4. Generator change

`generate_response()` gains `context: Optional[ConversationContext]`.

- For `Rewritten` turns, the `<user_request>` block carries the **original**
  message, followed by a line `Resolved as: <rewritten sentence>`, so the
  reply acknowledges what the person actually said ("Sure, here's the way to
  REWE") rather than the template.
- For `ResultQuestion` turns, add a `PREVIOUS RESULTS:` block built from the
  context's `results`, `stores` and `share_url` in the same compact format as
  `_offerhopper_context()`, and instruct the model to answer the question from
  that block only, ending with the map link on its own line when one exists.
  Keep the existing rule that nothing outside the supplied blocks may be
  invented. If the LLM call fails, fall back to a deterministic sentence that
  lists the previous results with prices (extend
  `_generate_fallback_response`), never to `CLARIFY`.
- Never include more than the last history turn's text in the prompt; the
  rest is only used for entity carry-over.

---

## 5. Training data

Add rows to `dataset/noah_dataset_v3.csv` (same columns as the existing rows)
for the rewrite targets the classifier currently declines, in both languages,
with the usual 10–20 paraphrases each, then retrain with
`python -m app.nlp.train` and check the holdout probes did not regress:

- German navigation: "Bring mich zu {merchant} in {city}", "Wie komme ich zu
  {merchant}?", "Zeig mir den Weg zu {merchant}" → `PLAN_ROUTE` /
  `GET_DIRECTIONS`.
- German opening hours: "Wann öffnet {merchant}?", "Öffnungszeiten von
  {merchant} in {city}" → `GET_OPENING_HOURS`.
- English plain product search without a price: "Where can I buy {product}
  in {city}?", "Find {product} offers near me", "Search for {product} near
  me" → `SEARCH_PRODUCT`.
- Distance: "How far is {merchant}?", "Wie weit ist {merchant} entfernt?" →
  `GET_MERCHANT_DETAILS` (or leave out of scope, but then never rewrite to
  it).

Do not add bare follow-ups ("take me there", "which one is cheapest?") as
training rows; without context they are genuinely ambiguous and belong in
`noah_holdout_probes.csv` as out-of-scope so the model keeps declining them
when no history is sent.

---

## 6. Tests

- `tests/test_response_contract.py`: add `test_history_is_optional_and_additive`
  in the style of the location test — with and without `history`, the key set
  is the 24 keys and `instruction` echoes the typed text.
- `tests/test_follow_up.py`: unit tests for the resolver — every row of the
  §1 table, the carry-over across an intervening navigation turn (basket →
  "take me there" → "add them to my list" still yields the products), a
  merchant from a maps turn is not mistaken for a card, failed/empty turns are
  skipped, and fresh messages pass through untouched.
- `tests/test_api_chat.py`: with OfferHopper monkeypatched as the existing
  tests do, drive the sequence basket → "Take me there" → "Add them to my
  shopping list" → "When does it open?" → "What about butter?" → "Which one
  is cheapest?" → "Show my Payback card" through the API with `history` built
  from each previous response, and assert `planner_actions` and the entity
  fields per the table; the result question must return `planner_actions []`
  with a reply naming the €0.99 milk and no OfferHopper call.

---

## 7. Verify against the running service before finishing

```bash
U=http://localhost:8000/api/chat; H='Content-Type: application/json'

# baseline unchanged
curl -s $U -H "$H" -d '{"instruction":"Take me there"}'                    # -> CHAT, planner_actions []
curl -s $U -H "$H" -d '{"instruction":"Show me my Payback barcode."}'      # -> DISPLAY_BARCODE

# with history: intent changes, entities carried
curl -s $U -H "$H" -d '{"instruction":"Take me there","history":[{"instruction":"Find the cheapest basket for milk, eggs and bread in Hamburg","planner_actions":["FIND_CHEAPEST_BASKET"],"entities":{"product":"Milk, Eggs, Bread","location":"Hamburg"},"stores":[{"name":"REWE","address":"Ballindamm, 40, 20095, Hamburg","latitude":53.55127,"longitude":9.99681}]}]}'
#   -> PLAN_ROUTE, entity_merchant REWE, entity_location Hamburg, instruction echoes "Take me there"

curl -s $U -H "$H" -d '{"instruction":"Add them to my shopping list","history":[{"instruction":"Find the cheapest basket for milk, eggs and bread in Hamburg","planner_actions":["FIND_CHEAPEST_BASKET"],"entities":{"product":"Milk, Eggs, Bread","location":"Hamburg"}},{"instruction":"Take me there","planner_actions":["PLAN_ROUTE"],"entities":{"merchant":"REWE","location":"Hamburg"}}]}'
#   -> BUILD_SHOPPING_LIST, entity_product contains milk, eggs, bread

curl -s $U -H "$H" -d '{"instruction":"Which one is cheapest?","history":[{"instruction":"Find the cheapest basket for milk, eggs and bread in Hamburg","planner_actions":["FIND_CHEAPEST_BASKET"],"entities":{"product":"Milk, Eggs, Bread","location":"Hamburg"},"results":[{"name":"Weihenstephan Barista Milch 1l","store":"REWE","price":0.99},{"name":"REWE Beste Wahl Eier Freilandhaltung 4 Stück","store":"REWE","price":1.49},{"name":"Harry Kürbiskernbrot 750g","store":"REWE","price":1.99}],"share_url":"https://offerhopper.ai/s/VcEZiHHo"}]}'
#   -> planner_actions [], offerhopperData null, response names the €0.99 milk, map link last, answered in < 3 s

curl -s $U -H "$H" -d '{"instruction":"Zeig mir die Karte","history":[{"instruction":"Show me my Payback barcode.","planner_actions":["DISPLAY_BARCODE"],"entities":{"loyalty_card":"PAYBACK","brand":"PAYBACK","category":"LOYALTY_CARD"}}]}'
#   -> OPEN_WALLET_CARD or DISPLAY_BARCODE, entity_loyalty_card PAYBACK, German reply
```

Acceptance: `pytest` green including the frozen-contract tests; every curl
above behaves as annotated; a request with no `history` is byte-for-byte the
same route as before; the response key set is still exactly 24 keys.

When done, summarise which files you changed, the rows added to the dataset
with the retrained metrics, and anything the Flutter client must change
(expected: send `history` built from its stored responses and drop its
client-side resolver).
