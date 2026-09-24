# Send conversation history to the Noah backend

You are working in the PayTo Flutter app. The Noah backend
(`https://noah-z7qr.onrender.com`, `POST /api/chat`) is now context-aware: when
the request carries the previous turns, the server resolves follow-ups itself
("take me there", "add them to my list", "which one is cheapest?"). Move that
responsibility out of the app.

Everything else in `noah_flutter_integration_prompt.md` still holds: the
request body is `instruction`, `location` and now `history`; the response is
always the same 24 keys; `instruction` still echoes what the user typed.

Work in the order below and verify each step with curl before wiring the UI.
Make sensible Flutter decisions and note them in your summary; ask only if
something here is contradictory.

---

## 1. Delete the client-side resolver

Remove `lib/noah/services/noah_follow_up_resolver.dart` and every call site.
Send exactly what the user typed as `instruction`. Do not fold the previous
exchange into the text and do not append context notes - both misroute on the
server, which is why this moved.

## 2. Store what each turn needs

Wherever responses are kept (`NoahChatProvider`), keep per turn:

- `instruction` - what the user typed (required)
- `response`, `planner_actions` - copied from the response
- `entities` - the nine `entity_*` fields of the response, without the
  prefix: `product, merchant, brand, category, price_min, price_max,
  location, radius, loyalty_card`
- a compact summary of `offerhopperData`, built once when the response
  arrives:
  - `results`: for every `route_segments[]` entry whose `to_name` is not
    `"user_location"`, for every `products_to_buy[]` item:
    `{ "name": selected_product ?? name, "store": to_store.name ?? to_name, "price": price }`
    - at most 10
  - `stores`: for every `optimized_route.stores[]` entry:
    `{ "name", "address": formatted_address, "latitude", "longitude" }`
  - `share_url` as is

Never store or send the full `offerhopperData` payload.

## 3. Send `history` on every request

```json
{
  "instruction": "Take me there",
  "location": { "latitude": 52.5219, "longitude": 13.4132 },
  "history": [
    {
      "instruction": "Find the cheapest basket for milk, eggs and bread in Hamburg",
      "response": "REWE on Ballindamm has the cheapest basket … https://offerhopper.ai/s/VcEZiHHo",
      "planner_actions": ["FIND_CHEAPEST_BASKET"],
      "entities": { "product": "Milk, Eggs, Bread", "merchant": null, "brand": null,
                    "category": null, "price_min": null, "price_max": null,
                    "location": "Hamburg", "radius": null, "loyalty_card": null },
      "results": [
        { "name": "Weihenstephan Barista Milch 1l", "store": "REWE", "price": 0.99 },
        { "name": "REWE Beste Wahl Eier Freilandhaltung 4 Stück", "store": "REWE", "price": 1.49 },
        { "name": "Harry Kürbiskernbrot 750g", "store": "REWE", "price": 1.99 }
      ],
      "share_url": "https://offerhopper.ai/s/VcEZiHHo",
      "stores": [ { "name": "REWE", "address": "Ballindamm, 40, 20095, Hamburg",
                    "latitude": 53.55127, "longitude": 9.99681 } ]
    }
  ]
}
```

Rules:

- Oldest first, newest last. Send the last 6 turns at most.
- Every field except `instruction` is optional. Nulls may be sent or omitted.
- Turns with an empty `planner_actions` (greetings, clarifying questions) may
  be skipped; the server ignores them anyway.
- Keep numbers as numbers: a `price` or `latitude` sent as a string is a
  `422`, exactly like a malformed `location`.
- Add a `NoahHistoryTurn` model with `toJson()` and build the list from the
  stored turns right before each request. The first message of a session
  sends no `history` (or an empty list).

## 4. Handle the response

Nothing new to open. Three cases to know about:

- **Rewritten follow-ups look like ordinary action responses.** "Take me
  there" after the basket comes back as `PLAN_ROUTE` with `entity_merchant`
  `REWE` and `entity_location` `Hamburg`; "Add them to my shopping list" as
  `BUILD_SHOPPING_LIST` with `entity_product` `"Milk, Eggs, Bread"`; "Zeig mir
  die Karte" after a wallet turn as `OPEN_WALLET_CARD` with
  `entity_loyalty_card` `PAYBACK`. Dispatch exactly as today. `instruction`
  still echoes what was typed.
- **Result questions** ("which one is cheapest?", "what's the total?", "lohnt
  sich die Fahrt?") return `planner_actions: []`, `domain: "CHAT"`,
  `intent: "FOLLOW_UP"`, `sub_intent: "RESULT_QUESTION"`,
  `offerhopperData: null`, and `response` is the answer with the map link on
  its last line. Render like any empty plan: text bubble, links tappable.
  `FOLLOW_UP` and `RESULT_QUESTION` are new values - if `intent` or
  `sub_intent` are parsed into enums, an unknown value must not throw.
- **A bare follow-up with no history** ("Take me there" as the first
  message) returns the empty plan with a clarifying question ("Happy to get
  you there - which store do you mean?"). Render as text; not an error.

Keep the 60 s timeout; result questions come back in well under a second plus
the reply generation.

## 5. Verify before wiring the UI

```bash
U=https://noah-z7qr.onrender.com/api/chat; H='Content-Type: application/json'

# follow-up with history                  -> PLAN_ROUTE, entity_merchant REWE, entity_location Hamburg, instruction "Take me there"
curl -s $U -H "$H" -d '{"instruction":"Take me there","history":[{"instruction":"Find the cheapest basket for milk, eggs and bread in Hamburg","planner_actions":["FIND_CHEAPEST_BASKET"],"entities":{"product":"Milk, Eggs, Bread","location":"Hamburg"},"stores":[{"name":"REWE","address":"Ballindamm, 40, 20095, Hamburg","latitude":53.55127,"longitude":9.99681}]}]}'

# products carried across a navigation turn -> BUILD_SHOPPING_LIST, entity_product "Milk, Eggs, Bread"
curl -s $U -H "$H" -d '{"instruction":"Add them to my shopping list","history":[{"instruction":"Find the cheapest basket for milk, eggs and bread in Hamburg","planner_actions":["FIND_CHEAPEST_BASKET"],"entities":{"product":"Milk, Eggs, Bread","location":"Hamburg"}},{"instruction":"Take me there","planner_actions":["PLAN_ROUTE"],"entities":{"merchant":"REWE","location":"Hamburg"}}]}'

# result question                          -> planner_actions [], sub_intent RESULT_QUESTION, response names the €0.99 milk, map link last
curl -s $U -H "$H" -d '{"instruction":"Which one is cheapest?","history":[{"instruction":"Find the cheapest basket for milk, eggs and bread in Hamburg","planner_actions":["FIND_CHEAPEST_BASKET"],"entities":{"product":"Milk, Eggs, Bread","location":"Hamburg"},"results":[{"name":"Weihenstephan Barista Milch 1l","store":"REWE","price":0.99},{"name":"REWE Beste Wahl Eier Freilandhaltung 4 Stück","store":"REWE","price":1.49},{"name":"Harry Kürbiskernbrot 750g","store":"REWE","price":1.99}],"share_url":"https://offerhopper.ai/s/VcEZiHHo"}]}'

# wallet follow-up in German               -> OPEN_WALLET_CARD or DISPLAY_BARCODE, entity_loyalty_card PAYBACK
curl -s $U -H "$H" -d '{"instruction":"Zeig mir die Karte","history":[{"instruction":"Show me my Payback barcode.","planner_actions":["DISPLAY_BARCODE"],"entities":{"loyalty_card":"PAYBACK","brand":"PAYBACK","category":"LOYALTY_CARD"}}]}'

# no history                               -> domain CHAT, planner_actions [], clarifying question
curl -s $U -H "$H" -d '{"instruction":"Take me there"}'
```

If the first curl comes back with `GET_OPENING_HOURS` and no merchant, the
backend deploy has not landed yet: run the backend locally
(`cd noah_backend && uvicorn app.main:app --port 8000`) and use
`http://localhost:8000/api/chat` instead.

Then drive the same sequence in the app: basket search → "Take me there" →
"Add them to my shopping list" → "When does it open?" → "What about butter?"
→ "Which one is cheapest?" → "Show my Payback card". Maps must open for REWE
in Hamburg, the list must receive milk, eggs and bread, and the result
question must render as a text bubble with a tappable map link and no route
card.

Acceptance: the resolver file is gone, every request after the first carries
`history`, no request ever includes `offerhopperData`, the two new
`intent`/`sub_intent` values parse without error, and the sequence above
behaves as described end to end.

When done, summarise which files you changed and where the history is built.
