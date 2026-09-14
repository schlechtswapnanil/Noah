# Integrate the Noah backend into the PayTo Flutter app

You are working in the PayTo Flutter app. Integrate it with the Noah assistant
backend, which is live at:

    https://noah-z7qr.onrender.com

Everything below is verified against that deployment. Do not rely on the older
`noah_flutter_implementation_plan.md` for the response shape — its
`OfferhopperRouteResult` model expects fields (`stops`, `total_basket_cost`,
`subtotal_eur`, …) that the server has never returned, so it parses to zeros and
an empty list. This document is the source of truth.

Work in this order, verifying each step against the live URL with curl before
moving on. Do not ask me clarifying questions unless something below is
contradictory; make sensible Flutter decisions and note them in your summary.

---

## 1. API contract

**Request** — `POST /api/chat`, `Content-Type: application/json`

```json
{ "instruction": "Zeig mir meine Payback-Karte" }
```

That is the whole request. No session id, no history: every call is independent.
Blank input returns `200 {"error": "Instruction cannot be empty."}`; a missing
field returns `422`.

**Response** — always exactly these 24 keys, in every case (success, decline,
tool failure). Never assume a key is absent; assume nullable values instead.

| key | type | notes |
|---|---|---|
| `instruction` | string | echo of the request |
| `domain` | string | `ACCOUNT` `CHAT` `KNOWLEDGE` `MERCHANT` `NAVIGATION` `OFFER` `PRODUCT` `RECOMMENDATION` `SHOPPING` `SYSTEM` `WALLET` |
| `intent` | string | |
| `sub_intent` | string | |
| `tool` | string | `none` `wallet_tool` `offers_tool` `maps_tool` `merchant_api` `offerhopper_mcp` `rag_engine` `recommendation_engine` `firestore` `planner` |
| `response_mode` | string | `functional` `hybrid` `navigation` `text` |
| `requires_memory` | bool | |
| `requires_rag` | bool | |
| `requires_recommendation` | bool | |
| `workflow_type` | string | `single_step` `multi_step` `sequential` |
| `planner_actions` | string[] | see §3; **may be empty** |
| `planner_action_count` | int | `== planner_actions.length` |
| `tool_sequence` | string[] | same length as `planner_actions`; `tool_sequence[i]` is the tool for `planner_actions[i]` |
| `entity_product` | string \| null | e.g. `"Milk, Eggs, Bread"` |
| `entity_merchant` | string \| null | upper-case: `REWE` `NETTO` `LIDL` `ALDI` `KAUFLAND` `EDEKA` `PENNY` `MÜLLER` `DM` `ROSSMANN` `GLOBUS` |
| `entity_brand` | string \| null | `PAYBACK` or `DeutschlandCard` for those cards |
| `entity_category` | string \| null | `LOYALTY_CARD` when a card was named |
| `entity_price_min` | number \| null | |
| `entity_price_max` | number \| null | euros |
| `entity_location` | string \| null | city name, 5-digit PLZ, or the literal `CURRENT_LOCATION` |
| `entity_radius` | number \| null | metres |
| `entity_loyalty_card` | string \| null | `PAYBACK` `DEUTSCHLANDCARD` `LIDL_PLUS` `NETTO_PLUS` `REWE_BONUS` `EDEKA_CARD` |
| `offerhopperData` | object \| null | see §4; **null unless an OfferHopper action ran and succeeded** |
| `response` | string | the user-facing sentence, already in the user's language (EN or DE) |

`response` is the only text you show the user. Never synthesise your own
wording from the other fields.

---

## 2. Behaviours the client must handle

**a. Empty `planner_actions` means "render the text and do nothing else".**
This is Noah greeting the user, answering a question conversationally, or —
when it is not confident enough to act — asking a clarifying question. It also
covers out-of-scope input ("what's the weather", "transfer 50 euros"). In all
these cases `domain` is `CHAT`, `response_mode` is `text`, `offerhopperData` is
`null`. Render `response` as a chat bubble. Do not show an error.

**b. Knowledge answers are plain text too.** `KNOWLEDGE` domain responses
(`ANSWER_FAQ`, `ANSWER_PAYTO_QUESTION`, `PROVIDE_APP_HELP`) carry one action but
nothing for the UI to open — render `response`.

**c. `offerhopperData` is null when the price service failed.** In that case
`response` already says so ("I couldn't reach the price service just now…" /
"Ich konnte den Preisdienst gerade nicht erreichen…"). Render the text and do
not open the route card. Do not treat it as an app error.

**d. Timeouts.** Set the HTTP client timeout to **at least 60 seconds**. The
service is on a free tier that spins down after ~15 min idle and takes ~50 s to
wake; OfferHopper queries add 3–20 s on top. On app start (or when the chat
screen opens) fire a `GET /health` and ignore the result — it warms the
container so the first real request is fast.

**e. Errors.** On network failure, timeout, or a non-200 status, show a short
bubble ("Noah is waking up, try again in a moment." / DE equivalent) with a
retry affordance. Never show a raw exception.

**f. Language.** Nothing to do — `response` is already in whichever of
English or German the user typed. Do not translate it.

---

## 3. Actions and what the UI does for each

Every value in `planner_actions` is one of the following. Dispatch in order.
When several actions are present (`multi_step` / `sequential`), perform each;
typically the first produces data and the second opens something.

| action(s) | UI |
|---|---|
| `DISPLAY_BARCODE` `OPEN_WALLET_CARD` | open the loyalty-card / barcode modal for `entity_loyalty_card` (fall back to `entity_brand`, then `entity_merchant`) |
| `DISPLAY_POINTS` `DISPLAY_REWARDS` | open the same card with the points / rewards tab |
| `ADD_LOYALTY_CARD` `REMOVE_LOYALTY_CARD` `LIST_WALLET_CARDS` | open the wallet screen |
| `PLAN_ROUTE` `OPEN_GOOGLE_MAPS` **`GET_DIRECTIONS`** | launch maps for `entity_merchant` ?? `entity_location`. **`GET_DIRECTIONS` was missing from the old dispatcher — add it to the same case group.** |
| `FIND_CHEAPEST_BASKET` `OPTIMIZE_SHOPPING_ROUTE` `SPLIT_BASKET_ACROSS_MERCHANTS` | if `offerhopperData != null`, open the route card (§4) |
| `SEARCH_PRODUCT` `SEARCH_PRODUCT_BY_PRICE` `SEARCH_PRODUCT_BY_BRAND` `SEARCH_PRODUCT_BY_CATEGORY` `SEARCH_OFFERS` `COMPARE_PRODUCTS` `CHECK_PRODUCT_AVAILABILITY` | if `offerhopperData != null`, show the products from §4 as result tiles (same data, list layout rather than route layout) |
| `SEARCH_DISCOUNTS` `SEARCH_CASHBACK` `OPEN_WEEKLY_FLYER` | open the offers / flyer screen for `entity_merchant` |
| `GET_OPENING_HOURS` `GET_MERCHANT_CONTACT` `GET_MERCHANT_DETAILS` `SEARCH_MERCHANT` `SEARCH_NEARBY_MERCHANTS` | open the merchant screen for `entity_merchant` if the app has one; otherwise just render `response` |
| `RECOMMEND_PRODUCTS` `RECOMMEND_OFFERS` `RECOMMEND_MERCHANTS` `GET_PERSONALIZED_RECOMMENDATIONS` | open the recommendations screen if one exists; otherwise render `response` |
| `SHOW_PROFILE` `OPEN_SETTINGS` `SHOW_PURCHASE_HISTORY` `SHOW_VISIT_HISTORY` | navigate to that screen |
| `BUILD_SHOPPING_LIST` | open the shopping list, pre-filled with `entity_product` if set |
| `SHOW_HELP` `REPORT_BUG` `SUBMIT_FEEDBACK` | navigate to that screen |
| `ANSWER_FAQ` `ANSWER_PAYTO_QUESTION` `PROVIDE_APP_HELP` | nothing to open — render `response` |

Unknown action strings must be ignored, not crash.

---

## 4. `offerhopperData` — the real shape

This is what the server returns (verified live). Only the fields the UI needs
are listed; parse leniently and ignore the rest.

```jsonc
{
  "success": true,
  "share_url": "https://offerhopper.ai/s/Fjpbl3wR",     // open in browser on tap
  "ai_description": "Visit 1 stores for maximum coverage.",
  "total_estimated_cost": 9.54,                        // euros, incl. travel/time
  "total_estimated_savings": 0.69,
  "optimized_route": {
    "estimated_total_cost": 9.54,
    "estimated_total_savings": 0.69,
    "total_distance_km": 3.504,
    "total_duration_minutes": 5.6,
    "total_time_with_shopping_minutes": 20.1,
    "stores": [
      { "name": "REWE", "place_id": "osm:W1000656439",
        "latitude": 53.55127, "longitude": 9.99681,
        "formatted_address": "Ballindamm, 40, 20095, Hamburg" }
    ],
    "route_segments": [
      {
        "from_name": "user_location",
        "to_name": "REWE",
        "to_store": { "name": "REWE", "latitude": 53.55127, "longitude": 9.99681,
                      "formatted_address": "Ballindamm, 40, 20095, Hamburg" },
        "distance_km": 2.432,
        "duration_minutes": 3.42,
        "travel_mode": "car",
        "estimated_cost_at_store": 4.47,            // basket subtotal at this stop
        "products_to_buy": [
          {
            "name": "Milch",                         // what the user asked for
            "selected_product": "Weihenstephan Barista Milch 1l",   // what to show
            "price": 0.99,
            "regular_price": 0.99,
            "discount_pct": 0,
            "quantity": 1.0,
            "unit": "pack",
            "total_cost": 0.99,
            "market_average": 1.045
          }
        ]
      },
      { "from_name": "REWE", "to_name": "user_location", "products_to_buy": [] }
    ]
  },
  "cost_analysis": {
    "product_cost": 4.47,
    "travel_cost": 2.17,
    "in_store_time_cost": 2.9,
    "verdict": { "headline": "not_worth_travel", "overall_key": "costly",
                 "savings_eur": 0.69 }
  }
}
```

Modelling rules:

- A **stop** is a `route_segments[]` entry whose `to_name != "user_location"`.
  The final segment is the trip home and has no products — skip it.
- Stop merchant name: `to_store.name` (fallback `to_name`). Address:
  `to_store.formatted_address`. Subtotal: `estimated_cost_at_store`.
- Item display name: `selected_product` (fallback `name`). Price: `price`.
  Show a strikethrough `regular_price` when `discount_pct > 0`.
- Card header: `total_estimated_cost` and `total_estimated_savings`;
  footer: `optimized_route.total_distance_km` and `total_duration_minutes`.
- `cost_analysis.verdict.headline` is one of a small set of snake_case keys
  (`not_worth_travel`, `worth_it`, …). Surface it as a one-line badge if
  cheap to do; otherwise ignore — `response` already reflects it in prose.
- All numbers are `num`; parse with `(x as num?)?.toDouble()`. Every field
  may be missing — default, never throw.

---

## 5. What to build or change

1. **`NoahApiService`** — base URL from a single config constant
   (`https://noah-z7qr.onrender.com`), `POST /api/chat`, 60 s timeout, a
   `warmUp()` that GETs `/health`, and a typed error for network/timeout/non-200.
2. **`NoahResponse` model** — all 24 keys from §1, every entity field nullable,
   `plannerActions` and `toolSequence` as `List<String>`, `offerhopperData` as
   `OfferhopperRouteResult?`.
3. **`OfferhopperRouteResult`, `OfferhopperStop`, `OfferhopperItem`** — rewrite
   from §4. Delete the old flat-shape parsers.
4. **`NoahActionDispatcher`** — the table in §3, including the empty-actions
   case, the `GET_DIRECTIONS` case, the `offerhopperData == null` guard, and
   ignoring unknown actions.
5. **Chat screen** — always render `response`; call `warmUp()` on open; show
   the friendly retry bubble from §2e on failure.
6. **Route card / result tiles** — built from the §4 model; `share_url` opens
   in the external browser.

Keep the request body exactly `{"instruction": ...}`. Do not add fields the
server will ignore, and do not depend on fields not listed here.

---

## 6. Verify against the live service before finishing

Run these and confirm the app handles each shape:

```bash
U=https://noah-z7qr.onrender.com/api/chat; H='Content-Type: application/json'

# wallet modal, EN                      -> DISPLAY_BARCODE, entity_loyalty_card=PAYBACK
curl -s $U -H "$H" -d '{"instruction":"Show me my Payback barcode."}'

# wallet modal, DE                      -> OPEN_WALLET_CARD, response in German
curl -s $U -H "$H" -d '{"instruction":"Zeig mir meine Lidl Plus Karte"}'

# maps                                  -> PLAN_ROUTE, entity_merchant=LIDL
curl -s $U -H "$H" -d '{"instruction":"Take me to the nearest Lidl."}'

# route card with live data (3-20 s)    -> FIND_CHEAPEST_BASKET, offerhopperData present
curl -s $U -H "$H" -d '{"instruction":"Find the cheapest basket for milk, eggs and bread in Hamburg"}'

# result tiles with live data           -> SEARCH_PRODUCT_BY_PRICE, entity_price_max=2
curl -s $U -H "$H" -d '{"instruction":"Where can I buy frozen pizzas for less than 2 euros near me?"}'

# two actions in sequence               -> [SEARCH_OFFERS, OPEN_WALLET_CARD], workflow_type=sequential
curl -s $U -H "$H" -d '{"instruction":"I am at Netto. What are the current offers on ice creams here below 3 euros? Then display my Payback card."}'

# knowledge, text only                  -> ANSWER_PAYTO_QUESTION, nothing to open
curl -s $U -H "$H" -d '{"instruction":"Does PayTo store my payment details?"}'

# empty actions: greeting               -> planner_actions=[], render response
curl -s $U -H "$H" -d '{"instruction":"Hello! Who are you?"}'

# empty actions: clarifying question    -> planner_actions=[], domain=CHAT
curl -s $U -H "$H" -d '{"instruction":"transfer 50 euros to my brother"}'
```

Acceptance: every response above parses without throwing, the right screen
opens (or nothing opens, for the last three), the route card shows a real
store, real product names and real prices, and no request uses a timeout
under 60 s.

When done, summarise which files you changed and any UI decisions you made
where the app did not already have a matching screen.
