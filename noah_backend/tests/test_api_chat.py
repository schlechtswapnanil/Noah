import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app.main import app
from app.planner.action_registry import ACTION_TO_TOOL, OFFERHOPPER_MCP_TOOL
from app.rag.retriever import retrieve, get_chunks

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_offerhopper_action_mappings():
    """Verify Offerhopper is mapped for grocery search, pricing, and basket optimization."""
    # Basket-level and product search tools MUST map to Offerhopper MCP
    assert ACTION_TO_TOOL["FIND_CHEAPEST_BASKET"] == OFFERHOPPER_MCP_TOOL
    assert ACTION_TO_TOOL["OPTIMIZE_SHOPPING_ROUTE"] == OFFERHOPPER_MCP_TOOL
    assert ACTION_TO_TOOL["SPLIT_BASKET_ACROSS_MERCHANTS"] == OFFERHOPPER_MCP_TOOL
    assert ACTION_TO_TOOL["SEARCH_PRODUCT"] == OFFERHOPPER_MCP_TOOL
    assert ACTION_TO_TOOL["SEARCH_PRODUCT_BY_PRICE"] == OFFERHOPPER_MCP_TOOL
    assert ACTION_TO_TOOL["COMPARE_PRODUCTS"] == OFFERHOPPER_MCP_TOOL

    # Single store directions MUST map to maps_tool
    assert ACTION_TO_TOOL["PLAN_ROUTE"] == "maps_tool"
    assert ACTION_TO_TOOL["OPEN_GOOGLE_MAPS"] == "maps_tool"


def test_chat_loyalty_barcode():
    response = client.post("/api/chat", json={"instruction": "Show me my Payback barcode."})
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "WALLET"
    assert "DISPLAY_BARCODE" in data["planner_actions"]
    assert "wallet_tool" in data["tool_sequence"]
    assert data["entity_loyalty_card"] == "PAYBACK"
    assert "barcode" in data["response"].lower() or "payback" in data["response"].lower()


def test_chat_product_search():
    response = client.post("/api/chat", json={"instruction": "Where can I buy frozen pizzas for less than €2 near me?"})
    assert response.status_code == 200
    data = response.json()
    assert "SEARCH_PRODUCT_BY_PRICE" in data["planner_actions"] or "SEARCH_PRODUCT" in data["planner_actions"]
    assert OFFERHOPPER_MCP_TOOL in data["tool_sequence"]
    assert data["entity_price_max"] == 2.0
    assert data["entity_location"] == "CURRENT_LOCATION"
    assert len(data["response"]) > 10


def test_chat_fetch_mayonnaise_offerhopper():
    response = client.post("/api/chat", json={"instruction": "fetch me mayonnaise below 2 euros near me"})
    assert response.status_code == 200
    data = response.json()
    assert data["entity_product"] == "Mayonnaise"
    assert data["entity_price_max"] == 2.0
    assert OFFERHOPPER_MCP_TOOL in data["tool_sequence"]
    assert data["offerhopperData"] is not None
    assert "mayo" in data["response"].lower() or "dm" in data["response"].lower() or "rewe" in data["response"].lower() or "found" in data["response"].lower()


def test_chat_navigation():
    response = client.post("/api/chat", json={"instruction": "Take me to the nearest Lidl."})
    assert response.status_code == 200
    data = response.json()
    assert "PLAN_ROUTE" in data["planner_actions"] or "OPEN_GOOGLE_MAPS" in data["planner_actions"]
    assert "maps_tool" in data["tool_sequence"]
    assert data["entity_merchant"] == "LIDL"
    assert "lidl" in data["response"].lower() or "route" in data["response"].lower() or "map" in data["response"].lower()


def test_chat_offerhopper_basket():
    response = client.post("/api/chat", json={"instruction": "Find the cheapest basket for milk, eggs, and bread in Berlin."})
    assert response.status_code == 200
    data = response.json()
    assert "FIND_CHEAPEST_BASKET" in data["planner_actions"]
    assert OFFERHOPPER_MCP_TOOL in data["tool_sequence"]
    assert data["entity_location"] == "Berlin"
    assert data["offerhopperData"] is not None
    assert "basket" in data["response"].lower() or "cheapest" in data["response"].lower() or "found" in data["response"].lower() or "€" in data["response"]


def test_chat_offerhopper_route_optimization():
    response = client.post("/api/chat", json={"instruction": "Plan the optimal shopping route for my grocery list in Hamburg."})
    assert response.status_code == 200
    data = response.json()
    assert "OPTIMIZE_SHOPPING_ROUTE" in data["planner_actions"]
    assert OFFERHOPPER_MCP_TOOL in data["tool_sequence"]
    assert data["entity_location"] == "Hamburg"


def test_chat_product_comparison_uses_offerhopper():
    response = client.post("/api/chat", json={"instruction": "Compare the prices of oat milk and almond milk."})
    assert response.status_code == 200
    data = response.json()
    assert "COMPARE_PRODUCTS" in data["planner_actions"]
    assert OFFERHOPPER_MCP_TOOL in data["tool_sequence"]


def test_chat_multi_step_compound():
    response = client.post("/api/chat", json={
        "instruction": "I'm at Netto. What are the current offers on ice creams here below €3? Then display my Payback card."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["workflow_type"] == "sequential"
    assert len(data["planner_actions"]) >= 2
    assert "SEARCH_OFFERS" in data["planner_actions"]
    assert "OPEN_WALLET_CARD" in data["planner_actions"]
    assert data["entity_merchant"] == "NETTO"
    assert data["entity_loyalty_card"] == "PAYBACK"
    assert data["entity_price_max"] == 3.0
    assert "netto" in data["response"].lower() or "payback" in data["response"].lower() or "ice cream" in data["response"].lower()


def test_chat_rag_payto_knowledge():
    response = client.post("/api/chat", json={"instruction": "What is PayTo and how do loyalty points work?"})
    assert response.status_code == 200
    data = response.json()
    assert data["requires_rag"] is True
    assert "rag_engine" in data["tool_sequence"]
    # Check that response is grounded and informative
    assert len(data["response"]) > 20
    assert "couldn't find" not in data["response"].lower() or "payto" in data["response"].lower()


def test_chat_greeting_conversation():
    response = client.post("/api/chat", json={"instruction": "Hello! Who are you?"})
    assert response.status_code == 200
    data = response.json()
    assert "noah" in data["response"].lower() or "help" in data["response"].lower()


def test_rag_retriever_indexes_user_facing_documents_only():
    """The corpus used to include Noah's own sprint plan, intent spec and API
    cost sheet, all of which were quotable to end users.  Internal documents
    must now retrieve nothing; the privacy policy and FAQ must still answer."""
    chunks = get_chunks()
    assert len(chunks) > 50
    assert not any("Week 1 Plan" in chunk or "APIs and Subscriptions" in chunk
                   or "noah_guide" in chunk for chunk in chunks)

    # The property is "no internal document is reachable", not "no answer at
    # all": a question about how Noah classifies intents is legitimately
    # answered by the user-facing capability document.
    INTERNAL = ("Week 1 Plan", "APIs and Subscriptions", "noah_guide",
                "Intent and Data Format", "dataset and output Update",
                "OfferHopper Integration", "Flutter App Integration")
    for query in ("hierarchical intent classification taxonomy Noah",
                  "What are the API subscription tiers and costs?",
                  "What is in the week 1 sprint plan?"):
        for result in retrieve(query):
            assert not any(name in result for name in INTERNAL), \
                f"internal document leaked for {query!r}: {result[:80]}"

    assert len(retrieve("Does Payto store my payment details?")) > 0


def test_rag_scopes_capability_questions_away_from_payto_docs():
    """"What can you do?" and "What is PayTo's policy?" are different shelves."""
    from app.rag.retriever import SCOPE_CAPABILITIES, SCOPE_PAYTO

    for query in ("Where do your prices come from?", "Can you remember our conversation?",
                  "Woher kommen deine Preise?", "Which languages do you support?"):
        results = retrieve(query, scope=SCOPE_CAPABILITIES)
        assert results, f"no capability answer for {query!r}"
        assert all("Noah_Capabilities" in r for r in results)

    for result in retrieve("Does PayTo store my payment details?", scope=SCOPE_PAYTO):
        assert "Noah_Capabilities" not in result


def test_chat_find_doner_under_5_euros():
    response = client.post("/api/chat", json={"instruction": "Find me a doner under 5 euros near me"})
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "PRODUCT"
    assert data["intent"] == "SEARCH"
    assert data["sub_intent"] == "SEARCH_BY_PRICE"
    assert "SEARCH_PRODUCT_BY_PRICE" in data["planner_actions"]
    assert OFFERHOPPER_MCP_TOOL in data["tool_sequence"]
    assert data["entity_product"] == "Doner"
    assert data["entity_price_max"] == 5.0
    assert data["entity_location"] == "CURRENT_LOCATION"
    assert data["offerhopperData"] is not None or len(data["response"]) > 10


def test_chat_kebab_search_price():
    response = client.post("/api/chat", json={"instruction": "Where can I get a kebab under 6 euros near me?"})
    assert response.status_code == 200
    data = response.json()
    assert "SEARCH_PRODUCT_BY_PRICE" in data["planner_actions"]
    assert OFFERHOPPER_MCP_TOOL in data["tool_sequence"]
    assert data["entity_product"] == "Kebab"
    assert data["entity_price_max"] == 6.0
    assert data["entity_location"] == "CURRENT_LOCATION"



# --------------------------------------------------------------------------
# Conversation context (`history`)
# --------------------------------------------------------------------------

def _summarise_offerhopper(data):
    """What the client stores from `offerhopperData` for the next request:
    (results, stores, share_url) - name/store/price and name/address/position,
    never the payload itself."""
    route = (data or {}).get("optimized_route") or {}
    results, stores = [], []
    for segment in route.get("route_segments") or []:
        store = (segment.get("to_store") or {}).get("name") or segment.get("to_name")
        if not store or store in ("user_location", "user_end_location"):
            continue
        for product in segment.get("products_to_buy") or []:
            results.append({"name": product.get("selected_product") or product.get("name"),
                            "store": store, "price": product.get("price")})
    for store in route.get("stores") or []:
        stores.append({"name": store.get("name"), "address": store.get("formatted_address"),
                       "latitude": store.get("latitude"), "longitude": store.get("longitude")})
    return results, stores, (data or {}).get("share_url")


def _turn(instruction, payload):
    """A `history` entry built from a previous response, as the client does."""
    results, stores, share_url = _summarise_offerhopper(payload.get("offerhopperData"))
    return {
        "instruction": instruction,
        "response": payload["response"],
        "planner_actions": payload["planner_actions"],
        "entities": {key[len("entity_"):]: value for key, value in payload.items()
                     if key.startswith("entity_")},
        "results": results, "stores": stores, "share_url": share_url,
    }


def test_follow_up_sequence_through_the_api(monkeypatch):
    """basket -> take me there -> add them to my list -> when does it open? ->
    what about butter? -> which one is cheapest? -> show my Payback card."""
    import app.tools.offerhopper as offerhopper_module
    from tests.conftest import fake_offerhopper

    calls = []

    def offerhopper(**kwargs):
        calls.append(kwargs["items"])
        # The butter search hits a service hiccup: a failed turn must be
        # skipped, so the result question still sees the basket's prices.
        if "butter" in kwargs["items"].lower():
            return {}
        return fake_offerhopper(**kwargs)

    monkeypatch.setattr(offerhopper_module, "call_offerhopper_mcp", offerhopper)
    history = []

    def ask(instruction):
        payload = client.post("/api/chat", json={"instruction": instruction, "history": history}).json()
        assert payload["instruction"] == instruction
        history.append(_turn(instruction, payload))
        return payload

    basket = ask("Find the cheapest basket for milk, eggs and bread in Hamburg")
    assert "FIND_CHEAPEST_BASKET" in basket["planner_actions"]
    assert basket["entity_product"] == "Milk, Eggs, Bread"
    assert basket["offerhopperData"] is not None
    assert calls == ["Milk, Eggs, Bread"]

    there = ask("Take me there")
    assert "PLAN_ROUTE" in there["planner_actions"]
    assert there["entity_merchant"] == "REWE"
    assert there["entity_location"] == "Hamburg"
    assert there["offerhopperData"] is None

    listed = ask("Add them to my shopping list")
    assert "BUILD_SHOPPING_LIST" in listed["planner_actions"]
    for item in ("milk", "eggs", "bread"):
        assert item in listed["entity_product"].lower()

    hours = ask("When does it open?")
    assert "GET_OPENING_HOURS" in hours["planner_actions"]
    assert hours["entity_merchant"] == "REWE"
    assert hours["entity_location"] == "Hamburg"

    butter = ask("What about butter?")
    assert "FIND_CHEAPEST_BASKET" in butter["planner_actions"]
    assert butter["entity_product"] == "Butter"
    assert butter["entity_location"] == "Hamburg"
    assert butter["offerhopperData"] is None          # the hiccup
    assert calls == ["Milk, Eggs, Bread", "Butter"]

    cheapest = ask("Which one is cheapest?")
    assert cheapest["planner_actions"] == []
    assert cheapest["planner_action_count"] == 0 and cheapest["tool_sequence"] == []
    assert cheapest["domain"] == "CHAT" and cheapest["response_mode"] == "text"
    assert cheapest["offerhopperData"] is None
    assert "0.99" in cheapest["response"] and "Milch" in cheapest["response"]
    assert cheapest["response"].rstrip().endswith("https://offerhopper.ai/s/VcEZiHHo")
    assert calls == ["Milk, Eggs, Bread", "Butter"]   # no OfferHopper call

    card = ask("Show my Payback card")
    assert "OPEN_WALLET_CARD" in card["planner_actions"] or "DISPLAY_BARCODE" in card["planner_actions"]
    assert card["entity_loyalty_card"] == "PAYBACK"


def test_bare_follow_ups_without_history_ask_instead_of_acting():
    for instruction in ("Take me there", "Add them to my shopping list", "When does it open?",
                        "Which one is cheapest?", "bring mich dorthin"):
        payload = client.post("/api/chat", json={"instruction": instruction}).json()
        assert payload["planner_actions"] == [], instruction
        assert payload["domain"] == "CHAT" and payload["offerhopperData"] is None
        assert payload["entity_merchant"] is None and payload["entity_product"] is None
        assert payload["response"].strip()


def test_wallet_follow_up_in_german():
    history = [{"instruction": "Show me my Payback barcode.",
                "planner_actions": ["DISPLAY_BARCODE"],
                "entities": {"loyalty_card": "PAYBACK", "brand": "PAYBACK", "category": "LOYALTY_CARD"}}]
    payload = client.post("/api/chat", json={"instruction": "Zeig mir die Karte", "history": history}).json()
    assert "OPEN_WALLET_CARD" in payload["planner_actions"] or "DISPLAY_BARCODE" in payload["planner_actions"]
    assert payload["entity_loyalty_card"] == "PAYBACK"
    assert payload["instruction"] == "Zeig mir die Karte"
    assert "payback" in payload["response"].lower()
    # (the reply's language is the LLM's job; the deterministic fallback used
    # when it is stubbed describes wallet actions in English)


def test_near_me_follow_up_uses_the_device_location(offline):
    """A rewritten "near me" resolves through the device position exactly as
    a typed "near me" does."""
    history = [{"instruction": "Find the cheapest basket for milk near me",
                "planner_actions": ["FIND_CHEAPEST_BASKET"],
                "entities": {"product": "Milk", "location": "CURRENT_LOCATION"}}]
    payload = client.post("/api/chat", json={
        "instruction": "What about butter?", "history": history,
        "location": {"latitude": 52.5219, "longitude": 13.4132},
    }).json()
    assert "FIND_CHEAPEST_BASKET" in payload["planner_actions"]
    assert payload["entity_product"] == "Butter"
    assert payload["entity_location"] == "CURRENT_LOCATION"
    assert offline[-1]["location"] == "52.52190,13.41320"

if __name__ == "__main__":
    tests = [
        test_health,
        test_offerhopper_action_mappings,
        test_chat_loyalty_barcode,
        test_chat_product_search,
        test_chat_fetch_mayonnaise_offerhopper,
        test_chat_find_doner_under_5_euros,
        test_chat_kebab_search_price,
        test_chat_navigation,
        test_chat_offerhopper_basket,
        test_chat_offerhopper_route_optimization,
        test_chat_product_comparison_uses_offerhopper,
        test_chat_multi_step_compound,
        test_chat_rag_payto_knowledge,
        test_chat_greeting_conversation,
        test_rag_retriever_pdf_and_web,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASSED: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__} -> {e}")

    print(f"\nTotal: {len(tests)}, Passed: {passed}, Failed: {len(tests) - passed}")


