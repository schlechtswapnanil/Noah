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


def test_rag_retriever_pdf_and_web():
    chunks = get_chunks()
    assert len(chunks) > 50
    # Test PDF chunk retrieval
    pdf_results = retrieve("hierarchical intent classification taxonomy Noah")
    assert len(pdf_results) > 0
    # Test Web FAQ chunk retrieval
    web_results = retrieve("Does Payto store my payment details?")
    assert len(web_results) > 0


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


