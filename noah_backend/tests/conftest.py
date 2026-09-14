"""Keep the suite offline and deterministic.

OfferHopper rate-limits hard and the LLM reply is sampled, so by default every
test gets a realistic fake OfferHopper payload (the shape documented in
noah_flutter_integration_prompt.md §4) and the deterministic reply generator.
Set ``NOAH_LIVE_TESTS=1`` to run against the real services instead.
"""

import os

import pytest

import app.llm.response_generator as generator_module
import app.tools.offerhopper as offerhopper_module

LIVE = os.getenv("NOAH_LIVE_TESTS") == "1"

SHARE_URL = "https://offerhopper.ai/s/VcEZiHHo"
STORE = {
    "name": "REWE", "place_id": "osm:W1000656439",
    "latitude": 53.55127, "longitude": 9.99681,
    "formatted_address": "Ballindamm, 40, 20095, Hamburg",
}
_PRICES = {
    "milk": 0.99, "milch": 0.99, "eggs": 1.49, "eier": 1.49, "bread": 1.99,
    "brot": 1.99, "butter": 2.29, "mayonnaise": 1.89, "oat milk": 1.29,
    "almond milk": 1.79, "ice creams": 2.49, "ice cream": 2.49, "doner": 4.50,
    "kebab": 4.50, "sweets": 0.79, "chocolate": 0.89,
}
_NAMES = {
    "milk": "Weihenstephan Barista Milch 1l",
    "eggs": "REWE Beste Wahl Eier Freilandhaltung 4 Stück",
    "bread": "Harry Kürbiskernbrot 750g",
    "butter": "Kerrygold Original Irische Butter 250g",
}


def fake_offerhopper(items: str, location: str, **kwargs) -> dict:
    """A plausible ``plan_optimal_shopping_route`` result for `items`."""
    products = []
    for raw in items.split(","):
        item = raw.strip()
        if not item:
            continue
        key = item.lower()
        price = _PRICES.get(key, 1.50)
        products.append({
            "name": item,
            "selected_product": _NAMES.get(key, f"{item} (Ja!)"),
            "price": price, "regular_price": price, "discount_pct": 0,
            "quantity": 1.0, "unit": "pack", "total_cost": price,
            "market_average": round(price * 1.05, 3), "is_synthetic": False,
            "alternatives": [
                {"name": f"{item} (Gut & Günstig)", "price": round(price + 0.30, 2),
                 "regular_price": round(price + 0.30, 2), "discount_pct": 0},
            ],
        })
    subtotal = round(sum(p["price"] for p in products), 2)
    return {
        "success": True,
        "share_url": SHARE_URL,
        "ai_description": "Visit 1 stores for maximum coverage.",
        "total_estimated_cost": round(subtotal + 2.17, 2),
        "total_estimated_savings": 0.69,
        "optimized_route": {
            "estimated_total_cost": round(subtotal + 2.17, 2),
            "estimated_total_savings": 0.69,
            "total_distance_km": 3.504,
            "total_duration_minutes": 5.6,
            "total_time_with_shopping_minutes": 20.1,
            "stores": [STORE],
            "route_segments": [
                {"from_name": "user_location", "to_name": "REWE", "to_store": STORE,
                 "distance_km": 2.432, "duration_minutes": 3.42, "travel_mode": "car",
                 "estimated_cost_at_store": subtotal, "products_to_buy": products},
                {"from_name": "REWE", "to_name": "user_location", "products_to_buy": []},
            ],
        },
        "cost_analysis": {
            "product_cost": subtotal, "travel_cost": 2.17, "in_store_time_cost": 2.9,
            "verdict": {"headline": "worth_it", "overall_key": "good", "savings_eur": 0.69},
        },
    }


def _no_llm(*args, **kwargs):
    raise RuntimeError("LLM disabled in tests (set NOAH_LIVE_TESTS=1 for live calls)")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Stub OfferHopper and the LLM unless the live suite was requested."""
    if LIVE:
        yield None
        return
    calls = []

    def stub(**kwargs):
        calls.append(kwargs)
        return fake_offerhopper(**kwargs)

    monkeypatch.setattr(offerhopper_module, "call_offerhopper_mcp", stub)
    monkeypatch.setattr(generator_module, "generate_text", _no_llm)
    yield calls
