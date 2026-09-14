"""Unit tests for the follow-up resolver (app/nlp/follow_up.py).

Every row of the rewrite table in noah_backend_context_prompt.md §1, plus the
carry-over rules: an intervening navigation turn does not lose the products,
a merchant from a maps turn is not mistaken for a loyalty card, failed and
empty turns are skipped, and fresh messages pass through untouched.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.nlp.follow_up import (MAX_RESULTS, MAX_TEXT, MAX_TURNS, Fresh, ResultQuestion,
                               Rewritten, Unresolved, build_context, resolve_follow_up)

RESULTS = [
    {"name": "Weihenstephan Barista Milch 1l", "store": "REWE", "price": 0.99},
    {"name": "REWE Beste Wahl Eier Freilandhaltung 4 Stück", "store": "REWE", "price": 1.49},
    {"name": "Harry Kürbiskernbrot 750g", "store": "REWE", "price": 1.99},
]
STORES = [{"name": "REWE", "address": "Ballindamm, 40, 20095, Hamburg",
           "latitude": 53.55127, "longitude": 9.99681}]

BASKET_HAMBURG = {
    "instruction": "Find the cheapest basket for milk, eggs and bread in Hamburg",
    "response": "REWE on Ballindamm has the cheapest basket … https://offerhopper.ai/s/VcEZiHHo",
    "planner_actions": ["FIND_CHEAPEST_BASKET"],
    "entities": {"product": "Milk, Eggs, Bread", "location": "Hamburg"},
    "results": RESULTS, "share_url": "https://offerhopper.ai/s/VcEZiHHo", "stores": STORES,
}
BASKET_NEAR_ME = {
    **BASKET_HAMBURG,
    "instruction": "Find the cheapest basket for milk, eggs and bread near me",
    "entities": {"product": "Milk, Eggs, Bread", "location": "CURRENT_LOCATION"},
}
PRICE_SEARCH_DE = {
    "instruction": "Wo gibt es Schokolade unter 2 Euro in meiner Nähe?",
    "planner_actions": ["SEARCH_PRODUCT_BY_PRICE"],
    "entities": {"product": "Schokolade", "price_max": 2.0, "location": "CURRENT_LOCATION"},
    "results": [{"name": "Milka Alpenmilch 100g", "store": "Penny", "price": 0.89}],
}
PLAIN_SEARCH = {
    "instruction": "Find the cheapest milk near me",
    "planner_actions": ["SEARCH_PRODUCT"],
    "entities": {"product": "Milk", "location": "CURRENT_LOCATION"},
}
WALLET_TURN = {
    "instruction": "Show me my Payback barcode.",
    "planner_actions": ["DISPLAY_BARCODE"],
    "entities": {"loyalty_card": "PAYBACK", "brand": "PAYBACK", "category": "LOYALTY_CARD"},
}
NAVIGATION_TURN = {
    "instruction": "Take me there",
    "planner_actions": ["PLAN_ROUTE"],
    "entities": {"merchant": "REWE", "location": "Hamburg"},
}


def resolve(text, history):
    return resolve_follow_up(text, build_context(history))


# --------------------------------------------------------------------------
# The §1 table
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text, history, expected", [
    ("take me there", [BASKET_HAMBURG], "Take me to REWE in Hamburg"),
    ("navigate there", [BASKET_HAMBURG], "Take me to REWE in Hamburg"),
    ("directions", [BASKET_HAMBURG], "Take me to REWE in Hamburg"),
    ("Take me there", [BASKET_NEAR_ME], "Take me to the nearest REWE"),
    ("bring mich dorthin", [BASKET_HAMBURG], "Navigiere mich zu REWE in Hamburg"),
    ("Bring mich dorthin", [BASKET_NEAR_ME], "Bring mich zum nächsten REWE"),
    ("when does it open?", [BASKET_HAMBURG], "When does REWE in Hamburg open?"),
    ("wann öffnet es?", [BASKET_HAMBURG], "Wann hat REWE in Hamburg geöffnet?"),
    ("add them to my shopping list", [BASKET_HAMBURG], "Add milk, eggs and bread to my shopping list"),
    ("setz sie auf meine Liste", [BASKET_HAMBURG], "Setze Milch, Eier und Brot auf meine Einkaufsliste"),
    ("what about butter?", [BASKET_HAMBURG], "Find the cheapest basket for butter in Hamburg"),
    ("und Eis?", [PRICE_SEARCH_DE], "Wo gibt es Eis unter 2 Euro in meiner Nähe?"),
    ("what about milk?", [PLAIN_SEARCH], "Where can I buy milk near me?"),
    ("show the barcode", [WALLET_TURN], "Show my Payback barcode"),
    ("Zeig mir die Karte", [WALLET_TURN], "Zeig mir meine Payback Karte"),
])
def test_table_rows_are_rewritten(text, history, expected):
    resolved = resolve(text, history)
    assert isinstance(resolved, Rewritten), resolved
    assert resolved.text == expected


@pytest.mark.parametrize("text", [
    "which one is cheapest?", "is it worth the trip?", "what's the total?",
    "show me alternatives", "cheapest?", "lohnt sich die Fahrt?", "Was ist mit dem Gesamtpreis?",
])
def test_result_questions_are_answered_from_context(text):
    resolved = resolve(text, [BASKET_HAMBURG])
    assert isinstance(resolved, ResultQuestion), resolved


def test_result_question_language_follows_the_message_then_the_last_turn():
    assert resolve("lohnt sich die Fahrt?", [BASKET_HAMBURG]).language == "de"
    assert resolve("which one is cheapest?", [BASKET_HAMBURG]).language == "en"
    # no marker at all: the previous turn decides
    assert resolve("cheapest?", [PRICE_SEARCH_DE]).language == "de"
    assert resolve("cheapest?", [BASKET_HAMBURG]).language == "en"


# --------------------------------------------------------------------------
# Carry-over rules
# --------------------------------------------------------------------------

def test_products_survive_an_intervening_navigation_turn():
    resolved = resolve("add them to my shopping list", [BASKET_HAMBURG, NAVIGATION_TURN])
    assert isinstance(resolved, Rewritten)
    assert resolved.text == "Add milk, eggs and bread to my shopping list"


def test_newest_merchant_and_location_win():
    lidl = {"instruction": "Take me to Lidl in Berlin", "planner_actions": ["PLAN_ROUTE"],
            "entities": {"merchant": "LIDL", "location": "Berlin"}}
    context = build_context([BASKET_HAMBURG, lidl])
    assert context.merchant == "LIDL" and context.location == "Berlin"
    assert resolve("when does it open?", [BASKET_HAMBURG, lidl]).text == "When does LIDL in Berlin open?"


def test_merchant_comes_from_stores_when_entities_have_none():
    context = build_context([BASKET_HAMBURG])
    assert context.merchant == "REWE"


def test_merchant_from_a_maps_turn_is_not_a_card():
    context = build_context([NAVIGATION_TURN])
    assert context.loyalty_card is None
    # so "show the barcode" cannot be resolved to a REWE Bonus barcode
    assert isinstance(resolve("show the barcode", [NAVIGATION_TURN]), Fresh)


def test_merchant_from_a_wallet_turn_names_the_card():
    lidl_card = {"instruction": "Show my Lidl card", "planner_actions": ["OPEN_WALLET_CARD"],
                 "entities": {"merchant": "LIDL"}}
    assert build_context([lidl_card]).loyalty_card == "LIDL_PLUS"
    assert resolve("show the barcode", [lidl_card]).text == "Show my Lidl Plus barcode"


def test_failed_and_empty_turns_are_skipped():
    declined = {"instruction": "asdkjhasd", "planner_actions": [], "entities": {"product": "Asdkjhasd"}}
    greeting = {"instruction": "hello", "planner_actions": []}
    failed_butter = {"instruction": "What about butter?", "planner_actions": ["FIND_CHEAPEST_BASKET"],
                     "entities": {"product": "Butter", "location": "Hamburg"}}   # no results: OfferHopper failed
    context = build_context([BASKET_HAMBURG, declined, greeting, failed_butter])
    assert context.products == "Butter"            # a failed search still says what was asked for
    assert context.results == RESULTS              # ...but the results carried are the last real ones
    assert context.merchant == "REWE"
    assert isinstance(resolve("which one is cheapest?", [BASKET_HAMBURG, declined, failed_butter]),
                      ResultQuestion)


def test_current_location_is_not_carried_as_a_typed_place():
    context = build_context([BASKET_NEAR_ME])
    assert context.location is None


# --------------------------------------------------------------------------
# Fresh messages and unresolved follow-ups
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Take me to the nearest Lidl.",
    "Show my Payback card",
    "Find the cheapest basket for milk and bread in Hamburg",
    "Where can I buy frozen pizzas for less than €2 near me?",
    "I'm at Netto. What are the current offers on ice creams here below €3? Then display my Payback card.",
    "show me this weeks deals",
    "das war super vielen dank",
    "add lasagne to my list",
    "Hello! Who are you?",
    "plan my shopping trip in hannover",
    "put lidl on the map for me",
    "welcher supermarkt ist am nächsten",
    "How far is REWE?",
    "Zeig mir die Karte",   # with no wallet turn "Karte" is the map, as before
])
def test_fresh_messages_pass_through_untouched(text):
    for history in ([], [BASKET_HAMBURG], [NAVIGATION_TURN]):
        resolved = resolve(text, history)
        assert isinstance(resolved, Fresh), (text, history, resolved)
        assert resolved.text == text


@pytest.mark.parametrize("text, family", [
    ("Take me there", "navigation"),
    ("When does it open?", "hours"),
    ("Add them to my shopping list", "list"),
    ("What about butter?", "product"),
    ("Which one is cheapest?", "results"),
    ("bring mich dorthin", "navigation"),
    ("setz sie auf meine Liste", "list"),
])
def test_bare_follow_ups_without_history_are_unresolved(text, family):
    resolved = resolve(text, [])
    assert isinstance(resolved, Unresolved), resolved
    assert resolved.family == family
    assert resolved.text == text


def test_follow_up_needing_a_merchant_after_a_turn_without_one_is_unresolved():
    resolved = resolve("take me there", [PLAIN_SEARCH])
    assert isinstance(resolved, Unresolved) and resolved.family == "navigation"


# --------------------------------------------------------------------------
# Request limits: truncate, never reject
# --------------------------------------------------------------------------

def test_history_is_capped_at_six_turns_newest_kept():
    old = {"instruction": "Take me to Lidl in Berlin", "planner_actions": ["PLAN_ROUTE"],
           "entities": {"merchant": "LIDL", "location": "Berlin"}}
    filler = [{"instruction": f"hello {i}", "planner_actions": []} for i in range(MAX_TURNS)]
    context = build_context([old] + filler)
    assert context.turns == MAX_TURNS
    assert context.merchant is None          # the Lidl turn fell off the front


def test_results_and_text_are_truncated():
    turn = {
        **BASKET_HAMBURG,
        "instruction": "x" * (MAX_TEXT + 500),
        "results": [{"name": f"item {i}", "store": "REWE", "price": 1.0 + i} for i in range(MAX_RESULTS + 5)],
    }
    context = build_context([turn])
    assert len(context.results) == MAX_RESULTS
    assert len(context.last_instruction) == MAX_TEXT


def test_pydantic_models_are_accepted_as_history():
    from app.api.chat import HistoryTurn
    turns = [HistoryTurn(**BASKET_HAMBURG)]
    assert resolve("take me there", turns).text == "Take me to REWE in Hamburg"


def test_results_block_lists_prices_cheapest_total_and_link():
    block = build_context([BASKET_HAMBURG]).results_block()
    assert "Weihenstephan Barista Milch 1l at REWE for €0.99" in block
    assert "Cheapest item: Weihenstephan Barista Milch 1l at €0.99" in block
    assert "Total of the listed items: €4.47" in block
    assert block.rstrip().endswith("https://offerhopper.ai/s/VcEZiHHo")
