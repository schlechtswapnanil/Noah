from .entity_extractor import extract_entities, KNOWN_PRODUCTS
from ..planner.action_registry import ACTION_TO_TOOL


BOOLEAN_FIELDS = {
    "requires_memory",
    "requires_rag",
    "requires_recommendation",
}


def classify(
    instruction: str,
    models: dict
):
    lowered = instruction.lower()
    result = {}

    # --------------------------------------------------------
    # SVM predictions
    # --------------------------------------------------------

    for field, model in models.items():

        prediction = model.predict(
            [instruction]
        )[0]

        if field in BOOLEAN_FIELDS:

            prediction = (
                str(prediction).lower()
                == "true"
            )

        elif prediction == "NULL":

            prediction = None

        result[field] = prediction

    # --------------------------------------------------------
    # Entity extraction
    # --------------------------------------------------------

    result["entities"] = extract_entities(
        instruction
    )

    # --------------------------------------------------------
    # Planner actions
    # --------------------------------------------------------

    raw_actions = result.get(
        "planner_actions"
    )

    if isinstance(raw_actions, str):

        actions = [
            x.strip()
            for x in raw_actions.split("|")
            if x.strip()
        ]

    else:
        actions = []

    # Preserve an explicit final navigation request even when the sequence SVM
    # predicts only its product-search prefix (a sparse multi-step phrasing).
    navigation_terms = ("navigate", "directions", "take me there", "open the map", "open maps")
    navigation_actions = {"OPEN_GOOGLE_MAPS", "GET_DIRECTIONS", "PLAN_ROUTE"}
    if (actions and any(term in lowered for term in navigation_terms)
            and not navigation_actions.intersection(actions)):
        actions.append("OPEN_GOOGLE_MAPS")
        result["workflow_type"] = "sequential"

    # Basket optimisation needs a different data source from a single-product
    # search. The supervised model can classify phrases such as "cheapest way
    # to buy" as SEARCH_PRODUCT, so prefer the basket action when the request
    # explicitly asks for the lowest total across a multi-item grocery list.
    basket_optimisation_terms = (
        "cheapest basket",
        "cheapest way to buy",
        "cheapest way to get",
        "lowest total",
        "lowest basket cost",
        "most cost-effective",
        "optimal shopping route",
        "split my grocery basket",
        "split my basket",
    )
    grocery_item_markers = ("milk", "bread", "butter", "eggs", "coffee", "pasta", "olive oil", "salmon")
    has_multi_item_basket = sum(marker in lowered for marker in grocery_item_markers) >= 2
    
    # German merchants for exclusion
    german_merchants = ("rewe", "netto", "lidl", "aldi", "kaufland", "edeka", "penny", "müller", "mueller", "dm", "rossmann", "globus", "hit")
    is_offers_on_grocery = (
        ("offers on" in lowered or "offers for" in lowered) 
        and any(item in lowered for item in KNOWN_PRODUCTS + list(grocery_item_markers))
        and not any(merchant in lowered for merchant in german_merchants)
    )
    
    if any(term in lowered for term in basket_optimisation_terms) or is_offers_on_grocery:
        if "route" in lowered:
            actions = ["OPTIMIZE_SHOPPING_ROUTE"]
        elif "split" in lowered:
            actions = ["SPLIT_BASKET_ACROSS_MERCHANTS"]
        elif has_multi_item_basket or "basket" in lowered or is_offers_on_grocery:
            actions = ["FIND_CHEAPEST_BASKET"]
        result["workflow_type"] = "single"

    # Preserve explicit sequential clauses even when a sparse multi-step class
    # is not selected by the supervised model.
    if "then" in lowered:
        if any(card in lowered for card in ("netto plus", "lidl plus", "rewe bonus", "edeka card", "payback", "deutschlandcard")) and "OPEN_WALLET_CARD" not in actions:
            actions.append("OPEN_WALLET_CARD")
        if any(term in lowered for term in ("route", "navigate", "directions")) and "OPEN_GOOGLE_MAPS" not in actions:
            actions.append("OPEN_GOOGLE_MAPS")
        if len(actions) > 1:
            result["workflow_type"] = "sequential"

    # Ensure PayTo / App information & FAQ questions route to RAG
    if any(phrase in lowered for phrase in ("what is payto", "how does payto", "does payto", "about payto", "payto faq", "privacy policy", "payto store", "who created payto")):
        result["requires_rag"] = True
        result["domain"] = "KNOWLEDGE"
        result["intent"] = "PAYTO_INFORMATION"
        result["sub_intent"] = "PAYTO_INFORMATION"
        result["tool"] = "rag_engine"
        actions = ["ANSWER_PAYTO_QUESTION"]
        result["workflow_type"] = "single"

    result["planner_actions"] = actions
    result["planner_action_count"] = len(actions)

    # --------------------------------------------------------
    # Tool sequence (synchronized with action registry)
    # --------------------------------------------------------
    if actions:
        result["tool_sequence"] = [
            ACTION_TO_TOOL.get(action, result.get("tool", "none"))
            for action in actions
        ]
    else:
        result["tool_sequence"] = []

    return result
