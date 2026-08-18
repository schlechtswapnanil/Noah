"""Noah natural-language response generation layer.

Produces short, user-friendly responses grounded in the PayTo action plan
and retrieved knowledge documents (PDFs + payto.one/FAQ).
"""

import json
import logging
import re
from typing import List, Optional

from .provider import generate_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Noah, the AI assistant inside the PayTo app.

Your goal is to communicate the result of PayTo's structured action plan to the user in a short, natural, and helpful sentence (1-2 sentences maximum).

STRICT RULES:
1. Never invent products, prices, discounts, stores, loyalty cards, or actions not present in the plan or context.
2. Treat the supplied PAYTO ACTION PLAN and LIVE OFFERHOPPER RESULTS as authoritative.
3. If LIVE OFFERHOPPER RESULTS with items and prices are provided, inform the user about the found products, prices, and stores (e.g. "I found mayonnaise at dm for €1.85.").
4. If the plan contains sequential actions, acknowledge them seamlessly (e.g., "Found mayonnaise at dm for €1.85 and opening your Payback card.").
5. If DOCUMENT CONTEXT is provided for a knowledge question, answer accurately and concisely using that context.
6. Keep responses direct, friendly, and natural. Avoid raw JSON or robot-like jargon.
"""


def _generate_fallback_response(
    instruction: str,
    plan: dict,
    rag_context: Optional[List[str]] = None,
    requires_rag: bool = False
) -> str:
    """Intelligent, deterministic rule-based fallback if the LLM is unavailable."""
    actions = plan.get("planner_actions", [])
    entities = plan.get("entities", {})
    workflow_type = plan.get("workflow_type", "single")

    product = entities.get("product")
    merchant = entities.get("merchant")
    brand = entities.get("brand") or entities.get("loyalty_card")
    price_max = entities.get("price_max")
    location = entities.get("location")

    # Helper formatters
    price_str = f" under €{price_max:g}" if price_max is not None else ""
    loc_str = " near you" if location == "CURRENT_LOCATION" else (f" in {location}" if location else "")
    merchant_str = f" at {merchant.title()}" if merchant else ""

    # Knowledge / RAG fallback
    if requires_rag:
        if rag_context:
            # Clean first chunk prefix
            first_chunk = rag_context[0]
            clean_text = re.sub(r"^\[.*?\]:\s*", "", first_chunk).strip()
            # Return first sentence or clean excerpt
            sentences = re.split(r"(?<=[.!?])\s+", clean_text)
            return sentences[0] if sentences else clean_text[:200]
        return "I couldn't find specific documentation on that topic in PayTo."

    # Conversational greetings / no actions
    if not actions:
        lowered = instruction.lower()
        if any(g in lowered for g in ["hi", "hello", "hey", "who are you"]):
            return "Hi! I'm Noah, your PayTo shopping and loyalty assistant. How can I help you today?"
        if any(t in lowered for t in ["thank", "thanks", "danke"]):
            return "You're welcome! Let me know if you need anything else."
        if any(b in lowered for b in ["bye", "goodbye", "tschüss"]):
            return "Goodbye! Happy shopping with PayTo."
        return "I'm ready to help you with offers, loyalty cards, product search, or basket optimization."

    # Live OfferHopper results fallback
    offerhopper_data = plan.get("offerhopperData")
    if offerhopper_data and isinstance(offerhopper_data, dict):
        from ..tools.offerhopper import format_offerhopper_summary
        oh_summary = format_offerhopper_summary(offerhopper_data)
        if oh_summary:
            remaining_actions = [
                a for a in actions
                if a not in (
                    "SEARCH_PRODUCT", "SEARCH_PRODUCT_BY_PRICE", "SEARCH_PRODUCT_BY_BRAND",
                    "SEARCH_PRODUCT_BY_CATEGORY", "SEARCH_OFFERS", "FIND_CHEAPEST_BASKET",
                    "OPTIMIZE_SHOPPING_ROUTE", "SPLIT_BASKET_ACROSS_MERCHANTS", "COMPARE_PRODUCTS"
                )
            ]
            if remaining_actions:
                other_phrases = []
                for act in remaining_actions:
                    if act in ("DISPLAY_BARCODE", "SHOW_BARCODE"):
                        target = brand.title() if brand else (merchant.title() if merchant else "loyalty")
                        other_phrases.append(f"displaying your {target} barcode")
                    elif act in ("OPEN_WALLET_CARD", "OPEN_CARD"):
                        target = brand.title() if brand else (merchant.title() if merchant else "loyalty")
                        other_phrases.append(f"opening your {target} card")
                    elif act in ("PLAN_ROUTE", "GET_DIRECTIONS", "OPEN_GOOGLE_MAPS"):
                        dest = merchant.title() if merchant else (location if location and location != "CURRENT_LOCATION" else "the store")
                        other_phrases.append(f"opening directions to {dest}")
                if other_phrases:
                    return f"{oh_summary} I'm also {', and '.join(other_phrases)}."
            return oh_summary

    # Multi-step sequential action phrases
    action_descriptions = []
    for act in actions:
        if act in ("DISPLAY_BARCODE", "SHOW_BARCODE"):
            target = brand.title() if brand else (merchant.title() if merchant else "loyalty")
            action_descriptions.append(f"displaying your {target} barcode")
        elif act in ("OPEN_WALLET_CARD", "OPEN_CARD"):
            target = brand.title() if brand else (merchant.title() if merchant else "loyalty")
            action_descriptions.append(f"opening your {target} card")
        elif act == "ADD_LOYALTY_CARD":
            target = brand.title() if brand else (merchant.title() if merchant else "loyalty")
            action_descriptions.append(f"adding your {target} card to your wallet")
        elif act == "DISPLAY_POINTS":
            target = brand.title() if brand else (merchant.title() if merchant else "loyalty")
            action_descriptions.append(f"checking your {target} points")
        elif act in ("SEARCH_PRODUCT", "SEARCH_PRODUCT_BY_PRICE", "SEARCH_PRODUCT_BY_BRAND", "SEARCH_PRODUCT_BY_CATEGORY"):
            prod_name = product.lower() if product else "products"
            action_descriptions.append(f"searching for {prod_name}{price_str}{merchant_str}{loc_str}")
        elif act in ("SEARCH_OFFERS", "SEARCH_DISCOUNTS", "OPEN_WEEKLY_FLYER"):
            prod_name = f" for {product.lower()}" if product else ""
            action_descriptions.append(f"checking the latest offers{prod_name}{merchant_str}")
        elif act in ("PLAN_ROUTE", "GET_DIRECTIONS", "OPEN_GOOGLE_MAPS"):
            dest = merchant.title() if merchant else (location if location and location != "CURRENT_LOCATION" else "your destination")
            action_descriptions.append(f"opening directions to {dest}")
        elif act == "FIND_CHEAPEST_BASKET":
            prod_name = f" for {product.lower()}" if product else " for your grocery items"
            action_descriptions.append(f"finding the cheapest basket{prod_name}{loc_str}")
        elif act == "OPTIMIZE_SHOPPING_ROUTE":
            action_descriptions.append(f"optimizing the best shopping route{loc_str}")
        elif act == "SPLIT_BASKET_ACROSS_MERCHANTS":
            action_descriptions.append("distributing your basket across stores for maximum savings")
        elif act == "COMPARE_PRODUCTS":
            prod_name = f" for {product.lower()}" if product else ""
            action_descriptions.append(f"comparing product options and prices{prod_name}")
        elif act == "GET_OPENING_HOURS":
            dest = merchant.title() if merchant else "the store"
            action_descriptions.append(f"checking opening hours for {dest}{loc_str}")
        elif act == "SHOW_PURCHASE_HISTORY":
            action_descriptions.append("pulling up your purchase history")
        elif act == "SHOW_PROFILE":
            action_descriptions.append("opening your PayTo profile")
        elif act == "SHOW_HELP":
            action_descriptions.append("opening the PayTo help center")
        else:
            clean_act = act.replace("_", " ").lower()
            action_descriptions.append(f"processing {clean_act}")

    if len(action_descriptions) == 1:
        phrase = action_descriptions[0]
        # Capitalize first letter and make into a natural sentence
        return f"I'm {phrase}."
    elif len(action_descriptions) == 2:
        return f"I'm {action_descriptions[0]}, and then {action_descriptions[1]}."
    else:
        return f"I'm {', '.join(action_descriptions[:-1])}, and then {action_descriptions[-1]}."


def generate_response(
    instruction: str,
    plan: dict,
    rag_context: Optional[List[str]] = None,
    requires_rag: bool = False
) -> str:
    """Generate a high-quality natural language string for the user."""
    rag_context = rag_context or []

    # If RAG is required and no context was found, provide clear message
    if requires_rag and not rag_context:
        return "I couldn't find specific documentation on that topic in PayTo."

    doc_section = (
        "\nDOCUMENT CONTEXT:\n" + "\n".join(rag_context)
        if rag_context
        else "No document context supplied."
    )

    offerhopper_data = plan.get("offerhopperData")
    oh_section = ""
    if offerhopper_data and isinstance(offerhopper_data, dict):
        from ..tools.offerhopper import format_offerhopper_summary
        oh_summary = format_offerhopper_summary(offerhopper_data)
        if oh_summary:
            oh_section = f"\nLIVE OFFERHOPPER STORE & PRICE RESULTS:\n{oh_summary}\n"

    user_prompt = f"""USER REQUEST:
{instruction}

PAYTO ACTION PLAN:
{json.dumps(plan, indent=2, ensure_ascii=False)}
{oh_section}{doc_section}

Generate Noah's short, natural response to the user mentioning the found prices/stores if available.
"""

    try:
        response = generate_text(SYSTEM_PROMPT, user_prompt)
        if response:
            return response
    except Exception:
        logger.exception("LLM response generation failed; using intelligent fallback.")

    return _generate_fallback_response(
        instruction=instruction,
        plan=plan,
        rag_context=rag_context,
        requires_rag=requires_rag
    )

