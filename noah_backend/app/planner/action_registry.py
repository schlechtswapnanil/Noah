"""The action vocabulary and tool mapping used by noah_dataset_20k_final.csv."""

# Offerhopper exposes one MCP tool, ``plan_optimal_shopping_route``: give it a
# shopping list and a German starting location and it returns the cheapest
# basket, the store(s), the route and - per item - three alternatives. It is
# the *only* live product and price source the backend has, so anything that
# has to name a real product or price goes through it, including "recommend
# me something": there is no separate recommendation engine to call.
OFFERHOPPER_MCP_TOOL = "offerhopper_mcp"

ACTION_TO_TOOL = {
    "ADD_LOYALTY_CARD": "wallet_tool", "ANSWER_FAQ": "rag_engine",
    "ANSWER_PAYTO_QUESTION": "rag_engine", "BUILD_SHOPPING_LIST": "planner",
    "CHECK_PRODUCT_AVAILABILITY": OFFERHOPPER_MCP_TOOL, "COMPARE_PRODUCTS": OFFERHOPPER_MCP_TOOL,
    "DISPLAY_BARCODE": "wallet_tool", "DISPLAY_POINTS": "wallet_tool",
    "DISPLAY_REWARDS": "wallet_tool", "FIND_CHEAPEST_BASKET": OFFERHOPPER_MCP_TOOL,
    "GET_DIRECTIONS": "maps_tool", "GET_MERCHANT_CONTACT": "merchant_api",
    "GET_MERCHANT_DETAILS": "merchant_api", "GET_OPENING_HOURS": "merchant_api",
    "GET_PERSONALIZED_RECOMMENDATIONS": OFFERHOPPER_MCP_TOOL,
    "LIST_WALLET_CARDS": "wallet_tool", "OPEN_GOOGLE_MAPS": "maps_tool",
    "OPEN_SETTINGS": "planner", "OPEN_WALLET_CARD": "wallet_tool",
    "OPEN_WEEKLY_FLYER": "offers_tool", "OPTIMIZE_SHOPPING_ROUTE": OFFERHOPPER_MCP_TOOL,
    "PLAN_ROUTE": "maps_tool", "PROVIDE_APP_HELP": "rag_engine",
    "RECOMMEND_MERCHANTS": "recommendation_engine", "RECOMMEND_OFFERS": OFFERHOPPER_MCP_TOOL,
    "RECOMMEND_PRODUCTS": OFFERHOPPER_MCP_TOOL, "REMOVE_LOYALTY_CARD": "wallet_tool",
    "REPORT_BUG": "planner", "SEARCH_CASHBACK": "offers_tool",
    "SEARCH_DISCOUNTS": "offers_tool", "SEARCH_MERCHANT": "merchant_api",
    "SEARCH_NEARBY_MERCHANTS": "merchant_api", "SEARCH_OFFERS": OFFERHOPPER_MCP_TOOL,
    "SEARCH_PRODUCT": OFFERHOPPER_MCP_TOOL, "SEARCH_PRODUCT_BY_BRAND": OFFERHOPPER_MCP_TOOL,
    "SEARCH_PRODUCT_BY_CATEGORY": OFFERHOPPER_MCP_TOOL, "SEARCH_PRODUCT_BY_PRICE": OFFERHOPPER_MCP_TOOL,
    "SHOW_HELP": "planner", "SHOW_PROFILE": "firestore",
    "SHOW_PURCHASE_HISTORY": "firestore", "SHOW_VISIT_HISTORY": "firestore",
    "SPLIT_BASKET_ACROSS_MERCHANTS": OFFERHOPPER_MCP_TOOL, "SUBMIT_FEEDBACK": "planner",
}

AVAILABLE_ACTIONS = {
    action: {"tool": tool, "description": action.replace("_", " ").title()}
    for action, tool in ACTION_TO_TOOL.items()
}


def is_valid_action(action: str) -> bool:
    return action in AVAILABLE_ACTIONS


def get_action(action: str):
    return AVAILABLE_ACTIONS.get(action)
