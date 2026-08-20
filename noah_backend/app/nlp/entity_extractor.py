import re

GERMAN_MERCHANTS = ["rewe", "netto", "lidl", "aldi", "kaufland", "edeka", "penny", "müller", "mueller", "dm", "rossmann", "globus", "hit"]
LOYALTY_CARDS = {
    "netto plus": ("NETTO", None, "NETTO_PLUS"),
    "lidl plus": ("LIDL", None, "LIDL_PLUS"),
    "rewe bonus": ("REWE", None, "REWE_BONUS"),
    "edeka card": ("EDEKA", None, "EDEKA_CARD"),
    "payback": (None, "PAYBACK", "PAYBACK"),
    "deutschlandcard": (None, "DeutschlandCard", "DEUTSCHLANDCARD"),
}
KNOWN_PRODUCTS = [
    "frozen pizza", "frozen pizzas", "ice creams", "ice cream", "vegan cheese", "oat milk",
    "almond milk", "olive oil", "pizza", "milk", "bread", "butter",
    "eggs", "coffee", "cereal", "pasta", "salmon", "chocolate", "apples", "bananas",
    "mayonnaise", "mayo", "ketchup", "mustard", "cheese", "yogurt",
    "doner", "döner", "kebab", "döner kebab", "doner kebab", "shawarma", "falafel",
    "currywurst", "burger", "sandwich", "salad", "sushi", "croissant", "bagel"
]



def _number(value: str) -> float:
    return float(value.replace(",", "."))


def extract_entities(instruction: str) -> dict:
    """Extract only textually present values; cards are not retailers."""
    text = instruction.lower()
    entities = {
        "product": None, "merchant": None, "brand": None,
        "category": None, "price_min": None, "price_max": None,
        "location": None, "radius": None, "loyalty_card": None
    }

    # 1. Prices
    currency = r"(?:€|eur(?:os?)?)?"
    between = re.search(rf"between\s*{currency}\s*(\d+(?:[.,]\d+)?)\s*(?:and|-)\s*{currency}\s*(\d+(?:[.,]\d+)?)", text)
    if between:
        entities["price_min"], entities["price_max"] = _number(between.group(1)), _number(between.group(2))
    
    maximum = re.search(rf"(?:under|below|less than|up to|max(?:imum)?)\s*{currency}\s*(\d+(?:[.,]\d+)?)", text)
    if maximum:
        entities["price_max"] = _number(maximum.group(1))

    within_price = re.search(r"\bwithin\s*(\d+(?:[.,]\d+)?)\s*(?:€|euros?|eur)\b", text)
    if within_price:
        entities["price_max"] = _number(within_price.group(1))

    minimum = re.search(rf"(?:above|over|more than|from)\s*{currency}\s*(\d+(?:[.,]\d+)?)", text)
    if minimum:
        entities["price_min"] = _number(minimum.group(1))

    # 2. Radius
    radius = re.search(r"(?:within|in|under)?\s*(\d+(?:[.,]\d+)?)\s*(km|m)\b", text)
    if radius:
        entities["radius"] = _number(radius.group(1)) * (1000 if radius.group(2) == "km" else 1)

    # 3. Location
    plz_match = re.search(r"\b\d{5}\b", instruction)
    if plz_match:
        entities["location"] = plz_match.group(0)
    elif any(term in text for term in ("near me", "nearby", "around me", "close to me", "at my location", "here")):
        entities["location"] = "CURRENT_LOCATION"
    elif re.search(r"\b(?:i'm|i am|im)\s+at\s+\w+", text):
        entities["location"] = "CURRENT_LOCATION"
    else:
        named_location = re.search(r"\b(?:in|near)\s+([A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß .'-]*?)(?:[,.!?]|$)", instruction)
        if named_location:
            value = named_location.group(1).strip()
            # Filter out non-location words
            if value and value.lower() not in {"me", "my location", "here", "the", "a", "our", "all"}:
                entities["location"] = value.title()

    # 4. Loyalty Cards & Merchants
    merchant_text = text
    for card, (merchant, brand, card_name) in LOYALTY_CARDS.items():
        if re.search(rf"\b{re.escape(card)}\b", text):
            entities.update(merchant=merchant, brand=brand, category="LOYALTY_CARD", loyalty_card=card_name)
            merchant_text = re.sub(rf"\b{re.escape(card)}\b", "", merchant_text)
            break

    if entities["merchant"] is None:
        for merchant in GERMAN_MERCHANTS:
            if re.search(rf"\b{merchant}\b", merchant_text):
                entities["merchant"] = merchant.upper()
                break

    # 5. Products & Grocery Baskets
    # Check for multi-item basket or compare patterns first
    basket_match = re.search(
        r"(?:basket\s+(?:for|with)|list\s+(?:for|with)|items?\s*:?|products?\s*:?|buy|prices?\s+of)\s+([a-zA-Z0-9\s,]+?)(?=\s+(?:in|at|near|under|below|within|to|for\s+less|\.|\?|$))",
        text
    )
    if basket_match:
        items_phrase = basket_match.group(1).strip(" ,.")
        # Filter out common stop phrases
        if items_phrase and not any(stop in items_phrase for stop in ["the cheapest", "lowest", "the lowest", "less than"]):
            entities["product"] = items_phrase.title()

    if entities["product"] is None:
        # Check known products list
        found_products = []
        for product in KNOWN_PRODUCTS:
            if re.search(rf"\b{re.escape(product)}\b", text):
                found_products.append(product.title())
        if found_products:
            # Join multiple products if found
            entities["product"] = ", ".join(found_products)

    if entities["product"] is None:
        generic_product = re.search(
            r"\b(?:find(?:\s+me)?|search(?:\s+for)?|look\s+for|fetch(?:\s+me)?|get(?:\s+me)?|show(?:\s+me)?|where\s+can\s+i\s+(?:buy|find|get))\s+(.+?)"
            r"(?=\s+(?:in|near|under|below|within|for\s+less|at)\b|\s*(?:,|and\s+navigate\b|$))",
            text,
        )
        if generic_product:
            value = generic_product.group(1).strip(" ,.")
            value = re.sub(r"^(?:a|an|the|some)\s+", "", value, flags=re.IGNORECASE).strip(" ,.")
            if value and value.lower() not in {"the", "a", "an", "some"}:
                entities["product"] = value.title()

    return entities

