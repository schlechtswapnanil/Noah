"""Extract slot values that are literally present in the instruction.

The previous version matched the product with a single greedy regex over the
raw text, so anything that was not on a 40-item English word list swallowed the
rest of the sentence: "whats the cheapest place to get nappies and washing
powder around 10115" came back with
``product = "Nappies And Washing Powder Around 10115"`` and the reply read
"... for nappies and washing powder around 10115 in 10115."

Slots are now filled most-specific first and each match is masked out of the
working text, so by the time the product is read the prices, merchants, cards,
locations and command verbs are already gone.  German phrasing is handled
alongside English because the app ships in Germany.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

GERMAN_MERCHANTS = ["rewe", "netto", "lidl", "aldi", "kaufland", "edeka", "penny",
                    "müller", "mueller", "dm", "rossmann", "globus", "hit"]
LOYALTY_CARDS = {
    "netto plus": ("NETTO", None, "NETTO_PLUS"),
    "lidl plus": ("LIDL", None, "LIDL_PLUS"),
    "rewe bonus": ("REWE", None, "REWE_BONUS"),
    "edeka card": ("EDEKA", None, "EDEKA_CARD"),
    "edeka karte": ("EDEKA", None, "EDEKA_CARD"),
    "payback": (None, "PAYBACK", "PAYBACK"),
    "deutschlandcard": (None, "DeutschlandCard", "DEUTSCHLANDCARD"),
}

KNOWN_PRODUCTS = [
    # English
    "frozen pizza", "frozen pizzas", "ice creams", "ice cream", "vegan cheese",
    "oat milk", "almond milk", "olive oil", "pizza", "milk", "bread", "butter",
    "eggs", "coffee", "cereal", "pasta", "salmon", "chocolate", "apples",
    "bananas", "mayonnaise", "mayo", "ketchup", "mustard", "cheese", "yogurt",
    "yoghurt", "flour", "sugar", "rice", "toilet paper", "laundry detergent",
    "washing powder", "nappies", "diapers", "beer", "wine", "water",
    "minced meat", "chicken", "margarine", "orange juice", "crisps", "biscuits",
    "doner", "döner", "kebab", "döner kebab", "doner kebab", "shawarma",
    "falafel", "currywurst", "burger", "sandwich", "salad", "sushi",
    "croissant", "bagel",
    # German
    "hafermilch", "mandelmilch", "tiefkühlpizza", "tiefkuehlpizza", "milch",
    "brot", "butter", "eier", "kaffee", "nudeln", "käse", "kaese", "joghurt",
    "olivenöl", "olivenoel", "hackfleisch", "bananen", "äpfel", "aepfel",
    "schokolade", "mehl", "zucker", "reis", "toilettenpapier", "waschmittel",
    "windeln", "bier", "wein", "wasser", "hähnchen", "haehnchen", "margarine",
    "orangensaft", "chips", "kekse", "sahne", "quark", "wurst", "schinken",
]
# Longest first so "frozen pizza" wins over "pizza" and "hafermilch" over "milch".
_PRODUCTS_BY_LENGTH = sorted(set(KNOWN_PRODUCTS), key=len, reverse=True)

# Words that describe the request rather than the goods.
_NON_PRODUCT_TERMS = {
    "barcode", "strichcode", "code", "card", "karte", "kart", "cards", "karten",
    "points", "punkte", "punktestand", "rewards", "prämien", "praemien",
    "wallet", "offers", "angebote", "deals", "rabatte", "discounts", "prospekt",
    "flyer", "profile", "profil", "account", "konto", "settings",
    "einstellungen", "history", "historie", "list", "liste", "route", "weg",
    "map", "karte", "store", "markt", "shop", "laden", "supermarkt", "filiale",
    "result", "ergebnis", "help", "hilfe", "balance", "guthaben", "stand",
    "way", "place", "thing", "something", "etwas", "alles", "nothing",
    "cheapest", "cheaper", "billigste", "billigsten", "günstigste",
    "guenstigste", "one", "ones", "eine", "eins", "andere", "second", "first",
    "zweite", "erste", "yes", "no", "ja", "nein", "ok", "okay",
}
_LEADING_STOPWORDS = re.compile(
    r"^(?:a|an|the|some|any|me|my|mir|mein|meine|meinen|meiner|ein|eine|einen|"
    r"einem|der|die|das|den|dem|noch|mal|bitte|please|schnell|günstig|guenstig|"
    r"günstige|guenstige|günstigen|guenstigen|billig|billige|billigen|cheap|"
    r"good|gute|guten)\s+",
    re.IGNORECASE,
)

# Command frames.  Everything up to and including the verb is not the product.
_COMMAND_PREFIX = re.compile(
    r"^.*?\b(?:find(?:\s+me)?|search(?:\s+for)?|look\s+for|fetch(?:\s+me)?|"
    r"get(?:\s+me)?|show(?:\s+me)?|buy|need|want|add|put|note\s+down|stick|"
    r"where\s+(?:can|do)\s+i\s+(?:buy|find|get)|is\s+there|do\s+they\s+have|"
    r"has\s+\w+\s+got|"
    r"suche?|such|finde?|zeig(?:e|en)?|brauche?|will|möchte|moechte|kaufen|"
    r"besorg(?:e|en)?|schreib|pack|notier|setz|hol|krieg(?:e)?|bekomme?|"
    r"wo\s+(?:kann|krieg|gibt|bekomme|finde|kaufe)\s*(?:ich|es)?)\b\s*",
    re.IGNORECASE,
)

_CURRENCY = r"(?:€|eur(?:os?)?|euros?)?"


def _number(value: str) -> float:
    return float(value.replace(",", "."))


def _mask(text: str, patterns: Iterable[str]) -> str:
    """Blank out already-consumed spans so later slots cannot re-read them."""
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _clean_product(value: str) -> Optional[str]:
    value = value.strip(" ,.!?-–—")
    previous = None
    while value and value != previous:
        previous = value
        value = _LEADING_STOPWORDS.sub("", value).strip(" ,.!?-–—")
    if not value or len(value) < 2:
        return None
    words = [w for w in re.split(r"\s+", value) if w]
    if not words or all(w.lower() in _NON_PRODUCT_TERMS for w in words):
        return None
    if len(words) > 6:
        return None
    return " ".join(words).title()


def extract_entities(instruction: str, planner_actions: Optional[list] = None) -> dict:
    """Return the slots literally present in `instruction`.

    `planner_actions` is optional context: a wallet or navigation plan has no
    product, so "Show me my Payback barcode." no longer reports
    ``product = "My Payback Barcode"``.
    """
    text = instruction.lower()
    entities = {
        "product": None, "merchant": None, "brand": None, "category": None,
        "price_min": None, "price_max": None, "location": None, "radius": None,
        "loyalty_card": None,
    }
    consumed: list[str] = []

    # ---------------------------------------------------------------- prices
    between = re.search(
        rf"(?:between|zwischen)\s*{_CURRENCY}\s*(\d+(?:[.,]\d+)?)\s*(?:and|-|und|bis)\s*"
        rf"{_CURRENCY}\s*(\d+(?:[.,]\d+)?)", text)
    if between:
        entities["price_min"] = _number(between.group(1))
        entities["price_max"] = _number(between.group(2))
        consumed.append(re.escape(between.group(0)))

    maximum = re.search(
        rf"(?:under|below|less\s+than|no\s+more\s+than|up\s+to|max(?:imum)?|"
        rf"unter|günstiger\s+als|guenstiger\s+als|weniger\s+als|bis\s+zu|"
        rf"höchstens|hoechstens|für\s+unter|fuer\s+unter)\s*{_CURRENCY}\s*"
        rf"(\d+(?:[.,]\d+)?)\s*{_CURRENCY}", text)
    if maximum:
        entities["price_max"] = _number(maximum.group(1))
        consumed.append(re.escape(maximum.group(0)))

    within_price = re.search(r"\bwithin\s*(\d+(?:[.,]\d+)?)\s*(?:€|euros?|eur)\b", text)
    if within_price:
        entities["price_max"] = _number(within_price.group(1))
        consumed.append(re.escape(within_price.group(0)))

    minimum = re.search(
        rf"(?:above|over|more\s+than|starting\s+(?:at|from)|über|ueber|ab)\s*"
        rf"{_CURRENCY}\s*(\d+(?:[.,]\d+)?)\s*{_CURRENCY}", text)
    if minimum:
        entities["price_min"] = _number(minimum.group(1))
        consumed.append(re.escape(minimum.group(0)))

    # Any remaining bare currency amount is a price mention, not a product.
    consumed.append(rf"\d+(?:[.,]\d+)?\s*(?:€|eur(?:os?)?|euros?|quid|cent)")
    consumed.append(rf"(?:€|eur)\s*\d+(?:[.,]\d+)?")

    # --------------------------------------------------------------- radius
    radius = re.search(
        r"(?:within|in|under|innerhalb\s+von|im\s+umkreis\s+von)?\s*"
        r"(\d+(?:[.,]\d+)?)\s*(km|m|kilometer|meter)\b", text)
    if radius:
        factor = 1000 if radius.group(2).startswith("k") else 1
        entities["radius"] = _number(radius.group(1)) * factor
        consumed.append(re.escape(radius.group(0)))

    # --------------------------------------------------- loyalty card, merchant
    working = _mask(text, consumed)
    for card, (merchant, brand, card_name) in LOYALTY_CARDS.items():
        if re.search(rf"\b{re.escape(card)}\b", working):
            entities.update(merchant=merchant, brand=brand,
                            category="LOYALTY_CARD", loyalty_card=card_name)
            working = _mask(working, [rf"\b{re.escape(card)}\b"])
            break

    if entities["merchant"] is None:
        for merchant in GERMAN_MERCHANTS:
            if re.search(rf"\b{merchant}\b", working):
                entities["merchant"] = merchant.upper()
                working = _mask(working, [rf"\b{merchant}\b"])
                break
    else:
        working = _mask(working, [rf"\b{m}\b" for m in GERMAN_MERCHANTS])

    # ------------------------------------------------------------- location
    postcode = re.search(r"\b\d{5}\b", instruction)
    near_me = (r"near\s+me|nearby|near\s+here|around\s+me|around\s+here|"
               r"close\s+to\s+me|closest|nearest|at\s+my\s+location|round\s+the\s+corner|"
               r"in\s+der\s+n(?:ä|ae)he|in\s+meiner\s+n(?:ä|ae)he|in\s+der\s+umgebung|"
               r"um\s+die\s+ecke|hier\s+in\s+der\s+n(?:ä|ae)he|bei\s+mir|"
               r"n(?:ä|ae)chste[nrm]?|hier")
    if postcode:
        entities["location"] = postcode.group(0)
        working = _mask(working, [r"\b\d{5}\b"])
    elif re.search(near_me, working) or re.search(r"\b(?:i'm|i am|im|ich bin)\s+(?:at|bei|in)\s+\w+", working):
        entities["location"] = "CURRENT_LOCATION"
    else:
        named = re.search(
            r"\b(?:in|near|around|nach|um)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß.'-]*"
            r"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.'-]*)?)(?:[,.!?]|\s|$)", instruction)
        if named:
            value = named.group(1).strip().strip(".,!?;:")
            if value and value.lower() not in {"me", "my", "der", "die", "das",
                                               "meiner", "the", "a", "our",
                                               "all", "here"}:
                entities["location"] = value.title()
    working = _mask(working, [near_me, r"\b(?:in|near|around|nach|um)\s+\w+\s*$"])

    # -------------------------------------------------------------- product
    wallet_or_navigation = {
        "DISPLAY_BARCODE", "OPEN_WALLET_CARD", "DISPLAY_POINTS", "DISPLAY_REWARDS",
        "ADD_LOYALTY_CARD", "REMOVE_LOYALTY_CARD", "LIST_WALLET_CARDS",
        "GET_DIRECTIONS", "PLAN_ROUTE", "OPEN_GOOGLE_MAPS", "SHOW_PROFILE",
        "OPEN_SETTINGS", "SHOW_PURCHASE_HISTORY", "SHOW_VISIT_HISTORY",
        "SHOW_HELP", "REPORT_BUG", "SUBMIT_FEEDBACK", "ANSWER_FAQ",
        "ANSWER_PAYTO_QUESTION", "PROVIDE_APP_HELP", "GET_OPENING_HOURS",
        "GET_MERCHANT_CONTACT", "GET_MERCHANT_DETAILS", "SEARCH_NEARBY_MERCHANTS",
        "SEARCH_MERCHANT",
    }
    # An empty plan means Noah is greeting the user or asking them to rephrase;
    # there is no product to read out of "yes" or "what's the weather tomorrow".
    if planner_actions is not None and not planner_actions:
        return entities
    if planner_actions and set(planner_actions).issubset(wallet_or_navigation):
        return entities

    found = []
    for product in _PRODUCTS_BY_LENGTH:
        match = re.search(rf"\b{re.escape(product)}\b", working)
        if match:
            found.append((match.start(), product))
    if found:
        # Keep the order the user said them, and drop any product that is a
        # substring of a longer match already taken ("milk" inside "oat milk").
        kept, covered = [], ""
        for _, product in sorted(found):
            if product not in covered:
                kept.append(product)
                covered += " " + product
        entities["product"] = ", ".join(p.title() for p in dict.fromkeys(kept))
        return entities

    remainder = _COMMAND_PREFIX.sub("", working, count=1)
    remainder = re.split(
        r"\b(?:in|at|near|bei|für|fuer|von|zu|and\s+then|und\s+dann|then|dann|,)\b",
        remainder, maxsplit=1)[0]
    entities["product"] = _clean_product(remainder)
    return entities
