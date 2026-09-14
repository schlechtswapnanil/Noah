"""Turn a follow-up into the self-contained sentence the user meant.

Every ``POST /api/chat`` used to be independent, so "take me there" after a
basket search re-ran the basket search (the classifier follows whatever
words it can see) and "take me there (REWE, Hamburg)" fell under the
confidence floor.  What *does* route is the sentence the user meant -
"Take me to REWE in Hamburg" - so that is what this module produces.

Two steps, both deterministic:

* :func:`build_context` folds the client's ``history`` (oldest first) into a
  :class:`ConversationContext`: the merchant, products, typed location,
  loyalty card and results carried forward, newest value winning.
* :func:`resolve_follow_up` classifies the new message as
  :class:`Fresh` (stands alone - classify it as today),
  :class:`Rewritten` (an anaphoric or bare follow-up turned into one of a
  fixed set of phrasings the route model is known to accept),
  :class:`ResultQuestion` (a question about what was just shown, answered
  from the carried results without the classifier or OfferHopper), or
  :class:`Unresolved` (a follow-up frame with nothing in the context to fill
  it - "take me there" with no history - which gets a clarifying question
  rather than a merchant-less navigation action).

The rewrite phrasings are the ones checked against the deployed classifier
one by one (see noah_backend_context_prompt.md §1) plus the rows added to
``dataset/noah_dataset_v3.csv`` for them.  Do not add phrasings here without
adding them there.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Union

from ..llm.response_generator import CARD_DISPLAY_NAMES, language_of
from .entity_extractor import KNOWN_PRODUCTS, extract_entities

logger = logging.getLogger(__name__)

# Request-side limits: reject nothing, truncate.
MAX_TURNS = 6
MAX_RESULTS = 10
MAX_TEXT = 2000

BASKET_ACTIONS = {"FIND_CHEAPEST_BASKET", "OPTIMIZE_SHOPPING_ROUTE",
                  "SPLIT_BASKET_ACROSS_MERCHANTS"}
PRICE_ACTIONS = {"SEARCH_PRODUCT_BY_PRICE"}
PRODUCT_ACTIONS = BASKET_ACTIONS | PRICE_ACTIONS | {
    "SEARCH_PRODUCT", "SEARCH_PRODUCT_BY_BRAND", "SEARCH_PRODUCT_BY_CATEGORY",
    "SEARCH_OFFERS", "COMPARE_PRODUCTS", "CHECK_PRODUCT_AVAILABILITY",
    "RECOMMEND_PRODUCTS", "RECOMMEND_OFFERS", "GET_PERSONALIZED_RECOMMENDATIONS",
}
WALLET_ACTIONS = {"DISPLAY_BARCODE", "OPEN_WALLET_CARD", "DISPLAY_POINTS",
                  "DISPLAY_REWARDS", "ADD_LOYALTY_CARD", "REMOVE_LOYALTY_CARD",
                  "LIST_WALLET_CARDS"}
CARD_CODES = set(CARD_DISPLAY_NAMES)
# A wallet turn that only named the merchant ("show my Lidl card") still tells
# us which card was meant.
MERCHANT_CARD = {"LIDL": "LIDL_PLUS", "NETTO": "NETTO_PLUS", "REWE": "REWE_BONUS",
                 "EDEKA": "EDEKA_CARD"}


# ---------------------------------------------------------------------------
# Carried context
# ---------------------------------------------------------------------------

@dataclass
class ConversationContext:
    """What the previous turns established, newest value winning."""

    merchant: Optional[str] = None
    products: Optional[str] = None        # "Milk, Eggs, Bread", as the extractor wrote it
    product_kind: Optional[str] = None    # "basket" | "price" | "search"
    price_max: Optional[float] = None
    location: Optional[str] = None        # a typed city or PLZ, never CURRENT_LOCATION
    loyalty_card: Optional[str] = None    # PAYBACK, LIDL_PLUS, ...
    results: List[dict] = field(default_factory=list)
    stores: List[dict] = field(default_factory=list)
    share_url: Optional[str] = None
    last_instruction: Optional[str] = None
    turns: int = 0

    @property
    def empty(self) -> bool:
        return self.turns == 0

    @property
    def language(self) -> Optional[str]:
        return language_of(self.last_instruction) if self.last_instruction else None

    def entities(self) -> dict:
        """The entity slots a result question inherits from the context."""
        return {
            "product": self.products, "merchant": self.merchant, "brand": None,
            "category": None, "price_min": None, "price_max": self.price_max,
            "location": self.location, "radius": None, "loyalty_card": None,
        }

    def results_block(self) -> str:
        """Compact view of the carried results, in the style the LLM already
        gets for live OfferHopper data (``_offerhopper_context``)."""
        if not self.results:
            return ""
        lines: List[str] = []
        for store in self.stores:
            name = store.get("name")
            if name:
                address = store.get("address")
                lines.append(f"Store: {name}" + (f" ({address})" if address else ""))
        priced = []
        for item in self.results:
            name, store, price = item.get("name"), item.get("store"), item.get("price")
            if not name:
                continue
            where = f" at {store}" if store else ""
            if price is None:
                lines.append(f"  - {name}{where} (no price given)")
            else:
                priced.append((float(price), name, store))
                lines.append(f"  - {name}{where} for €{float(price):.2f}")
        if priced:
            cheapest = min(priced)
            priciest = max(priced)
            lines.append(f"Cheapest item: {cheapest[1]} at €{cheapest[0]:.2f}")
            if len(priced) > 1:
                lines.append(f"Most expensive item: {priciest[1]} at €{priciest[0]:.2f}")
                lines.append(f"Total of the listed items: €{sum(p for p, _, _ in priced):.2f}")
        if self.share_url:
            lines.append("Map link (include it at the end of the reply, on its own): "
                         f"{self.share_url}")
        return "\n".join(lines)


def _as_dict(turn: Any) -> Optional[dict]:
    if turn is None:
        return None
    if hasattr(turn, "model_dump"):
        return turn.model_dump()
    if isinstance(turn, dict):
        return turn
    return None


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[:MAX_TEXT]
    return value


def _clean_turn(turn: dict) -> dict:
    """Apply the size caps: last 6 turns, 10 results, 2 000 characters per text."""
    cleaned: dict = {}
    cleaned["instruction"] = _clip(str(turn.get("instruction") or ""))
    cleaned["response"] = _clip(turn.get("response"))
    cleaned["share_url"] = _clip(turn.get("share_url"))
    actions = turn.get("planner_actions") or []
    cleaned["planner_actions"] = [str(a) for a in actions if isinstance(a, str)]
    entities = _as_dict(turn.get("entities")) or {}
    cleaned["entities"] = {k: _clip(v) for k, v in entities.items()}
    results = []
    for item in (turn.get("results") or [])[:MAX_RESULTS]:
        item = _as_dict(item)
        if item:
            results.append({k: _clip(v) for k, v in item.items()})
    cleaned["results"] = results
    stores = []
    for store in (turn.get("stores") or [])[:MAX_RESULTS]:
        store = _as_dict(store)
        if store:
            stores.append({k: _clip(v) for k, v in store.items()})
    cleaned["stores"] = stores
    return cleaned


def _card_from(entities: dict, actions: Iterable[str]) -> Optional[str]:
    card = entities.get("loyalty_card")
    if card and str(card).upper() in CARD_CODES:
        return str(card).upper()
    brand = entities.get("brand")
    if brand and str(brand).upper() in CARD_CODES:
        return str(brand).upper()
    # A merchant names a card only when the turn was about the wallet - the
    # REWE from "take me to REWE" is a store, not a REWE Bonus card.
    if set(actions) and set(actions) <= WALLET_ACTIONS and entities.get("merchant"):
        return MERCHANT_CARD.get(str(entities["merchant"]).upper())
    return None


def build_context(history: Optional[Iterable[Any]]) -> ConversationContext:
    """Fold `history` (oldest first) into the context the resolver needs."""
    context = ConversationContext()
    turns = [t for t in (_as_dict(t) for t in (history or [])) if t]
    turns = [_clean_turn(t) for t in turns[-MAX_TURNS:]]
    context.turns = len(turns)
    if not turns:
        return context

    context.last_instruction = turns[-1]["instruction"] or None
    for turn in reversed(turns):
        actions = turn["planner_actions"]
        entities = turn["entities"]
        # A declined, conversational or failed turn established nothing.
        if not actions:
            continue
        if context.merchant is None:
            merchant = entities.get("merchant") or next(
                (s.get("name") for s in turn["stores"] if s.get("name")), None)
            if merchant:
                context.merchant = str(merchant)
        if context.products is None and entities.get("product") \
                and set(actions) & PRODUCT_ACTIONS:
            context.products = str(entities["product"])
            price_max = entities.get("price_max")
            context.price_max = float(price_max) if price_max is not None else None
            if set(actions) & BASKET_ACTIONS:
                context.product_kind = "basket"
            elif set(actions) & PRICE_ACTIONS and context.price_max is not None:
                context.product_kind = "price"
            else:
                context.product_kind = "search"
        location = entities.get("location")
        if context.location is None and location and location != "CURRENT_LOCATION":
            context.location = str(location)
        if context.loyalty_card is None:
            card = _card_from(entities, actions)
            if card:
                context.loyalty_card = card
        if not context.results and turn["results"]:
            context.results = turn["results"]
            context.stores = turn["stores"]
            context.share_url = turn["share_url"]
    return context


# ---------------------------------------------------------------------------
# Resolution outcomes
# ---------------------------------------------------------------------------

@dataclass
class Fresh:
    """The message stands alone; classify it exactly as before."""
    text: str


@dataclass
class Rewritten:
    """A follow-up turned into a self-contained sentence; classify `text`."""
    text: str
    language: str
    family: str


@dataclass
class ResultQuestion:
    """A question about the results just shown; answer from the context."""
    text: str
    language: str


@dataclass
class Unresolved:
    """A follow-up frame with nothing in the context to fill it."""
    text: str
    language: str
    family: str


Resolution = Union[Fresh, Rewritten, ResultQuestion, Unresolved]


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

# Demonstratives count as anaphors only when they stand in for a thing
# ("add those to my list", "that one"), not as determiners ("this week's
# deals").  A short whitelist of what may follow a pronominal demonstrative is
# more robust than trying to enumerate nouns.
_EN_ANAPHOR = re.compile(
    r"\b(?:there|it|its|them|one|ones|here|same)\b"
    r"|\b(?:that|those|these|this)\b(?=\s*(?:$|[?!.,]|(?:one|ones|to|on|in|for|too|also|"
    r"please|as|instead|again|at|from|with|and|or|then|open|worth|cheaper|cheapest|"
    r"the)\b))",
    re.IGNORECASE)
_DE_ANAPHOR = re.compile(
    r"\b(?:dort|dorthin|dahin|da|hin|es|sie|davon|dazu|damit|dieselbe|gleiche|selbe|"
    r"ihn|ihm|denen)\b"
    r"|\b(?:das|die|diese|dieses|diesen|dies)\b(?=\s*(?:$|[?!.,]|(?:auch|bitte|noch|mal|"
    r"dann|auf|zu|in|an|da|dort|hin|dorthin|und|oder|dazu|mir|mich|alles|alle|"
    r"eine|einer|hier|offen|günstiger|billiger)\b))",
    re.IGNORECASE)

_RESULT_WORDS = re.compile(
    r"\b(?:alternatives?|others?|another|more|options?|else|total\w*|cheaper|cheapest|"
    r"worth|expensive|priciest|pricier|sum|altogether|prices?|costs?|"
    r"alternativen?|andere[srn]?|weitere[sn]?|mehr|insgesamt|gesamt\w*|günstiger|"
    r"guenstiger|günstigste[nrs]?|guenstigste[nrs]?|billiger|billigste[nrs]?|teurer|"
    r"teuerste[nrs]?|lohnt|zusammen|preis|preise|kosten)\b",
    re.IGNORECASE)

_WHAT_ABOUT = re.compile(
    r"^(?:and|also|what about|how about|und|was ist mit|wie w(?:ä|ae)r'?s mit|"
    r"wie w(?:ä|ae)re es mit|auch)\s+(.+?)\s*[?!.]*$",
    re.IGNORECASE)
_TRAILING_ALSO = re.compile(r"^(.+?)\s+(?:too|also|as well|auch)\s*[?!.]*$", re.IGNORECASE)
_TOPIC_NOISE = re.compile(
    r"^(?:the|some|a|an|any|der|die|das|dem|den|ein|eine|einen|einem|noch|mal|bitte|"
    r"please|maybe|vielleicht|lieber|rather)\s+", re.IGNORECASE)
_TOPIC_TAIL = re.compile(r"\s+(?:too|also|as well|auch|bitte|please|then|dann)$", re.IGNORECASE)
_COMMAND_START = re.compile(
    r"^(?:show|open|find|take|bring|navigate|add|put|search|get|give|tell|plan|"
    r"zeig|zeige|öffne|oeffne|finde|such|suche|bring|navigiere|setz|setze|füg|füge|"
    r"pack|hol|plane|gib|sag)\b",
    re.IGNORECASE)

_NAV_CUE_EN = re.compile(
    r"\b(?:take me|bring me|get me|drive me|walk me|navigate|navigation|directions?|"
    r"how do i get|guide me|go there|way there|lead me)\b", re.IGNORECASE)
_NAV_CUE_DE = re.compile(
    r"\b(?:bring mich|fahr mich|fahre mich|f(?:ü|ue)hr mich|f(?:ü|ue)hre mich|"
    r"navigier\w*|navigation|wie komme ich|wie komm ich|dorthin|dahin|los geht)\b",
    re.IGNORECASE)

_HOURS_CUE_EN = re.compile(
    r"\bwhen\b.*\b(?:open|opens|close|closes|shut|shuts)\b"
    r"|\b(?:opening hours|hours|how late|until when|till when|what time|still open|"
    r"open (?:now|today|tomorrow|right now|yet|already)|is (?:it|that|this one|this|"
    r"the store|the shop) open|opens|closes|closing|closed)\b",
    re.IGNORECASE)
_HOURS_CUE_DE = re.compile(
    r"\bwann\b.*\b(?:öffnet|oeffnet|auf|offen|zu|geöffnet|geoeffnet|schließt|schliesst|"
    r"aufmacht|zumacht|aufhat|zuhat)\b"
    r"|\b(?:öffnungszeiten|oeffnungszeiten|offen|geöffnet|geoeffnet|schließt|schliesst|"
    r"geschlossen|bis wann|wie lange)\b",
    re.IGNORECASE)

_LIST_CUE_EN = re.compile(
    r"\b(?:add|put|stick|note|write|save|throw)\b.*\blist\b"
    r"|\b(?:to|on|onto) (?:my|the) (?:shopping )?list\b|\bshopping list\b",
    re.IGNORECASE)
_LIST_CUE_DE = re.compile(
    r"\b(?:setz|setze|pack|packe|schreib|schreibe|f(?:ü|ue)g|f(?:ü|ue)ge|tu|tue|nimm|"
    r"speicher|speichere|notier|notiere)\b.*\bliste\b"
    r"|\bauf (?:meine|die|meiner|der) (?:einkaufs)?liste\b|\bzur (?:einkaufs)?liste\b|"
    r"\beinkaufsliste\b",
    re.IGNORECASE)
_LIST_ANAPHOR = re.compile(
    r"\b(?:them|those|these|that|it|all|everything|both|the items|the products|"
    r"the lot|sie|die|das|alle|alles|davon|damit|beides|die sachen|die artikel|"
    r"die produkte|die drei|die beiden)\b",
    re.IGNORECASE)

_WALLET_CUE = re.compile(
    r"\b(?:barcode|bar code|qr|code|scan|scanner|card|points|rewards|balance|wallet|"
    r"strichcode|scannen|karte|punkte|punktestand|pr(?:ä|ae)mien|guthaben)\b",
    re.IGNORECASE)
_WALLET_BARCODE = re.compile(r"\b(?:barcode|bar code|qr|code|scan|scanner|strichcode|scannen)\b",
                             re.IGNORECASE)
_WALLET_POINTS = re.compile(r"\b(?:points|punkte|punktestand|balance|guthaben)\b", re.IGNORECASE)
_WALLET_REWARDS = re.compile(r"\b(?:rewards|pr(?:ä|ae)mien)\b", re.IGNORECASE)

_DISTANCE_CUE = re.compile(
    r"\b(?:how far|far away|far is|is it far|distance|wie weit|entfernt|entfernung|weit weg)\b",
    re.IGNORECASE)

_PRICE_MENTION = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:€|eur(?:os?)?|euros?|quid|cent)\b|(?:€|eur)\s*\d", re.IGNORECASE)
# Capitalised place after the prepositions the prompt lists (the extractor
# covers in/near/around/nach/um; bei/zu/to are added here).
_PLACE = re.compile(
    r"\b(?:in|near|around|nach|bei|zu|to)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß.'-]+)")
_PLACE_STOP = {"my", "me", "the", "a", "our", "all", "here", "there", "it", "that", "this",
               "them", "those", "these", "der", "die", "das", "dem", "den", "meiner",
               "meinem", "meine", "mein", "mir", "ihm", "ihr", "hause", "haus"}
_PRODUCT_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(set(KNOWN_PRODUCTS), key=len, reverse=True))
    + r")\b", re.IGNORECASE)

# Products the extractor knows, in both languages, so a German follow-up after
# an English basket reads "Setze Milch, Eier und Brot ..." and vice versa.
_EN_TO_DE = {
    "milk": "Milch", "eggs": "Eier", "bread": "Brot", "butter": "Butter",
    "coffee": "Kaffee", "pasta": "Nudeln", "cheese": "Käse", "yogurt": "Joghurt",
    "yoghurt": "Joghurt", "olive oil": "Olivenöl", "minced meat": "Hackfleisch",
    "frozen pizza": "Tiefkühlpizza", "frozen pizzas": "Tiefkühlpizza",
    "bananas": "Bananen", "apples": "Äpfel", "chocolate": "Schokolade",
    "oat milk": "Hafermilch", "almond milk": "Mandelmilch", "flour": "Mehl",
    "sugar": "Zucker", "rice": "Reis", "toilet paper": "Toilettenpapier",
    "laundry detergent": "Waschmittel", "washing powder": "Waschmittel",
    "nappies": "Windeln", "diapers": "Windeln", "beer": "Bier", "wine": "Wein",
    "water": "Wasser", "chicken": "Hähnchen", "margarine": "Margarine",
    "orange juice": "Orangensaft", "crisps": "Chips", "biscuits": "Kekse",
    "ice cream": "Eis", "ice creams": "Eis", "cereal": "Müsli",
    "mayonnaise": "Mayonnaise", "ketchup": "Ketchup", "mustard": "Senf",
    "salmon": "Lachs", "pizza": "Pizza", "cream": "Sahne", "sausage": "Wurst",
    "ham": "Schinken", "salad": "Salat", "sweets": "Süßigkeiten",
    "salty snacks": "salzige Snacks", "snacks": "Snacks", "drinks": "Getränke",
    "fruit": "Obst", "vegetables": "Gemüse", "tea": "Tee", "toiletries": "Drogerieartikel",
}
_DE_TO_EN = {}
for _en, _de in _EN_TO_DE.items():
    _DE_TO_EN.setdefault(_de.lower(), _en)
_DE_TO_EN.update({"tiefkuehlpizza": "frozen pizza", "kaese": "cheese", "olivenoel": "olive oil",
                  "aepfel": "apples", "haehnchen": "chicken", "muesli": "cereal",
                  "suessigkeiten": "sweets", "getraenke": "drinks", "gemuese": "vegetables"})


def _translate(product: str, language: str) -> str:
    key = product.strip().lower()
    if language == "de":
        return _EN_TO_DE.get(key, product.strip())
    return _DE_TO_EN.get(key, product.strip())


def _items(products: str, language: str) -> str:
    """"Milk, Eggs, Bread" -> "milk, eggs and bread" / "Milch, Eier und Brot"."""
    parts = [p.strip() for p in products.split(",") if p.strip()]
    if language == "de":
        parts = [_translate(p, "de") for p in parts]
        parts = [p[:1].upper() + p[1:] for p in parts]
        joiner = " und "
    else:
        parts = [_translate(p, "en").lower() for p in parts]
        joiner = " and "
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + joiner + parts[-1]


def _card_name(code: str) -> str:
    return CARD_DISPLAY_NAMES.get(code.upper(), code.replace("_", " ").title())


def _price(value: float) -> str:
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Classification of the message
# ---------------------------------------------------------------------------

def _has_anaphor(text: str) -> bool:
    return bool(_EN_ANAPHOR.search(text) or _DE_ANAPHOR.search(text))


def _word_count(text: str) -> int:
    return len(re.findall(r"[\wäöüÄÖÜß'-]+", text))


def _stands_alone(text: str) -> bool:
    """The message names what it is about, so it can be classified as is."""
    entities = extract_entities(text)
    if entities["merchant"] or entities["loyalty_card"] or entities["location"]:
        return True
    place = _PLACE.search(text)
    if place and place.group(1).lower() not in _PLACE_STOP:
        return True
    if _PRODUCT_PATTERN.search(text) or _PRICE_MENTION.search(text):
        return True
    return _word_count(text) > 10 and not _has_anaphor(text)


def _what_about_topic(text: str) -> Optional[str]:
    """"What about butter?" -> "butter"; None when the message is not of that shape."""
    match = _WHAT_ABOUT.match(text) or _TRAILING_ALSO.match(text)
    if not match:
        return None
    topic = match.group(1).strip(" ,.!?")
    previous = None
    while topic != previous:
        previous = topic
        topic = _TOPIC_NOISE.sub("", topic)
        topic = _TOPIC_TAIL.sub("", topic).strip(" ,.!?")
    if not topic or not (1 <= _word_count(topic) <= 4) or _COMMAND_START.match(topic):
        return None
    entities = extract_entities(topic)
    if entities["merchant"] or entities["loyalty_card"] or entities["location"] \
            or _PRICE_MENTION.search(topic):
        return None
    return topic


def _bare(text: str, cue: re.Pattern) -> bool:
    """Short and anaphoric, or so short there is nothing but the cue."""
    return _has_anaphor(text) or _word_count(text) <= 5


def _product_sentence(product: str, context: ConversationContext, language: str) -> str:
    city = context.location
    if language == "de":
        item = _translate(product, "de")
        item = item[:1].upper() + item[1:]
        where = f"in {city}" if city else "in meiner Nähe"
        if context.product_kind == "price" and context.price_max is not None:
            return f"Wo gibt es {item} unter {_price(context.price_max)} Euro {where}?"
        if context.product_kind == "basket":
            return f"Finde den günstigsten Einkaufskorb für {item} {where}"
        return f"Wo kann ich {item} {where} kaufen?"
    item = _translate(product, "en").lower()
    where = f"in {city}" if city else "near me"
    if context.product_kind == "price" and context.price_max is not None:
        return f"Where can I buy {item} under {_price(context.price_max)} euros {where}?"
    if context.product_kind == "basket":
        return f"Find the cheapest basket for {item} {where}"
    return f"Where can I buy {item} {where}?"


def _navigation_sentence(context: ConversationContext, language: str) -> str:
    merchant, city = context.merchant, context.location
    if language == "de":
        return (f"Navigiere mich zu {merchant} in {city}" if city
                else f"Bring mich zum nächsten {merchant}")
    return (f"Take me to {merchant} in {city}" if city
            else f"Take me to the nearest {merchant}")


def _hours_sentence(context: ConversationContext, language: str) -> str:
    merchant, city = context.merchant, context.location
    if language == "de":
        return (f"Wann hat {merchant} in {city} geöffnet?" if city
                else f"Wann hat {merchant} geöffnet?")
    return (f"When does {merchant} in {city} open?" if city
            else f"When does {merchant} open?")


def _distance_sentence(context: ConversationContext, language: str) -> str:
    merchant, city = context.merchant, context.location
    if language == "de":
        return (f"Wie weit ist {merchant} in {city} von hier entfernt?" if city
                else f"Wie weit ist {merchant} entfernt?")
    return (f"How far away is {merchant} in {city}?" if city
            else f"How far is {merchant}?")


def _list_sentence(context: ConversationContext, language: str) -> str:
    items = _items(context.products or "", language)
    if language == "de":
        return f"Setze {items} auf meine Einkaufsliste"
    return f"Add {items} to my shopping list"


def _wallet_sentence(text: str, context: ConversationContext, language: str) -> str:
    card = _card_name(context.loyalty_card or "")
    if _WALLET_BARCODE.search(text):
        return f"Zeig mir meinen {card} Barcode" if language == "de" else f"Show my {card} barcode"
    if _WALLET_POINTS.search(text):
        return (f"Wie viele Punkte habe ich bei {card}?" if language == "de"
                else f"How many points do I have on my {card} card?")
    if _WALLET_REWARDS.search(text):
        return f"Zeig mir meine {card} Prämien" if language == "de" else f"Show my {card} rewards"
    return f"Zeig mir meine {card} Karte" if language == "de" else f"Show my {card} card"


def _llm_rewrite(text: str, context: ConversationContext, language: str) -> Optional[str]:
    """Optional: ask the configured Groq model for one self-contained sentence.

    Off unless FOLLOW_UP_LLM_REWRITE=1.  Measure the added latency before
    turning it on; the rules above cover the measured follow-ups without it.
    """
    if os.getenv("FOLLOW_UP_LLM_REWRITE", "0").strip() != "1" or context.empty:
        return None
    from ..llm.provider import generate_text
    system = (
        "You rewrite one follow-up message from a shopping-app conversation into a "
        "single self-contained sentence that means the same thing, using only the "
        "facts in CONTEXT. Same language as the follow-up. No new facts, no "
        "explanation, no quotes. If unsure, return the follow-up unchanged."
    )
    facts = {
        "previous request": context.last_instruction, "store": context.merchant,
        "products": context.products, "place": context.location,
        "loyalty card": _card_name(context.loyalty_card) if context.loyalty_card else None,
    }
    user = "CONTEXT:\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items() if v) \
        + f"\n\nFOLLOW-UP: {text}\n\nREWRITE:"
    try:
        rewritten = generate_text(system, user, temperature=0.0, max_tokens=60)
    except Exception:
        logger.warning("follow-up LLM rewrite failed", exc_info=True)
        return None
    rewritten = (rewritten or "").strip().strip('"').strip()
    if not rewritten or rewritten.lower() == text.lower() or "\n" in rewritten:
        return None
    return rewritten


def resolve_follow_up(instruction: str, context: Optional[ConversationContext]) -> Resolution:
    """Decide whether `instruction` stands alone, and rewrite it if not."""
    text = (instruction or "").strip()
    context = context or ConversationContext()
    language = language_of(text, fallback=context.language)

    # "What about butter?" is elliptical even though it names a product: the
    # *kind* of request comes from the turn before.
    topic = _what_about_topic(text)
    if topic is not None and not (_RESULT_WORDS.search(topic) or _has_anaphor(topic)):
        if context.product_kind:
            return Rewritten(_product_sentence(topic, context, language), language, "product")
        return Unresolved(text, language, "product")

    if _stands_alone(text):
        return Fresh(text)

    # Wallet before hours: "open it" after a card turn is the card, not a store.
    if context.loyalty_card and _WALLET_CUE.search(text) and _bare(text, _WALLET_CUE):
        return Rewritten(_wallet_sentence(text, context, language), language, "wallet")

    nav = _NAV_CUE_EN if language == "en" else _NAV_CUE_DE
    if (nav.search(text) or _NAV_CUE_EN.search(text) or _NAV_CUE_DE.search(text)) \
            and _bare(text, nav):
        if context.merchant:
            return Rewritten(_navigation_sentence(context, language), language, "navigation")
        return Unresolved(text, language, "navigation")

    if (_HOURS_CUE_EN.search(text) or _HOURS_CUE_DE.search(text)) and _bare(text, _HOURS_CUE_EN):
        if context.merchant:
            return Rewritten(_hours_sentence(context, language), language, "hours")
        return Unresolved(text, language, "hours")

    if _DISTANCE_CUE.search(text) and _bare(text, _DISTANCE_CUE):
        if context.merchant:
            return Rewritten(_distance_sentence(context, language), language, "distance")
        return Unresolved(text, language, "distance")

    if (_LIST_CUE_EN.search(text) or _LIST_CUE_DE.search(text)) and _LIST_ANAPHOR.search(text):
        if context.products:
            return Rewritten(_list_sentence(context, language), language, "list")
        return Unresolved(text, language, "list")

    if _word_count(text) <= 10 and (_RESULT_WORDS.search(text)
                                    or (topic is not None)
                                    or (_has_anaphor(text) and text.endswith("?"))):
        if context.results:
            return ResultQuestion(text, language)
        if _RESULT_WORDS.search(text) or topic is not None:
            return Unresolved(text, language, "results")

    if _has_anaphor(text):
        rewritten = _llm_rewrite(text, context, language)
        if rewritten:
            return Rewritten(rewritten, language, "llm")

    return Fresh(text)
