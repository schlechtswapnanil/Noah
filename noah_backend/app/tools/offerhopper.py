"""Offerhopper MCP Client for multi-item grocery basket and route optimization."""

import logging
import json
import time
from typing import Any, Dict, Optional
# pyrefly: ignore [missing-import]
import httpx

logger = logging.getLogger(__name__)

OFFERHOPPER_MCP_URL = "https://mcp.offerhopper.ai/mcp"

GERMAN_CITY_TO_PLZ = {
    "berlin": "10117",
    "hamburg": "20095",
    "münchen": "80331",
    "munich": "80331",
    "köln": "50667",
    "cologne": "50667",
    "frankfurt": "60311",
    "stuttgart": "70173",
    "düsseldorf": "40210",
    "dusseldorf": "40210",
    "dortmund": "44135",
    "essen": "45127",
    "leipzig": "04109",
    "bremen": "28195",
    "dresden": "01067",
    "hannover": "30159",
    "hanover": "30159",
    "nürnberg": "90402",
    "nuremberg": "90402",
    "duisburg": "47051",
    "bochum": "44787",
    "wuppertal": "42103",
    "bielefeld": "33602",
    "bonn": "53111",
    "münster": "48143",
    "munster": "48143",
    "karlsruhe": "76133",
    "mannheim": "68161",
    "augsburg": "86150",
    "wiesbaden": "65183",
    "gelsenkirchen": "45879",
    "mönchengladbach": "41061",
    "braunschweig": "38100",
    "chemnitz": "09111",
    "aachen": "52062",
    "kiel": "24103",
    "halle": "06108",
    "magdeburg": "39104",
    "freiburg": "79098",
    "krefeld": "47798",
    "lübeck": "23552",
    "lubeck": "23552",
    "oberhausen": "46045",
    "erfurt": "99084",
    "mainz": "55116",
    "rostock": "18055",
    "kassel": "34117",
    "hagen": "58095",
    "saarbrücken": "66111",
    "saarbruecken": "66111",
    "hamm": "59065",
    "mülheim": "45468",
    "mulheim": "45468",
    "ludwigshafen": "67059",
    "leverkusen": "51373",
    "oldenburg": "26122",
    "potsdam": "14467",
    "solingen": "42651",
    "heidelberg": "69117",
    "herne": "44623",
    "neuss": "41460",
    "darmstadt": "64283",
    "paderborn": "33098",
    "regensburg": "93047",
    "ingolstadt": "85049",
    "würzburg": "97070",
    "wuerzburg": "97070",
    "fürth": "90762",
    "fuerth": "90762",
    "wolfsburg": "38440",
    "offenbach": "63065",
    "ulm": "89073",
    "heilbronn": "74072",
    "pforzheim": "75175",
    "göttingen": "37073",
    "goettingen": "37073",
    "bottrop": "46236",
    "trier": "54290",
    "recklinghausen": "45657",
    "reutlingen": "72764",
    "bremerhaven": "27568",
    "koblenz": "56068",
    "bergisch gladbach": "51465",
    "jena": "07743",
    "remscheid": "42853",
    "erlangen": "91054",
    "moers": "47441",
    "siegen": "57072",
    "hildesheim": "31134",
    "salzgitter": "38226",
}


def resolve_location_to_plz(location: Optional[str]) -> str:
    """Resolve user-supplied location (city, current location, or raw string) to German PLZ."""
    if not location or location == "CURRENT_LOCATION":
        return "80331"  # Default postcode (Munich central)

    loc_lower = location.lower().strip()
    if loc_lower in GERMAN_CITY_TO_PLZ:
        return GERMAN_CITY_TO_PLZ[loc_lower]

    return location


def call_offerhopper_mcp(
    items: str,
    location: str,
    travel_mode: str = "car",
    max_radius_km: Optional[float] = None,
    hour_cost: float = 12.0,
) -> Dict[str, Any]:
    """Execute plan_optimal_shopping_route on the live Offerhopper MCP server."""
    # Resolve location to PLZ code for higher accuracy
    resolved_loc = resolve_location_to_plz(location)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "plan_optimal_shopping_route",
            "arguments": {
                "items": items,
                "location": resolved_loc,
                "travel_mode": travel_mode,
                "hour_cost": hour_cost,
                **({"max_radius_km": max_radius_km} if max_radius_km is not None else {}),
            },
        },
    }

    # Streamable HTTP requires accepting application/json and text/event-stream
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "payto-noah-backend/1.0"
    }

    # One retry on the failures that clear themselves - a 429 from a burst of
    # requests, a gateway hiccup, a slow response - so a single blip does not
    # turn into "the price service is unavailable" in front of the user.
    for attempt in range(2):
        try:
            with httpx.Client(timeout=25.0) as client:
                init_resp = client.post(
                    OFFERHOPPER_MCP_URL,
                    json={
                        "jsonrpc": "2.0",
                        "id": 0,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-03-26",
                            "clientInfo": {"name": "payto-noah-backend", "version": "1.0"},
                        },
                    },
                    headers=headers,
                )
                call_headers = dict(headers)
                session_id = init_resp.headers.get("mcp-session-id")
                if session_id:
                    call_headers["mcp-session-id"] = session_id

                resp = client.post(OFFERHOPPER_MCP_URL, json=payload, headers=call_headers)
                if resp.status_code in (429, 502, 503, 504) and attempt == 0:
                    wait = min(float(resp.headers.get("retry-after") or 2), 5.0)
                    logger.warning("Offerhopper returned %s; retrying in %.0fs", resp.status_code, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return _parse_tool_response(resp.text)
        except (httpx.TimeoutException, httpx.TransportError) as error:
            if attempt == 0:
                logger.warning("Offerhopper transport error (%s); retrying", error)
                time.sleep(2)
                continue
            logger.exception("Failed to communicate with Offerhopper MCP")
        except Exception:
            logger.exception("Failed to communicate with Offerhopper MCP")
            break

    return {}


def _parse_tool_response(body: str) -> Dict[str, Any]:
    """Read the JSON-RPC result out of the event stream.

    Three outcomes, and the caller has to tell them apart:
      * a route  -> the parsed result dict (``success`` is True);
      * the tool declining, e.g. no item could be matched -> OfferHopper sends
        ``result.isError: true`` with a plain-text message such as
        "Optimization failed: error_invalid_list". Returned as
        ``{"success": False, "error": <text>}``. This used to raise inside
        json.loads, get swallowed, and come back as an empty dict - so "nothing
        matched" was reported to the user as "the price service is down";
      * nothing usable -> ``{}``, meaning the transport failed.
    """
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except ValueError:
            continue
        if "error" in event:
            return {"success": False, "error": str(event["error"].get("message", event["error"]))}
        result = event.get("result")
        if not isinstance(result, dict):
            continue
        texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        if result.get("isError"):
            return {"success": False, "error": " ".join(texts).strip() or "tool error"}
        for text in texts:
            try:
                return json.loads(text)
            except ValueError:
                continue
        return result
    return {}


def format_offerhopper_summary(data: Dict[str, Any]) -> str:
    """Format OfferHopper live results into a concise, human-readable summary with stores and prices."""
    if not data or not isinstance(data, dict):
        return ""

    route = data.get("optimized_route") or {}
    segments = route.get("route_segments") or []
    picks = []
    
    for seg in segments:
        store = (seg.get("to_store") or {}).get("name") or seg.get("to_name")
        for prod in seg.get("products_to_buy") or []:
            name = prod.get("selected_product") or prod.get("name") or prod.get("display_title")
            price = prod.get("price")
            if name and price is not None:
                if store and store != "user_end_location":
                    picks.append(f"{name} at {store} for €{price:.2f}")
                else:
                    picks.append(f"{name} for €{price:.2f}")

    if picks:
        total = data.get("total_estimated_cost") or route.get("estimated_total_cost")
        savings = data.get("total_estimated_savings") or route.get("estimated_total_savings")
        summary = f"Found {', '.join(picks[:4])}."
        if savings and savings > 0.05:
            summary += f" (Estimated savings: €{savings:.2f})"
        return summary

    if data.get("ai_description"):
        return str(data["ai_description"])

    return ""

