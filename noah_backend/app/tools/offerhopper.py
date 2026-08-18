"""Offerhopper MCP Client for multi-item grocery basket and route optimization."""

import logging
import json
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

    try:
        with httpx.Client(timeout=30.0) as client:
            # Initialize session for streamable HTTP transport
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
                headers=headers
            )
            session_id = init_resp.headers.get("mcp-session-id")
            
            call_headers = dict(headers)
            if session_id:
                call_headers["mcp-session-id"] = session_id

            resp = client.post(OFFERHOPPER_MCP_URL, json=payload, headers=call_headers)
            resp.raise_for_status()
            
            # OfferHopper remote MCP server returns a text/event-stream
            # We parse the event stream to retrieve the final jsonrpc result
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    try:
                        event_data = json.loads(line[5:].strip())
                        if "result" in event_data:
                            result_data = event_data["result"]
                            if isinstance(result_data, dict) and "content" in result_data:
                                for item in result_data["content"]:
                                    if item.get("type") == "text":
                                        return json.loads(item["text"])
                            return result_data
                    except Exception:
                        continue
    except Exception as e:
        logger.exception(f"Failed to communicate with Offerhopper MCP: {e}")

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

