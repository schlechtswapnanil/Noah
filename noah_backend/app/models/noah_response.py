from typing import Any, Optional

from pydantic import BaseModel


class Entities(BaseModel):

    product: Optional[str] = None

    merchant: Optional[str] = None

    brand: Optional[str] = None

    category: Optional[str] = None

    price_min: Optional[float] = None

    price_max: Optional[float] = None

    location: Optional[str] = None

    radius: Optional[float] = None

    loyalty_card: Optional[str] = None


class NoahResponse(BaseModel):

    domain: Optional[str] = None

    intent: Optional[str] = None

    sub_intent: Optional[str] = None

    tool: Optional[str] = None

    response_mode: Optional[str] = None

    entity_category: Optional[str] = None

    requires_memory: bool = False

    requires_rag: bool = False

    requires_recommendation: bool = False

    workflow_type: Optional[str] = None

    planner_actions: list[str] = []

    planner_action_count: int = 0

    tool_sequence: list[str] = []

    entities: Entities = Entities()

    response: str
