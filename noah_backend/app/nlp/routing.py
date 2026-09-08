"""Shared route-scoring logic.

Both the trainer and the request path score routes through this module so the
confidence the threshold is calibrated against is the same number the API
compares at request time.

Routes are ``sub_intent`` + action sequence, which means one intent can be
spread over several labels ("show the offers at Netto" versus "show the offers
at Netto and then my card").  A compound request splits its probability mass
between those, and neither half clears the threshold on its own even though the
model is in no doubt about the *intent*.  Confidence is therefore measured per
sub-intent - the decision Noah has to be sure about - while the action sequence
is chosen within the winning sub-intent.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def score_routes(model, registry: dict, instructions: Sequence[str]):
    """Return ``(routes, confidences)`` for each instruction."""
    probabilities = model.predict_proba(list(instructions))
    classes = list(model.classes_)
    sub_intents = [registry[c]["sub_intent"] if c in registry else None for c in classes]

    routes, confidences = [], []
    for row in probabilities:
        totals = defaultdict(float)
        for sub_intent, probability in zip(sub_intents, row):
            if sub_intent is not None:
                totals[sub_intent] += float(probability)
        if not totals:
            routes.append(classes[int(row.argmax())])
            confidences.append(float(row.max()))
            continue

        best_sub_intent = max(totals, key=totals.get)
        best_index, best_probability = None, -1.0
        for index, (sub_intent, probability) in enumerate(zip(sub_intents, row)):
            if sub_intent == best_sub_intent and probability > best_probability:
                best_index, best_probability = index, float(probability)
        routes.append(classes[best_index])
        confidences.append(totals[best_sub_intent])

    return np.array(routes), np.array(confidences)
