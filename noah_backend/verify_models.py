import json
from app.nlp.model_loader import load_models
from app.nlp.classifier import classify
from app.planner.planner import create_plan

models = load_models()

queries = [
    'Find me a doner under 5 euros near me',
    'Where can I get a kebab under 6 euros near me?',
    'Find the cheapest basket for milk, eggs, and bread in Berlin.',
    'Plan the optimal shopping route for my grocery list in Hamburg.',
    'Compare the prices of oat milk and almond milk.',
    'Where can I buy frozen pizzas for less than 2 euros near me?',
    'Show me my Payback barcode.'
]

for q in queries:
    p = classify(q, models)
    plan = create_plan(p)
    print('========================================')
    print('QUERY:', q)
    print('DOMAIN / INTENT / SUB-INTENT:', f"{p['domain']} / {p['intent']} / {p['sub_intent']}")
    print('PLANNER ACTIONS             :', plan['planner_actions'])
    print('TOOL SEQUENCE               :', plan['tool_sequence'])
    print('ENTITIES                    :', plan['entities'])
