import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_queries = [
    'Show me my Payback barcode.',
    'Where can I buy frozen pizzas for less than €2 near me?',
    'Take me to the nearest Lidl.',
    "I'm at Netto. What are the current offers on ice creams here below €3? Then display my Payback card.",
    'Find the cheapest basket for milk, eggs, and bread in Berlin.',
    'Plan the optimal shopping route for my grocery list in Hamburg.',
    'Compare the prices of oat milk and almond milk.',
    'What is PayTo and how do loyalty points work?',
    'Does Payto store my payment details?',
    'Hello! Who are you?'
]

for q in test_queries:
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/chat',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'instruction': q}).encode('utf-8')
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print('=' * 70)
        print('INSTRUCTION :', res['instruction'])
        print('DOMAIN/INTENT:', f"{res.get('domain')}/{res.get('intent')}/{res.get('sub_intent')}")
        print('ACTIONS     :', res.get('planner_actions'))
        print('TOOLS       :', res.get('tool_sequence'))
        print('RESPONSE    :', res.get('response'))
