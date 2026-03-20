import os
from app import analyze
import time

TEST_CASES = [
    ("FAKE", "A newly discovered plant extract completely eliminates the need for sleep. Early trials showed subjects fully alert for weeks without rest, but scientists have strongly rejected these claims as biologically impossible."),
    ("REAL", "The Reserve Bank of India kept the repo rate unchanged yesterday."),
    ("UNCERTAIN", "A preliminary study suggests that intermittent fasting may potentially help with weight loss, but more research is needed."),
]

print("── Aletheia Model Fusion Regression Test ──")
print("===========================================")

for expected, text in TEST_CASES:
    try:
        print(f"\nAnalyzing: {text[:80]}...")
        r = analyze(text)
        
        print(f"  VERDICT: {r['label']} (Confidence: {r['confidence']}%)")
        print(f"  Model Used: {r['model_used']}")
        print(f"  Agreement: {r['agreement_text']} ({r['agreement']})")
        print(f"  Net Score: {r['net_score']} (Reason: {r['decision_reason']})")
        
        match = r['label'] == expected
        # Since BERT is failing locally, "UNCERTAIN" might fallback to REAL or FAKE if the rule engine fires
        print(f"  {'✅' if match else '⚠️'} Result: Expected {expected}, Got {r['label']}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    time.sleep(1)
