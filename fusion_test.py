import sys
import os

# Mock the environment or set HF_TOKEN if needed
# os.environ["HF_TOKEN"] = "your_token"

from app import analyze
import time

TEST_CASES = [
    ("REAL", "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent... CITING stable inflation."),
    ("FAKE", "According to unnamed sources familiar with the matter, researchers discovered a natural compound reversing diabetes. Findings not published as pharmaceutical companies allegedly blocking release to protect profits. No peer-reviewed evidence provided."),
    ("FAKE", "A report claims the World Health Organization has been secretly stockpiling evidence that common household chemicals enhance immunity. According to anonymous insiders, this information has been withheld from the public for over a decade. Independent verification has not been possible."),
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
        print(f"  BERT: {r['bert_label']} ({r['bert_confidence']}%)")
        print(f"  SVM : {r['svm_label']} ({r['svm_confidence']}%)")
        print(f"  Net Score: {r['net_score']} (Reason: {r['decision_reason']})")
        
        match = r['label'] == expected
        print(f"  {'✅' if match else '❌'} Result: {'MATCH' if match else 'MISMATCH'}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    time.sleep(1)
