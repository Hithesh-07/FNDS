# quick_test.py
from predict import load_model, predict

model, vectorizer, scaler = load_model()

tests = [
    # (expected, text)

    # ── Should be REAL ─────────────────────────────────────
    ("REAL", "The Reserve Bank increased interest rates by 0.25% to control inflation."),
    ("REAL", "Apple reported quarterly revenue of 94.8 billion dollars beating estimates."),
    ("REAL", "NASA James Webb telescope captured images of galaxy 4.6 billion light years away."),
    ("REAL", "The Supreme Court ruled in favor of environmental regulations today."),
    ("REAL", "India GDP grew by 6.5 percent in last fiscal quarter according to government data."),

    # ── Should be FAKE ─────────────────────────────────────
    ("FAKE", "Scientists confirm cure all diseases suppressed by global organizations."),
    ("FAKE", "SHOCKING: Bill Gates admits vaccines contain microchips EXPOSED!"),
    ("FAKE", "Government HIDING 5G cancer link, whistleblower LEAKS proof!"),
    ("FAKE", "Miracle cure for cancer BANNED by Big Pharma, doctors CENSORED!"),
    ("FAKE", "NASA FAKED moon landing, new leaked footage PROVES cover-up!"),

    # ── Should be UNCERTAIN ────────────────────────────────
    ("UNCERTAIN", "Some researchers suggest lifestyle changes may improve immunity, more evidence required."),
    ("UNCERTAIN", "Preliminary findings suggest the drug could be effective, clinical trials ongoing."),
    ("UNCERTAIN", "Some experts believe economy may slow, though others dispute this."),
    ("UNCERTAIN", "Reports suggest company may be planning layoffs, not yet confirmed."),
    ("UNCERTAIN", "Studies indicate coffee might reduce Alzheimers risk but more research needed."),
]

print("\n" + "="*65)
print("  TruthLens — Balance Test (15 cases)")
print("="*65 + "\n")

correct = 0
results = {"REAL": [0,0], "FAKE": [0,0], "UNCERTAIN": [0,0]}

for expected, text in tests:
    r      = predict(text, model, vectorizer, scaler)
    pred   = r["label"]
    conf   = r["confidence"]
    match  = pred == expected
    correct += int(match)
    results[expected][0] += int(match)
    results[expected][1] += 1
    status = "✅" if match else "❌"
    print(f"  {status} Expected: {expected:<10} Got: {pred:<10} ({conf}%)")
    print(f"     {text[:65]}...")
    print()

print("="*65)
print(f"  TOTAL:    {correct}/15")
print(f"  REAL:     {results['REAL'][0]}/{results['REAL'][1]}")
print(f"  FAKE:     {results['FAKE'][0]}/{results['FAKE'][1]}")
print(f"  UNCERTAIN:{results['UNCERTAIN'][0]}/{results['UNCERTAIN'][1]}")

score_pct = (correct/15)*100
if score_pct == 100:
    print("\n  🏆 PERFECT BALANCE!")
elif score_pct >= 87:
    print("\n  🥇 EXCELLENT — Production ready!")
elif score_pct >= 73:
    print("\n  🥈 GOOD — Minor tuning needed")
else:
    print("\n  🥉 NEEDS MORE TUNING")
print("="*65)
