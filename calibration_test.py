from predict import load_model, predict
import os

model, vectorizer, scaler = load_model()

tests = [
    ("REAL",
     "India central bank maintained benchmark interest rate while monitoring inflation. Officials stated inflation has shown signs of easing. Economists noted the decision reflects balance between growth and price stability."),

    ("FAKE",
     "A report circulating online claims researchers discovered a compound reversing chronic diseases. According to unnamed sources, findings not published because major institutions are allegedly preventing research from being released. No verifiable evidence or peer-reviewed studies provided."),

    ("UNCERTAIN",
     "A recent observational study suggests lifestyle changes may influence long-term health outcomes. Researchers emphasize findings are preliminary based on correlations not controlled experiments. Experts caution further research needed. Several institutions announced plans to conduct more detailed studies."),

    ("REAL",
     "Apple reported quarterly revenue of 94.8 billion dollars beating analyst expectations for fourth quarter."),

    ("FAKE",
     "Scientists confirm natural compound cures all diseases but findings suppressed by global organizations protecting treatment industries."),

    ("REAL",
     "The Reserve Bank increased interest rates by 0.25 percent to control inflation amid rising global prices."),

    ("FAKE",
     "According to unnamed whistleblower, government has been hiding evidence of alien contact since 1947. No official confirmation and documents cannot be verified but sources claim cover-up ongoing."),

    ("UNCERTAIN",
     "Some experts suggest remote work may improve productivity in certain industries though evidence remains mixed and more longitudinal studies are required to draw firm conclusions."),
]

print("\n" + "="*65)
print("  TruthLens — Final Calibration Test (8 cases)")
print("="*65 + "\n")

correct = 0
for i, (expected, text) in enumerate(tests, 1):
    r     = predict(text, model, vectorizer, scaler)
    pred  = r["label"]
    conf  = r["confidence"]
    cred  = r.get("credibility_score", 0)
    match = pred == expected
    correct += int(match)
    status = "✅" if match else "❌"

    print(f"  [{i}] {status} Expected: {expected:<10} Got: {pred:<10} "
          f"Conf: {conf}%  CredScore: {cred}")
    print(f"       {text[:70]}...")
    print()

print("="*65)
score_pct = (correct/8)*100
print(f"  SCORE: {correct}/8  ({score_pct:.0f}%)")

if score_pct == 100:   print("  🏆 PERFECT CALIBRATION!")
elif score_pct >= 87:  print("  🥇 EXCELLENT — Production ready!")
elif score_pct >= 75:  print("  🥈 GOOD — Minor tuning needed")
else:                  print("  🥉 STILL NEEDS TUNING")
print("="*65 + "\n")
