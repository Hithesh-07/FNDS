# test_uncertainty.py
from predict import load_model, predict as svm_predict
from decision_engine import run_decision_engine, calculate_uncertainty_score

model, vectorizer, scaler = load_model()

tests = [
    # (expected, text)
    ("UNCERTAIN",
     "A preliminary study suggests that certain lifestyle changes may improve immunity, though researchers caution that more evidence is required before drawing conclusions."),

    ("UNCERTAIN",
     "Some experts suggest intermittent fasting might offer metabolic benefits, but evidence remains mixed and further longitudinal research is needed."),

    ("UNCERTAIN",
     "Initial findings indicate the compound could reduce inflammation, however scientists note the observational study cannot confirm causation."),

    ("FAKE",
     "SHOCKING: Scientists confirm 5G towers cause cancer and the government is HIDING the truth from the public! Leaked documents EXPOSED!"),

    ("REAL",
     "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent following a scheduled monetary policy committee review."),

    ("FAKE",
     "According to unnamed sources, researchers discovered a cure for all diseases but findings were suppressed by pharmaceutical companies. No peer-reviewed evidence provided."),

    ("UNCERTAIN",
     "Preliminary research suggests certain dietary patterns may be associated with reduced cognitive decline, though experts caution that more controlled studies are required to confirm these findings."),

    ("REAL",
     "Apple reported quarterly revenue of 94.8 billion dollars beating analyst expectations for the fourth fiscal quarter."),
]

print("\n" + "="*65)
print("  Uncertainty Detection Test")
print("="*65 + "\n")

correct = 0
for expected, text in tests:
    svm = svm_predict(text, model, vectorizer, scaler)
    dec = run_decision_engine(
        text, svm["label"], svm["confidence"],
        svm["fake_prob"], svm["real_prob"]
    )
    pred    = dec["final_label"]
    conf    = dec["final_confidence"]
    unc_sc  = dec["uncertain_score"]
    net_sc  = dec["net_score"]
    match   = pred == expected
    correct += int(match)

    print(f"  {'✅' if match else '❌'} Expected: {expected:<10} "
          f"Got: {pred:<10} ({conf}%) "
          f"[uncertain:{unc_sc} net:{net_sc}]")
    print(f"     {text[:70]}...")
    print()

print("="*65)
score_pct = (correct/len(tests))*100
print(f"  SCORE: {correct}/{len(tests)} ({score_pct:.0f}%)")
if score_pct >= 87: print("  ✅ Uncertainty detection working!")
else:               print("  ❌ Still needs tuning")
print("="*65)
