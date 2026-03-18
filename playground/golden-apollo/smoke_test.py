from predict import load_model, predict
import os

m, v, s = load_model()

tests = [
    "Scientists confirm a natural compound can cure all diseases but findings are suppressed.",
    "Some researchers suggest certain lifestyle changes may improve immunity though more evidence is required.",
    "The Reserve Bank increased interest rates by 0.25 percent to control inflation."
]

print("\n--- CALIBRATION SMOKE TEST (SVM) ---")
for t in tests:
    r = predict(t, m, v, s)
    label = r["label"]
    conf = r["confidence"]
    print(f"[{label:^10}] Conf: {conf:>5}% | Text: {t[:60]}...")

# Also test BERT if available
if os.path.exists("model/bert_model"):
    from bert_predict import bert_predict
    print("\n--- CALIBRATION SMOKE TEST (BERT) ---")
    for t in tests:
        r = bert_predict(t)
        label = r["label"]
        conf = r["confidence"]
        print(f"[{label:^10}] Conf: {conf:>5}% | Text: {t[:60]}...")
