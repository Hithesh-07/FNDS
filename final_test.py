from predict import load_model, predict

model, vectorizer, scaler = load_model()

tests = [
    ("REAL",
     "A recent report by a global economic organization highlights emerging markets expected to experience moderate growth next fiscal year supported by improvements in domestic consumption and investment. Analysts emphasize policymakers should remain vigilant and prepared to adjust strategies in response to evolving global dynamics."),

    ("FAKE",
     "An online article claims researchers identified a natural method reversing multiple chronic illnesses. The report suggests findings have not been officially published and are being withheld due to pressure from major industry stakeholders. According to unnamed individuals familiar with the research attempts to share results have been blocked. No verifiable data or peer-reviewed evidence has been provided."),

    ("UNCERTAIN",
     "A preliminary study suggests certain environmental factors may influence cognitive performance over time although researchers caution findings are based on limited data and should not be interpreted as conclusive. Experts note additional controlled studies are required to better understand the relationship and determine whether observed effects are consistent across different populations."),

    ("FAKE",
     "A report circulating on several platforms claims global institutions have been quietly developing technologies capable of influencing environmental conditions on large scale. While supporters argue leaked documents indicate advanced progress independent verification has not been possible and experts have expressed skepticism regarding feasibility of such claims."),

    ("REAL",
     "A new policy initiative introduced by the government aims to expand renewable energy capacity by incentivizing private sector investment and improving grid infrastructure. Officials state the program is part of broader strategy to reduce carbon emissions and enhance energy security. Industry experts responded positively although some raised concerns about implementation challenges and long-term financing."),
]

print("\n" + "="*65)
print("  TruthLens — Final Decision Engine Test (5 cases)")
print("="*65 + "\n")

correct = 0
for i, (expected, text) in enumerate(tests, 1):
    r      = predict(text, model, vectorizer, scaler)
    pred   = r["label"]
    conf   = r["confidence"]
    reason = r.get("decision_reason", "unknown")
    net    = r.get("net_score", 0)
    match  = pred == expected
    correct += int(match)
    status = "✅" if match else "❌"

    print(f"  [{i}] {status} Expected: {expected:<10} "
          f"Got: {pred:<10} Conf: {conf}%")
    print(f"       Reason: {reason}  |  NetScore: {net}")
    print(f"       {text[:65]}...")
    print()

print("="*65)
score_pct = (correct / 5) * 100
print(f"  SCORE: {correct}/5  ({score_pct:.0f}%)")

if score_pct == 100:  print("  🏆 PERFECT — System fully calibrated!")
elif score_pct >= 80: print("  🥇 EXCELLENT — Nearly production ready!")
elif score_pct >= 60: print("  🥈 GOOD — One more round of tuning")
else:                 print("  🥉 NEEDS MORE WORK")
print("="*65)
