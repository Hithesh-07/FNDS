from predict import predict
import time

TEST_CASES = [
    ("TEST 1 — CLEAR REAL", "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent for the sixth consecutive meeting, citing stable inflation and steady economic growth. The decision was announced following a three-day monetary policy committee meeting."),
    ("TEST 2 — OBVIOUS FAKE", "SHOCKING: Scientists have CONFIRMED that 5G towers are secretly emitting radiation that causes cancer and the government is HIDING this from the public. Whistleblowers have LEAKED documents proving the cover-up!"),
    ("TEST 3 — SOPHISTICATED FAKE", "According to unnamed sources familiar with the matter, a group of independent researchers has discovered a natural compound capable of reversing Type 2 diabetes within 30 days. The findings have not been published as major pharmaceutical companies are allegedly blocking the release to protect their insulin profits. No peer-reviewed evidence has been provided."),
    ("TEST 4 — UNCERTAIN", "A preliminary observational study suggests that increased screen time may be associated with reduced attention spans in adolescents, though researchers caution that the findings are based on limited data. Experts note that further controlled studies are needed before any conclusions can be drawn."),
    ("TEST 5 — REAL (complex)", "Apple Inc reported quarterly revenue of 124.3 billion dollars for the first fiscal quarter, beating analyst expectations. The company attributed growth to strong iPhone sales in emerging markets and continued expansion of its services division. CEO Tim Cook said the results reflect strong consumer demand despite global economic pressures."),
    ("TEST 6 — TRICKY FAKE", "A report circulating on multiple platforms claims that the World Health Organization has been secretly stockpiling evidence that common household chemicals, when combined, produce a substance that enhances human immunity. According to anonymous insiders, this information has been withheld from the public for over a decade. Independent verification of these claims has not been possible."),
    ("TEST 7 — REAL (Indian news)", "The Indian government announced a new production-linked incentive scheme worth 76000 crore rupees aimed at boosting domestic semiconductor manufacturing. The scheme is expected to attract major global chipmakers and create thousands of jobs over the next five years, according to the Ministry of Electronics."),
    ("TEST 8 — UNCERTAIN (scientific)", "Some nutrition experts suggest that intermittent fasting may offer metabolic benefits for certain individuals, although evidence from long-term studies remains mixed. While short-term results appear promising, researchers emphasize that dietary interventions should be personalized and that more longitudinal research is required."),
    ("TEST 9 — FAKE (conspiracy style)", "Leaked internal documents allegedly reveal that global financial institutions have been coordinating to artificially suppress gold prices for decades in order to maintain confidence in fiat currencies. Anonymous sources within the banking sector claim the manipulation involves multiple central banks but verification of these documents has not been independently confirmed."),
    ("TEST 10 — REAL (global news)", "NASA successfully completed the first test flight of its next generation Artemis lunar lander, marking a significant milestone in the agency's plan to return humans to the Moon by 2026. The flight test lasted approximately six hours and all primary objectives were met according to mission control officials.")
]

for idx, (name, text) in enumerate(TEST_CASES, 1):
    print(f"\n--- {idx}. {name} ---")
    try:
        res = predict(text)
        print(f"Result: {res['label']} ({res['confidence']}%) Model: {res['model_used']}")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(1) # to prevent rate limit
