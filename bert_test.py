"""
bert_test.py — 30-case automated test suite for BERT model

Run:
    python bert_test.py

Target: 28/30 or higher.
If below 25/30, increase EPOCHS in bert_train.py and retrain.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from bert_predict import bert_predict

test_cases = [
    ("FAKE", "SHOCKING: Scientists PROVE drinking bleach cures cancer and doctors are HIDING it!"),
    ("FAKE", "BREAKING: Bill Gates admits COVID vaccines contain microchips to track population EXPOSED!"),
    ("FAKE", "Pope Francis SECRETLY endorses flat earth theory in leaked Vatican document bombshell!"),
    ("FAKE", "5G towers are KILLING birds and government is COVERING IT UP says whistleblower!"),
    ("FAKE", "NASA FAKED the moon landing in 1969, new leaked footage PROVES Hollywood studio used!"),
    ("FAKE", "A new study found that COVID vaccine reduces human fertility by 30 percent."),
    ("FAKE", "Senior White House officials confirmed Biden signed secret executive order banning guns."),
    ("FAKE", "Indian government quietly passed law allowing police to arrest citizens without warrant."),
    ("FAKE", "Leaked WHO memo reveals masks were never effective and officials knew from beginning."),
    ("FAKE", "Harvard economists warned India will face complete economic collapse within 18 months."),
    ("FAKE", "URGENT: Hollywood elites running child trafficking ring from underground tunnels EXPOSED!"),
    ("FAKE", "George Soros FUNDS secret army to overthrow democratic governments worldwide!"),
    ("FAKE", "BANNED VIDEO: Doctors CURED diabetes using one fruit but Big Pharma SILENCING them!"),
    ("FAKE", "Alien spacecraft LANDED in Arizona desert, US military covering up contact for 30 years!"),
    ("FAKE", "CHEMTRAILS CONFIRMED: Retired pilot LEAKS proof governments spraying mind control chemicals!"),
    ("REAL", "The Federal Reserve raised interest rates by 25 basis points citing persistent inflation."),
    ("REAL", "Apple reported quarterly revenue of 94.8 billion dollars beating analyst expectations."),
    ("REAL", "NASA James Webb Space Telescope captured new images of galaxy cluster 4.6 billion light years away."),
    ("REAL", "United Nations climate report warned global sea levels could rise by one meter by 2100."),
    ("REAL", "India Supreme Court ruled electoral bonds scheme violated right to information of citizens."),
    ("REAL", "Study in The Lancet found ultra processed foods linked to increased risk of heart disease."),
    ("REAL", "SpaceX successfully landed its Starship rocket for the first time after several failed attempts."),
    ("REAL", "Reserve Bank of India kept repo rate unchanged at 6.5 percent for sixth consecutive meeting."),
    ("REAL", "Microsoft announced layoffs affecting 1900 employees in gaming division after Activision acquisition."),
    ("REAL", "New strain of bird flu H5N1 detected in dairy cattle across 12 US states according to CDC."),
    ("REAL", "WHO declared mpox a global health emergency as cases surge across Central Africa."),
    ("REAL", "Pentagon confirmed unidentified aerial phenomena observed by Navy pilots between 2019 and 2021."),
    ("REAL", "Scientists warned Amazon rainforest approaching tipping point beyond which it cannot recover."),
    ("REAL", "Pakistan imposed emergency economic measures after foreign exchange reserves dropped critically."),
    ("REAL", "Cybersecurity firm reported data breach exposed personal information of 560 million Ticketmaster customers."),
    ("UNCERTAIN", "Some researchers suggest certain foods may improve brain health, though more evidence is required."),
    ("UNCERTAIN", "Studies indicate coffee might reduce Alzheimer's risk, but scientists say further research is needed."),
    ("UNCERTAIN", "Preliminary findings suggest the new drug could be effective, though clinical trials are ongoing."),
    ("UNCERTAIN", "Some experts believe the economy may slow down next year, while others dispute this assessment."),
    ("UNCERTAIN", "Reports suggest the company may be planning layoffs, though this has not been officially confirmed."),
    ("UNCERTAIN", "A controversial new study claims certain exercises might reverse aging, but results are disputed."),
    ("UNCERTAIN", "Sources indicate there could be a policy change soon, though officials have not yet confirmed."),
    ("UNCERTAIN", "Some scientists argue the findings are promising, while others say the methodology is unclear."),
    ("UNCERTAIN", "The treatment appears to work in some patients, but researchers say more trials are needed."),
    ("UNCERTAIN", "It is believed the new regulations may come into effect next month, pending final approval."),
]


def run_tests():
    print("\n" + "=" * 65)
    print("  BERT (RoBERTa) -- TruthLens Test Suite (30 cases)")
    print("=" * 65 + "\n")

    correct = 0
    wrong_cases = []

    for i, (expected, text) in enumerate(test_cases, 1):
        result = bert_predict(text)
        pred   = result["label"]
        conf   = result["confidence"]

        if pred == expected:
            correct += 1
            status = "[OK]"
        else:
            wrong_cases.append((i, expected, pred, conf, text[:60]))
            status = "[FAIL]"

        print(f"  [{i:02d}] {status} Expected: {expected:<5} | Got: {pred:<10} | Conf: {conf}%")
        print(f"       {text[:70]}{'...' if len(text) > 70 else ''}")
        print()

    pct = correct / len(test_cases) * 100
    print("=" * 65)
    print(f"  SCORE: {correct}/{len(test_cases)}  ({pct:.1f}%)")

    if correct == 30:
        print("  PERFECT SCORE!")
    elif correct >= 28:
        print("  EXCELLENT!")
    elif correct >= 25:
        print("  GOOD!")
    else:
        print("  NEEDS IMPROVEMENT — consider increasing EPOCHS in bert_train.py")

    if wrong_cases:
        print(f"\n  Wrong predictions ({len(wrong_cases)}):")
        for n, exp, got, conf, txt in wrong_cases:
            print(f"     [{n:02d}] Expected {exp}, got {got} ({conf}%) -- {txt}...")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_tests()
