"""
test_model.py — 20 automated test cases for TruthLens

Run after training:
    python test_model.py

Target: 18/20 or higher.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from predict import load_model, cached_predict

load_model()

TEST_CASES = [
    ("REAL", "The Federal Reserve raised interest rates by 0.25 percent on Wednesday."),
    ("REAL", "NASA successfully launched the Artemis mission to the Moon."),
    ("REAL", "Apple reported quarterly earnings of 90 billion dollars beating estimates."),
    ("REAL", "The World Health Organization approved a new malaria vaccine for children."),
    ("REAL", "India's GDP grew by 6.5 percent in the last fiscal quarter according to government data."),
    ("FAKE", "SHOCKING: Scientists PROVE that 5G towers cause cancer and the government is HIDING it!"),
    ("FAKE", "BREAKING: Bill Gates admits microchips are inside COVID vaccines EXPOSED by whistleblower!"),
    ("FAKE", "Pope Francis SECRETLY endorses flat earth theory in leaked Vatican document!"),
    ("FAKE", "Donald Trump won 2020 election, voting machines were rigged says INSIDER source!"),
    ("FAKE", "Drinking bleach cures COVID-19, doctors BANNED from telling you the TRUTH!"),
    ("REAL", "The Supreme Court ruled in favor of environmental regulations today."),
    ("REAL", "Tesla announced a new battery technology with 500 mile range."),
    ("FAKE", "Hollywood actress EXPOSED for running secret satanic cult in California!"),
    ("FAKE", "Government CHEMTRAILS confirmed by whistleblower pilot, full COVER-UP revealed!"),
    ("REAL", "Scientists discover new species of deep sea fish near Pacific Ocean floor."),
    ("REAL", "UK parliament passed new legislation on data privacy and AI regulation."),
    ("FAKE", "URGENT: Mainstream media HIDING alien contact, NASA astronaut LEAKS footage!"),
    ("FAKE", "George Soros FUNDS antifa to DESTROY America says leaked document!"),
    ("REAL", "The United Nations climate report warns of rising sea levels by 2050."),
    ("REAL", "Microsoft acquires gaming company in deal worth 15 billion dollars."),
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
    total = len(TEST_CASES)
    print("\n" + "=" * 70)
    print(f"  TruthLens — Automated Model Test Suite ({total} cases)")
    print("=" * 70 + "\n")

    correct = 0
    for i, (expected, text) in enumerate(TEST_CASES, 1):
        result    = cached_predict(text)
        predicted = result.get("label", "ERROR")
        conf      = result.get("confidence", 0)
        conf_lvl  = result.get("confidence_level", "?")

        is_correct = (predicted == expected)
        match_sym = "[OK]" if is_correct else (
            "[~]" if predicted == "UNCERTAIN" and expected in ("REAL", "FAKE") else "[FAIL]"
        )
        if is_correct:
            correct += 1

        print(f"  {i:>2}. {match_sym} Expected:{expected:<10} Got:{predicted:<10} "
              f"Conf:{conf:>5.1f}% ({conf_lvl})")
        print(f"      {text[:80]}{'...' if len(text) > 80 else ''}\n")

    score = correct
    pct   = score / total * 100
    print("=" * 70)
    print(f"  FINAL SCORE : {score}/{total}  ({pct:.0f}%)")
    if score >= (total * 0.85):
        print(f"  [PASS] Target achieved: {score}/{total} (85%+)!")
    else:
        print(f"  [NOTE] Below 85% target — consider retraining or refining logic.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_tests()
