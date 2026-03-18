import os
import sys
from colorama import Fore, Style, init

init(autoreset=True)

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════╗
║        🗞️  FAKE NEWS DETECTOR  v1.0           ║
║     Powered by Logistic Regression + NLP      ║
╚══════════════════════════════════════════════╝{Style.RESET_ALL}
"""

MENU = f"""
{Fore.YELLOW}Choose an option:
  [1] Train / Retrain the model
  [2] Analyze a news article (type or paste)
  [3] Analyze from a .txt file
  [4] Run batch test on sample headlines
  [5] Show model info
  [0] Exit
{Style.RESET_ALL}"""


def print_result(result: dict):
    label = result["label"]
    conf  = result["confidence"]

    if label == "FAKE":
        color = Fore.RED
        icon  = "🚨"
    else:
        color = Fore.GREEN
        icon  = "✅"

    print("\n" + "─" * 48)
    print(f"  {icon}  Verdict    : {color}{Style.BRIGHT}{label}{Style.RESET_ALL}")
    print(f"  📊  Confidence : {conf}%")
    print(f"  📉  Fake Prob  : {result['fake_prob']}%")
    print(f"  📈  Real Prob  : {result['real_prob']}%")
    if "keywords" in result:
        print(f"  🔑  Keywords   : {', '.join(result['keywords'])}")
    print("─" * 48 + "\n")


def option_train():
    from train import train
    train()


def option_analyze():
    from predict import load_model, predict
    model, vectorizer = load_model()

    print(f"\n{Fore.CYAN}Paste your news text below.")
    print("(Type END on a new line when done)\n")

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)

    text = " ".join(lines).strip()
    if not text:
        print(f"{Fore.RED}⚠️  No text entered.")
        return

    result = predict(text, model, vectorizer)
    print_result(result)


def option_file():
    from predict import load_model, predict
    model, vectorizer = load_model()

    path = input("\nEnter path to .txt file: ").strip()
    if not os.path.exists(path):
        print(f"{Fore.RED}⚠️  File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n📄 File loaded ({len(text)} characters). Analyzing...")
    result = predict(text, model, vectorizer)
    print_result(result)


def option_batch():
    from predict import load_model, predict
    model, vectorizer = load_model()

    samples = [
        ("REAL", "The Federal Reserve raised interest rates by 0.25% on Wednesday, citing continued inflation concerns and a strong labor market."),
        ("FAKE", "BREAKING: Scientists confirm that drinking bleach cures all diseases. Government hiding this secret for decades!"),
        ("REAL", "NASA's James Webb Space Telescope has captured new images of a galaxy cluster 4.6 billion light years away."),
        ("FAKE", "Pope Francis endorsed Donald Trump for President of the United States in a surprise announcement from the Vatican."),
        ("REAL", "The World Health Organization released updated guidelines on antibiotic resistance, urging hospitals to limit overuse."),
        ("FAKE", "Hillary Clinton runs a secret child trafficking ring from a Washington D.C. pizzeria basement."),
    ]

    print(f"\n{Fore.CYAN}{'─'*60}")
    print(f"  Running batch test on {len(samples)} sample headlines...")
    print(f"{'─'*60}{Style.RESET_ALL}\n")

    correct = 0
    for i, (true_label, headline) in enumerate(samples, 1):
        result  = predict(headline, model, vectorizer)
        pred    = result["label"]
        conf    = result["confidence"]
        match   = pred == true_label
        correct += int(match)

        status = f"{Fore.GREEN}✅ CORRECT" if match else f"{Fore.RED}❌ WRONG"
        label_color = Fore.GREEN if pred == "REAL" else Fore.RED

        print(f"  [{i}] {Fore.WHITE}{headline[:65]}...")
        print(f"       Expected: {true_label}  |  "
              f"Predicted: {label_color}{pred}{Style.RESET_ALL}  |  "
              f"Confidence: {conf}%  |  {status}{Style.RESET_ALL}")
        print()

    acc = (correct / len(samples)) * 100
    print(f"{Fore.CYAN}{'─'*60}")
    print(f"  Batch Accuracy: {correct}/{len(samples)} = {acc:.1f}%")
    print(f"{'─'*60}{Style.RESET_ALL}\n")


def option_info():
    import joblib
    try:
        model = joblib.load("model/model.pkl")
        vec   = joblib.load("model/vectorizer.pkl")
        print(f"\n{Fore.CYAN}── Model Info ──────────────────────────────")
        print(f"  Type          : {type(model).__name__}")
        if hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'estimator'):
            print(f"  Base Estimator: {type(model.estimators_[0].estimator).__name__}")
        else:
            if hasattr(model, 'max_iter'): print(f"  Max Iter      : {model.max_iter}")
            if hasattr(model, 'solver'):  print(f"  Solver        : {model.solver}")
            if hasattr(model, 'C'):       print(f"  C (Reg)       : {model.C}")
        print(f"  Vocabulary    : {len(vec.vocabulary_)} tokens")
        print(f"  N-gram Range  : {vec.ngram_range}")
        print(f"  Max Features  : {vec.max_features}")
        print(f"{'─'*44}{Style.RESET_ALL}\n")
    except FileNotFoundError:
        print(f"{Fore.RED}⚠️  No trained model found. Run option [1] first.")


def main():
    print(BANNER)

    # Auto-check if model exists
    if not os.path.exists("model/model.pkl"):
        print(f"{Fore.YELLOW}⚠️  No trained model found.")
        ans = input("   Train the model now? (y/n): ").strip().lower()
        if ans == "y":
            option_train()
        else:
            print(f"{Fore.RED}   Cannot predict without a trained model. Exiting.")
            sys.exit(0)

    while True:
        print(MENU)
        choice = input("Enter choice: ").strip()

        if   choice == "1": option_train()
        elif choice == "2": option_analyze()
        elif choice == "3": option_file()
        elif choice == "4": option_batch()
        elif choice == "5": option_info()
        elif choice == "0":
            print(f"\n{Fore.CYAN}👋 Goodbye!\n")
            break
        else:
            print(f"{Fore.RED}⚠️  Invalid option. Try again.")


if __name__ == "__main__":
    main()
