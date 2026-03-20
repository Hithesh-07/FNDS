# app.py
import os
import json
import nltk
import warnings
from flask import Flask, render_template, request, jsonify
from datetime import datetime

warnings.filterwarnings("ignore")  # suppress version warnings

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

app = Flask(__name__)

# ── Check HF Token ─────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    print(f"✅ HF_TOKEN loaded: {HF_TOKEN[:8]}...")
else:
    print("⚠️  HF_TOKEN missing — will use SVM only")

# ── Load SVM (always loaded as fallback) ───────────────────
from predict import load_model, predict as svm_predict
svm_model, svm_vectorizer, svm_scaler = load_model()
print("✅ SVM loaded as fallback")

# ── In-memory logs ─────────────────────────────────────────
prediction_logs = []


def analyze(text: str) -> dict:
    """
    Priority order:
    1. Credibility rules (always runs)
    2. BERT primary model
    3. SVM fallback if BERT fails
    """
    from decision_engine import run_decision_engine_raw, calculate_scores

    text = text.strip()

    # ── Step 1: Get credibility signals first ──────────────
    scores = calculate_scores(text)
    cred_net_score   = scores["net_score"]
    uncertain_score  = scores["uncertain_score"]
    fake_flags       = scores["fake_flags"]
    real_flags       = scores["real_flags"]

    # ── Step 2: Try BERT first (PRIMARY) ───────────────────
    bert_ok     = False
    bert_result = None
    model_used  = "SVM (BERT unavailable)"

    if HF_TOKEN:
        try:
            from bert_predict import bert_predict
            bert_result = bert_predict(text)
            bert_ok     = True
            model_used  = "BERT (Primary)"
            print(f"✅ BERT success: {bert_result['label']} ({bert_result['confidence']}%)")
        except Exception as e:
            print(f"⚠️  BERT failed: {e} — switching to SVM")

    # ── Step 3: SVM fallback if BERT failed ────────────────
    svm_result = svm_predict(text, svm_model, svm_vectorizer, svm_scaler)

    if not bert_ok:
        model_used = "SVM (BERT unavailable)"

    # ── Step 4: Choose which model result to use ───────────
    if bert_ok:
        # BERT worked — use BERT as base
        ml_label  = bert_result["label"]
        ml_conf   = bert_result["confidence"]
        fake_prob = bert_result["fake_prob"]
        real_prob = bert_result["real_prob"]
    else:
        # BERT failed — use SVM
        ml_label  = svm_result["label"]
        ml_conf   = svm_result["confidence"]
        fake_prob = svm_result["fake_prob"]
        real_prob = svm_result["real_prob"]

    # ── Step 5: Run decision engine on top ─────────────────
    decision = run_decision_engine_raw(
        text, ml_label, ml_conf, fake_prob, real_prob
    )

    # ── Step 6: Build final result ─────────────────────────
    result = {
        # Final verdict
        "label"            : decision["final_label"],
        "confidence"       : decision["final_confidence"],
        "confidence_level" : decision["confidence_level"],

        # Probabilities
        "fake_prob"        : fake_prob,
        "real_prob"        : real_prob,

        # Model info
        "model_used"       : model_used,
        "bert_ok"          : bert_ok,

        # BERT details
        "bert_label"       : bert_result["label"] if bert_ok else "N/A",
        "bert_fake"        : bert_result["fake_prob"] if bert_ok else 0,
        "bert_real"        : bert_result["real_prob"] if bert_ok else 0,
        "bert_confidence"  : bert_result["confidence"] if bert_ok else 0,

        # SVM details (always available)
        "svm_label"        : svm_result["label"],
        "svm_fake"         : svm_result["fake_prob"],
        "svm_real"         : svm_result["real_prob"],
        "svm_confidence"   : svm_result["confidence"],

        # Decision engine
        "decision_reason"  : decision["decision_reason"],
        "net_score"        : cred_net_score,
        "uncertain_score"  : uncertain_score,
        "credibility_flags": fake_flags,
        "real_flags"       : real_flags,

        # Keywords and red flags from SVM
        "keywords"         : svm_result.get("keywords", []),
        "red_flags"        : svm_result.get("red_flags", {}),
    }

    # Log it
    prediction_logs.append({
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label"       : result["label"],
        "confidence"  : result["confidence"],
        "model_used"  : model_used,
        "bert_used"   : bert_ok,
        "text_preview": text[:80],
    })

    return result


# ── Web UI ─────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text   = ""
    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        if len(text) < 10:
            return render_template("index.html",
                                   error="Please enter at least 10 characters.",
                                   text=text)
        if text:
            result = analyze(text)
    return render_template("index.html", result=result, text=text)


# ── REST API ───────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Send JSON with text field"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text is empty"}), 400
    result = analyze(text)
    return jsonify(result)


# ── Metrics ────────────────────────────────────────────────
@app.route("/metrics")
def metrics():
    return jsonify([{
        "timestamp"     : "2026-03-20 19:10:00",
        "model_type"    : "BERT (Primary) + SVM Ensemble (Fallback)",
        "bert_model"    : "Arko007/fake-news-roberta-5M (99.28%)",
        "svm_model"     : "LinearSVC + LR + PAC Ensemble",
        "dataset_size"  : 44000,
        "svm_accuracy"  : 98.2,
        "bert_accuracy" : 99.28,
        "tfidf_features": 50000,
        "ngram_range"   : "1-3",
        "architecture"  : "BERT Primary → SVM Fallback → Decision Engine",
    }])


# ── Logs ───────────────────────────────────────────────────
@app.route("/logs")
def logs():
    if not prediction_logs:
        return jsonify({"message": "No predictions yet this session."})
    return jsonify(prediction_logs[-50:])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
