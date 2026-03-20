# app.py
import os
import json
import nltk
from flask import Flask, render_template, request, jsonify
from datetime import datetime

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
try:
    from predict import load_model, predict as svm_predict
    svm_model, svm_vectorizer, svm_scaler = load_model()
    print("✅ SVM loaded successfully")
    SVM_OK = True
except Exception as e:
    print(f"❌ SVM load failed: {e}")
    SVM_OK = False

# ── In-memory logs ─────────────────────────────────────────
prediction_logs = []


def analyze(text: str) -> dict:
    print(f"→ analyze() start len={len(text)}")

    # ── SVM (always runs) ──────────────────────────────────
    try:
        svm_result = svm_predict(
            text, svm_model, svm_vectorizer, svm_scaler
        )
        print(f"→ SVM OK: {svm_result['label']}")
    except Exception as e:
        print(f"❌ SVM crashed: {e}")
        import traceback
        traceback.print_exc()
        # Return safe default instead of crashing
        return {
            "label"            : "UNCERTAIN",
            "confidence"       : 50.0,
            "confidence_level" : "LOW",
            "fake_prob"        : 50.0,
            "real_prob"        : 50.0,
            "model_used"       : "ERROR - SVM failed",
            "decision_reason"  : "svm_error",
            "net_score"        : 0,
            "uncertain_score"  : 0,
            "credibility_flags": [],
            "real_flags"       : [],
            "bert_ok"          : False,
            "bert_label"       : "N/A",
            "bert_fake"        : 0,
            "bert_real"        : 0,
            "bert_confidence"  : 0,
            "svm_label"        : "ERROR",
            "svm_fake"         : 50,
            "svm_real"         : 50,
            "svm_confidence"   : 50,
            "keywords"         : [],
            "red_flags"        : {},
            "error"            : str(e),
        }

    # ── Decision engine ────────────────────────────────────
    decision = None
    try:
        from decision_engine import run_decision_engine
        decision = run_decision_engine(
            text,
            svm_result["label"],
            svm_result["confidence"],
            svm_result["fake_prob"],
            svm_result["real_prob"]
        )
        print(f"→ Decision OK: {decision['final_label']}")
    except Exception as e:
        print(f"❌ Decision engine crashed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to raw SVM result
        decision = {
            "final_label"      : svm_result["label"],
            "final_confidence" : svm_result["confidence"],
            "confidence_level" : "MEDIUM",
            "decision_reason"  : "svm_direct_fallback",
            "net_score"        : 0,
            "fake_flags"       : [],
            "real_flags"       : [],
            "uncertain_score"  : 0,
        }

    # ── BERT (optional) ────────────────────────────────────
    bert_ok     = False
    bert_result = None
    if HF_TOKEN:
        try:
            from bert_predict import bert_predict
            bert_result = bert_predict(text)
            bert_ok     = True
            print(f"→ BERT OK: {bert_result['label']}")
        except Exception as e:
            print(f"→ BERT failed: {e}")

    # ── Final probabilities ────────────────────────────────
    if bert_ok and bert_result:
        final_fake = bert_result["fake_prob"]
        final_real = bert_result["real_prob"]
        model_used = "BERT (Primary) ✅"
    else:
        final_fake = svm_result["fake_prob"]
        final_real = svm_result["real_prob"]
        model_used = "SVM (BERT unavailable)"

    result = {
        "label"            : decision["final_label"],
        "confidence"       : decision["final_confidence"],
        "confidence_level" : decision["confidence_level"],
        "fake_prob"        : final_fake,
        "real_prob"        : final_real,
        "model_used"       : model_used,
        "decision_reason"  : decision["decision_reason"],
        "net_score"        : decision.get("net_score", 0),
        "uncertain_score"  : decision.get("uncertain_score", 0),
        "credibility_flags": decision.get("fake_flags", []),
        "real_flags"       : decision.get("real_flags", []),
        "bert_ok"          : bert_ok,
        "bert_label"       : bert_result["label"] if bert_ok else "N/A",
        "bert_fake"        : bert_result["fake_prob"] if bert_ok else 0,
        "bert_real"        : bert_result["real_prob"] if bert_ok else 0,
        "bert_confidence"  : bert_result["confidence"] if bert_ok else 0,
        "svm_label"        : svm_result["label"],
        "svm_fake"         : svm_result["fake_prob"],
        "svm_real"         : svm_result["real_prob"],
        "svm_confidence"   : svm_result["confidence"],
        "keywords"         : svm_result.get("keywords", []),
        "red_flags"        : svm_result.get("red_flags", {}),
    }

    prediction_logs.append({
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label"       : result["label"],
        "confidence"  : result["confidence"],
        "model_used"  : model_used,
        "bert_used"   : bert_ok,
        "text_preview": text[:80],
    })

    print(f"→ Final: {result['label']} {result['confidence']}%")
    return result


@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Send JSON with text field"}), 400
        text = data["text"].strip()
        if not text or len(text) < 5:
            return jsonify({"error": "Text too short"}), 400
        result = analyze(text)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error"            : str(e),
            "label"            : "ERROR",
            "confidence"       : 0,
            "confidence_level" : "LOW",
            "fake_prob"        : 0,
            "real_prob"        : 0,
            "model_used"       : "ERROR",
            "decision_reason"  : "error",
            "net_score"        : 0,
            "credibility_flags": [],
            "bert_ok"          : False,
            "svm_label"        : "ERROR",
            "svm_fake"         : 0,
            "svm_real"         : 0,
            "svm_confidence"   : 0,
            "keywords"         : [],
            "red_flags"        : {},
        }), 500


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error  = None
    text   = ""
    if request.method == "POST":
        try:
            text = request.form.get("news_text", "").strip()
            if len(text) < 10:
                error = "Please enter at least 10 characters."
            else:
                result = analyze(text)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"Analysis failed: {str(e)}"
    return render_template("index.html",
                           result=result, text=text, error=error)


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/metrics")
def metrics():
    return jsonify([{
        "model_type"    : "BERT Primary + SVM Fallback",
        "bert_model"    : "Arko007/fake-news-roberta-5M (99.28%)",
        "svm_accuracy"  : 98.2,
        "bert_accuracy" : 99.28,
        "architecture"  : "BERT → SVM Fallback → Decision Engine",
    }])


@app.route("/logs")
def logs():
    if not prediction_logs:
        return jsonify({"message": "No predictions yet."})
    return jsonify(prediction_logs[-50:])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
