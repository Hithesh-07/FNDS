import os
import json
import nltk
from flask import Flask, render_template, request, jsonify, redirect
from datetime import datetime

# Download NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

app = Flask(__name__)

# ── Environment ────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    print(f"HF_TOKEN loaded: {HF_TOKEN[:8]}...")
else:
    print("WARNING: No HF_TOKEN")

# ── LAZY model loading — does NOT run at startup ───────────
# This is the key fix — models load on first request not on boot
_svm_model      = None
_svm_vectorizer = None
_svm_scaler     = None
_svm_loaded     = False

def get_svm():
    global _svm_model, _svm_vectorizer, _svm_scaler, _svm_loaded
    if not _svm_loaded:
        try:
            from predict import load_model
            _svm_model, _svm_vectorizer, _svm_scaler = load_model()
            _svm_loaded = True
            print("✅ SVM loaded on first request")
        except Exception as e:
            print(f"❌ SVM load failed: {e}")
            raise
    return _svm_model, _svm_vectorizer, _svm_scaler

# ── In-memory logs ─────────────────────────────────────────
prediction_logs = []

# ── ROUTES — defined immediately so Gunicorn can bind ──────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze_page():
    result = None
    error  = None
    text   = ""
    if request.method == "POST":
        try:
            text = request.form.get("news_text", "").strip()
            if not text or len(text) < 5:
                error = "Please enter at least a few words to analyze."
            else:
                result = run_analysis(text)
                return render_template("report.html",
                                       result=result, text=text)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"Analysis failed: {str(e)}"
    return render_template("analyze.html",
                           result=result, text=text, error=error)

@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Send JSON with text field"}), 400
        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text is empty"}), 400
        result = run_analysis(text)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "label": "ERROR"}), 500

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/how_it_works")
def how_it_works_redirect():
    return redirect("/how-it-works", code=301)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/metrics")
def metrics():
    return render_template("metrics.html")

@app.route("/metrics/json")
def metrics_json():
    return jsonify([{
        "timestamp"     : "2026-03-21 00:00:00",
        "bert_model"    : "Monk3ydluffy/truthlens-bert",
        "bert_accuracy" : 99.2,
        "svm_accuracy"  : 98.2,
        "dataset_size"  : 44000,
        "architecture"  : "BERT Primary → SVM Fallback → Decision Engine",
    }])

@app.route("/logs")
def logs():
    if not prediction_logs:
        return jsonify({"message": "No predictions yet."})
    return jsonify(prediction_logs[-50:])

@app.route("/health")
def health():
    return jsonify({"status": "ok", "svm_loaded": _svm_loaded, "version": "v3.0-lazy"}), 200

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ── Main analysis function ─────────────────────────────────
def run_analysis(text: str) -> dict:
    print(f"→ run_analysis() len={len(text)}")

    # Load SVM lazily
    svm_model, svm_vectorizer, svm_scaler = get_svm()

    # SVM prediction
    try:
        from predict import predict as svm_predict
        svm_result = svm_predict(text, svm_model, svm_vectorizer, svm_scaler)
        print(f"→ SVM: {svm_result['label']} ({svm_result['confidence']}%)")
    except Exception as e:
        print(f"❌ SVM error: {e}")
        # Build basic fallback result if SVM fails
        svm_result = {"label": "ERROR", "confidence": 0, "fake_prob": 0, "real_prob": 0, "keywords": [], "red_flags": {}}

    # Decision engine
    try:
        from decision_engine import run_decision_engine
        decision = run_decision_engine(
            text,
            svm_result["label"],
            svm_result["confidence"],
            svm_result["fake_prob"],
            svm_result["real_prob"]
        )
    except Exception as e:
        print(f"❌ Decision engine error: {e}")
        decision = {
            "final_label"      : svm_result["label"],
            "final_confidence" : svm_result["confidence"],
            "confidence_level" : "MEDIUM",
            "decision_reason"  : "svm_direct",
            "net_score"        : 0,
            "fake_flags"       : [],
            "real_flags"       : [],
            "uncertain_score"  : 0,
        }

    # BERT (optional)
    bert_ok     = False
    bert_result = None
    if HF_TOKEN:
        try:
            from bert_predict import bert_predict_with_timeout
            bert_result = bert_predict_with_timeout(text, timeout_seconds=20)
            bert_ok     = True
            print(f"→ BERT: {bert_result['label']} ({bert_result['confidence']}%)")
        except Exception as e:
            print(f"→ BERT failed: {e}")

    # BERT-first final verdict
    if bert_ok and bert_result:
        bert_conf = bert_result["confidence"]
        if bert_conf >= 70:
            final_label = bert_result["label"]
            final_conf  = min(bert_conf, 95.0)
            final_fake  = bert_result["fake_prob"]
            final_real  = bert_result["real_prob"]
            model_used  = "BERT (Primary)"
            reason      = "bert_stable"
        elif bert_conf >= 55:
            blended_fake = round(bert_result["fake_prob"]*0.8 + svm_result["fake_prob"]*0.2, 2)
            blended_real = round(bert_result["real_prob"]*0.8 + svm_result["real_prob"]*0.2, 2)
            final_label  = "FAKE" if blended_fake > blended_real else "REAL"
            final_conf   = round(min(max(blended_fake, blended_real), 95.0), 2)
            final_fake   = blended_fake
            final_real   = blended_real
            model_used   = "BERT+SVM Blend"
            reason      = "weighted_consensus"
        else:
            final_label = "UNCERTAIN"
            final_conf  = bert_conf
            final_fake  = bert_result["fake_prob"]
            final_real  = bert_result["real_prob"]
            model_used  = "BERT (Low Confidence)"
            reason      = "low_prob_uncertainty"

        # Apply decision engine override for strong credibility signals
        net = decision.get("net_score", 0)
        if abs(net) >= 7:
            final_label = decision["final_label"]
            final_conf  = decision["final_confidence"]
            model_used  = "Rules Override"
            reason      = decision.get("decision_reason", "credibility_override")
    else:
        final_label = decision["final_label"]
        final_conf  = decision["final_confidence"]
        final_fake  = svm_result["fake_prob"]
        final_real  = svm_result["real_prob"]
        model_used  = "SVM (BERT unavailable)"
        reason      = decision.get("decision_reason", "svm_fallback")

    final_conf = round(min(max(final_conf, 50.0), 95.0), 2)
    if final_conf >= 85: conf_level = "HIGH"
    elif final_conf >= 68: conf_level = "MEDIUM"
    else: conf_level = "LOW"

    result = {
        "label"            : final_label,
        "confidence"       : final_conf,
        "confidence_level" : conf_level,
        "fake_prob"        : final_fake,
        "real_prob"        : final_real,
        "model_used"       : model_used,
        "decision_reason"  : reason,
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
        "label"       : final_label,
        "confidence"  : final_conf,
        "model_used"  : model_used,
        "text_preview": text[:80],
    })

    print(f"→ Final: {final_label} {final_conf}% via {model_used}")
    return result


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
