# app.py — Bulletproof startup version

import os
import sys

# ── NLTK downloads — must be first ────────────────────────
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("omw-1.4",   quiet=True)
    print("✅ NLTK ready")
except Exception as e:
    print(f"⚠️ NLTK warning: {e}")

from flask import Flask, render_template, request, jsonify, redirect
from datetime import datetime

app = Flask(__name__)

# ── Environment ────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
print(f"HF_TOKEN: {'loaded' if HF_TOKEN else 'MISSING'}")

# ── Global state ───────────────────────────────────────────
prediction_logs = []
_svm_loaded     = False
_svm_model      = None
_svm_vectorizer = None
_svm_scaler     = None


def get_svm():
    """Load SVM lazily on first request."""
    global _svm_model, _svm_vectorizer, _svm_scaler, _svm_loaded
    if not _svm_loaded:
        from predict import load_model
        _svm_model, _svm_vectorizer, _svm_scaler = load_model()
        _svm_loaded = True
        print("✅ SVM loaded")
    return _svm_model, _svm_vectorizer, _svm_scaler


# ══════════════════════════════════════════════════════════
# ALL ROUTES — defined before any heavy imports
# ══════════════════════════════════════════════════════════

@app.route("/health")
def health():
    return {"status": "ok", "svm_loaded": _svm_loaded}, 200

@app.route("/ping")
def ping():
    return "pong", 200

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
            if len(text) < 10:
                error = "Please enter at least 10 characters."
                return render_template("analyze.html",
                                       error=error, text=text)
            
            result = run_analysis(text)
            # Show Intelligence Report after analysis
            return render_template("report.html",
                                   result=result, text=text)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"Analysis failed: {str(e)}"
            return render_template("analyze.html",
                                   error=error, text=text)
    return render_template("analyze.html", text=text)

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
        return jsonify({"error": str(e), "label": "ERROR",
                        "confidence": 0}), 500

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/how_it_works")
def how_it_works_old():
    return redirect("/how-it-works", code=301)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/metrics")
def metrics_page():
    return render_template("metrics.html")

@app.route("/metrics/json")
def metrics_json():
    return jsonify([{
        "bert_model"    : "Monk3ydluffy/truthlens-bert",
        "bert_accuracy" : 99.2,
        "svm_accuracy"  : 98.2,
        "dataset_size"  : 44000,
    }])

@app.route("/logs")
def logs():
    return jsonify(prediction_logs[-50:] if prediction_logs
                   else {"message": "No predictions yet."})

@app.errorhandler(404)
def not_found(e):
    try:
        return render_template("404.html"), 404
    except:
        return "<h1>404 - Page Not Found</h1><a href='/'>Home</a>", 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ══════════════════════════════════════════════════════════
# ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════

def run_analysis(text: str) -> dict:
    print(f"→ run_analysis() len={len(text)}")

    # Step 1: SVM
    svm_model, svm_vectorizer, svm_scaler = get_svm()
    try:
        from predict import predict as svm_predict
        svm_result = svm_predict(text, svm_model,
                                  svm_vectorizer, svm_scaler)
        print(f"→ SVM: {svm_result['label']} ({svm_result['confidence']}%)")
    except Exception as e:
        print(f"❌ SVM error: {e}")
        raise Exception(f"SVM failed: {e}")

    # Step 2: Decision engine
    try:
        from decision_engine import run_decision_engine
        decision = run_decision_engine(
            text,
            svm_result["label"],
            svm_result["confidence"],
            svm_result["fake_prob"],
            svm_result["real_prob"]
        )
        print(f"→ Decision: {decision['final_label']} "
              f"reason={decision['decision_reason']}")
    except Exception as e:
        print(f"❌ Decision engine error: {e}")
        import traceback
        traceback.print_exc()
        decision = {
            "final_label"      : svm_result["label"],
            "final_confidence" : svm_result["confidence"],
            "confidence_level" : "MEDIUM",
            "decision_reason"  : "svm_fallback",
            "net_score"        : 0,
            "fake_flags"       : [],
            "real_flags"       : [],
            "uncertain_score"  : 0,
        }

    # Step 3: BERT (optional — never crashes the app)
    bert_ok     = False
    bert_result = None
    if HF_TOKEN:
        try:
            from bert_predict import bert_predict_with_timeout
            bert_result = bert_predict_with_timeout(
                text, timeout_seconds=20
            )
            bert_ok = True
            print(f"→ BERT: {bert_result['label']} "
                  f"({bert_result['confidence']}%)")
        except Exception as e:
            print(f"→ BERT failed (using SVM): {e}")

    # Step 4: BERT-first verdict logic
    if bert_ok and bert_result:
        bc = bert_result["confidence"]
        if bc >= 70:
            final_label = bert_result["label"]
            final_conf  = min(bc, 95.0)
            final_fake  = bert_result["fake_prob"]
            final_real  = bert_result["real_prob"]
            model_used  = "BERT (Primary)"
        elif bc >= 55:
            bf = round(bert_result["fake_prob"]*0.8 +
                       svm_result["fake_prob"]*0.2, 2)
            br = round(bert_result["real_prob"]*0.8 +
                       svm_result["real_prob"]*0.2, 2)
            final_label = "FAKE" if bf > br else "REAL"
            final_conf  = round(min(max(bf, br), 95.0), 2)
            final_fake  = bf
            final_real  = br
            model_used  = "BERT+SVM Blend"
        else:
            final_label = "UNCERTAIN"
            final_conf  = bc
            final_fake  = bert_result["fake_prob"]
            final_real  = bert_result["real_prob"]
            model_used  = "BERT (Low Confidence)"

        # Strong rules override BERT
        if abs(decision.get("net_score", 0)) >= 7:
            final_label = decision["final_label"]
            final_conf  = decision["final_confidence"]
            model_used  = "Rules Override"
    else:
        final_label = decision["final_label"]
        final_conf  = decision["final_confidence"]
        final_fake  = svm_result["fake_prob"]
        final_real  = svm_result["real_prob"]
        model_used  = "SVM (BERT unavailable)"

    final_conf = round(min(max(float(final_conf), 50.0), 95.0), 2)
    if final_conf >= 85:   conf_level = "HIGH"
    elif final_conf >= 68: conf_level = "MEDIUM"
    else:                  conf_level = "LOW"

    result = {
        "label"            : final_label,
        "confidence"       : final_conf,
        "confidence_level" : conf_level,
        "fake_prob"        : round(float(final_fake), 2),
        "real_prob"        : round(float(final_real), 2),
        "model_used"       : model_used,
        "decision_reason"  : decision.get("decision_reason", "unknown"),
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
        "svm_fake"         : round(float(svm_result["fake_prob"]), 2),
        "svm_real"         : round(float(svm_result["real_prob"]), 2),
        "svm_confidence"   : round(float(svm_result["confidence"]), 2),
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


# ══════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════
print("✅ Flask app ready — all routes defined")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
