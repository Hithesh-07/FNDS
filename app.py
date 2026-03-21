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
    print(f"HF_TOKEN loaded: {HF_TOKEN[:8]}...")
else:
    print("HF_TOKEN missing - will use SVM only")

# ── Load SVM (always loaded as fallback) ───────────────────
try:
    from predict import load_model, predict as svm_predict
    svm_model, svm_vectorizer, svm_scaler = load_model()
    print("SVM loaded successfully")
    SVM_OK = True
except Exception as e:
    print(f"❌ SVM load failed: {e}")
    SVM_OK = False

# ── In-memory logs ─────────────────────────────────────────
prediction_logs = []


def analyze(text: str) -> dict:
    print(f"→ analyze() start len={len(text)}")

    # ── Step 1: Always run SVM ─────────────────────────────
    try:
        svm_result = svm_predict(
            text, svm_model, svm_vectorizer, svm_scaler
        )
        print(f"→ SVM: {svm_result['label']} ({svm_result['confidence']}%)")
    except Exception as e:
        print(f"❌ SVM failed: {e}")
        import traceback
        traceback.print_exc()
        svm_result = {
            "label": "UNCERTAIN", "confidence": 50.0,
            "fake_prob": 50.0, "real_prob": 50.0,
            "keywords": [], "red_flags": {}
        }

    # ── Step 2: Try BERT (PRIMARY) ─────────────────────────
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

    # ── Step 3: Decision engine credibility rules ──────────
    decision = None
    try:
        from decision_engine import run_decision_engine
        # Feed BERT result if available, else SVM
        base_label = bert_result["label"] if bert_ok else svm_result["label"]
        base_conf  = bert_result["confidence"] if bert_ok else svm_result["confidence"]
        base_fake  = bert_result["fake_prob"] if bert_ok else svm_result["fake_prob"]
        base_real  = bert_result["real_prob"] if bert_ok else svm_result["real_prob"]

        decision = run_decision_engine(
            text, base_label, base_conf, base_fake, base_real
        )
        print(f"→ Decision: {decision['final_label']} reason={decision['decision_reason']}")
    except Exception as e:
        print(f"❌ Decision engine failed: {e}")
        import traceback
        traceback.print_exc()
        decision = {
            "final_label"      : base_label if bert_ok else svm_result["label"],
            "final_confidence" : base_conf if bert_ok else svm_result["confidence"],
            "confidence_level" : "MEDIUM",
            "decision_reason"  : "fallback_direct",
            "net_score"        : 0,
            "fake_flags"       : [],
            "real_flags"       : [],
            "uncertain_score"  : 0,
        }

    # ── Step 4: BERT-FIRST final verdict ───────────────────
    #
    # This is the key logic — BERT dominates
    #
    if bert_ok and bert_result:
        bert_conf  = bert_result["confidence"]
        bert_label = bert_result["label"]
        bert_fake  = bert_result["fake_prob"]
        bert_real  = bert_result["real_prob"]

        if bert_conf >= 70:
            # ── BERT is confident → BERT wins completely ───
            # Check if decision engine also agrees
            engine_label = decision["final_label"]

            if engine_label == bert_label:
                # BERT + Rules agree → high confidence
                final_label = bert_label
                final_conf  = min(bert_conf + 5, 95.0)
                final_fake  = bert_fake
                final_real  = bert_real
                model_used  = "BERT (Primary) + Rules confirmed"
                reason      = "bert_rules_agree"
            else:
                # BERT confident but rules disagree
                # Rules override only if very strong (net_score >= 7)
                net = decision.get("net_score", 0)
                if abs(net) >= 7:
                    final_label = decision["final_label"]
                    final_conf  = decision["final_confidence"]
                    final_fake  = bert_fake
                    final_real  = bert_real
                    model_used  = "Rules Override"
                    reason      = "rules_override_bert"
                else:
                    # BERT wins over weak rule disagreement
                    final_label = bert_label
                    final_conf  = bert_conf
                    final_fake  = bert_fake
                    final_real  = bert_real
                    model_used  = "BERT (Primary)"
                    reason      = "bert_dominant"

        elif bert_conf >= 55:
            # ── BERT moderately confident → weighted blend ─
            # BERT 80%, SVM 20%
            blended_fake = round(bert_fake * 0.80 +
                                  svm_result["fake_prob"] * 0.20, 2)
            blended_real = round(bert_real * 0.80 +
                                  svm_result["real_prob"] * 0.20, 2)

            if blended_fake > blended_real:
                final_label = "FAKE"
                final_conf  = round(min(blended_fake, 95.0), 2)
            else:
                final_label = "REAL"
                final_conf  = round(min(blended_real, 95.0), 2)

            final_fake  = blended_fake
            final_real  = blended_real
            model_used  = "BERT (80%) + SVM (20%) Blend"
            reason      = "bert_svm_weighted_blend"

            # Apply decision engine on top of blend
            net = decision.get("net_score", 0)
            if abs(net) >= 6:
                final_label = decision["final_label"]
                final_conf  = decision["final_confidence"]
                reason      = "rules_override_blend"

        else:
            # ── BERT not confident → UNCERTAIN ─────────────
            # Don't use SVM — show UNCERTAIN honestly
            gap = abs(bert_fake - bert_real)

            if gap < 20:
                final_label = "UNCERTAIN"
                final_conf  = round(bert_conf, 2)
                final_fake  = bert_fake
                final_real  = bert_real
                model_used  = "BERT (Low Confidence → UNCERTAIN)"
                reason      = "bert_low_confidence_uncertain"
            else:
                # Gap is meaningful even if confidence low
                # Use BERT but show as medium confidence
                final_label = bert_label
                final_conf  = round(min(bert_conf, 72.0), 2)
                final_fake  = bert_fake
                final_real  = bert_real
                model_used  = "BERT (Moderate)"
                reason      = "bert_moderate"

    else:
        # ── BERT completely failed → SVM fallback ──────────
        # Apply decision engine on SVM result
        final_label = decision["final_label"]
        final_conf  = decision["final_confidence"]
        final_fake  = svm_result["fake_prob"]
        final_real  = svm_result["real_prob"]
        model_used  = "SVM (BERT unavailable)"
        reason      = decision["decision_reason"]

    # ── Step 5: Final confidence level ────────────────────
    final_conf = round(min(max(final_conf, 50.0), 95.0), 2)

    if final_conf >= 85:
        conf_level = "HIGH"
    elif final_conf >= 68:
        conf_level = "MEDIUM"
    else:
        conf_level = "LOW"

    # ── Step 6: Build result ───────────────────────────────
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
        "bert_used"   : bert_ok,
        "text_preview": text[:80],
    })

    print(f"→ FINAL: {final_label} {final_conf}% via {model_used}")
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
            else:
                result = analyze(text)
                # Redirect to report page with result
                return render_template("report.html",
                                       result=result, text=text)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"Analysis failed: {str(e)}"
    return render_template("analyze.html",
                           result=result, text=text, error=error)

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/about")
def about():
    return render_template("about.html")

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
