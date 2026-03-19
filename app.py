"""
app.py — Aletheia Flask Application

Automatically uses BERT (RoBERTa) if fine-tuned model exists,
otherwise falls back to SVM ensemble.
"""
import os
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load Models ──────────────────────────────────────────
from predict import load_model, cached_predict
load_model()

from bert_predict import bert_predict

def analyze(text: str) -> dict:
    """
    Hybrid Decision Flow (User Requested):
    1. Check Rules (Fake Score)
    2. Check Uncertainty
    3. Check SVM Confidence (>80)
    4. Fallback to BERT
    """
    # Get SVM and Decision Engine results
    svm_result = cached_predict(text)
    
    # Extract scores for hierarchy checks
    red_flags = svm_result.get("red_flags", {})
    fake_score = red_flags.get("sensational_words", 0) 
    uncertain_score = svm_result.get("uncertain_score", 0)
    net_score = svm_result.get("net_score", 0)

    # 1. Rules Final (Threshold Sync v2.2)
    if fake_score >= 8 or (fake_score >= 5 and svm_result["confidence"] < 90):
        svm_result["model_used"] = "Priority Rules (Fake Signal Override)"
        svm_result["label"] = "FAKE"
        return _map_results(svm_result)

    # 2. Uncertainty Detected (Score >= 5)
    if uncertain_score >= 5 or svm_result["label"] == "UNCERTAIN":
        svm_result["model_used"] = "Uncertainty Check (High Hedge)"
        svm_result["label"] = "UNCERTAIN"
        return _map_results(svm_result)

    # 3. SVM High Confidence (> 80%)
    if svm_result["confidence"] > 80:
        svm_result["model_used"] = "SVM Ensemble (High Confidence)"
        return _map_results(svm_result)

    # 4. Fallback to BERT (Pipeline)
    try:
        bert_result = bert_predict(text)
        # Inherit all credibility data from SVM for context
        bert_result["model_used"] = "RoBERTa (Deep Context Check)"
        bert_result["decision_reason"] = "bert_refinement"
        bert_result["keywords"] = svm_result.get("keywords", [])
        bert_result["red_flags"] = svm_result.get("red_flags", {})
        bert_result["credibility_flags"] = svm_result.get("credibility_flags", [])
        bert_result["real_flags"] = svm_result.get("real_flags", [])
        bert_result["net_score"] = svm_result.get("net_score", 0)
        bert_result["uncertain_score"] = svm_result.get("uncertain_score", 0)
        bert_result["confidence_level"] = "HIGH" if bert_result["confidence"] >= 85 else "MEDIUM"
        return _map_results(bert_result)
    except Exception as e:
        print(f"BERT Error: {e}")
        svm_result["model_used"] = "SVM Ensemble (BERT Error Fallback)"
        return _map_results(svm_result)

def _map_results(result: dict) -> dict:
    # Ensure net_score is present
    if "net_score" not in result:
        red = result.get("red_flags", {})
        result["net_score"] = result.get("net_score", 0)
    
    # credibility_flags and real_flags are now passed directly from predict()
    # No synthesis needed for v2.2+

    # Final reason mapping
    result["decision_reason"] = result.get("model_used", "ML_CONSENSUS")
    
    return result


# ── Web UI ─────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ── REST API ───────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Send JSON with a 'text' field."}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 50000:
        return jsonify({"error": "Text too long (max 50,000 chars)."}), 400

    result = analyze(text)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


# ── Model info ─────────────────────────────────────────────
@app.route("/metrics")
def metrics():
    if not os.path.exists("metrics_log.json"):
        return jsonify({"error": "No metrics log yet. Train the model first."})
    with open("metrics_log.json") as f:
        return jsonify(json.load(f))


@app.route("/logs")
def prediction_logs():
    log_file = "logs/predictions.json"
    if not os.path.exists(log_file):
        return jsonify({"error": "No prediction logs yet."})
    with open(log_file, encoding="utf-8") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
