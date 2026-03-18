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

from bert_predict import predict_bert

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
    
    # Extract scores from decision engine
    fake_score = svm_result.get("net_score", 0) 
    uncertain_score = svm_result.get("uncertain_score", 0)

    # 1. Rules Final (Fake Score >= 3)
    if fake_score >= 3:
        svm_result["model_used"] = "Priority Rules (Fake Signal Override)"
        svm_result["label"] = "FAKE"
        return svm_result

    # 2. Uncertainty Detected (Score >= 5)
    if uncertain_score >= 5:
        svm_result["model_used"] = "Uncertainty Check (High Hedge)"
        svm_result["label"] = "UNCERTAIN"
        return svm_result

    # 3. SVM High Confidence (> 80%)
    if svm_result["confidence"] > 80:
        svm_result["model_used"] = "SVM Ensemble (High Confidence)"
        return svm_result

    # 4. Fallback to BERT (Pipeline)
    try:
        bert_result = predict_bert(text)
        # Merge BERT result with SVM structure for UI consistency
        bert_result["model_used"] = "RoBERTa (Deep Context Check)"
        bert_result["decision_reason"] = "bert_refinement"
        # Keep SVM keywords and flags for the dashboard
        bert_result["keywords"] = svm_result.get("keywords", [])
        bert_result["credibility_flags"] = svm_result.get("credibility_flags", [])
        bert_result["real_flags"] = svm_result.get("real_flags", [])
        bert_result["red_flags"] = svm_result.get("red_flags", {})
        bert_result["confidence_level"] = "HIGH" if bert_result["confidence"] >= 85 else "MEDIUM"
        return bert_result
    except Exception as e:
        print(f"BERT Error: {e}")
        svm_result["model_used"] = "SVM Ensemble (BERT Error Fallback)"
        return svm_result


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
