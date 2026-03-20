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
from predict import predict as analyze  # Direct mapping to v4.0 Hybrid Engine
from bert_predict import bert_predict


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
