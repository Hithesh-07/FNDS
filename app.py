"""
app.py — Aletheia Flask Application v5.0
Hybrid BERT + Rule-Based Engine with in-memory prediction logs.
"""
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, abort

app = Flask(__name__)

from predict import predict as analyze

# ── In-Memory Prediction Log (survives within a session) ──
_prediction_log = []

# ── Hardcoded Training Metrics (from last training run) ──
METRICS_LOG = [
    {
        "timestamp": "2026-03-19 18:42:11",
        "model_type": "Ensemble (SVM + LR + PAC)",
        "dataset_path": "data/news.csv",
        "dataset_size": 40000,
        "accuracy": 98.2,
        "f1_score": 98.1,
        "precision": 98.3,
        "recall": 98.0,
    }
]


# ── Web UI ─────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


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

    # Store in memory log
    _prediction_log.insert(0, {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "label": result.get("label"),
        "confidence": result.get("confidence"),
        "model_used": result.get("model_used"),
        "text_snippet": text[:80] + ("..." if len(text) > 80 else "")
    })
    if len(_prediction_log) > 100:
        _prediction_log.pop()

    return jsonify(result)


# ── Model Metrics ──────────────────────────────────────────
@app.route("/metrics")
def metrics():
    return jsonify(METRICS_LOG)


# ── Prediction Logs ────────────────────────────────────────
@app.route("/logs")
def prediction_logs():
    if not _prediction_log:
        return jsonify({"message": "No predictions made yet in this session.", "logs": []})
    return jsonify({"count": len(_prediction_log), "logs": _prediction_log})


# ── Error Pages ────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Internal server error. Please try again."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
