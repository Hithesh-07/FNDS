# app.py — Security-hardened version

import os
import sys
import logging
from collections import deque

# ── Logging setup — structured logging, no print() ─────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("aletheia")

# ── NLTK downloads — must be first ────────────────────────
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("omw-1.4",   quiet=True)
    log.info("NLTK ready")
except Exception as e:
    log.warning("NLTK warning: %s", e)

from flask import Flask, render_template, request, jsonify, redirect, Response
from flask_wtf.csrf import CSRFProtect
from datetime import datetime

# ══════════════════════════════════════════════════════════
# APP FACTORY & SECURITY CONFIG
# ══════════════════════════════════════════════════════════

app = Flask(__name__)

# ── SECRET_KEY — required for sessions & CSRF ─────────────
# Generate a default for development, but MUST be set in production
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    os.urandom(32).hex()  # random per-restart if not set
)

# ── Request size limit — prevent payload abuse (1 MB) ─────
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB

# ── CSRF Protection ───────────────────────────────────────
csrf = CSRFProtect(app)

# ── Rate Limiting ─────────────────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per hour", "50 per minute"],
        storage_uri="memory://",
    )
    log.info("Rate limiter active")
except ImportError:
    # Graceful fallback if flask-limiter not installed yet
    log.warning("flask-limiter not installed — rate limiting disabled")
    from contextlib import contextmanager

    class _NoopLimiter:
        """Stub so @limiter.limit() decorators don't crash."""
        def limit(self, *a, **kw):
            def decorator(f):
                return f
            return decorator
        def exempt(self, f):
            return f

    limiter = _NoopLimiter()

# ── Environment ────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

# ── Global state — bounded deque to prevent memory leak ────
MAX_LOG_ENTRIES = 100
prediction_logs = deque(maxlen=MAX_LOG_ENTRIES)

_svm_loaded     = False
_svm_model      = None
_svm_vectorizer = None
_svm_scaler     = None

# ── Input validation constants ────────────────────────────
MAX_TEXT_LENGTH = 50_000
MIN_TEXT_LENGTH = 10


def get_svm():
    """Load SVM lazily on first request."""
    global _svm_model, _svm_vectorizer, _svm_scaler, _svm_loaded
    if not _svm_loaded:
        from predict import load_model
        _svm_model, _svm_vectorizer, _svm_scaler = load_model()
        _svm_loaded = True
        log.info("SVM model loaded")
    return _svm_model, _svm_vectorizer, _svm_scaler


# ══════════════════════════════════════════════════════════
# SECURITY HEADERS — applied to every response
# ══════════════════════════════════════════════════════════

@app.after_request
def set_security_headers(response):
    """Inject security headers into every HTTP response."""
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    # Prevent MIME-type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # XSS protection (legacy browsers)
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Referrer policy — don't leak full URL to third parties
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Permissions policy — disable unnecessary browser features
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), payment=()"
    )
    # Content Security Policy — allow only trusted sources
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
            "https://cdn.tailwindcss.com https://www.clarity.ms; "
        "style-src 'self' 'unsafe-inline' "
            "https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com; "
        "img-src 'self' data:; "
        "connect-src 'self' https://www.clarity.ms; "
        "frame-ancestors 'none';"
    )
    # HSTS — force HTTPS (Render handles TLS)
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


# ══════════════════════════════════════════════════════════
# ALL ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/health")
@limiter.exempt
def health():
    return {"status": "ok", "svm_loaded": _svm_loaded}, 200


@app.route("/ping")
@limiter.exempt
def ping():
    return "pong", 200


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
@limiter.limit("10 per minute;50 per hour")
def analyze_page():
    result = None
    error  = None
    text   = ""
    if request.method == "POST":
        try:
            text = request.form.get("news_text", "").strip()
            if len(text) < MIN_TEXT_LENGTH:
                error = "Please enter at least 10 characters."
                return render_template("analyze.html",
                                       error=error, text=text)
            if len(text) > MAX_TEXT_LENGTH:
                error = f"Text too long. Maximum {MAX_TEXT_LENGTH:,} characters."
                return render_template("analyze.html",
                                       error=error, text=text)

            result = run_analysis(text)
            return render_template("report.html",
                                   result=result, text=text)
        except Exception as e:
            log.error("Analysis failed: %s", e, exc_info=True)
            # SECURITY: Never expose internal error details to users
            error = "Analysis failed. Please try again later."
            return render_template("analyze.html",
                                   error=error, text=text)
    return render_template("analyze.html", text=text)


@app.route("/predict", methods=["POST"])
@limiter.limit("20 per minute;100 per hour")
@csrf.exempt  # API endpoint — uses JSON, not forms
def api_predict():
    try:
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Send JSON with text field"}), 400
        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text is empty"}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({
                "error": f"Text too long. Max {MAX_TEXT_LENGTH:,} characters."
            }), 400
        result = run_analysis(text)
        return jsonify(result)
    except Exception as e:
        log.error("API predict failed: %s", e, exc_info=True)
        # SECURITY: Generic error only — no stack traces
        return jsonify({"error": "Internal server error", "label": "ERROR",
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
@limiter.limit("10 per minute")
def metrics_json():
    return jsonify([{
        "bert_model"    : "Monk3ydluffy/truthlens-bert",
        "bert_accuracy" : 99.2,
        "svm_accuracy"  : 98.2,
        "dataset_size"  : 44000,
    }])


@app.route("/logs")
@limiter.limit("5 per minute")
def logs():
    """Admin-only endpoint — requires ADMIN_TOKEN in query param."""
    token = request.args.get("token", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 403

    # Sanitize logs — strip text previews for safety
    safe_logs = []
    for entry in list(prediction_logs)[-50:]:
        safe_logs.append({
            "timestamp":  entry.get("timestamp"),
            "label":      entry.get("label"),
            "confidence": entry.get("confidence"),
            "model_used": entry.get("model_used"),
        })
    return jsonify(safe_logs)


@app.errorhandler(404)
def not_found(e):
    try:
        return render_template("404.html"), 404
    except Exception:
        return "<h1>404 - Page Not Found</h1><a href='/'>Home</a>", 404


@app.errorhandler(500)
def server_error(e):
    log.error("500 error: %s", e)
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    """Custom response for rate-limited requests."""
    log.warning("Rate limit exceeded: %s from %s",
                request.path, request.remote_addr)
    if request.path.startswith("/predict") or request.path.startswith("/metrics"):
        return jsonify({
            "error": "Rate limit exceeded. Please slow down.",
            "retry_after": e.description
        }), 429
    return render_template("error.html"), 429


@app.route('/google5f589252169489ad.html')
def google_verify():
    return 'google-site-verification: google5f589252169489ad.html'


@app.route('/sitemap.xml')
def sitemap():
    base_url = request.url_root.rstrip('/')
    pages = [
        {"path": "", "priority": "1.0"},
        {"path": "/analyze", "priority": "0.9"},
        {"path": "/how-it-works", "priority": "0.8"},
        {"path": "/about", "priority": "0.8"},
        {"path": "/metrics", "priority": "0.5"}
    ]
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for page in pages:
        xml += f'  <url>\n    <loc>{base_url}{page["path"]}</loc>\n    <priority>{page["priority"]}</priority>\n  </url>\n'
    xml += '</urlset>'
    return Response(xml, mimetype='application/xml')


@app.route('/robots.txt')
def robots():
    """Prevent crawlers from indexing sensitive endpoints."""
    txt = (
        "User-agent: *\n"
        "Allow: /\n"
        "Allow: /analyze\n"
        "Allow: /how-it-works\n"
        "Allow: /about\n"
        "Disallow: /logs\n"
        "Disallow: /predict\n"
        "Disallow: /metrics/json\n"
        "Disallow: /health\n"
        "Disallow: /ping\n"
        f"\nSitemap: {request.url_root.rstrip('/')}/sitemap.xml\n"
    )
    return Response(txt, mimetype='text/plain')


# ══════════════════════════════════════════════════════════
# ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════

def run_analysis(text: str) -> dict:
    log.info("run_analysis() len=%d", len(text))

    # Step 1: SVM
    svm_model, svm_vectorizer, svm_scaler = get_svm()
    try:
        from predict import predict as svm_predict
        svm_result = svm_predict(text, svm_model,
                                  svm_vectorizer, svm_scaler)
        log.info("SVM: %s (%.1f%%)", svm_result['label'],
                 svm_result['confidence'])
    except Exception as e:
        log.error("SVM error: %s", e, exc_info=True)
        raise Exception("Analysis engine unavailable")

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
        log.info("Decision: %s reason=%s",
                 decision['final_label'], decision['decision_reason'])
    except Exception as e:
        log.error("Decision engine error: %s", e, exc_info=True)
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
            log.info("BERT: %s (%.1f%%)",
                     bert_result['label'], bert_result['confidence'])
        except Exception as e:
            log.info("BERT failed (using SVM): %s", e)

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

    # Log prediction — text_preview stripped for security
    prediction_logs.append({
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label"       : final_label,
        "confidence"  : final_conf,
        "model_used"  : model_used,
    })

    log.info("Final: %s %.1f%% via %s", final_label, final_conf, model_used)
    return result


# ══════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════
log.info("Flask app ready — all routes defined, security hardened")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
