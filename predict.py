# predict.py — v4.0 Lean Serverless (Vercel Optimized)

from preprocess import clean_text, extract_features
from bert_predict import bert_predict
import json

# ── Credibility Signal Definitions ───────────────────────
# NEGATIVE SIGNALS (Misinformation / Sensationalism)
FAKE_TRIGGERS = [
    "shocking", "secret", "exposed", "breaking", "urgent",
    "hoax", "conspiracy", "conspiracies", "bombshell", "cover-up", 
    "leaked", "leak", "banned", "censored", "suppressed", "suppress",
    "they don't want you to know", "what they aren't telling",
    "mainstream media", "big pharma", "deep state", "new world order",
    "illuminati", "government hiding", "doctors hate", "one weird trick",
    "miracle cure", "cure all", "cures everything", "100% proven",
    "scientists confirm cure", "breakthrough in intelligence", 
    "replace traditional", "financial interest", "direct bias",
    "extraordinary claims", "gain traction", "social media influencers"
]

# UNCERTAINTY SIGNALS (Hedging / Scientific skepticism)
HEDGE_WORDS = [
    "may", "might", "could", "suggest", "suggests", "suggested",
    "unclear", "unconfirmed", "not confirmed", "more research",
    "more evidence", "further study", "further research",
    "experts say", "preliminary", "alleged", "allegedly", 
    "reportedly", "rumored", "some researchers", "some scientists",
    "some experts", "possibly", "potentially", "appears to",
    "seems to", "not yet proven", "under investigation",
    "debated", "controversial", "disputed", "unverified",
    "lack of peer review", "not yet published", "small sample size",
    "fewer than 100", "not sufficient", "strongly disputed",
    "awaiting more robust", "definitive conclusions", "skepticism"
]

def get_signal_data(text_lower: str):
    """Scans text for signals and returns (score, phrases)."""
    h_score = 0
    f_score = 0
    detected_neg = []
    detected_hedge = []

    for word in FAKE_TRIGGERS:
        if word in text_lower:
            f_score += 2
            detected_neg.append(word)
    
    for word in HEDGE_WORDS:
        if word in text_lower:
            h_score += 2
            detected_hedge.append(word)
            
    return h_score, f_score, detected_hedge, detected_neg

def predict(text: str) -> dict:
    """
    v4.0 Hybrid Engine: Rules + BERT API (Zero local ML binaries)
    """
    text_lower = text.lower()
    h_score, f_score, detected_hedge, detected_neg = get_signal_data(text_lower)

    # 1. API-Based BERT Prediction (RoBERTa)
    bert_result = bert_predict(text)
    
    # Extract BERT metrics
    bert_label = bert_result.get("label", "UNCERTAIN")
    bert_confidence = bert_result.get("confidence", 0.0)
    fake_prob = bert_result.get("fake_prob", 50.0)
    real_prob = bert_result.get("real_prob", 50.0)

    # 2. Rule-Based Bias Scaling
    # Real signals counter
    real_signal_score = 0
    real_signal_words = [
        "reserve bank", "interest rates", "inflation", "gdp", "fiscal",
        "revenue", "quarterly", "nasa", "telescope", "galaxy",
        "supreme court", "regulations", "environmental", "government data"
    ]
    detected_real = []
    for w in real_signal_words:
        if w in text_lower:
            real_signal_score += 3
            detected_real.append(w)

    # 3. Consensus Logic (Weighted v4.0)
    final_label = bert_label
    final_confidence = bert_confidence

    # Boost/Override Logic based on Signals
    if bert_label == "REAL" and f_score >= 6:
        # Override BERT if fake news buzzwords are overwhelming
        final_label = "FAKE"
        final_confidence = 75.0
    elif bert_label == "FAKE" and real_signal_score >= 6:
        # Override BERT if high-quality verifyable nouns are found
        final_label = "REAL"
        final_confidence = 75.0
    elif h_score >= 6 and (bert_confidence < 85):
        # Override to Uncertain if hedging is extreme
        final_label = "UNCERTAIN"
        final_confidence = max(fake_prob, real_prob)

    # Sync probabilities to UI for the final label
    if final_label == "REAL":
        real_prob = max(real_prob, 70.0)
        fake_prob = 100.0 - real_prob
    elif final_label == "FAKE":
        fake_prob = max(fake_prob, 70.0)
        real_prob = 100.0 - fake_prob
    elif final_label == "UNCERTAIN":
        fake_prob = 50.0
        real_prob = 50.0

    # UI Meta Data
    words      = text.split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = (caps_count / max(len(words), 1)) * 100
    excl_count = text.count("!")

    # Credibility Flags Assembly
    credibility_flags = []
    for f in detected_neg:
        credibility_flags.append(f"{f.title()} Signal")
    for h in detected_hedge:
        credibility_flags.append(f"{h.title()} (Uncertainty)")
    
    real_flags = []
    for r in detected_real:
        real_flags.append(f"{r.title()} Verified")

    return {
        "label"             : final_label,
        "confidence"        : round(final_confidence, 2),
        "confidence_level"  : "HIGH" if final_confidence >= 80 else "MEDIUM" if final_confidence >= 60 else "LOW",
        "fake_prob"         : round(fake_prob, 2),
        "real_prob"         : round(real_prob, 2),
        "gap"               : round(abs(fake_prob - real_prob), 2),
        "keywords"          : {"fake": [], "real": []}, # Reduced for speed
        "credibility_flags" : credibility_flags,
        "real_flags"        : real_flags,
        "net_score"         : (real_signal_score - f_score),
        "uncertain_score"   : h_score,
        "red_flags"         : {
            "sensational_words" : f_score,
            "caps_ratio"        : round(caps_ratio, 1),
            "exclamation_marks" : excl_count,
            "hedge_words"       : h_score,
        },
        "model_used"        : f"v4.0 Consensus ({bert_result.get('model_used', 'API')})"
    }

def cached_predict(text: str) -> dict:
    """Helper for Flask app."""
    return predict(text)
