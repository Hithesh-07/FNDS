# predict.py — Full upgraded version

import joblib
import numpy as np
import scipy.sparse as sp
from preprocess import clean_text, extract_features

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

def load_model():
    model      = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    scaler     = joblib.load("model/scaler.pkl") if \
                 __import__("os").path.exists("model/scaler.pkl") else None
    return model, vectorizer, scaler

def get_top_keywords(text, vectorizer, n=8):
    """Returns top TF-IDF weighted words, filtered of noise."""
    NOISE_WORDS = {
        "cell", "local", "show", "google", "said", "also",
        "would", "could", "make", "take", "come", "go",
        "get", "use", "just", "like", "know", "time",
        "year", "day", "way", "man", "new", "one", "two",
        "say", "says", "think", "want", "need", "look"
    }
    cleaned       = clean_text(text)
    vec           = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    scores        = vec.toarray()[0]
    top_idx       = np.argsort(scores)[::-1][:n*2]
    top_words     = [
        feature_names[i]
        for i in top_idx
        if scores[i] > 0 and feature_names[i] not in NOISE_WORDS
    ]
    return top_words[:n]

def cached_predict(text: str) -> dict:
    model, vectorizer, scaler = load_model()
    return predict(text, model, vectorizer, scaler)

def predict(text: str, model, vectorizer, scaler=None) -> dict:

    text_lower = text.lower()
    h_score, f_score, detected_hedge, detected_neg = get_signal_data(text_lower)

    # ── Get base probabilities ─────────────────────────────
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    
    # Handcrafted styling features if scaler exists
    if scaler:
        hand_feat = np.array([list(extract_features(text).values())], dtype=float)
        hand_scaled = scaler.transform(hand_feat)
        X_final = sp.hstack([vec, sp.csr_matrix(hand_scaled)])
    else:
        X_final = vec

    proba   = model.predict_proba(X_final)[0]

    fake_prob = round(float(proba[0]) * 100, 2)
    real_prob = round(float(proba[1]) * 100, 2)

    # Cap at 95% max
    fake_prob = min(fake_prob, 95.0)
    real_prob = min(real_prob, 95.0)

    # ── Probability gap — KEY metric ──────────────────────
    gap       = abs(fake_prob - real_prob)
    raw_label = "REAL" if real_prob > fake_prob else "FAKE"

    # Caps Logic
    words      = text.split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = (caps_count / max(len(words), 1)) * 100

    # Map signals to the legacy names for decision tree
    fake_signal_score = f_score 
    hedge_signal_score = h_score

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

    # ── DECISION TREE (Calibrated v2.2) ─────────────────────
    
    # Priority 0: Extreme Real Certainty (Trust the Model)
    if real_prob > 90 and fake_signal_score < 4:
        final_label = "REAL"
        final_confidence = real_prob

    # Priority 1: Clear Real Signals (Rule-based Boost)
    elif real_signal_score >= 5 and fake_signal_score < 2:
        final_label = "REAL"
        final_confidence = min(max(real_prob, 85.0) + real_signal_score, 95.0)

    # Priority 2: Strong Fake Evidence (Rule-based Override)
    # Only override confident REAL if fake evidence is overwhelming (>=8)
    elif fake_signal_score >= 8 or (fake_signal_score >= 5 and real_prob < 80):
        final_label = "FAKE"
        final_confidence = min(max(fake_prob, 80.0) + fake_signal_score, 95.0)

    # Priority 3: Sensationalism + Low ML confidence
    elif fake_signal_score >= 3 and real_prob < 70:
        final_label = "FAKE"
        final_confidence = min(max(fake_prob, 75.0), 90.0)

    # Priority 4: Uncertainty Logic
    elif gap < 15 or hedge_signal_score >= 5:
        final_label = "UNCERTAIN"
        final_confidence = min(max(fake_prob, real_prob), 82.0)
    
    # Priority 5: Hedge signals vs confident model
    elif hedge_signal_score >= 3 and max(fake_prob, real_prob) < 85:
        final_label = "UNCERTAIN"
        final_confidence = min(max(fake_prob, real_prob), 78.0)

    # Priority 6: Default (Trust ML)
    else:
        final_label = raw_label
        final_confidence = max(fake_prob, real_prob)

    final_confidence = round(final_confidence, 2)
    conf_level = "HIGH" if final_confidence >= 85 else "MEDIUM" if final_confidence >= 65 else "LOW"

    keywords = get_top_keywords(text, vectorizer)
    excl_count = text.count("!")

    # ── Final Probability Alignment ────────────────────────
    # If a rule overrode the ML, we sync the probabilities to avoid UI confusion
    if final_label == "REAL" and raw_label != "REAL":
        real_prob = final_confidence
        fake_prob = 100.0 - final_confidence
    elif final_label == "FAKE" and raw_label != "FAKE":
        fake_prob = final_confidence
        real_prob = 100.0 - final_confidence
    elif final_label == "UNCERTAIN":
        fake_prob = 50.0
        real_prob = 50.0

    # Credibility Flags Assembly
    # Negative signals (Fake + Hedge)
    credibility_flags = []
    for f in detected_neg:
        credibility_flags.append(f"{f.title()} Signal")
    for h in detected_hedge:
        credibility_flags.append(f"{h.title()} (Uncertainty)")
    
    # Positive signals
    real_flags = []
    for r in detected_real:
        real_flags.append(f"{r.title()} Verified")

    return {
        "label"             : final_label,
        "confidence"        : final_confidence,
        "confidence_level"  : conf_level,
        "fake_prob"         : round(fake_prob, 2),
        "real_prob"         : round(real_prob, 2),
        "gap"               : round(abs(fake_prob - real_prob), 2),
        "keywords"          : {"fake": keywords if final_label == "FAKE" else [], "real": keywords if final_label == "REAL" else []},
        "credibility_flags" : credibility_flags,
        "real_flags"        : real_flags,
        "net_score"         : (real_signal_score - fake_signal_score),
        "uncertain_score"   : hedge_signal_score,
        "red_flags"         : {
            "sensational_words" : fake_signal_score,
            "caps_ratio"        : round(caps_ratio, 1),
            "exclamation_marks" : excl_count,
            "hedge_words"       : hedge_signal_score,
        }
    }
