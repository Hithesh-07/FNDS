# predict.py — Full upgraded version

import joblib
import numpy as np
import scipy.sparse as sp
from preprocess import clean_text, extract_features

# ── Hedging / Uncertainty language (Legacy/Optional) ──────
HEDGE_WORDS = [
    "may", "might", "could", "suggest", "suggests",
    "suggested", "unclear", "unconfirmed", "not confirmed",
    "more research", "more evidence", "further study",
    "further research", "indicate", "indicated", "experts say",
    "preliminary", "alleged", "reportedly", "rumored",
    "some researchers", "some scientists", "some experts",
    "possibly", "potentially", "appears to", "seems to",
    "according to some", "not yet proven", "under investigation",
    "debated", "controversial", "disputed", "unverified",
    "sources say", "anonymous sources", "we cannot confirm",
    "it is believed", "it is thought", "widely believed",
    "no evidence", "little evidence", "mixed results"
]

# ── Sensational / Fake trigger words (Legacy/Optional) ────
FAKE_TRIGGERS = [
    "shocking", "secret", "exposed", "breaking", "urgent",
    "hoax", "conspiracy", "bombshell", "cover-up", "leaked",
    "banned", "censored", "suppressed", "hiding", "coverup",
    "they don't want you to know", "what they aren't telling",
    "mainstream media won't", "big pharma", "deep state",
    "new world order", "illuminati", "government hiding",
    "doctors hate", "one weird trick", "miracle cure",
    "cure all", "cures everything", "100% proven",
    "scientists confirm cure", "suppress", "suppressing"
]

from decision_engine import run_decision_engine

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

    # Step 1: Get ML probabilities
    cleaned   = clean_text(text)
    vec       = vectorizer.transform([cleaned])
    
    # Handcrafted styling features if scaler exists
    if scaler:
        hand_feat = np.array([list(extract_features(text).values())], dtype=float)
        hand_scaled = scaler.transform(hand_feat)
        X_final = sp.hstack([vec, sp.csr_matrix(hand_scaled)])
    else:
        X_final = vec

    proba   = model.predict_proba(X_final)[0]
    fake_prob = min(round(float(proba[0]) * 100, 2), 95.0)
    real_prob = min(round(float(proba[1]) * 100, 2), 95.0)
    ml_label  = "REAL" if real_prob > fake_prob else "FAKE"
    ml_conf   = max(fake_prob, real_prob)

    # Step 2: Run decision engine (rules first)
    decision  = run_decision_engine(
        text, ml_label, ml_conf, fake_prob, real_prob
    )

    # Step 3: Get keywords
    keywords  = get_top_keywords(text, vectorizer)

    # Step 4: Red flags
    words      = text.split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = round((caps_count / max(len(words), 1)) * 100, 1)
    excl_count = text.count("!")

    return {
        "label"            : decision["final_label"],
        "confidence"       : decision["final_confidence"],
        "confidence_level" : decision["confidence_level"],
        "fake_prob"        : fake_prob,
        "real_prob"        : real_prob,
        "keywords"         : keywords,
        "decision_reason"  : decision["decision_reason"],
        "credibility_flags": decision["fake_flags"],
        "real_flags"       : decision["real_flags"],
        "net_score"        : decision["net_score"],
        "credibility_score": decision["net_score"],
        "uncertain_score"  : decision["uncertain_score"],
        "red_flags"        : {
            "sensational_words"  : decision["fake_score"],
            "caps_ratio"         : caps_ratio,
            "exclamation_marks"  : excl_count,
            "credibility_issues" : decision["net_score"],
        }
    }
