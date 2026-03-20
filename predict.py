import joblib
import os
from preprocess import clean_text, extract_features

def get_handcrafted_features(text):
    """Sync helper to get stylistic feature list."""
    return list(extract_features(text).values())

def load_model():
    """Loads the SVM ensemble and preprocessors from disk."""
    model_path = "model/model.pkl"
    vec_path   = "model/vectorizer.pkl"
    scl_path   = "model/scaler.pkl"
    
    # Check for mandatory files: model and vectorizer
    if not all(os.path.exists(p) for p in [model_path, vec_path]):
        print("Missing mandatory model files (model.pkl or vectorizer.pkl)! Run train.py first.")
        return None, None, None
        
    model = joblib.load(model_path)
    vec   = joblib.load(vec_path)
    
    scaler = None
    if os.path.exists(scl_path):
        scaler = joblib.load(scl_path)
        
    return model, vec, scaler

# ── FAKE TRIGGERS — sensational / misinformation keywords ──
FAKE_TRIGGERS = [
    # Classic conspiracy / sensationalism
    "shocking", "secret", "secrets", "exposed", "expose", "breaking",
    "urgent", "hoax", "conspiracy", "conspiracies", "bombshell", "bombshells",
    "cover-up", "coverup", "leaked", "leak", "banned", "censored",
    "suppressed", "suppress",
    # Phrases
    "they don't want you to know", "what they aren't telling",
    "mainstream media", "big pharma", "deep state", "new world order",
    "illuminati", "government hiding", "doctors hate", "one weird trick",
    "miracle cure", "cure all", "cures everything", "100% proven",
    "scientists confirm cure", "replace traditional", "financial interest",
    "you won't believe", "share before deleted", "they're hiding",
    "wake up", "sheeple", "plandemic", "fake vaccine", "mind control",
    "5g causes", "microchip", "globalist", "satanic",
    "secretly tested", "no evidence", "unnamed sources", "not published",
    "hidden agenda", "miracle", "exclusive proof", "stunning revelation",
    "rigged election", "stolen election", "insider reveals",
    "whistleblower", "truth they hide", "media blackout",
    "suppressed cure", "they are hiding", "banned video",
    "they deleted this", "fake pandemic", "crisis actors",
]

# ── REAL SIGNALS — authoritative / credible news phrases ──
REAL_SIGNALS = [
    # Government / official sources
    "government announced", "government said", "government stated",
    "officials announced", "officials stated", "officials said",
    "ministry of", "minister announced", "minister said",
    "president announced", "prime minister", "government initiative",
    "official statement", "press release", "official data",
    "parliament", "congress", "senate", "legislation",
    # Economic / financial
    "reserve bank", "interest rates", "inflation", "gdp", "fiscal policy",
    "revenue", "quarterly results", "budget", "central bank",
    "stock market", "trade deficit", "economic growth",
    # Scientific institutions
    "nasa", "telescope", "galaxy", "space agency",
    "study published", "peer reviewed", "journal of", "university of",
    "research published", "clinical trial", "scientific study",
    "according to researchers", "scientists found", "scientists say",
    "scientists discovered", "health ministry", "world health organization",
    # Legal / courts
    "supreme court", "court ruled", "court ordered",
    "regulations", "law passed", "bill passed", "signed into law",
]

# ── HEDGE WORDS — uncertainty language ──
HEDGE_WORDS = [
    "may", "might", "could", "suggest", "suggests", "suggested",
    "unclear", "unconfirmed", "not confirmed", "more research",
    "more evidence", "further study", "further research",
    "allegedly", "reportedly", "rumored", "possibly", "potentially",
    "appears to", "seems to", "not yet proven", "under investigation",
    "debated", "controversial", "disputed", "unverified",
    "small sample size", "not sufficient", "strongly disputed",
    "skepticism", "under review", "some evidence", "limited evidence",
]


def get_signal_data(text: str, text_lower: str):
    """
    Scan text for signals. Returns matched phrases as they appear in the original text.
    """
    h_score = 0
    f_score = 0
    real_signal_score = 0
    detected_neg = []    # matched fake phrases
    detected_hedge = []  # matched hedge phrases
    detected_real = []   # matched real phrases

    for phrase in FAKE_TRIGGERS:
        if phrase in text_lower:
            f_score += 2
            detected_neg.append(phrase)

    for phrase in HEDGE_WORDS:
        if phrase in text_lower:
            h_score += 1
            detected_hedge.append(phrase)

    for phrase in REAL_SIGNALS:
        if phrase in text_lower:
            real_signal_score += 3
            detected_real.append(phrase)

    return h_score, f_score, real_signal_score, detected_hedge, detected_neg, detected_real


def rule_based_predict(f_score: int, real_signal_score: int, h_score: int) -> dict:
    """Fast rule-based classifier — only used when BERT is unavailable."""
    net = real_signal_score - f_score

    if f_score >= 6:
        label = "FAKE"
        conf = min(65 + f_score * 1.5, 92)
    elif real_signal_score >= 3:
        label = "REAL"
        conf = min(62 + real_signal_score, 90)
    elif net > 0:
        label = "REAL"
        conf = 62.0
    elif net < 0:
        label = "FAKE"
        conf = 63.0
    elif h_score >= 6:
        label = "UNCERTAIN"
        conf = 55.0
    else:
        label = "UNCERTAIN"
        conf = 55.0

    fake_prob = conf if label == "FAKE" else (100 - conf if label == "REAL" else 50.0)
    real_prob = conf if label == "REAL" else (100 - conf if label == "FAKE" else 50.0)

    return {
        "label": label,
        "confidence": round(conf, 2),
        "fake_prob": round(fake_prob, 2),
        "real_prob": round(real_prob, 2),
        "model_used": "Rule-Based Engine (BERT Fallback)"
    }


def predict(text: str, model=None, vectorizer=None, scaler=None) -> dict:
    """
    SVM/Rule component for Fusion Engine.
    1. Extract credibility signals.
    2. Run SVM prediction (if model provided).
    3. Return combined probabilities.
    """
    text_lower = text.lower()
    h_score, f_score, real_signal_score, detected_hedge, detected_neg, detected_real = get_signal_data(text, text_lower)

    # ── SVM Prediction ───────────────────────────────────────
    if model and vectorizer and scaler:
        try:
            cleaned = clean_text(text)
            tfidf   = vectorizer.transform([cleaned])
            
            # Re-stacking features properly (matching train.py)
            hand    = get_handcrafted_features(text)
            hand_scaled = scaler.transform([hand])
            
            import scipy.sparse as sp
            X_final = sp.hstack([tfidf, sp.csr_matrix(hand_scaled)])
            
            proba = model.predict_proba(X_final)[0] 
            
            fake_prob = proba[0] * 100
            real_prob = proba[1] * 100
            conf      = max(fake_prob, real_prob)
            label     = "FAKE" if fake_prob > real_prob else "REAL"
        except Exception as e:
            print(f"SVM proba failed: {e}")
            fallback  = rule_based_predict(f_score, real_signal_score, h_score)
            fake_prob = fallback["fake_prob"]
            real_prob = fallback["real_prob"]
            conf      = fallback["confidence"]
            label     = fallback["label"]
    else:
        fallback  = rule_based_predict(f_score, real_signal_score, h_score)
        fake_prob = fallback["fake_prob"]
        real_prob = fallback["real_prob"]
        conf      = fallback["confidence"]
        label     = fallback["label"]

    # UI Metadata
    words      = text.split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = (caps_count / max(len(words), 1)) * 100
    excl_count = text.count("!")

    return {
        "label"      : label,
        "confidence" : round(conf, 2),
        "fake_prob"  : round(fake_prob, 2),
        "real_prob"  : round(real_prob, 2),
        "keywords"   : {"fake": list(set(detected_neg)), "real": list(set(detected_real))},
        "red_flags"  : {
            "sensational_words" : f_score,
            "caps_ratio"        : round(caps_ratio, 1),
            "exclamation_marks" : excl_count,
            "hedge_words"       : h_score,
        }
    }


def cached_predict(text: str) -> dict:
    return predict(text)
