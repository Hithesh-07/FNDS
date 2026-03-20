# predict.py — v5.1 Hybrid Engine: BERT + Rule-Based Fallback + Keyword Highlights

from preprocess import clean_text, extract_features
from bert_predict import bert_predict

# ── Fake News Signal Words (for highlighting + scoring) ───
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
    "extraordinary claims", "gain traction", "social media influencers",
    "you won't believe", "share before it's deleted", "they're hiding",
    "wake up", "sheeple", "plandemic", "fake vaccine", "mind control",
    "5g causes", "microchip vaccine", "globalist", "satanic",
    "secretly", "no evidence", "unnamed sources", "not published",
    "unbelievable", "explosive", "scandal", "corrupt", "rigged",
    "miracle", "exclusive", "stunning", "hidden agenda",
]

# ── Real News Signals (for highlighting + scoring) ────────
REAL_SIGNALS = [
    "reserve bank", "interest rates", "inflation", "gdp", "fiscal",
    "revenue", "quarterly", "nasa", "telescope", "galaxy",
    "supreme court", "regulations", "environmental", "government data",
    "according to official", "press release", "confirmed by",
    "study published", "peer reviewed", "journal of", "university of",
    "reuters", "associated press", "official statement", "ministry of",
    "world health organization", "united nations", "cdc says",
    "statistics show", "data from", "research shows", "scientists say",
    "published in", "health ministry", "official data", "government report",
]

# ── Uncertainty Signals ───────────────────────────────────
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
    "not sufficient", "strongly disputed", "awaiting more robust",
    "skepticism", "under review",
]


def get_signal_data(text_lower: str):
    h_score = 0
    f_score = 0
    real_signal_score = 0
    detected_neg = []
    detected_hedge = []
    detected_real = []

    for word in FAKE_TRIGGERS:
        if word in text_lower:
            f_score += 2
            detected_neg.append(word)

    for word in HEDGE_WORDS:
        if word in text_lower:
            h_score += 1
            detected_hedge.append(word)

    for word in REAL_SIGNALS:
        if word in text_lower:
            real_signal_score += 3
            detected_real.append(word)

    return h_score, f_score, real_signal_score, detected_hedge, detected_neg, detected_real


def rule_based_predict(f_score: int, real_signal_score: int, h_score: int) -> dict:
    """Fast rule-based classifier. Used when BERT is unavailable."""
    net = real_signal_score - f_score

    if f_score >= 6:
        label = "FAKE"
        confidence = min(65 + f_score * 1.5, 92)
    elif real_signal_score >= 6:
        label = "REAL"
        confidence = min(65 + real_signal_score, 90)
    elif h_score >= 6 and f_score <= 2:
        label = "UNCERTAIN"
        confidence = 55.0
    elif net > 0:
        label = "REAL"
        confidence = 60.0
    elif net < 0:
        label = "FAKE"
        confidence = 63.0
    else:
        label = "UNCERTAIN"
        confidence = 55.0

    fake_prob = confidence if label == "FAKE" else (100 - confidence if label == "REAL" else 50.0)
    real_prob = confidence if label == "REAL" else (100 - confidence if label == "FAKE" else 50.0)

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "fake_prob": round(fake_prob, 2),
        "real_prob": round(real_prob, 2),
        "model_used": "Rule-Based Engine (BERT Fallback)"
    }


def predict(text: str) -> dict:
    """
    v5.1 Hybrid Engine:
    1. Extract credibility signals for keyword highlighting.
    2. Try BERT via HF Router (fast, 20s timeout).
    3. If BERT fails → use Rule-Based fallback (NEVER returns empty UNCERTAIN by default).
    4. Apply override logic to adjust BERT label when signals are overwhelming.
    """
    text_lower = text.lower()
    h_score, f_score, real_signal_score, detected_hedge, detected_neg, detected_real = get_signal_data(text_lower)

    # ── BERT Prediction ───────────────────────────────────────
    bert_result = bert_predict(text)
    bert_ok = bert_result is not None

    if bert_ok:
        bert_label      = bert_result.get("label", "UNCERTAIN")
        bert_confidence = bert_result.get("confidence", 0.0)
        fake_prob       = bert_result.get("fake_prob", 50.0)
        real_prob       = bert_result.get("real_prob", 50.0)
        model_tag       = bert_result.get("model_used", "RoBERTa")
    else:
        print("BERT unavailable → using Rule-Based fallback")
        fallback        = rule_based_predict(f_score, real_signal_score, h_score)
        bert_label      = fallback["label"]
        bert_confidence = fallback["confidence"]
        fake_prob       = fallback["fake_prob"]
        real_prob       = fallback["real_prob"]
        model_tag       = fallback["model_used"]

    # ── Override Logic ────────────────────────────────────────
    final_label      = bert_label
    final_confidence = bert_confidence

    if bert_ok:
        if bert_label == "REAL" and f_score >= 6:
            final_label      = "FAKE"
            final_confidence = 75.0
        elif bert_label == "FAKE" and real_signal_score >= 6:
            final_label      = "REAL"
            final_confidence = 75.0
        elif h_score >= 6 and bert_confidence < 85:
            final_label      = "UNCERTAIN"
            final_confidence = max(fake_prob, real_prob)

    # ── Sync probabilities ─────────────────────────────────────
    if final_label == "REAL":
        real_prob = max(real_prob, 70.0)
        fake_prob = 100.0 - real_prob
    elif final_label == "FAKE":
        fake_prob = max(fake_prob, 70.0)
        real_prob = 100.0 - fake_prob
    elif final_label == "UNCERTAIN":
        fake_prob = 50.0
        real_prob = 50.0

    # ── UI Metadata ────────────────────────────────────────────
    words      = text.split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = (caps_count / max(len(words), 1)) * 100
    excl_count = text.count("!")

    # ── Keyword Highlights ─────────────────────────────────────
    # Build lists for frontend text highlighting
    fake_keywords = list(set(detected_neg))   # unique trigger words found
    real_keywords = list(set(detected_real))  # unique real signal words found

    credibility_flags = [f"{f.title()} Signal" for f in detected_neg]
    credibility_flags += [f"{h.title()} (Uncertainty)" for h in detected_hedge]
    real_flags = [f"{r.title()} Verified" for r in detected_real]

    return {
        "label"             : final_label,
        "confidence"        : round(final_confidence, 2),
        "confidence_level"  : "HIGH" if final_confidence >= 80 else "MEDIUM" if final_confidence >= 60 else "LOW",
        "fake_prob"         : round(fake_prob, 2),
        "real_prob"         : round(real_prob, 2),
        "gap"               : round(abs(fake_prob - real_prob), 2),
        "keywords"          : {"fake": fake_keywords, "real": real_keywords},  # for highlighting!
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
        "model_used"        : f"v5.1 Hybrid ({model_tag})",
        "decision_reason"   : model_tag
    }


def cached_predict(text: str) -> dict:
    return predict(text)
