# predict.py — v5.2 Hybrid Engine: BERT + Expanded Rule-Based Fallback

from preprocess import clean_text, extract_features
from bert_predict import bert_predict

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
    "cdc", "who said", "expert says", "experts say", "experts believe",
    "industry experts", "analysts say", "analysts believe",
    # Journalism / source attribution
    "reuters", "associated press", "according to", "confirmed by",
    "statistics show", "data from", "report says", "report found",
    "survey found", "survey shows", "poll shows", "study shows",
    "study found", "research shows", "research found",
    # Legal / courts
    "supreme court", "court ruled", "court said", "court ordered",
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


def predict(text: str) -> dict:
    """
    v5.2 Hybrid Engine:
    1. Extract credibility signals from text.
    2. Try BERT via HF Router.
    3. If BERT fails → rule-based fallback (never blind UNCERTAIN).
    4. Override BERT if rule signals overwhelmingly disagree.
    5. Return matched keywords for frontend text highlighting.
    """
    text_lower = text.lower()
    h_score, f_score, real_signal_score, detected_hedge, detected_neg, detected_real = get_signal_data(text, text_lower)

    # ── BERT ─────────────────────────────────────────────────
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

    # ── Override Logic (applied for BERT and rule results) ────
    final_label      = bert_label
    final_confidence = bert_confidence

    if bert_label == "REAL" and f_score >= 6:
        final_label      = "FAKE"
        final_confidence = 75.0
    elif bert_label == "FAKE" and real_signal_score >= 6:
        final_label      = "REAL"
        final_confidence = 78.0
    elif bert_label in ("REAL", "FAKE") and h_score >= 8 and bert_confidence < 75:
        final_label      = "UNCERTAIN"
        final_confidence = max(fake_prob, real_prob)

    # ── Sync probs ────────────────────────────────────────────
    if final_label == "REAL":
        real_prob = max(real_prob, 70.0)
        fake_prob = 100.0 - real_prob
    elif final_label == "FAKE":
        fake_prob = max(fake_prob, 70.0)
        real_prob = 100.0 - fake_prob
    elif final_label == "UNCERTAIN":
        fake_prob = 50.0
        real_prob = 50.0

    # ── UI Metadata ───────────────────────────────────────────
    words      = text.split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = (caps_count / max(len(words), 1)) * 100
    excl_count = text.count("!")

    credibility_flags = [f"{f.title()} Signal" for f in detected_neg]
    credibility_flags += [f"{h.title()} (Uncertainty)" for h in detected_hedge]
    real_flags = [f"{r.title()} Verified" for r in detected_real]

    # Keywords for frontend text highlighting
    fake_keywords = list(set(detected_neg))
    real_keywords = list(set(detected_real))

    return {
        "label"             : final_label,
        "confidence"        : round(final_confidence, 2),
        "confidence_level"  : "HIGH" if final_confidence >= 80 else "MEDIUM" if final_confidence >= 60 else "LOW",
        "fake_prob"         : round(fake_prob, 2),
        "real_prob"         : round(real_prob, 2),
        "gap"               : round(abs(fake_prob - real_prob), 2),
        "keywords"          : {"fake": fake_keywords, "real": real_keywords},
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
        "model_used"        : f"v5.2 Hybrid ({model_tag})",
        "decision_reason"   : model_tag
    }


def cached_predict(text: str) -> dict:
    return predict(text)
