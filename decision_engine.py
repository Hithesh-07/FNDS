"""
decision_engine.py — Context-Aware Credibility Scoring Engine v6.0
Supports Priority Pipeline (BERT Primary -> SVM Fallback)
"""

def score_according_to(text_lower: str) -> int:
    score = 0
    credible_according = [
        "according to official", "according to the government", "according to the ministry",
        "according to the reserve bank", "according to the central bank", "according to researchers at",
        "according to the report", "according to data", "according to the study", "according to nasa",
        "according to who", "according to the united nations", "according to the supreme court",
    ]
    suspicious_according = [
        "according to unnamed", "according to anonymous", "according to sources",
        "according to insiders", "according to individuals", "according to a report circulating",
        "according to unverified",
    ]
    for phrase in credible_according:
        if phrase in text_lower: score -= 2
    for phrase in suspicious_according:
        if phrase in text_lower: score += 3
    return score


def score_organization_mentions(text_lower: str) -> int:
    score = 0
    trusted_orgs = [
        "world health organization", "nasa", "united nations", "reserve bank",
        "central bank", "supreme court", "ministry", "european union",
        "world bank", "imf", "federal reserve", "parliament",
    ]
    conspiracy_context = [
        "secretly", "secret", "hiding", "hidden", "suppressed", "suppress",
        "cover", "withheld", "blocked", "preventing", "allegedly", "quietly",
        "leaked", "stockpiling", "anonymous insiders", "unnamed",
    ]
    org_found     = any(org in text_lower for org in trusted_orgs)
    conspiracy_found = any(c in text_lower for c in conspiracy_context)

    if org_found and not conspiracy_found:
        score -= 3
    elif org_found and conspiracy_found:
        score += 4
    return score


STRONG_FAKE_SIGNALS = {
    "secretly stockpiling": 4, "secretly developing": 3, "has been secretly": 4,
    "have been secretly": 4, "withheld from the public": 4, "withheld from public": 4,
    "anonymous insiders": 3, "claim the manipulation": 3, "independent verification has": 4,
    "verification has not been": 4, "feasibility of such claims": 3,
    "scientists have strongly rejected"  : 4, "strongly rejected these claims"     : 4,
    "experts have rejected"              : 3, "doctors have rejected"              : 3,
    "no scientific basis"                : 4, "no scientific evidence"             : 4,
    "biologically impossible"            : 4, "scientifically impossible"          : 4,
}

# Added for the 9/10 failure cases
UNCERTAIN_SIGNALS = [
    "preliminary study", "suggests that", "more research is needed",
    "may potentially", "small sample size", "needs verification",
    "indicates a possibility", "further investigation"
]


def detect_pseudoscience(text_lower: str) -> int:
    score = 0
    bio_impossible = [
        "eliminate the need for sleep", "without sleep", "no sleep needed",
        "cure all diseases", "cures all", "reverses aging completely",
        "regrow organs", "regrow limbs", "unlimited energy",
        "defy gravity", "100% effective cure", "guaranteed cure",
    ]
    for claim in bio_impossible:
        if claim in text_lower:
            score += 4
    return score


def calculate_scores(text: str) -> dict:
    """
    Step 1 of the new analyze() flow.
    Calculates raw credibility scores before model runs.
    """
    text_lower = text.lower()
    net_score = 0
    uncertain_score = 0
    fake_flags = []
    real_flags = []

    net_score += score_according_to(text_lower)
    net_score += score_organization_mentions(text_lower)
    net_score += detect_pseudoscience(text_lower)

    for phrase, val in STRONG_FAKE_SIGNALS.items():
        if phrase in text_lower:
            net_score += val
            fake_flags.append(phrase.title())

    for phrase in UNCERTAIN_SIGNALS:
        if phrase in text_lower:
            uncertain_score += 2

    return {
        "net_score": net_score,
        "uncertain_score": uncertain_score,
        "fake_flags": fake_flags,
        "real_flags": real_flags
    }


def run_decision_engine_raw(text: str, ml_label: str, ml_conf: float, fake_prob: float, real_prob: float) -> dict:
    """
    Step 5 of the new analyze() flow.
    Applies overrides and determines final verdict.
    """
    scores = calculate_scores(text)
    net_score = scores["net_score"]
    uncertain_score = scores["uncertain_score"]

    final_label = ml_label
    final_conf = ml_conf
    reason = "ML model prediction"

    # Override: High negative signals
    if net_score >= 7:
        final_label = "FAKE"
        final_conf = min(70 + net_score, 94.0)
        reason = "Strong credibility flags (FAKE)"
    elif net_score >= 4 and ml_label == "REAL":
        final_label = "FAKE"
        final_conf = 65.0
        reason = "Credibility flags override REAL prediction"
    
    # Override: High positive signals
    elif net_score <= -5 and ml_label == "FAKE":
        final_label = "REAL"
        final_conf = 70.0
        reason = "Trusted source override FAKE prediction"

    # Override: Uncertainty
    if uncertain_score >= 4 and final_conf < 85:
        final_label = "UNCERTAIN"
        final_conf = 50.0
        reason = "Preliminary/Uncertain language detected"

    # Confidence Labeling
    if final_conf >= 85:
        level = "HIGH"
    elif final_conf >= 65:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "final_label": final_label,
        "final_confidence": round(final_conf, 2),
        "confidence_level": level,
        "decision_reason": reason
    }
