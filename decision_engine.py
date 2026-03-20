"""
decision_engine.py — Context-Aware Credibility Scoring Engine v5.4
Fixes contextual bugs for 'according to' and 'organization mentions'.
"""

def score_according_to(text_lower: str) -> int:
    """
    'according to' only counts as REAL if followed by credible source.
    'according to unnamed' or 'according to anonymous' = FAKE signal instead.
    """
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
    """
    Org names only count as real signals if used credibly.
    If paired with 'secretly', 'hiding', 'suppressed' etc = FAKE.
    """
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
}

def run_decision_engine_raw(text: str) -> dict:
    """
    Returns only the net score and flags for fusion engine.
    """
    text_lower = text.lower()
    net_score = 0
    fake_flags = []
    real_flags = []

    # Score from contextual functions
    net_score += score_according_to(text_lower)
    net_score += score_organization_mentions(text_lower)
    
    # Simple flags for UI
    for phrase, val in STRONG_FAKE_SIGNALS.items():
        if phrase in text_lower:
            net_score += val
            fake_flags.append(phrase.title())

    if net_score >= 4:
        decision_reason = "High negative signals detected"
    elif net_score <= -3:
        decision_reason = "Credible source verified"
    else:
        decision_reason = "Standard fusion processing"

    return {
        "net_score": net_score,
        "fake_flags": fake_flags,
        "real_flags": real_flags,
        "decision_reason": decision_reason
    }
