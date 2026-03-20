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
    "scientists have strongly rejected"  : 4, "strongly rejected these claims"     : 4,
    "experts have rejected"              : 3, "doctors have rejected"              : 3,
    "no scientific basis"                : 4, "no scientific evidence"             : 4,
    "biologically impossible"            : 4, "scientifically impossible"          : 4,
    "cannot be bypassed"                 : 2, "eliminate the need for sleep"       : 4,
    "eliminate the need for"             : 3, "completely eliminate"               : 2,
    "without any negative"               : 2, "without causing any"                : 2,
    "fully alert for weeks"              : 3, "weeks without rest"                 : 3,
    "newly discovered"                   : 2, "plant extract"                      : 2,
    "natural extract"                    : 2, "miracle extract"                    : 3,
    "early trials showed"                : 2, "early trials"                       : 1,
    "claimed benefits"                   : 2, "claimed results"                    : 2,
}

def detect_pseudoscience(text_lower: str) -> int:
    """
    Detects pseudoscientific claims that contradict
    established biology/medicine/physics.
    These are almost always fake news.
    """
    score = 0

    # Extraordinary biological claims
    bio_impossible = [
        "eliminate the need for sleep", "without sleep", "no sleep needed",
        "cure all diseases", "cures all", "reverses aging completely",
        "regrow organs", "regrow limbs", "unlimited energy",
        "defy gravity", "100% effective cure", "guaranteed cure",
        "no side effects whatsoever", "completely safe for everyone",
        "ancient secret cure", "forbidden cure",
    ]

    # Contradiction patterns
    contradiction_pairs = [
        ("sleep is a biological necessity", "eliminate"),
        ("scientists rejected", "claims"),
        ("experts dismissed", "claims"),
        ("no evidence", "cure"),
        ("no evidence", "treatment"),
        ("rejected by", "researchers"),
    ]

    for claim in bio_impossible:
        if claim in text_lower:
            score += 4
            print(f"  Pseudoscience detected: '{claim}'")

    for word1, word2 in contradiction_pairs:
        if word1 in text_lower and word2 in text_lower:
            score += 3
            print(f"  Contradiction detected: '{word1}' + '{word2}'")

    return score

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
    
    pseudo_score = detect_pseudoscience(text_lower)
    net_score += pseudo_score
    
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
