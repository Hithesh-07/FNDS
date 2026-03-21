import os

# decision_engine.py

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

def calculate_fake_score(text_lower: str) -> int:
    score = 0
    suspicious_according = [
        "according to unnamed", "according to anonymous", "according to sources",
        "according to insiders", "according to individuals", "according to a report circulating",
        "according to unverified",
    ]
    for phrase in suspicious_according:
        if phrase in text_lower: score += 3

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
    org_found = any(org in text_lower for org in trusted_orgs)
    conspiracy_found = any(c in text_lower for c in conspiracy_context)
    if org_found and conspiracy_found:
        score += 4

    bio_impossible = [
        "eliminate the need for sleep", "without sleep", "no sleep needed",
        "cure all diseases", "cure for all diseases", "cures all", "reverses aging completely",
        "regrow organs", "regrow limbs", "unlimited energy",
        "defy gravity", "100% effective cure", "guaranteed cure",
    ]
    for claim in bio_impossible:
        if claim in text_lower:
            score += 4

    for phrase, val in STRONG_FAKE_SIGNALS.items():
        if phrase in text_lower:
            score += val

    return score


def calculate_real_score(text_lower: str) -> int:
    score = 0
    credible_according = [
        "according to official", "according to the government", "according to the ministry",
        "according to the reserve bank", "according to the central bank", "according to researchers at",
        "according to the report", "according to data", "according to the study", "according to nasa",
        "according to who", "according to the united nations", "according to the supreme court",
    ]
    for phrase in credible_according:
        if phrase in text_lower: score += 2

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
    org_found = any(org in text_lower for org in trusted_orgs)
    conspiracy_found = any(c in text_lower for c in conspiracy_context)
    if org_found and not conspiracy_found:
        score += 3

    return score


def get_fake_flags(text_lower: str) -> list:
    flags = []
    for phrase in STRONG_FAKE_SIGNALS.keys():
        if phrase in text_lower:
            flags.append(phrase.title())
    return flags


def get_real_flags(text_lower: str) -> list:
    return []


def calculate_uncertainty_score(text_lower: str) -> int:
    """
    Scores how uncertain/hedged the language is.
    High score = article should be UNCERTAIN not FAKE or REAL.
    """
    score = 0

    # ── Tier 1: Very strong uncertainty signals (score 4 each) ──
    tier1 = [
        "more research is needed",
        "more evidence is needed",
        "more evidence required",
        "further research is needed",
        "further studies are needed",
        "further investigation is needed",
        "not yet confirmed",
        "cannot be confirmed",
        "could not be confirmed",
        "not yet proven",
        "remains unproven",
        "should not be interpreted as conclusive",
        "based on limited data",
        "preliminary findings",
        "preliminary study",
        "preliminary research",
        "observational study",
        "inconclusive results",
        "inconclusive evidence",
        "evidence remains mixed",
        "mixed results",
        "mixed evidence",
    ]

    # ── Tier 2: Medium uncertainty signals (score 2 each) ──────
    tier2 = [
        "suggest that",
        "suggests that",
        "suggested that",
        "suggest ",
        "may indicate",
        "may suggest",
        "may be associated",
        "may have",
        "might indicate",
        "might suggest",
        "could indicate",
        "could suggest",
        "appears to",
        "seems to",
        "according to some",
        "some researchers",
        "some scientists",
        "some experts",
        "some studies",
        "certain studies",
        "experts caution",
        "researchers caution",
        "scientists caution",
        "warrants further",
        "requires further",
        "needs further",
        "not yet fully understood",
        "not fully understood",
        "debate continues",
        "experts disagree",
        "scientists disagree",
        "disputed finding",
        "disputed claim",
        "controversial finding",
        "awaiting peer review",
        "has not been peer reviewed",
        "preprint",
        "early stage",
        "early stages",
        "early results",
        "initial results",
        "initial findings",
    ]

    # ── Tier 3: Weak uncertainty signals (score 1 each) ─────────
    tier3 = [
        "may ", "might ", "could ", "possibly ",
        "potentially ", "perhaps ", "tentatively ",
        "reportedly ", "allegedly ", "apparently ",
        "it seems ", "it appears ", "believed to ",
        "thought to ", "expected to ", "likely to ",
        "unclear ", "uncertain ", "unknown ",
    ]

    for phrase in tier1:
        if phrase in text_lower:
            score += 4

    for phrase in tier2:
        if phrase in text_lower:
            score += 2

    for phrase in tier3:
        count = text_lower.count(phrase)
        score += min(count, 3)  # max 3 per tier3 word

    return score


def run_decision_engine(text, ml_label, ml_conf,
                        fake_prob, real_prob) -> dict:
    """
    4-layer decision engine.
    Priority: Rules → Uncertainty → ML model
    """
    text_lower = text.lower()
    gap        = abs(fake_prob - real_prob)

    # ── Calculate all scores ───────────────────────────────
    fake_score    = calculate_fake_score(text_lower)
    real_score    = calculate_real_score(text_lower)
    net_score     = fake_score - real_score
    uncert_score  = calculate_uncertainty_score(text_lower)

    fake_flags    = get_fake_flags(text_lower)
    real_flags    = get_real_flags(text_lower)

    print(f"  Scores → fake:{fake_score} real:{real_score} "
          f"net:{net_score} uncertain:{uncert_score} gap:{gap:.1f}")

    # ══════════════════════════════════════════════════════
    # DECISION TREE — ORDER MATTERS
    # ══════════════════════════════════════════════════════

    # RULE 0: Very strong UNCERTAINTY signals
    # Check BEFORE fake rules — hedged language ≠ fake news
    if uncert_score >= 8 and net_score < 6:
        # Strong uncertainty + not enough fake signals
        return _result("UNCERTAIN",
                       min(max(fake_prob, real_prob), 74.0),
                       fake_score, real_score, net_score,
                       uncert_score, fake_flags, real_flags,
                       "strong_uncertainty_detected")

    # RULE 1: Extreme fake credibility evidence
    # Only fire if fake signal is MUCH stronger than uncertainty
    if net_score >= 8 and net_score > uncert_score:
        confidence = min(65.0 + net_score * 2, 93.0)
        return _result("FAKE", confidence,
                       fake_score, real_score, net_score,
                       uncert_score, fake_flags, real_flags,
                       "credibility_override_fake")

    # RULE 2: Moderate fake evidence AND ML also says fake
    if net_score >= 4 and ml_label == "FAKE" and uncert_score < 5:
        confidence = min(ml_conf + net_score * 2, 93.0)
        return _result("FAKE", confidence,
                       fake_score, real_score, net_score,
                       uncert_score, fake_flags, real_flags,
                       "combined_fake_signal")

    # RULE 3: Model says REAL with strong credibility backing
    if real_score >= 4 and fake_score == 0 and uncert_score < 4:
        confidence = min(ml_conf + real_score, 93.0)
        return _result("REAL", confidence,
                       fake_score, real_score, net_score,
                       uncert_score, fake_flags, real_flags,
                       "credibility_real")

    # RULE 4: Model is internally confused (small probability gap)
    if gap < 15:
        return _result("UNCERTAIN", max(fake_prob, real_prob),
                       fake_score, real_score, net_score,
                       uncert_score, fake_flags, real_flags,
                       "ml_gap_uncertain")

    # RULE 5: Moderate uncertainty + model overconfident about REAL
    if uncert_score >= 5 and ml_label == "REAL" and ml_conf > 75:
        return _result("UNCERTAIN",
                       min(max(fake_prob, real_prob), 72.0),
                       fake_score, real_score, net_score,
                       uncert_score, fake_flags, real_flags,
                       "uncertainty_overrides_real")

    # RULE 6: Mild uncertainty — reduce confidence but keep label
    if uncert_score >= 3 and ml_conf > 85:
        adjusted_conf = ml_conf - (uncert_score * 3)
        if adjusted_conf < 60:
            return _result("UNCERTAIN", adjusted_conf,
                           fake_score, real_score, net_score,
                           uncert_score, fake_flags, real_flags,
                           "uncertainty_reduces_confidence")

    # RULE 7: Default — trust the ML model
    return _result(ml_label, ml_conf,
                   fake_score, real_score, net_score,
                   uncert_score, fake_flags, real_flags,
                   "ml_default")


def _result(label, confidence, fake_score, real_score,
            net_score, uncertain_score, fake_flags,
            real_flags, reason) -> dict:
    confidence = round(min(max(float(confidence), 50.0), 95.0), 2)
    if confidence >= 85:   conf_level = "HIGH"
    elif confidence >= 68: conf_level = "MEDIUM"
    else:                  conf_level = "LOW"
    return {
        "final_label"      : label,
        "final_confidence" : confidence,
        "confidence_level" : conf_level,
        "fake_score"       : fake_score,
        "real_score"       : real_score,
        "net_score"        : net_score,
        "uncertain_score"  : uncertain_score,
        "fake_flags"       : fake_flags[:6],
        "real_flags"       : real_flags[:6],
        "decision_reason"  : reason,
    }
