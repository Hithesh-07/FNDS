# decision_engine.py
# This runs BEFORE ML models
# Rules take priority over BERT/SVM

def run_decision_engine(text: str, ml_label: str,
                        ml_confidence: float,
                        fake_prob: float,
                        real_prob: float) -> dict:
    """
    3-layer decision system:
    Layer 1: Credibility scoring (fake signals vs real signals)
    Layer 2: Uncertainty detection
    Layer 3: ML model result (BERT/SVM)

    Rules override ML when evidence is strong enough.
    """

    text_lower = text.lower()
    gap        = abs(fake_prob - real_prob)

    # ════════════════════════════════════════════════════
    # LAYER 1 — CREDIBILITY SCORING
    # ════════════════════════════════════════════════════

    fake_score = 0
    real_score = 0
    fake_flags = []
    real_flags = []

    # ── Strong fake signals (weight 3) ──────────────────
    strong_fake = {
        "no verifiable evidence"    : 3,
        "no verifiable data"        : 3,
        "not been published"        : 3,
        "withheld due to"           : 3,
        "being withheld"            : 3,
        "allegedly preventing"      : 3,
        "attempts to share"         : 2,
        "attempts have been blocked": 3,
        "blocked from publishing"   : 3,
        "pressure from"             : 2,
        "industry stakeholders"     : 2,
        "quietly developing"        : 2,
        "leaked documents"          : 3,
        "leaked document"           : 3,
        "suppressed"                : 3,
        "suppress"                  : 3,
        "cover-up"                  : 3,
        "cover up"                  : 3,
        "deep state"                : 4,
        "big pharma"                : 3,
        "new world order"           : 4,
        "they don't want"           : 3,
        "government hiding"         : 3,
        "miracle cure"              : 3,
        "cure all diseases"         : 4,
        "cure all"                  : 3,
        "reverses aging"            : 3,
        "circulating online"        : 2,
        "circulating on platforms"  : 2,
        "circulating on several"    : 2,
        "viral claim"               : 2,
        "forwarded message"         : 3,
        "independent verification has not been possible" : 4,
        "cannot be independently verified" : 4,
        "experts have expressed skepticism": 2,
    }

    # ── Medium fake signals (weight 2) ───────────────────
    medium_fake = {
        "unnamed sources"           : 2,
        "unnamed individuals"       : 2,
        "anonymous sources"         : 2,
        "no evidence"               : 2,
        "no proof"                  : 2,
        "not verified"              : 2,
        "unverified"                : 2,
        "cannot be confirmed"       : 2,
        "could not be confirmed"    : 2,
        "could not be verified"     : 2,
        "no peer-reviewed"          : 2,
        "not peer reviewed"         : 2,
        "no studies"                : 2,
        "claims have generated"     : 1,
        "significant interest"      : 1,
        "skepticism"                : 1,
        "feasibility"               : 1,
    }

    # ── Strong real signals (weight 2-3) ─────────────────
    strong_real = {
        "peer-reviewed"             : 3,
        "peer reviewed"             : 3,
        "published in the"          : 3,
        "study published"           : 3,
        "official statement"        : 3,
        "press conference"          : 2,
        "data shows"                : 2,
        "statistics show"           : 2,
        "basis points"              : 3,
        "quarterly earnings"        : 3,
        "quarterly revenue"         : 3,
        "annual report"             : 3,
        "financial results"         : 3,
        "according to official"     : 2,
        "confirmed by"              : 2,
        "spokesperson said"         : 2,
        "minister said"             : 2,
        "minister stated"           : 2,
        "officials state"           : 2,
        "officials stated"          : 2,
        "government announced"      : 2,
        "central bank"              : 2,
        "reserve bank"              : 2,
        "federal reserve"           : 2,
        "supreme court"             : 2,
        "parliament"                : 2,
        "policy initiative"         : 2,
        "aims to"                   : 1,
        "renewable energy"          : 1,
        "carbon emissions"          : 1,
        "grid infrastructure"       : 1,
        "fiscal year"               : 2,
        "monetary policy"           : 2,
        "domestic consumption"      : 2,
        "commodity prices"          : 2,
        "financial conditions"      : 2,
        "emerging markets"          : 1,
        "policymakers"              : 2,
        "analysts emphasize"        : 2,
        "industry experts"          : 1,
        "implementation challenges" : 1,
    }

    # Calculate scores
    for phrase, weight in strong_fake.items():
        if phrase in text_lower:
            fake_score += weight
            fake_flags.append(f"{phrase} (+{weight})")

    for phrase, weight in medium_fake.items():
        if phrase in text_lower:
            fake_score += weight
            fake_flags.append(f"{phrase} (+{weight})")

    for phrase, weight in strong_real.items():
        if phrase in text_lower:
            real_score += weight
            real_flags.append(f"{phrase} (+{weight})")

    net_score = fake_score - real_score

    # ════════════════════════════════════════════════════
    # LAYER 2 — UNCERTAINTY DETECTION
    # ════════════════════════════════════════════════════

    uncertain_score = 0

    uncertain_phrases = {
        "preliminary"               : 3,
        "preliminary study"         : 4,
        "preliminary findings"      : 4,
        "limited data"              : 3,
        "not conclusive"            : 3,
        "not be interpreted as conclusive" : 4,
        "further research needed"   : 3,
        "more research needed"      : 3,
        "more evidence required"    : 3,
        "additional studies"        : 3,
        "additional controlled"     : 3,
        "controlled studies required": 3,
        "should not be interpreted" : 3,
        "caution that"              : 2,
        "experts caution"           : 2,
        "researchers caution"       : 2,
        "based on limited"          : 3,
        "based on correlations"     : 3,
        "not causal"                : 3,
        "may influence"             : 2,
        "may have"                  : 1,
        "might"                     : 1,
        "suggest that"              : 2,
        "suggests that"             : 2,
        "some experts suggest"      : 3,
        "some researchers suggest"  : 3,
        "observational study"       : 3,
        "across different populations": 2,
        "underlying mechanisms"     : 2,
    }

    for phrase, weight in uncertain_phrases.items():
        if phrase in text_lower:
            uncertain_score += weight

    # ════════════════════════════════════════════════════
    # LAYER 3 — FINAL DECISION
    # Runs in strict priority order
    # ════════════════════════════════════════════════════

    # DECISION 1: Very strong fake credibility evidence
    # Multiple "no evidence" + "unnamed sources" + "suppressed" etc
    if net_score >= 6:
        confidence = min(65.0 + net_score * 2, 93.0)
        return _result("FAKE", confidence, fake_score,
                       real_score, net_score, uncertain_score,
                       fake_flags, real_flags, "credibility_override")

    # DECISION 2: Moderate fake evidence AND ML also says fake
    if net_score >= 3 and ml_label == "FAKE":
        confidence = min(ml_confidence + net_score * 2, 93.0)
        return _result("FAKE", confidence, fake_score,
                       real_score, net_score, uncertain_score,
                       fake_flags, real_flags, "combined_fake")

    # DECISION 3: Moderate fake evidence but ML says real
    # Well-written misinformation case
    if net_score >= 4 and ml_label == "REAL":
        confidence = min(60.0 + net_score * 2, 82.0)
        return _result("FAKE", confidence, fake_score,
                       real_score, net_score, uncertain_score,
                       fake_flags, real_flags, "credibility_fake_override")

    # DECISION 4: Strong uncertainty signals
    # Don't let model override clear uncertainty language
    if uncertain_score >= 8 and net_score < 3:
        confidence = min(ml_confidence, 74.0)
        return _result("UNCERTAIN", confidence, fake_score,
                       real_score, net_score, uncertain_score,
                       fake_flags, real_flags, "uncertainty_override")

    # DECISION 5: Moderate uncertainty + model overconfident
    if uncertain_score >= 5 and ml_label == "REAL" and ml_confidence > 80:
        return _result("UNCERTAIN", 72.0, fake_score,
                       real_score, net_score, uncertain_score,
                       fake_flags, real_flags, "uncertainty_moderate")

    # DECISION 6: Strong real credibility evidence
    if real_score >= 4 and fake_score == 0:
        confidence = min(ml_confidence + real_score, 93.0)
        return _result("REAL", confidence, fake_score,
                       real_score, net_score, uncertain_score,
                       fake_flags, real_flags, "credibility_real")

    # DECISION 7: ML model confident + no red flags
    if gap > 35 and net_score < 2 and uncertain_score < 4:
        return _result(ml_label, ml_confidence, fake_score,
                       real_score, net_score, uncertain_score,
                       fake_flags, real_flags, "ml_confident")

    # DECISION 8: ML internally confused (small gap)
    if gap < 15:
        return _result("UNCERTAIN", ml_confidence, fake_score,
                       real_score, net_score, uncertain_score,
                       fake_flags, real_flags, "ml_uncertain")

    # DECISION 9: Default — trust ML
    return _result(ml_label, ml_confidence, fake_score,
                   real_score, net_score, uncertain_score,
                   fake_flags, real_flags, "ml_default")


def _result(label, confidence, fake_score, real_score,
            net_score, uncertain_score, fake_flags,
            real_flags, reason):
    """Helper to build consistent result dict."""
    confidence = round(min(max(confidence, 50.0), 95.0), 2)

    if confidence >= 85:
        conf_level = "HIGH"
    elif confidence >= 65:
        conf_level = "MEDIUM"
    else:
        conf_level = "LOW"

    return {
        "final_label"       : label,
        "final_confidence"  : confidence,
        "confidence_level"  : conf_level,
        "fake_score"        : fake_score,
        "real_score"        : real_score,
        "net_score"         : net_score,
        "uncertain_score"   : uncertain_score,
        "fake_flags"        : fake_flags[:6],
        "real_flags"        : real_flags[:6],
        "decision_reason"   : reason,
    }
