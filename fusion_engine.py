# fusion_engine.py — v1.0 Model Fusion Core
# Runs BERT + SVM simultaneously
# Merges outputs using weighted consensus
# Applies credibility layer on top

BERT_WEIGHT = 0.72
SVM_WEIGHT  = 0.28

# NEW: SVM real bias correction
# SVM is trained to lean REAL — correct for this
SVM_FAKE_BIAS_CORRECTION = 8.0  # add 8% to SVM fake_prob


# Minimum confidence gap for BERT to override SVM
BERT_OVERRIDE_THRESHOLD = 25


def fuse_predictions(bert_result: dict,
                     svm_result: dict,
                     credibility_net_score: int) -> dict:
    """
    Merges BERT + SVM predictions into final verdict.

    Priority order:
    1. Credibility rules (if very strong)
    2. Model agreement consensus
    3. BERT dominates if much more confident
    4. Weighted average otherwise
    """

    b_fake = bert_result.get("fake_prob", 50.0)
    b_real = bert_result.get("real_prob", 50.0)

    # Apply SVM bias correction before fusion
    s_fake_raw = svm_result.get("fake_prob", 50.0)
    s_real_raw = svm_result.get("real_prob", 50.0)

    # SVM tends to underestimate fake — correct it
    s_fake = min(s_fake_raw + SVM_FAKE_BIAS_CORRECTION, 95.0)
    s_real = max(s_real_raw - SVM_FAKE_BIAS_CORRECTION, 5.0)

    # Re-normalize
    total  = s_fake + s_real
    s_fake = round((s_fake / total) * 100, 2)
    s_real = round((s_real / total) * 100, 2)

    b_conf = bert_result.get("confidence", 50.0)
    s_conf = svm_result.get("confidence", 50.0)

    b_label = "FAKE" if b_fake > b_real else "REAL"
    s_label = "FAKE" if s_fake > s_real else "REAL"

    # ── Step 1: Weighted fusion ──────────────────────────
    fused_fake = round(b_fake * BERT_WEIGHT + s_fake * SVM_WEIGHT, 2)
    fused_real = round(b_real * BERT_WEIGHT + s_real * SVM_WEIGHT, 2)

    fused_label = "FAKE" if fused_fake > fused_real else "REAL"
    fused_conf  = round(max(fused_fake, fused_real), 2)

    # ── Step 2: Agreement analysis ───────────────────────
    if b_label == s_label:
        agreement       = "FULL"
        agreement_text  = "Both models agree"
        # Boost confidence when both agree
        fused_conf      = min(fused_conf + 8, 95.0)
    else:
        agreement       = "SPLIT"
        agreement_text  = "Models disagree"
        # Reduce confidence when models disagree
        fused_conf      = fused_conf - 15

        # Check if BERT is MUCH more confident → let it lead
        bert_gap = abs(b_fake - b_real)
        svm_gap  = abs(s_fake - s_real)

        if bert_gap > svm_gap + BERT_OVERRIDE_THRESHOLD:
            fused_label    = b_label
            fused_conf     = b_conf - 10
            agreement      = "BERT_LEADS"
            agreement_text = "BERT overrides SVM"
        elif svm_gap > bert_gap + BERT_OVERRIDE_THRESHOLD:
            fused_label    = s_label
            fused_conf     = s_conf - 10
            agreement      = "SVM_LEADS"
            agreement_text = "SVM overrides BERT"

    # ── Step 3: Uncertainty gate ─────────────────────────
    # If fused confidence too low → UNCERTAIN
    if fused_conf < 55:
        fused_label    = "UNCERTAIN"
        agreement_text = "Insufficient confidence"

    # ── Step 4: Credibility override ────────────────────
    # Strong credibility evidence overrides both models
    if credibility_net_score >= 7:
        fused_label    = "FAKE"
        fused_conf     = min(65 + credibility_net_score, 93.0)
        agreement_text = "Credibility override"
    elif credibility_net_score >= 4 and fused_label == "REAL":
        fused_label    = "FAKE"
        fused_conf     = min(60 + credibility_net_score, 85.0)
        agreement_text = "Credibility flags fake"
    elif credibility_net_score <= -4 and fused_label == "FAKE":
        fused_label    = "REAL"
        fused_conf     = min(abs(credibility_net_score) * 5 + 60, 90.0)
        agreement_text = "Credibility confirms real"

    fused_conf = round(min(max(fused_conf, 50.0), 95.0), 2)

    # ── Step 5: Confidence level label ──────────────────
    if fused_conf >= 85:
        conf_level = "HIGH"
    elif fused_conf >= 68:
        conf_level = "MEDIUM"
    else:
        conf_level = "LOW"

    return {
        "label"            : fused_label,
        "confidence"       : fused_conf,
        "confidence_level" : conf_level,
        "agreement"        : agreement,
        "agreement_text"   : agreement_text,
        "fused_fake"       : fused_fake,
        "fused_real"       : fused_real,
        "bert_fake"        : b_fake,
        "bert_real"        : b_real,
        "bert_confidence"  : b_conf,
        "bert_label"       : b_label,
        "svm_fake"         : s_fake,
        "svm_real"         : s_real,
        "svm_confidence"   : s_conf,
        "svm_label"        : s_label,
    }


def run_fusion(text: str,
               bert_predict_fn,
               svm_predict_fn,
               svm_model,
               svm_vectorizer,
               svm_scaler,
               decision_engine_fn) -> dict:
    """
    Main entry point. Runs everything and returns final result.
    """
    errors = []

    # ── Run BERT ─────────────────────────────────────────
    try:
        bert_result = bert_predict_fn(text)
        if bert_result is None:
             raise ValueError("BERT returned None")
        bert_ok     = True
    except Exception as e:
        print(f"BERT error: {e}")
        bert_ok     = False
        bert_result = None
        errors.append(f"BERT: {str(e)[:50]}")

    # ── Run SVM ──────────────────────────────────────────
    try:
        svm_result = svm_predict_fn(text, svm_model, svm_vectorizer, svm_scaler)
        svm_ok     = True
    except Exception as e:
        print(f"SVM error: {e}")
        svm_result = {"fake_prob": 50, "real_prob": 50,
                      "confidence": 50, "label": "UNCERTAIN"}
        svm_ok     = False
        errors.append(f"SVM: {str(e)[:50]}")

    # ── Run Decision Engine ───────────────────────────────
    decision    = decision_engine_fn(text)
    cred_score  = decision.get("net_score", 0)

    # ── Fuse results ─────────────────────────────────────
    global BERT_WEIGHT, SVM_WEIGHT, SVM_FAKE_BIAS_CORRECTION
    if not bert_ok:
        # BERT failed — use SVM with full weight but stronger bias correction
        old_bert_weight = BERT_WEIGHT
        old_svm_weight = SVM_WEIGHT
        old_bias = SVM_FAKE_BIAS_CORRECTION
        
        BERT_WEIGHT = 0.0
        SVM_WEIGHT  = 1.0
        SVM_FAKE_BIAS_CORRECTION = 12.0  # stronger correction
        
        # Give mock values to fuse_predictions just so variables don't fail parsing
        fusion = fuse_predictions({"fake_prob": 50, "real_prob": 50, "confidence": 50, "label": "UNCERTAIN"}, svm_result, cred_score)
        
        # Restore globals for next run
        BERT_WEIGHT = old_bert_weight
        SVM_WEIGHT = old_svm_weight
        SVM_FAKE_BIAS_CORRECTION = old_bias
    else:
        fusion = fuse_predictions(bert_result, svm_result, cred_score)

    # ── Determine model status ────────────────────────────
    if bert_ok and svm_ok:
        model_status = "BERT + SVM Fusion"
    elif bert_ok:
        model_status = "BERT Only (SVM failed)"
    elif svm_ok:
        model_status = "SVM Only (BERT unavailable)"
    else:
        model_status = "Rules Only (both failed)"

    return {
        # Final verdict
        "label"            : fusion["label"],
        "confidence"       : fusion["confidence"],
        "confidence_level" : fusion["confidence_level"],

        # Agreement info
        "agreement"        : fusion["agreement"],
        "agreement_text"   : fusion["agreement_text"],
        "model_used"       : model_status,

        # Individual model results
        "bert_label"       : fusion["bert_label"],
        "bert_fake"        : fusion["bert_fake"],
        "bert_real"        : fusion["bert_real"],
        "bert_confidence"  : fusion["bert_confidence"],
        "bert_ok"          : bert_ok,

        "svm_label"        : fusion["svm_label"],
        "svm_fake"         : fusion["svm_fake"],
        "svm_real"         : fusion["svm_real"],
        "svm_confidence"   : fusion["svm_confidence"],
        "svm_ok"           : svm_ok,

        # Fused probabilities
        "fake_prob"        : fusion["fused_fake"],
        "real_prob"        : fusion["fused_real"],

        # Credibility layer
        "net_score"        : cred_score,
        "credibility_flags": decision.get("fake_flags", []),
        "real_flags"       : decision.get("real_flags", []),
        "decision_reason"  : decision.get("decision_reason", "fusion"),

        # Keywords and red flags from SVM
        "keywords"         : svm_result.get("keywords", {}),
        "red_flags"        : svm_result.get("red_flags", {}),

        # Debug
        "errors"           : errors,
    }
