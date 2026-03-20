import os
import requests
import time

# ── Use the NEW HuggingFace router URL ────────────────────
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

# Best fake news model — already fine-tuned
BERT_MODEL = "hamzab/roberta-fake-news-classification"
BERT_URL   = f"https://router.huggingface.co/models/{BERT_MODEL}"

HEADERS = {
    "Authorization"    : f"Bearer {HF_TOKEN}",
    "Content-Type"     : "application/json",
    "x-wait-for-model" : "true",
}


def parse_bert_response(data) -> dict:
    """
    Handles every possible HuggingFace response format.
    """
    try:
        # Unwrap nested lists
        if isinstance(data, list) and isinstance(data[0], list):
            data = data[0]
        if isinstance(data, list):
            items = data
        else:
            raise Exception(f"Unexpected format: {type(data)}")

        scores = {}
        for item in items:
            label = str(item.get("label", "")).upper().strip()
            score = float(item.get("score", 0))

            # Normalize all possible label formats
            if label in ["FAKE", "LABEL_1", "LABEL_0",
                         "NEGATIVE", "0", "FALSE"]:
                # Check which one means FAKE for this model
                # hamzab model: FAKE=FAKE, REAL=REAL
                if label == "FAKE":
                    scores["FAKE"] = score
                elif label == "REAL":
                    scores["REAL"] = score
                elif label == "LABEL_0":
                    scores["REAL"] = score
                elif label == "LABEL_1":
                    scores["FAKE"] = score
            elif label in ["REAL", "POSITIVE", "1", "TRUE"]:
                scores["REAL"] = score

        # If still empty try direct assignment
        if not scores:
            for item in items:
                label = str(item.get("label","")).upper()
                score = float(item.get("score", 0))
                if "FAKE" in label or "FALSE" in label:
                    scores["FAKE"] = score
                elif "REAL" in label or "TRUE" in label:
                    scores["REAL"] = score

        if not scores:
            raise Exception(f"Could not parse labels from: {items}")

        fake_prob = round(scores.get("FAKE", 0.0) * 100, 2)
        real_prob = round(scores.get("REAL", 0.0) * 100, 2)

        # Normalize to 100%
        total = fake_prob + real_prob
        if total > 0 and abs(total - 100) > 1:
            fake_prob = round((fake_prob / total) * 100, 2)
            real_prob = round((real_prob / total) * 100, 2)

        # Cap at 95%
        fake_prob = min(fake_prob, 95.0)
        real_prob = min(real_prob, 95.0)

        label      = "FAKE" if fake_prob > real_prob else "REAL"
        confidence = round(max(fake_prob, real_prob), 2)
        
        conf_level = "HIGH" if confidence >= 85 else "MEDIUM" if confidence >= 70 else "LOW"

        return {
            "label"      : label,
            "confidence" : confidence,
            "confidence_level" : conf_level,
            "fake_prob"  : fake_prob,
            "real_prob"  : real_prob,
            "gap"        : round(abs(fake_prob - real_prob), 2),
            "model_used" : "BERT (RoBERTa)"
        }

    except Exception as e:
        raise Exception(f"parse_bert_response failed: {e} | raw: {str(data)[:200]}")


def bert_predict(text: str) -> dict:
    """
    Calls HuggingFace API with retry logic.
    Never returns 50/50 — raises exception on failure
    so fusion engine uses SVM instead.
    """

    # Truncate to 512 tokens worth of text
    text_input = str(text)[:1500]

    last_error = None

    for attempt in range(3):
        try:
            response = requests.post(
                BERT_URL,
                headers = HEADERS,
                json    = {"inputs": text_input},
                timeout = 20
            )

            # Model loading — wait and retry
            if response.status_code == 503:
                print(f"  BERT loading (attempt {attempt+1}/3)...")
                time.sleep(4)
                continue

            if response.status_code != 200:
                raise Exception(
                    f"API returned {response.status_code}: {response.text[:100]}"
                )

            data = response.json()

            # Parse response — handle all formats
            result = parse_bert_response(data)
            print(f"  BERT success: {result['label']} ({result['confidence']}%)")
            return result

        except requests.exceptions.Timeout:
            last_error = "Timeout"
            print(f"  BERT timeout (attempt {attempt+1}/3)")
            time.sleep(2)
            continue

        except Exception as e:
            last_error = str(e)
            print(f"  BERT error (attempt {attempt+1}/3): {e}")
            time.sleep(2)
            continue

    # All retries failed — raise so fusion uses SVM
    raise Exception(f"BERT failed after 3 attempts: {last_error}")
