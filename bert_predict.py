# bert_predict.py

import os
import requests
import time

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
BERT_MODEL = "omykhailiv/bert-fake-news-recognition"
BERT_URL   = f"https://api-inference.huggingface.co/models/{BERT_MODEL}"


def bert_predict(text: str) -> dict:
    if not HF_TOKEN:
        raise Exception("HF_TOKEN not set in environment variables")

    headers = {
        "Authorization"    : f"Bearer {HF_TOKEN}",
        "Content-Type"     : "application/json",
        "x-wait-for-model" : "true",
    }

    # This model works best with 6-12 words (headline style)
    # For long articles, use first 200 words
    words      = text.strip().split()
    text_input = " ".join(words[:200])

    last_error = None

    for attempt in range(3):
        try:
            print(f"  BERT attempt {attempt+1}/3...")
            response = requests.post(
                BERT_URL,
                headers = headers,
                json    = {"inputs": text_input},
                timeout = 25
            )
            print(f"  BERT status: {response.status_code}")

            if response.status_code == 503:
                print("  Model loading, waiting 5s...")
                time.sleep(5)
                continue

            if response.status_code == 401:
                raise Exception("Invalid HF_TOKEN")

            if response.status_code == 404:
                raise Exception("Model not found")

            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text[:200]}")

            data = response.json()
            print(f"  BERT raw: {str(data)[:200]}")
            return parse_bert_response(data)

        except requests.exceptions.Timeout:
            last_error = "Timeout"
            print(f"  Timeout (attempt {attempt+1})")
            time.sleep(3)
            continue
        except Exception as e:
            last_error = str(e)
            print(f"  Error (attempt {attempt+1}): {e}")
            time.sleep(2)
            continue

    raise Exception(f"BERT failed: {last_error}")


def parse_bert_response(data) -> dict:
    """
    omykhailiv model returns:
    LABEL_0 = FAKE (false news)
    LABEL_1 = REAL (true news)
    Score = confidence probability
    """
    try:
        # Unwrap nested list
        if isinstance(data, list) and isinstance(data[0], list):
            data = data[0]

        items     = data
        fake_prob = 0.0
        real_prob = 0.0

        for item in items:
            label = str(item.get("label", "")).upper().strip()
            score = float(item.get("score", 0))
            print(f"  Parsing → Label: {label}  Score: {score:.4f}")

            # omykhailiv model specific labels
            if label == "LABEL_0":
                fake_prob = score * 100    # LABEL_0 = FAKE
            elif label == "LABEL_1":
                real_prob = score * 100    # LABEL_1 = REAL

            # Fallback for other formats
            elif "FAKE" in label or "FALSE" in label:
                fake_prob = score * 100
            elif "REAL" in label or "TRUE" in label:
                real_prob = score * 100

        # Normalize to 100%
        total = fake_prob + real_prob
        if total > 0:
            fake_prob = round((fake_prob / total) * 100, 2)
            real_prob = round((real_prob / total) * 100, 2)
        else:
            raise Exception(f"Could not parse labels: {items}")

        # Cap at 95%
        fake_prob  = min(fake_prob, 95.0)
        real_prob  = min(real_prob, 95.0)
        label      = "FAKE" if fake_prob > real_prob else "REAL"
        confidence = round(max(fake_prob, real_prob), 2)

        print(f"  ✅ BERT result: {label} (fake:{fake_prob}% real:{real_prob}%)")

        return {
            "label"      : label,
            "confidence" : confidence,
            "fake_prob"  : fake_prob,
            "real_prob"  : real_prob,
        }

    except Exception as e:
        raise Exception(f"Parse failed: {e} | raw: {str(data)[:300]}")
