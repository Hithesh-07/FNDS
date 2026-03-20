# bert_predict.py — Using 99.28% accuracy model

import os
import requests
import time

HF_TOKEN   = os.environ.get("HF_TOKEN", "")

# Primary: 99.28% accuracy — trained on 5M articles
BERT_MODEL = "Arko007/fake-news-roberta-5M"
BERT_URL   = f"https://api-inference.huggingface.co/models/{BERT_MODEL}"

# Fallback if primary fails
FALLBACK_MODEL = "jy46604790/Fake-News-Bert-Detect"
FALLBACK_URL   = f"https://api-inference.huggingface.co/models/{FALLBACK_MODEL}"


def bert_predict(text: str) -> dict:
    if not HF_TOKEN:
        raise Exception("HF_TOKEN not set")

    headers = {
        "Authorization"    : f"Bearer {HF_TOKEN}",
        "Content-Type"     : "application/json",
        "x-wait-for-model" : "true",
    }

    words      = text.strip().split()
    text_input = " ".join(words[:300])

    # Try primary model first
    try:
        result = call_api(BERT_URL, text_input, headers, "Primary")
        return result
    except Exception as e:
        print(f"  Primary model failed: {e}")
        print(f"  Trying fallback model...")

    # Try fallback model
    try:
        result = call_api(FALLBACK_URL, text_input, headers, "Fallback")
        return result
    except Exception as e:
        raise Exception(f"Both models failed: {e}")


def call_api(url: str, text: str, headers: dict, name: str) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            print(f"  {name} BERT attempt {attempt+1}/3...")
            response = requests.post(
                url,
                headers = headers,
                json    = {"inputs": text},
                timeout = 25
            )
            print(f"  {name} status: {response.status_code}")

            if response.status_code == 503:
                print(f"  {name} loading, waiting 5s...")
                time.sleep(5)
                continue

            if response.status_code == 401:
                raise Exception("Invalid HF_TOKEN")

            if response.status_code == 404:
                raise Exception(f"{name} model not found")

            if response.status_code != 200:
                raise Exception(f"API {response.status_code}: {response.text[:100]}")

            data = response.json()
            print(f"  {name} raw: {str(data)[:150]}")
            return parse_bert_response(data)

        except requests.exceptions.Timeout:
            last_error = "Timeout"
            time.sleep(3)
        except Exception as e:
            last_error = str(e)
            if "HF_TOKEN" in str(e) or "Invalid" in str(e):
                raise
            time.sleep(2)

    raise Exception(f"{name} failed: {last_error}")


def parse_bert_response(data) -> dict:
    try:
        if isinstance(data, list) and isinstance(data[0], list):
            data = data[0]

        items     = data
        fake_prob = 0.0
        real_prob = 0.0

        for item in items:
            label = str(item.get("label", "")).upper().strip()
            score = float(item.get("score", 0))
            print(f"  Label: {label}  Score: {score:.4f}")

            # Arko007 model: LABEL_0=FAKE, LABEL_1=REAL
            if label == "LABEL_0":
                fake_prob = score * 100
            elif label == "LABEL_1":
                real_prob = score * 100
            # jy46604790 fallback: direct FAKE/REAL
            elif "FAKE" in label:
                fake_prob = score * 100
            elif "REAL" in label:
                real_prob = score * 100

        total = fake_prob + real_prob
        if total > 0:
            fake_prob = round((fake_prob / total) * 100, 2)
            real_prob = round((real_prob / total) * 100, 2)
        else:
            raise Exception(f"Could not parse: {items}")

        fake_prob  = min(fake_prob, 95.0)
        real_prob  = min(real_prob, 95.0)
        label      = "FAKE" if fake_prob > real_prob else "REAL"
        confidence = round(max(fake_prob, real_prob), 2)

        print(f"  ✅ BERT: {label} (fake:{fake_prob}% real:{real_prob}%)")

        return {
            "label"      : label,
            "confidence" : confidence,
            "fake_prob"  : fake_prob,
            "real_prob"  : real_prob,
        }

    except Exception as e:
        raise Exception(f"Parse failed: {e}")
