import requests
import os
import time

# Hugging Face Inference Router (correct format with /hf-inference/ prefix)
API_URL = "https://router.huggingface.co/hf-inference/models/hamzab/roberta-fake-news-classification"
HF_TOKEN = os.getenv("HF_TOKEN", "")


def query(payload):
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=25)

            if response.status_code == 503:
                # Model loading — wait and retry
                try:
                    body = response.json()
                    wait = min(float(body.get("estimated_time", 10)), 15)
                except Exception:
                    wait = 10
                print(f"Model loading, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                print(f"HF API HTTP {response.status_code}: {response.text[:120]}")
                return None

            output = response.json()

            if isinstance(output, dict) and "error" in output:
                print(f"HF API Error: {output['error']}")
                return None

            return output

        except requests.exceptions.Timeout:
            print(f"HF API timeout (attempt {attempt+1})")
            time.sleep(2)
        except Exception as e:
            print(f"Requests error: {e}")
            time.sleep(2)

    return None


def bert_predict(text: str) -> dict:
    """
    RoBERTa via HuggingFace Router.
    Returns result dict or None (caller uses rule-based fallback).
    """
    try:
        payload = {"inputs": text[:512], "options": {"wait_for_model": True}}
        output = query(payload)

        if output is None:
            return None

        # Handle both [[{...}]] and [{...}] shapes
        if isinstance(output, list) and len(output) > 0:
            inner = output[0]
            if isinstance(inner, list) and len(inner) > 0:
                result = inner[0]
            elif isinstance(inner, dict):
                result = inner
            else:
                print(f"Unexpected output shape: {output}")
                return None
        else:
            print(f"Unexpected output type: {output}")
            return None

        label_raw = result.get("label", "").lower()
        score = float(result.get("score", 0.5))

        # LABEL_1 / real / true → REAL, else FAKE
        is_real = any(w in label_raw for w in ["real", "true", "label_1", "fact", "joy", "positive"])

        if is_real:
            label = "REAL"
            real_prob = round(score * 100, 2)
            fake_prob = round((1 - score) * 100, 2)
        else:
            label = "FAKE"
            fake_prob = round(score * 100, 2)
            real_prob = round((1 - score) * 100, 2)

        confidence = round(score * 100, 2)
        conf_level = "HIGH" if confidence >= 85 else "MEDIUM" if confidence >= 70 else "LOW"

        return {
            "label": label,
            "confidence": confidence,
            "confidence_level": conf_level,
            "fake_prob": fake_prob,
            "real_prob": real_prob,
            "gap": abs(fake_prob - real_prob),
            "model_used": "RoBERTa (HF Router)"
        }

    except Exception as e:
        print(f"BERT Inference Error: {e}")
        return None
