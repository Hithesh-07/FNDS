import requests
import os
import time

# Hugging Face Inference Router URL for the model (NEW API as of 2025)
API_URL = "https://router.huggingface.co/models/hamzab/roberta-fake-news-classification"
HF_TOKEN = os.getenv("HF_TOKEN", "")


def query(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            output = response.json()
            if isinstance(output, dict) and 'error' in output:
                if 'estimated_time' in output:
                    wait_time = min(float(output['estimated_time']), 15)
                    print(f"Model is loading, waiting {wait_time}s... (attempt {attempt+1})")
                    time.sleep(wait_time)
                    continue
                else:
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
    BERT (RoBERTa) based prediction via Hugging Face Inference Router.
    Returns None-safe dict; caller should check if result is None to use fallback.
    """
    try:
        payload = {"inputs": text[:512], "options": {"wait_for_model": True}}
        output = query(payload)

        if output is None:
            return None  # Signal caller to use fallback

        # API returns: [[{"label": "...", "score": ...}, ...]]
        if isinstance(output, list) and len(output) > 0:
            if isinstance(output[0], list):
                result = output[0][0]
            else:
                result = output[0]
            label_raw = result["label"].lower()
            score = result["score"]
        else:
            print(f"Unexpected API response shape: {output}")
            return None

        # Normalize: LABEL_1 / real / true → REAL, else FAKE
        is_real = any(w in label_raw for w in ["real", "true", "label_1", "fact", "joy"])

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
        return None  # Signal caller to use fallback
