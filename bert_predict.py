import requests
import os
import time

# Option 1 — Best for fake news
BERT_MODEL = "hamzab/roberta-fake-news-classification"
BERT_API_URL = f"https://router.huggingface.co/models/{BERT_MODEL}"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def call_bert_api(text: str, max_retries: int = 3) -> dict:
    headers = {
        "Content-Type": "application/json",
        "x-wait-for-model": "true"
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    for attempt in range(max_retries):
        try:
            response = requests.post(
                BERT_API_URL,
                headers=headers,
                json={"inputs": text[:512]},
                timeout=20
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "error" in data:
                    print(f"API returned error in JSON: {data}")
                    time.sleep(3)
                    continue
                return data
            elif response.status_code == 503:
                # Model loading — wait and retry
                print(f"Model loading, retry {attempt+1}/{max_retries}")
                time.sleep(10) # 10 seconds is usually enough for a roberta model
                continue
            else:
                print(f"API error {response.status_code}: {response.text}")
                break

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt+1}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
        except Exception as e:
            print(f"Request error: {e}")
            break

    raise Exception("BERT API failed after all retries")


def parse_bert_response(response_data) -> dict:
    """
    Handles multiple HuggingFace response formats
    """
    try:
        # Unwrap nested list if needed
        if isinstance(response_data, list) and len(response_data) > 0:
            if isinstance(response_data[0], list):
                response_data = response_data[0]
            elif isinstance(response_data[0], dict) and "label" not in response_data[0] and len(response_data) == 1:
                 # sometimes it wraps the dict in a list and inside there's a list
                 pass

        if not isinstance(response_data, list) and isinstance(response_data, dict):
            # sometimes it just returns a single dict like {"label": "FAKE", "score": 0.9}
            response_data = [response_data]

        if not response_data or not isinstance(response_data[0], dict) or "label" not in response_data[0]:
            print(f"Unexpected data shape: {response_data}")
            raise Exception("Unexpected response data shape")

        # Build scores dict
        scores = {}
        for item in response_data:
            label = str(item.get("label", "")).upper()
            score = item.get("score", 0.5)

            # Normalize label names
            if label in ["FAKE", "LABEL_1", "NEGATIVE", "0"]:
                scores["FAKE"] = score
            elif label in ["REAL", "LABEL_0", "POSITIVE", "1"]:
                scores["REAL"] = score

        # If only one label is returned (e.g. just {"label": "FAKE", "score": 0.99})
        if "FAKE" in scores and "REAL" not in scores:
            scores["REAL"] = 1.0 - scores["FAKE"]
        elif "REAL" in scores and "FAKE" not in scores:
            scores["FAKE"] = 1.0 - scores["REAL"]
        elif "FAKE" not in scores and "REAL" not in scores:
             # Default fallback if labels are totally weird
             scores["FAKE"] = 0.5
             scores["REAL"] = 0.5

        fake_prob = round(scores["FAKE"] * 100, 2)
        real_prob = round(scores["REAL"] * 100, 2)

        # Normalize to 100%
        total = fake_prob + real_prob
        if total > 0:
            fake_prob = round((fake_prob / total) * 100, 2)
            real_prob = round((real_prob / total) * 100, 2)

        return {
            "fake_prob" : min(fake_prob, 99.9),
            "real_prob" : min(real_prob, 99.9),
            "label"     : "FAKE" if fake_prob > real_prob else "REAL",
            "confidence": min(max(fake_prob, real_prob), 99.9)
        }

    except Exception as e:
        raise Exception(f"Failed to parse BERT response: {e}")

def bert_predict(text: str) -> dict:
    try:
        raw_response = call_bert_api(text)
        parsed = parse_bert_response(raw_response)
        
        confidence = parsed["confidence"]
        conf_level = "HIGH" if confidence >= 85 else "MEDIUM" if confidence >= 70 else "LOW"
        
        parsed["confidence_level"] = conf_level
        parsed["gap"] = round(abs(parsed["fake_prob"] - parsed["real_prob"]), 2)
        parsed["model_used"] = "BERT (RoBERTa)"
        
        return parsed
    except Exception as e:
        print(f"bert_predict final error: {e}")
        return None
