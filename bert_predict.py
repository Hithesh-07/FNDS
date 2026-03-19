import requests
import json
import os
import time

# Hugging Face Inference API URL for the model
API_URL = "https://api-inference.huggingface.co/models/hamzab/roberta-fake-news-classification"
# Recommend adding your HF token to Vercel/Local environment variables
HF_TOKEN = os.getenv("HF_TOKEN", "") 

def query(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def bert_predict(text: str) -> dict:
    """
    BERT (RoBERTa) based prediction using Hugging Face Inference API.
    Used for serverless deployments (Vercel) to keep the package small.
    """
    try:
        # RoBERTa has a 512 token limit
        payload = {"inputs": text[:512], "options": {"wait_for_model": True}}
        output = query(payload)
        
        # API returns a list of lists of dicts: [[{"label": "...", "score": ...}, ...]]
        if isinstance(output, list) and len(output) > 0:
            result = output[0][0]
            label_raw = result["label"].lower()
            score = result["score"]
        else:
            raise ValueError(f"Unexpected API response: {output}")

        # Normalize label based on model output:
        # 'LABEL_1' or 'Real' for Truth, 'LABEL_0' or 'Fake' for False
        is_real = any(word in label_raw for word in ["real", "true", "label_1", "fact", "joy"])
        
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
            "model_used": "RoBERTa (HF Inference API)"
        }
    except Exception as e:
        print(f"BERT Inference Error: {e}")
        return {
            "label": "UNCERTAIN",
            "confidence": 0.0,
            "fake_prob": 50.0,
            "real_prob": 50.0,
            "confidence_level": "LOW",
            "model_used": "BERT (API Fallback)"
        }
