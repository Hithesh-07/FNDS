from transformers import pipeline
import sys

# Ensure UTF-8 for console
sys.stdout.reconfigure(encoding='utf-8')

# Load model once lazily
_classifier = None

def _load_model():
    global _classifier
    if _classifier is None:
        try:
            print("Loading local BERT model from ./model/bert_model...")
            _classifier = pipeline(
                "text-classification",
                model="./model/bert_model"
            )
            print("Local BERT model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load BERT model: {e}")
            return False
    return True

def bert_predict(text: str) -> dict:
    """
    BERT (RoBERTa) based prediction using transformers pipeline.
    """
    if not _load_model():
        return {
            "label": "UNCERTAIN",
            "confidence": 0.0,
            "fake_prob": 50.0,
            "real_prob": 50.0,
            "confidence_level": "LOW",
            "model_used": "BERT (Load Fail Fallback)"
        }

    # RoBERTa has a 512 token limit
    try:
        result = _classifier(text[:512])[0]
        label_raw = result["label"].lower()
        score = result["score"]

        # Normalize label based on model output:
        # hamzab/roberta-fake-news-classification returns 'Fake' or 'Real'
        # but common labels include 'TRUE', 'REAL', 'LABEL_1' for truth.
        is_real = any(word in label_raw for word in ["real", "true", "label_1", "fact"])
        
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
            "gap": abs(fake_prob - real_prob)
        }
    except Exception as e:
        print(f"BERT Inference Error: {e}")
        return {
            "label": "UNCERTAIN",
            "confidence": 0.0,
            "fake_prob": 50.0,
            "real_prob": 50.0,
            "confidence_level": "LOW"
        }
