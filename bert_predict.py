from transformers import pipeline

# Load model once (very important)
print("Loading BERT model...")
classifier = pipeline(
    "text-classification",
    model="hamzab/roberta-fake-news-classification"
)

def predict_bert(text: str) -> dict:
    # RoBERTa has a 512 token limit
    result = classifier(text[:512])[0]

    label_raw = result["label"].lower()
    score = result["score"]

    # Normalize label
    if "real" in label_raw:
        label = "REAL"
        real_prob = score * 100
        fake_prob = (1 - score) * 100
    else:
        label = "FAKE"
        fake_prob = score * 100
        real_prob = (1 - score) * 100

    confidence = max(fake_prob, real_prob)

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "fake_prob": round(fake_prob, 2),
        "real_prob": round(real_prob, 2),
        "keywords": []
    }
