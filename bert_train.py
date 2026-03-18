"""
bert_train.py — Fine-tunes roberta-base on our fake news dataset.
Saves model to model/bert_model/

Run:
    python bert_train.py

On CPU: ~30-60 minutes | On GPU: ~5-10 minutes
Downloads roberta-base (~500MB) on first run.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import os
import json
from datetime import datetime

# ── Config ─────────────────────────────────────────────────
MODEL_NAME   = "roberta-base"
MAX_LEN      = 256        # max tokens per article
BATCH_SIZE   = 8          # keep low for CPU
EPOCHS       = 3
LR           = 2e-5
SAVE_PATH    = "model/bert_model"
DATA_PATH    = "data/news.csv"

# ── Device ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Dataset Class ──────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length      = self.max_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt"
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(),
            "attention_mask" : encoding["attention_mask"].squeeze(),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── Load Data ──────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH).dropna(subset=["content", "label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use 20000 samples max for speed on CPU
if device.type == "cpu":
    df = df.sample(min(20000, len(df)), random_state=42)
    print(f"CPU mode: using {len(df)} samples for speed")
else:
    print(f"GPU mode: using full {len(df)} samples")

X = df["content"].tolist()
y = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Tokenizer ──────────────────────────────────────────────
print("Loading RoBERTa tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

train_dataset = NewsDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset  = NewsDataset(X_test,  y_test,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ──────────────────────────────────────────────────
print("Loading RoBERTa model...")
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = 2
)
model.to(device)

# ── Optimizer & Scheduler ──────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = total_steps // 10,
    num_training_steps = total_steps
)

# ── Training Loop ──────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
best_accuracy = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = outputs.logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        loop.set_postfix(
            loss = f"{total_loss/total:.4f}",
            acc  = f"{correct/total*100:.2f}%"
        )

    # ── Evaluate after each epoch ──────────────────────────
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask
            )

            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")

    print(f"\nEpoch {epoch+1} Results:")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=["FAKE", "REAL"]))

    # Save best model
    if acc > best_accuracy:
        best_accuracy = acc
        os.makedirs(SAVE_PATH, exist_ok=True)
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"  Best model saved! Accuracy: {acc*100:.2f}%")

# ── Save Metrics ───────────────────────────────────────────
log = {
    "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type"   : "RoBERTa (roberta-base fine-tuned)",
    "dataset_size" : len(df),
    "epochs"       : EPOCHS,
    "best_accuracy": round(best_accuracy * 100, 2),
    "device"       : str(device)
}

history = []
if os.path.exists("metrics_log.json"):
    with open("metrics_log.json") as f:
        history = json.load(f)
history.append(log)
with open("metrics_log.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nTraining complete! Best accuracy: {best_accuracy*100:.2f}%")
print(f"Model saved to {SAVE_PATH}")
