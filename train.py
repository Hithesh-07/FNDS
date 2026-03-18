"""
train.py — Elite Ensemble Trainer for TruthLens

Uses 3-model VotingClassifier (SVM + LR + PAC) combined with
handcrafted stylistic features for ~99% accuracy.

Run:
    python train.py
"""
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import scipy.sparse as sp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from preprocess import clean_text, extract_features, load_and_prepare

sys.stdout.reconfigure(encoding='utf-8')


def get_handcrafted(texts):
    """Convert texts into a numeric matrix of stylistic features."""
    rows = [list(extract_features(t).values()) for t in texts]
    return np.array(rows, dtype=float)


def train():
    # ── Load Data ──────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    data_path = "data/news.csv" if os.path.exists("data/news.csv") else "data/news_augmented.csv"
    print(f"   Using: {data_path}")
    df = load_and_prepare(data_path)

    print("\n[2/6] Cleaning text (may take 1-2 min for large datasets)...")
    df["cleaned"] = df["content"].apply(clean_text)

    # Drop rows that cleaned to nothing
    df = df[df["cleaned"].str.strip().str.len() > 0]

    X_text = df["cleaned"].values
    X_raw  = df["content"].values   # raw text for handcrafted features
    y      = df["label"].values

    # ── Train/Test Split ───────────────────────────────────────
    X_text_train, X_text_test, X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_text, X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_text_train)} | Test: {len(X_text_test)}")

    # ── TF-IDF Vectorization ───────────────────────────────────
    print("\n[3/6] TF-IDF Vectorizing (max_features=50000, ngram=(1,3))...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        analyzer="word"
    )
    tfidf_train = vectorizer.fit_transform(X_text_train)
    tfidf_test  = vectorizer.transform(X_text_test)

    # ── Handcrafted Features ───────────────────────────────────
    print("[4/6] Extracting handcrafted stylistic features...")
    hand_train = get_handcrafted(X_raw_train)
    hand_test  = get_handcrafted(X_raw_test)

    scaler = StandardScaler()
    hand_train_scaled = scaler.fit_transform(hand_train)
    hand_test_scaled  = scaler.transform(hand_test)

    # Combine TF-IDF + handcrafted
    X_train_final = sp.hstack([tfidf_train, sp.csr_matrix(hand_train_scaled)])
    X_test_final  = sp.hstack([tfidf_test,  sp.csr_matrix(hand_test_scaled)])

    # ── Ensemble Model ─────────────────────────────────────────
    print("\n[5/6] Training Ensemble (SVM + LogisticRegression + PassiveAggressive)...")
    svm = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, class_weight="balanced"), cv=5)
    lr  = LogisticRegression(max_iter=1000, C=1.5, solver="lbfgs", class_weight="balanced", n_jobs=-1)
    pac = CalibratedClassifierCV(PassiveAggressiveClassifier(max_iter=1000, class_weight="balanced"), cv=5)

    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("lr", lr), ("pac", pac)],
        voting="soft"
    )
    ensemble.fit(X_train_final, y_train)

    # ── Evaluate ───────────────────────────────────────────────
    print("\n[6/6] Evaluating...")
    y_pred = ensemble.predict(X_test_final)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 55)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  F1 Score  : {f1  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print("=" * 55)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"               Predicted FAKE   Predicted REAL")
    print(f"  Actual FAKE      {cm[0][0]:<12}   {cm[0][1]}")
    print(f"  Actual REAL      {cm[1][0]:<12}   {cm[1][1]}")

    # ── Misclassified Samples ─────────────────────────────────
    print("\nMisclassified Samples (up to 5):")
    misses = y_test != y_pred
    miss_texts  = X_text_test[misses]
    miss_actual = y_test[misses]
    miss_pred   = y_pred[misses]
    if len(miss_texts) == 0:
        print("   No misclassifications! Perfect on test set.")
    else:
        for i in range(min(5, len(miss_texts))):
            al = "REAL" if miss_actual[i] == 1 else "FAKE"
            pl = "REAL" if miss_pred[i]   == 1 else "FAKE"
            print(f"   [FAIL] Actual:{al} | Predicted:{pl}")
            print(f"          {miss_texts[i][:120]}...\n")

    # ── Top Words ─────────────────────────────────────────────
    try:
        feature_names = vectorizer.get_feature_names_out()
        # LR has direct coef — most interpretable
        lr_model = ensemble.named_estimators_["lr"]
        coefs = lr_model.coef_[0][:len(feature_names)]
        top_fake_idx = np.argsort(coefs)[:20]
        top_real_idx = np.argsort(coefs)[::-1][:20]
        print("\nTop 20 FAKE-leaning words:")
        print("  " + ", ".join(feature_names[i] for i in top_fake_idx))
        print("\nTop 20 REAL-leaning words:")
        print("  " + ", ".join(feature_names[i] for i in top_real_idx))
    except Exception:
        pass

    # ── Save Artefacts ────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    joblib.dump(ensemble,   "model/model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    joblib.dump(scaler,     "model/scaler.pkl")
    print("\nSaved: model/model.pkl, model/vectorizer.pkl, model/scaler.pkl")

    # ── Metrics Log ───────────────────────────────────────────
    log_entry = {
        "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type"   : "Ensemble (SVM + LR + PAC)",
        "dataset_path" : data_path,
        "dataset_size" : len(df),
        "accuracy"     : round(acc  * 100, 2),
        "f1_score"     : round(f1   * 100, 2),
        "precision"    : round(prec * 100, 2),
        "recall"       : round(rec  * 100, 2),
    }
    log_path  = "metrics_log.json"
    history   = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            history = json.load(f)
    history.append(log_entry)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Metrics saved to {log_path}\n")


if __name__ == "__main__":
    train()
