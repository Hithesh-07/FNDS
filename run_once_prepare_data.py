import pandas as pd
import os
import sys

# Ensure Windows console can print emojis
sys.stdout.reconfigure(encoding='utf-8')

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake["label"] = 0   # 0 = Fake
real["label"] = 1   # 1 = Real

df = pd.concat([fake, real], ignore_index=True)
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df[["content", "label"]].dropna()

os.makedirs("data", exist_ok=True)
df.to_csv("data/news.csv", index=False)
print(f"✅ Dataset ready! {len(df)} rows saved to data/news.csv")
