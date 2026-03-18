"""
prepare_data.py — Multi-dataset merger for TruthLens

Loads all available datasets, standardises to (content, label),
merges, deduplicates, balances, and saves to data/news.csv.

Datasets supported:
  1. Kaggle Fake/True (Fake.csv + True.csv) — always loaded if present
  2. LIAR dataset     (train.tsv)           — loaded if present
  3. Any extra CSV with 'content'/'text' and 'label' columns

Run:
    python prepare_data.py
"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# ── Helper ─────────────────────────────────────────────────────
LIAR_LABEL_MAP = {
    "true":         1,
    "mostly-true":  1,
    "half-true":    1,
    "false":        0,
    "barely-true":  0,
    "pants-fire":   0,
}


def load_kaggle(fake_path="Fake.csv", true_path="True.csv"):
    """Load the standard Kaggle Fake/True dataset pair."""
    dfs = []
    for path, lbl in [(fake_path, 0), (true_path, 1)]:
        if not os.path.exists(path):
            print(f"   [SKIP] {path} not found.")
            continue
        df = pd.read_csv(path)
        df["content"] = df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")
        df["label"] = lbl
        dfs.append(df[["content", "label"]])
        print(f"   [OK]   Loaded {path}: {len(df)} rows")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_liar(path="liar_dataset/train.tsv"):
    """Load LIAR dataset (TSV) and map 6-class labels → binary."""
    if not os.path.exists(path):
        print(f"   [SKIP] LIAR dataset not found at {path}.")
        return pd.DataFrame()
    # LIAR columns: id, label, statement, subject, speaker, job, state, party,
    #               barely_true_counts, false_counts, half_true_counts,
    #               mostly_true_counts, pants_on_fire_counts, context
    cols = [
        "id", "label", "statement", "subject", "speaker", "job",
        "state", "party", "barely_true_counts", "false_counts",
        "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
    ]
    df = pd.read_csv(path, sep="\t", names=cols)
    df["label"] = df["label"].map(LIAR_LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["content"] = df["statement"].fillna("")
    result = df[["content", "label"]]
    print(f"   [OK]   Loaded LIAR {path}: {len(result)} rows")
    return result


def load_generic_csv(path: str):
    """Load any CSV that has a text/content column and a label (0/1) column."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]
    content_col = next((c for c in ["content", "text", "news", "body"] if c in df.columns), None)
    label_col   = next((c for c in ["label", "class", "fake"] if c in df.columns), None)
    if not content_col or not label_col:
        print(f"   [SKIP] {path}: can't identify content/label columns.")
        return pd.DataFrame()
    df = df.rename(columns={content_col: "content", label_col: "label"})
    df = df[["content", "label"]].dropna()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").dropna().astype(int)
    print(f"   [OK]   Loaded {path}: {len(df)} rows")
    return df


def merge_and_save(output_path: str = "data/news.csv"):
    print("\n[1/4] Loading all dataset sources...")
    frames = []

    # Dataset 1 — Kaggle
    frames.append(load_kaggle("Fake.csv", "True.csv"))

    # Dataset 2 — LIAR
    frames.append(load_liar("liar_dataset/train.tsv"))

    # Dataset 3 — Augmented synthetic data (if already generated)
    frames.append(load_generic_csv("data/news_augmented.csv"))

    # Filter empty frames
    frames = [f for f in frames if not f.empty]
    if not frames:
        print("[ERROR] No data found. Place Fake.csv + True.csv in project root.")
        return

    print("\n[2/4] Merging all sources...")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["content", "label"])
    df["content"] = df["content"].astype(str).str.strip()

    print(f"\n[3/4] Cleaning...")
    # Remove very short texts
    before = len(df)
    df = df[df["content"].apply(lambda x: len(x.split()) >= 10)]
    print(f"   Removed {before - len(df)} short texts (< 10 words).")

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(subset=["content"], inplace=True)
    print(f"   Removed {before - len(df)} duplicate rows.")

    # Balance classes
    fake_df  = df[df["label"] == 0]
    real_df  = df[df["label"] == 1]
    min_size = min(len(fake_df), len(real_df))

    df = pd.concat([
        fake_df.sample(min_size, random_state=42),
        real_df.sample(min_size, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[4/4] Saving to {output_path}...")
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print(f"  TOTAL ROWS    : {len(df)}")
    print(f"  FAKE (label=0): {(df['label'] == 0).sum()}")
    print(f"  REAL (label=1): {(df['label'] == 1).sum()}")
    print("=" * 50)
    print(f"[DONE] Saved to {output_path}\n")


if __name__ == "__main__":
    merge_and_save()
