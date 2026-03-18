import re
import sys
import nltk
import pandas as pd
from functools import lru_cache

# Ensure Windows console handles all characters
sys.stdout.reconfigure(encoding='utf-8')

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

STOPWORDS   = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

# Words that are strong signals of fake news — NEVER remove these
SENSATIONAL_WORDS = {
    "shocking", "secret", "exposed", "breaking", "urgent", "hoax",
    "conspiracy", "bombshell", "coverup", "leaked", "banned", "censored",
    "hidden", "miracle", "exclusive", "stunning", "unbelievable",
    "incredible", "explosive", "scandal", "corrupt", "rigged"
}

@lru_cache(maxsize=50000)
def lemmatize_word(w: str) -> str:
    return lemmatizer.lemmatize(w)


def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove URLs (http, www, .com links)
    3. Remove HTML tags (<p>, <br> etc)
    4. Remove punctuation and special characters
    5. Remove numbers
    6. Remove extra whitespace
    7. Tokenize
    8. Remove stopwords (but KEEP sensational words)
    9. Remove very short words (len < 3) unless sensational
    10. Lemmatize every token using WordNetLemmatizer
    11. Return cleaned string
    """
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # Remove URLs
    text = re.sub(r"<[^>]+>", "", text)                    # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)               # Remove punctuation & numbers
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text)   # Deduplicate consecutive words
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for w in text.split():
        # Always keep sensational words — they're key signals
        if w in SENSATIONAL_WORDS:
            tokens.append(w)
        elif w not in STOPWORDS and len(w) >= 3:
            tokens.append(lemmatize_word(w))

    return " ".join(tokens)


def extract_features(text: str) -> dict:
    """
    Handcrafted feature extraction for fake news detection.
    These go alongside TF-IDF vectors to catch stylistic patterns.
    """
    if not text:
        return {k: 0 for k in [
            "caps_ratio", "exclamation_count", "question_count",
            "quote_count", "avg_word_length", "sentence_count", "sensational_words"
        ]}

    raw = str(text)
    words = raw.split()
    num_words = max(len(words), 1)

    # Ratio of uppercase letters (fake news shouts)
    alpha_chars = [c for c in raw if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)

    # Punctuation counts
    exclamation_count = raw.count("!")
    question_count    = raw.count("?")
    quote_count       = raw.count('"')

    # Average word length
    avg_word_length = sum(len(w) for w in words) / num_words

    # Sentence count
    try:
        sentence_count = len(sent_tokenize(raw))
    except Exception:
        sentence_count = max(1, raw.count("."))

    # Sensational word count (original casing)
    lower_raw = raw.lower()
    sensational_words = sum(1 for sw in SENSATIONAL_WORDS if sw in lower_raw)

    return {
        "caps_ratio"         : round(caps_ratio, 4),
        "exclamation_count"  : exclamation_count,
        "question_count"     : question_count,
        "quote_count"        : quote_count,
        "avg_word_length"    : round(avg_word_length, 2),
        "sentence_count"     : sentence_count,
        "sensational_words"  : sensational_words,
    }


def load_and_prepare(path: str = "data/news.csv") -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["content", "label"])

    # Remove very short texts (< 10 words)
    before_short = len(df)
    df = df[df["content"].apply(lambda x: len(str(x).split()) >= 10)]
    print(f"   Removed {before_short - len(df)} rows with < 10 words.")

    # Remove duplicates
    before_dup = len(df)
    df.drop_duplicates(subset=["content"], inplace=True)
    print(f"   Removed {before_dup - len(df)} duplicate rows.")

    # Balance classes
    fake_df  = df[df["label"] == 0]
    real_df  = df[df["label"] == 1]
    min_size = min(len(fake_df), len(real_df))

    if min_size == 0:
        print("   WARNING: One class is empty, returning unbalanced data.")
        return df

    df = pd.concat([
        fake_df.sample(min_size, random_state=42),
        real_df.sample(min_size, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"   Balanced: {min_size} FAKE + {min_size} REAL = {len(df)} total")
    return df
