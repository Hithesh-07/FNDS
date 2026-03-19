# FNDS

# 🗞️ TruthLens — AI-Powered Fake News Detection System

TruthLens is a full-stack AI application designed to detect and analyze fake news using a hybrid approach that combines machine learning, deep learning, and rule-based reasoning.

It goes beyond basic classification by incorporating contextual understanding (BERT), credibility analysis, and uncertainty handling to provide more reliable predictions in real-world scenarios.

---

## 🚀 Features

* 🔍 **Fake News Detection** — Classifies news as **REAL**, **FAKE**, or **UNCERTAIN**
* 🧠 **Hybrid AI System**

  * SVM (fast baseline model)
  * BERT/RoBERTa (deep contextual understanding)
  * Rule-based credibility engine
* ⚖️ **Uncertainty Detection** — Avoids overconfident wrong predictions
* 📊 **Confidence Scores** — Shows prediction probabilities
* 🧾 **Keyword Insights** — Highlights influential words
* 🌐 **Web Interface** — User-friendly UI built with Flask
* 🔗 **REST API** — Programmatic access via `/predict`

---

## 🧠 How It Works

TruthLens uses a multi-layer decision pipeline:

1. **Credibility Rules Layer**

   * Detects signals like:

     * “no evidence”
     * “unnamed sources”
     * “not published”
   * Overrides model when strong fake patterns are detected

2. **SVM Model (TF-IDF)**

   * Fast baseline prediction
   * Handles simple cases efficiently

3. **BERT / RoBERTa Model**

   * Handles complex, context-heavy inputs
   * Understands tone, structure, and semantics

4. **Uncertainty Engine**

   * Flags ambiguous or low-confidence predictions as **UNCERTAIN**

---

## 🧱 Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** scikit-learn (SVM, TF-IDF)
* **Deep Learning:** Hugging Face Transformers (BERT / RoBERTa)
* **NLP:** NLTK
* **Frontend:** HTML, CSS, Jinja2
* **Deployment:** Gunicorn + Render

---

## 📁 Project Structure

```
truthlens/
│
├── app.py                  # Flask app
├── train.py                # SVM training
├── predict.py              # Prediction logic
├── preprocess.py           # Text cleaning
│
├── model/
│   ├── model.pkl           # SVM model
│   ├── vectorizer.pkl      # TF-IDF
│   └── bert_model/         # BERT model
│
├── templates/
│   └── index.html          # UI
├── static/
│   └── style.css           # Styling
│
├── data/
│   └── news.csv            # Dataset
│
├── requirements.txt
└── Procfile
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/truthlens.git
cd truthlens
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Train the Model (SVM)

```bash
python train.py
```

---

### 4. Run the Application

```bash
python app.py
```

Open in browser:

```
http://localhost:10000
```

---

## 🧪 API Usage

### Endpoint:

```
POST /predict
```

### Example:

```bash
curl -X POST http://localhost:10000/predict \
-H "Content-Type: application/json" \
-d '{"text": "Scientists claim a miracle cure has been discovered"}'
```

### Response:

```json
{
  "label": "FAKE",
  "confidence": 92.3,
  "fake_prob": 92.3,
  "real_prob": 7.7,
  "keywords": ["miracle", "cure", "claim"]
}
```

---

## ⚠️ Limitations

* Does not perform real-time fact verification from live sources
* May struggle with highly nuanced or emerging misinformation
* Accuracy depends on dataset quality and model tuning

---

## 🚀 Future Improvements

* 🌐 Real-time news verification (API integration)
* 🧠 Improved BERT fine-tuning on misinformation datasets
* 📱 Mobile app / Chrome extension
* 🔐 User authentication & history tracking
* 📊 Analytics dashboard

---

## 📌 Use Cases

* Fake news detection for social media
* Educational tool for media literacy
* API for content moderation systems
* News verification assistant

---

## 👨‍💻 Author

**Hithesh**
Aspiring AI Developer | Machine Learning Enthusiast

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub and share your feedback!
