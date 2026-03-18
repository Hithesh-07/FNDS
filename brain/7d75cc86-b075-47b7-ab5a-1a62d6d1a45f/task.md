# Task Checklist

## Completed
- [x] Full ensemble SVM pipeline (20/20 test score)
- [x] Handcrafted features + confidence tiers + red flags
- [x] Data augmentation (17,946 rows)

## Deployment Preparation
- [x] Create `Procfile` (gunicorn)
- [x] Update `app.py` port logic (10000)
- [x] Refine `requirements.txt`

**Status**: READY FOR DEPLOY. 🏛️🌎

## Hybrid Intelligence (v2.1)
- [x] Pre-trained RoBERTa integration (`bert_predict.py`)
- [x] Strict Multi-Layer Decision Flow (`app.py`)
- [x] Rules First (Fake Score >= 3)
- [x] Uncertainty Override
- [x] SVM High-Confidence Threshold (>80)
- [x] BERT Fallback for complex cases

**Status**: Deploying & Monitoring... ⏳
- [x] Create `bert_train.py` — RoBERTa fine-tuning on news.csv
- [x] Create `bert_predict.py` — BERT inference with confidence tiers
- [x] Create `bert_test.py` — 30-case automated test suite
- [x] Update `app.py` — BERT primary + SVM fallback
- [x] Update `templates/index.html` — model badge, verdict states, disclaimer

## UNCERTAIN Logic & UI
- [x] Update `predict.py` with uncertainty logic, hedge words, and fake triggers
- [x] Restore stylistic feature extraction in `predict.py` to fix feature mismatch
- [x] Update `bert_predict.py` with identical uncertainty logic
- [x] Update `static/style.css` with `verdict-uncertain` and alert styles
- [x] Update `templates/index.html` with hedge word flags and warning banners
- [x] Update `test_model.py` and `bert_test.py` with 10 new UNCERTAIN test cases

## Model Calibration & Balancing
- [x] Calibrate UNCERTAIN thresholds and rules
- [x] Optimize SVM/BERT handoff (SVM > 80 else BERT)

## Priority Decision Engine Overhaul
- [x] Shift to "Rules First" architecture
- [x] 100% Calibration Verified

## Aletheia: Rebranding & UI Color Overhaul
- [x] Global rebranding ("TruthLens" -> "aletheia")
- [x] Color Overhaul (Light: #F8FAFC | Dark: #0B0F19)
- [x] Premium "aletheia" UX Refinements (small-caps, "Run" button)

**Status**: MISSION ACCOMPLISHED. 🏛️⚖️
