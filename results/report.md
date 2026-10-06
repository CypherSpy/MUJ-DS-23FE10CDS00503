# HinglishLens Evaluation Report

**Evaluation Split:** `test` (60 samples)

## 1. Sentiment Classification Performance

| System | Accuracy | Macro F1 | Positive F1 | Negative F1 | Neutral F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LogisticRegression** | 0.85 | 0.85 | 0.8649 | 0.7805 | 0.9048 |
| **LLM (zero_shot)** | TBD | TBD | TBD | TBD | TBD |
| **LLM (few_shot)** | TBD | TBD | TBD | TBD | TBD |

## 2. Token-Level Language Identification (LID)

| System | Token Accuracy | HI F1 | EN F1 | NE F1 | UNIV F1 | OTHER F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM (zero_shot)** | TBD | TBD | TBD | TBD | TBD | TBD |
| **LLM (few_shot)** | TBD | TBD | TBD | TBD | TBD | TBD |

## 3. Operational & Cost Statistics

| Mode | Calls | Cache Hits | Mean Latency (ms) | P95 Latency (ms) | Input Tokens | Output Tokens | Repaired | Degraded |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **zero_shot** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **few_shot** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---
**Note:** If LLM cells show `TBD`, please set your API key environment variable (`export GEMINI_API_KEY=...`) and execute:
```bash
python -m hinglishlens evaluate --modes zero_shot few_shot --split test
```
