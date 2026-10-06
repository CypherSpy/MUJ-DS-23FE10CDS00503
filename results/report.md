# HinglishLens Evaluation Report

**Evaluation Split:** `test` (5 samples)

## 1. Sentiment Classification Performance

| System | Accuracy | Macro F1 | Positive F1 | Negative F1 | Neutral F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LogisticRegression** | 1.0 | 0.3333 | 1.0 | 0.0 | 0.0 |
| **LLM (zero_shot)** | 1.0 | 0.3333 | 1.0 | 0.0 | 0.0 |
| **LLM (few_shot)** | 1.0 | 0.3333 | 1.0 | 0.0 | 0.0 |

## 2. Token-Level Language Identification (LID)

| System | Token Accuracy | HI F1 | EN F1 | NE F1 | UNIV F1 | OTHER F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM (zero_shot)** | 0.9603 | 0.9735 | 0.9899 | 0.8571 | 1.0 | 0.0 |
| **LLM (few_shot)** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

## 3. Operational & Cost Statistics

| Mode | Calls | Cache Hits | Mean Latency (ms) | P95 Latency (ms) | Input Tokens | Output Tokens | Repaired | Degraded |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **zero_shot** | 5 | 5 | 0.98 | 1.15 | 4162 | 1008 | 0 | 0 |
| **few_shot** | 5 | 2 | 27454.34 | 68335.98 | 10582 | 639 | 0 | 0 |

---
**Note:** If LLM cells show `TBD`, please set your API key environment variable (`export GEMINI_API_KEY=...`) and execute:
```bash
python -m hinglishlens evaluate --modes zero_shot few_shot --split test
```
