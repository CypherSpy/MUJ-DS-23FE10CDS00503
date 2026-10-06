# HinglishLens: Code-Mixed (Hinglish) Text Normalizer & Analyzer

Instructions to the coding agent (Antigravity): Read this whole file before writing any code. Implement the project end to end exactly as specified, run every verification command in section 12, commit after each phase, and finish with the final report described in section 13. This is a graded university submission, so correctness, honesty and code quality matter more than extra features. Do not add features that are not listed here until the Definition of Done (section 11) is fully met.

1. Mission

Build a solo NLP course project that uses an LLM through API calls to analyze Hinglish: Hindi written in Roman script and mixed with English (code-mixed text), as used on social media, chats and reviews. Example: yaar ye movie bakwaas thi but acting mast thi.

For each input text the system returns, in one validated structured result:

Token-level language identification (HI / EN / NE / UNIV / OTHER)
Text normalization (spelling variants and SMS shorthand mapped to a canonical Roman form)
Sentiment (positive / negative / neutral) with confidence
Emotion and sarcasm flag
English translation
Code-mixing metrics computed in code (Code-Mixing Index, language distribution, switch points)

A Streamlit web app, a CLI, and an evaluation harness (LLM zero-shot vs LLM few-shot vs classical TF-IDF baseline) sit on top.
