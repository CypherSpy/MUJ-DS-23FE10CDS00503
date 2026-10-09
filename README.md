HinglishLens
Code-Mixed (Hinglish) Text Normalizer & Analyzer powered by LLM APIs
HinglishLens analyzes Hinglish, meaning Hindi written in Roman script and mixed with English, such as "yaar ye movie bakwaas thi but acting mast thi". A single LLM API call, wrapped in deterministic code for tokenization, validation and metrics, produces token-level language identification, text normalization, sentiment, emotion, sarcasm detection, English translation, and code-mixing metrics (CMI). It ships with a Streamlit app, a CLI, and an evaluation harness that compares zero-shot prompting, few-shot prompting and a classical TF-IDF baseline.

<img width="938" height="434" alt="Screenshot 2026-10-09 145225" src="https://github.com/user-attachments/assets/100ed213-3d36-4461-b5e1-2e7907109cb1" />
<img width="648" height="431" alt="Screenshot 2026-10-09 145241" src="https://github.com/user-attachments/assets/bbace130-88da-4707-90ee-5d995e89eb28" />
At a glance
	
Task	
Code-mixed NLP: language identification, normalization, sentiment, emotion, sarcasm, translation
LLM access	API calls through one provider-agnostic client (Google Gemini, OpenAI, Anthropic)
Prompt file	prompts.yaml: versioned system prompt, rules, output schema, 5 few-shot examples
Configuration file	config.yaml: model, decoding, retries, cache, paths, prompt mode
Interfaces	Streamlit web app, command-line interface, batch CSV processing
Evaluation	TF-IDF + Logistic Regression baseline vs LLM zero-shot vs LLM few-shot, scored with accuracy and macro-F1
Reliability	Pydantic schema validation, one self-repair retry, exponential backoff with jitter, SQLite response cache, graceful degradation
Table of contents
For evaluators: where to look
Demo output
Why Hinglish is hard
Features
Architecture and design decisions
Project structure
Quick start
Configuration reference
Prompt engineering
Evaluation
Rubric mapping
Testing
Deployment on Replit
Troubleshooting
Limitations and ethics
Future work
1. For evaluators: where to look
Required item	Where it is
Project source code	hinglishlens/ package, app.py
Prompt file	prompts.yaml
Configuration file	config.yaml
LLM API implementation	hinglishlens/llm_client.py (providers, retries), hinglishlens/analyzer.py (validation, repair, post-processing)
Everything needed to run	requirements.txt, .env.example, data/, .replit
Evaluation results	results/ and Section 10
Grading criteria, point by point	Section 11

Fastest way to see it work: follow the three commands in Section 7, then run python -m hinglishlens analyze "your sentence".

2. Demo output

[!NOTE] The block below illustrates the format of a result. It is built from the annotated few-shot example fs4 in prompts.yaml. Real model outputs can differ slightly, so replace it with your own run's output if you like.

Input

wah bhai wah, 2 ghante wait karwaya aur bola table nahi hai 👏

Token-level language tags (HI Hindi, EN English, UNIV language-independent)

wah/HI  bhai/HI  wah/HI  ,/UNIV  2/UNIV  ghante/HI  wait/EN  karwaya/HI  aur/HI  bola/HI  table/EN  nahi/HI  hai/HI  👏/UNIV
Field	Value
Sentiment	negative (confidence 0.90)
Sarcasm	yes: praise ("wah bhai wah", clapping) contradicts a long wait and no table
Emotion	anger
English translation	"Wow, bro, wow, made me wait 2 hours and then said there's no table 👏"
Code-Mixing Index (CMI)	18.18
Language switch points	4
Tag distribution	HI 9 · EN 2 · NE 0 · UNIV 3 · OTHER 0

Normalization examples

Original	Normalized
kyaa baat hai! mujhe toh bohot pasand aaya yeh phone	kya baat hai! mujhe toh bahut pasand aaya yeh phone
kal meeting subah 10 bje h, agenda mail kr diya	kal meeting subah 10 baje hai, agenda mail kar diya
3. Why Hinglish is hard

Hinglish is the everyday language of Indian social media, chats and product reviews, and it breaks pipelines built for monolingual text.

Challenge	Example	Why it is difficult
Transliteration and informal spelling	kya, kyaa, kiya; bahut, bohot, boht	Hindi has no single standard Roman spelling, so the same word appears in many forms
Code-mixing and code-switching	delivery 3 din late aayi	Hindi function words and English content words alternate inside one sentence
SMS shorthand	plz, tmrw, h, kr, bje	Abbreviations are ambiguous without context
Context-dependent language of a word	me = Hindi mein ("in") or English "myself"; to = Hindi toh	Spelling alone cannot decide the language
Sarcasm	wah bhai wah, 2 ghante wait karwaya	Literal praise carries negative intent

Models trained on clean English or Devanagari Hindi degrade on such text. HinglishLens combines structured LLM prompting with deterministic linguistic rules to handle it.

4. Features
Capability	What it does
Token-level language identification	Tags every token as HI, EN, NE (named entity), UNIV (punctuation, digits, emoji, @mentions, #hashtags) or OTHER
Text normalization	Maps spelling variants and SMS shorthand to one canonical Roman form while preserving the original spacing
Sentiment analysis	positive, negative or neutral, with a confidence score from 0.0 to 1.0
Emotion detection	joy, anger, sadness, fear, surprise, disgust or neutral
Sarcasm detection	Sentiment follows the intended meaning, not the literal words
Machine translation	Natural English that keeps tone and named entities
Code-mixing metrics	Code-Mixing Index (CMI), switch points and tag distribution, computed in code
Multi-provider LLM support	Google Gemini (google-genai), OpenAI (openai), Anthropic (anthropic), selected in config.yaml
Resilient pipeline	Schema validation, self-repair retry, exponential backoff with jitter, SQLite caching, thread-pool batch inference
Benchmark harness	Baseline vs zero-shot vs few-shot, with confusion matrices and operational statistics
5. Architecture and design decisions
5.1 Data flow
Yes
No
Valid
Invalid
Raw Hinglish text
tokenizer.pyregex tokens + characterspans
prompt_loader.pysystem prompt + few-shot+ numbered tokens
cache.pySQLite hit?
JSON extraction +Pydantic validation
llm_client.pyAPI call + retry + backoff
Post-processing in codeforce UNIV, rebuildnormalized text,CMI, switch points
One repair retry,then degrade
AnalysisResult
Streamlit app / CLI /evaluation
5.2 What happens to one request
Tokenize the text with a Unicode-aware regex and keep each token's character span.
Build the prompt from prompts.yaml: system prompt, optional few-shot examples, and the numbered token list (0=Aaj 1=ka 2=match ...).
Check the cache. A hit skips the API call entirely.
Call the LLM through the provider-agnostic client, with retries on transient errors only.
Extract and validate the JSON against a Pydantic schema, including a check that every token index appears exactly once.
Repair or degrade. If validation fails, re-ask once with the exact error. If it still fails, fall back to heuristic tags and flag the result as degraded instead of crashing.
Post-process in code: force language-independent tokens to UNIV, rebuild the normalized sentence from spans, and compute CMI, switch points and tag counts.
5.3 Design decisions
Decision	Why	Benefit
Code tokenizes, the LLM only labels	The model receives numbered tokens and returns one tag per index	No hallucinated or merged tokens, exact alignment, token-level accuracy is measurable
Compact output schema	Short keys (i, l, n); n is omitted when the normalized form equals the original	Fewer output tokens, lower latency and cost
Normalized sentence rebuilt from spans	The LLM never re-emits the full sentence	Original spacing and punctuation are preserved exactly
Deterministic overrides	Tokens without letters, @mentions and #hashtags are forced to UNIV in code	Zero tagging errors on language-independent tokens
Metrics computed in code	CMI, switch points and distribution are calculated in Python	Exact, reproducible, free of LLM estimation error
Prompt and settings kept out of the code	Prompt text only in prompts.yaml, tunables only in config.yaml	Easy to audit, version and tune without touching Python
Cache key includes prompt version, mode and model	Any change invalidates stale entries automatically	Repeated runs are free and reproducible
5.4 Code-Mixing Index

Treating HI and EN as the two matrix languages, and UNIV, NE and OTHER as language-independent tokens (count u), with n total tokens:

CMI = 0                                              if n == u
CMI = 100 × (1 − max(count_HI, count_EN) / (n − u))  otherwise

A switch point is a change between HI and EN in the token sequence, ignoring language-independent tokens in between.

6. Project structure
hinglishlens/
├── README.md
├── config.yaml              ← configuration file
├── prompts.yaml             ← prompt file
├── app.py                   ← Streamlit web app
├── requirements.txt
├── .env.example             ← API key template (no real keys)
├── .replit, .streamlit/     ← deployment configuration
├── hinglishlens/
│   ├── config.py            ← load and validate config.yaml
│   ├── tokenizer.py         ← regex tokenizer, span-preserving rebuild
│   ├── schemas.py           ← Pydantic models
│   ├── prompt_loader.py     ← builds messages from prompts.yaml
│   ├── llm_client.py        ← Gemini / OpenAI / Anthropic clients, retry and backoff
│   ├── cache.py             ← SQLite response cache
│   ├── analyzer.py          ← core pipeline (single and batch)
│   ├── cmi.py               ← CMI, switch points, distribution
│   ├── baseline.py          ← TF-IDF + Logistic Regression
│   ├── evaluate.py          ← evaluation harness and report writer
│   └── cli.py               ← command-line interface
├── data/                    ← train/test sentiment sets, LID gold set, demo samples
├── results/                 ← committed evaluation outputs
└── tests/                   ← offline pytest suite
7. Quick start

Prerequisites: Python 3.11 or newer, and an API key for at least one provider.

bash
# 1. Get the code
git clone https://github.com/your-username/hinglishlens.git
cd hinglishlens

# 2. Create an environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env               # then edit .env
Provider	Environment variable	Set in config.yaml
Google Gemini (default)	GEMINI_API_KEY	llm.provider: gemini
OpenAI	OPENAI_API_KEY	llm.provider: openai
Anthropic	ANTHROPIC_API_KEY	llm.provider: anthropic

You can also switch provider without editing the file: LLM_PROVIDER=openai python -m hinglishlens analyze "...".

Command-line interface
bash
# Analyze one sentence
python -m hinglishlens analyze "yaar ye movie bakwaas thi but acting mast thi"

# Full structured result as JSON
python -m hinglishlens analyze "Aaj ka match toh ekdum zabardast tha" --json

# Batch inference over a file (one sentence per line)
python -m hinglishlens batch --input data/sample_inputs.txt --output results/batch.csv

# Run the evaluation harness
python -m hinglishlens evaluate --modes zero_shot few_shot --split test
Web app
bash
streamlit run app.py

The app has three tabs: Analyze (colored token chips, sentiment, CMI chart), Batch (CSV upload and download) and Evaluation (saved results and confusion matrices).

8. Configuration reference

Every tunable lives in config.yaml. Nothing is hard-coded in Python.

Section	Key settings	Purpose
llm	provider, models, temperature (0.0), max_output_tokens, timeout_s, json_mode	Which model to call and how it decodes
llm.retries	max_attempts (4), base_delay_s, max_delay_s, jitter	Exponential backoff for rate limits, timeouts and server errors
llm	max_repair_attempts (1)	Re-ask once with the validation error if the JSON is invalid
prompt	file, mode (zero_shot or few_shot)	Which prompt file and prompting strategy to use
pipeline	max_workers (4), max_input_chars (600)	Batch concurrency and input length cap
cache	enabled, path	SQLite response cache
data	train, test, lid_gold, samples	Dataset locations
evaluation	output_dir, seed (42)	Where reports go and the random seed
logging	level, file	Log verbosity and destination
<details> <summary><b>Show the full <code>config.yaml</code></b></summary>
yaml
app:
  name: HinglishLens
  version: "1.0.0"

llm:
  provider: gemini            # gemini | openai | anthropic
  models:
    gemini: gemini-2.5-flash
    openai: gpt-4o-mini
    anthropic: claude-haiku-4-5-20251001
  temperature: 0.0
  max_output_tokens: 1500
  timeout_s: 30
  json_mode: true
  retries:
    max_attempts: 4
    base_delay_s: 1.0
    max_delay_s: 16.0
    jitter: true
  max_repair_attempts: 1

prompt:
  file: prompts.yaml
  mode: few_shot              # zero_shot | few_shot

pipeline:
  max_workers: 4
  max_input_chars: 600

cache:
  enabled: true
  path: .cache/llm_cache.sqlite

data:
  train: data/sentiment_train.csv
  test: data/sentiment_test.csv
  lid_gold: data/lid_gold.jsonl
  samples: data/sample_inputs.txt

evaluation:
  output_dir: results
  seed: 42

logging:
  level: INFO
  file: logs/hinglishlens.log
</details>
9. Prompt engineering

prompts.yaml holds the system prompt, rules, output schema, user template and five few-shot examples, under an explicit version that is part of the cache key.

9.1 Techniques used
Technique	Where in the prompt	Effect
Role prompting	Opening of the system prompt	Frames the model as a computational linguist for Hinglish
Explicit label definitions	Tag section for HI, EN, NE, UNIV, OTHER	Removes ambiguity about what each tag means
Context-over-spelling rule	"Tag by the language the word functions as in this sentence"	Resolves cases like to / toh and me / mein
Normalization rules with examples	kyaa→kya, bohot→bahut, h→hai, bje→baje, plz→please	Consistent canonical spellings; never translates or changes meaning
Sarcasm rule	Sentiment section	Sentiment follows intended meaning; sets sarcasm: true
Tie-break rules	Mixed polarity or purely factual text → neutral	Stable behavior on ambiguous inputs
Structured output contract	JSON format with one entry per token index	Machine-validated output, no free text
Token-saving schema	Short keys, optional n	Shorter outputs, faster and cheaper calls
Few-shot demonstrations	Five annotated examples	Anchors tag boundaries, normalization and sarcasm handling
9.2 Few-shot examples
#	Input	Demonstrates
1	Aaj ka match toh ekdum zabardast tha, Kohli ne kamaal kar diya 🔥	Positive sentiment, named entity (NE), emoji (UNIV)
2	bhai delivery 3 din late aayi aur packing bhi kharab thi, worst service	Negative complaint, English nouns inside Hindi syntax, digits
3	kyaa baat hai! mujhe toh bohot pasand aaya yeh phone	Spelling normalization (kyaa→kya, bohot→bahut)
4	wah bhai wah, 2 ghante wait karwaya aur bola table nahi hai 👏	Sarcasm: praise used to complain
5	kal meeting subah 10 bje h, agenda mail kr diya	SMS shorthand and a neutral, informational message
9.3 Zero-shot vs few-shot

The same system prompt is used in both modes. zero_shot sends no examples, and few_shot prepends the five examples as user/assistant turns. The mode is a single line in config.yaml (prompt.mode), which makes the ablation in Section 10 a controlled comparison.

10. Evaluation
10.1 Protocol
Item	Detail
Sentiment test set	data/sentiment_test.csv: 60 texts, 20 per class
Language-ID gold set	data/lid_gold.jsonl: 25 sentences with token-aligned gold tags
Systems compared	TF-IDF + Logistic Regression baseline, LLM zero-shot, LLM few-shot
Metrics	Accuracy, macro-F1, per-class F1, confusion matrix, token accuracy for language ID
Leakage control	The prompt was tuned only on the training split. Few-shot examples are checked by a unit test not to appear in the test set
Reproducibility	Fixed seed (42), temperature 0.0, cached responses

Baseline: character n-gram TF-IDF (2 to 5, char_wb, sublinear term frequency) feeding a class-balanced Logistic Regression.

10.2 Sentiment classification
System	Accuracy	Macro-F1	Positive F1	Negative F1	Neutral F1
TF-IDF + Logistic Regression	0.8500	0.8500	0.8649	0.7805	0.9048
LLM, zero-shot	TBD	TBD	TBD	TBD	TBD
LLM, few-shot	TBD	TBD	TBD	TBD	TBD

[!WARNING] The LLM rows are pending. Run python -m hinglishlens evaluate --modes zero_shot few_shot with an API key, then copy the numbers from results/report.md. Do not submit with TBD cells.

10.3 Token-level language identification
System	Token accuracy	HI F1	EN F1	NE F1	UNIV F1	OTHER F1
LLM, zero-shot	TBD	TBD	TBD	TBD	TBD	TBD
LLM, few-shot	TBD	TBD	TBD	TBD	TBD	TBD
10.4 Operational statistics
Mode	API calls	Cache hits	Mean latency	p95 latency	Input tokens	Output tokens	Repaired	Degraded
zero-shot	TBD	TBD	TBD	TBD	TBD	TBD	TBD	TBD
few-shot	TBD	TBD	TBD	TBD	TBD	TBD	TBD	TBD
10.5 Error analysis

Add three to five real misclassified examples from results/predictions_*.csv after the evaluation run.

Input	Gold	Predicted	Likely reason
TBD	TBD	TBD	TBD
10.6 Artifacts

Each evaluation run writes results/metrics.json, results/report.md, results/predictions_<system>.csv and results/confusion_<system>.png.

[!NOTE] The datasets are small and self-constructed, so treat the numbers as indicative rather than as a benchmark. A strong baseline score on a small, templated set is expected, and the more informative comparison is where each system fails (sarcasm, unseen spellings, heavy English mixing).

11. Rubric mapping
Evaluation criterion	How the project addresses it	Files
Quality and efficiency of the code	Modular package, type hints, Pydantic models, exponential backoff with jitter, SQLite response caching, thread-pool batch inference, compact output schema, offline pytest suite, ruff-clean	hinglishlens/, tests/
Functionality and implementation of the LLM API	One abstract client over Gemini, OpenAI and Anthropic, native JSON mode where available, robust JSON extraction, schema validation, self-repair retry, graceful degraded mode	llm_client.py, analyzer.py, schemas.py
Effectiveness and efficiency of the prompt file	Versioned prompts.yaml, pre-tokenization, explicit tag and normalization rules, sarcasm and tie-break rules, five few-shot examples, measured zero-shot vs few-shot ablation	prompts.yaml, prompt_loader.py, evaluate.py
Overall project quality and execution	Baseline comparison, reproducible evaluation CLI, committed results, interactive app, deployment configuration, documentation	baseline.py, evaluate.py, results/, app.py, .replit
12. Testing

The suite runs offline with a fake LLM client, so it needs no API key and finishes in seconds.

bash
python -m pytest -q        # unit tests
python -m ruff check .     # lint
Test module	Verifies
test_tokenizer.py	Spans reconstruct the text; emoji, hashtags, mentions, apostrophes and Devanagari; spacing preserved on rebuild
test_cmi.py	CMI and switch-point edge cases
test_schemas.py	Rejects missing or duplicate token indices, invalid labels, confidence outside 0 to 1
test_prompt_loader.py	Few-shot tokens match the tokenizer; no leakage into the test set
test_cache.py	Round trip, key invalidation on prompt version or mode, corrupted entries ignored
test_llm_client_retry.py	Transient errors retried, permanent errors not
test_analyzer.py	Success, fenced JSON, repair path, degraded path, batch order, empty input
test_baseline.py	Baseline trains and predicts
13. Deployment on Replit
In Replit choose Create App → Import from GitHub and paste the repository URL.
Open Secrets and add your key, for example GEMINI_API_KEY. For another provider also add LLM_PROVIDER=openai or anthropic.
Press Run. The .replit file starts streamlit run app.py on port 8501.
To publish, press Deploy (Autoscale). If websockets misbehave, switch to a Reserved VM.

[!NOTE] Replit deployments may require a paid plan. If unavailable, the Run URL or Streamlit Community Cloud works as a demo. API keys live only in Secrets or .env, never in the repository.

14. Troubleshooting
Symptom	Cause and fix
"Missing API key" error	Set the matching variable (GEMINI_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY) in .env or Replit Secrets
Frequent rate-limit errors	Lower pipeline.max_workers in config.yaml; retries with backoff are already automatic
"Model not found"	Update the model name under llm.models to one available to your account
App shows no provider	No API key is set in the environment
Results look stale after editing the prompt	Bump version in prompts.yaml; it is part of the cache key. Or delete .cache/
ModuleNotFoundError: hinglishlens	Run commands from the repository root
15. Limitations and ethics
Small, self-constructed datasets. Results are indicative, not a published benchmark.
Subjective sarcasm labels. Sarcasm depends on pragmatic and social context, and annotators can disagree.
LLM nondeterminism. Even at temperature 0.0, provider updates can change outputs slightly. The cache and fixed prompt version help reproducibility.
Possible bias. LLMs may carry cultural or dialect biases that affect sentiment and emotion judgments.
Privacy. Inputs are sent to the selected LLM provider's API. Do not submit personal or confidential text. No inputs are stored beyond the local response cache, which is excluded from version control.
Cost. API calls consume quota. Caching and the compact schema reduce usage.
16. Future work
Devanagari-to-Roman transliteration as a pre-processing step
Evaluation on a public code-mixed benchmark such as SentiMix
Calibration of confidence scores
A fine-tuned small open model for offline inference

Author: Your name · Roll number · Course name
