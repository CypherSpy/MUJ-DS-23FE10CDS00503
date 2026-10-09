"""HinglishLens Streamlit Web Application."""

import os
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

from hinglishlens.analyzer import HinglishAnalyzer
from hinglishlens.config import Config, load_config
from hinglishlens.llm_client import PermanentLLMError

# Suppress harmless sklearn metric warnings
warnings.filterwarnings("ignore")

# Sync Streamlit Secrets to os.environ for Streamlit Cloud deployment
if hasattr(st, "secrets"):
    try:
        for secret_key in [
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "LLM_PROVIDER",
            "PROMPT_MODE",
        ]:
            if secret_key in st.secrets and not os.getenv(secret_key):
                os.environ[secret_key] = str(st.secrets[secret_key])
    except Exception:  # noqa: BLE001, S110
        pass

# Resolve base directory
BASE_DIR = Path(__file__).resolve().parent

# Page Configuration
st.set_page_config(
    page_title="HinglishLens - Hinglish Text Normalizer & Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Token Chips and Badges
TAG_COLORS = {
    "HI": "#FF8C00",  # Dark Orange
    "EN": "#1E90FF",  # Dodger Blue
    "NE": "#2E8B57",  # Sea Green
    "UNIV": "#708090",  # Slate Grey
    "OTHER": "#DC143C",  # Crimson
}

SENTIMENT_COLORS = {
    "positive": "#28a745",
    "negative": "#dc3545",
    "neutral": "#6c757d",
}


def get_available_providers() -> list[str]:
    """Return list of providers with API keys present in environment."""
    providers = []
    if os.getenv("GEMINI_API_KEY"):
        providers.append("gemini")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    return providers


@st.cache_resource
def get_base_config() -> Config:
    """Load and cache base configuration."""
    return load_config()


@st.cache_resource
def get_analyzer(provider: str, mode: str, enable_cache: bool) -> HinglishAnalyzer:
    """Instantiate and cache HinglishAnalyzer resource."""
    cfg = load_config()
    cfg.llm.provider = provider
    cfg.prompt.mode = mode
    cfg.cache.enabled = enable_cache
    return HinglishAnalyzer(config=cfg)


def render_token_chips(tokens) -> str:
    """Render HTML for language-colored token chips with tooltip for normalized forms."""
    chips_html = []
    for tok in tokens:
        color = TAG_COLORS.get(tok.lang, "#708090")
        has_norm = tok.normalized and tok.normalized != tok.text
        title_attr = (
            f'title="Normalized: {tok.normalized}"'
            if has_norm
            else f'title="Lang: {tok.lang}"'
        )
        norm_label = f" <small>({tok.normalized})</small>" if has_norm else ""

        chip = (
            f'<span {title_attr} style="background-color: {color}; color: white; '
            f"padding: 5px 10px; border-radius: 6px; margin: 3px; display: inline-block; "
            f'font-family: sans-serif; font-size: 14px; font-weight: 500;">'
            f"{tok.text}{norm_label} "
            f'<span style="font-size: 10px; opacity: 0.85; text-transform: uppercase;">[{tok.lang}]</span>'
            f"</span>"
        )
        chips_html.append(chip)
    return f'<div style="line-height: 2.2;">{"".join(chips_html)}</div>'


def main():
    st.title("🔍 HinglishLens")
    st.subheader("Code-Mixed (Hinglish) Text Normalizer & Analyzer")
    st.markdown(
        "Analyze Hinglish (Hindi + English code-mixed text): Token-level Language Identification, "
        "Spelling & SMS Normalization, Sentiment, Emotion, Sarcasm, and Code-Mixing Index (CMI)."
    )

    base_cfg = get_base_config()
    available_providers = get_available_providers()

    # Sidebar Controls
    st.sidebar.header("⚙️ Settings")

    if available_providers:
        selected_provider = st.sidebar.selectbox(
            "LLM Provider",
            available_providers,
            index=0,
            help="Provider selected based on active environment API keys",
        )
    else:
        st.sidebar.warning("⚠️ No API Key Detected in Environment")
        st.sidebar.info(
            "Set `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` in `.env` "
            "or Streamlit Cloud Secrets to enable live LLM calls."
        )
        selected_provider = "gemini"

    prompt_mode = st.sidebar.radio(
        "Prompt Mode",
        ["few_shot", "zero_shot"],
        index=0 if base_cfg.prompt.mode == "few_shot" else 1,
        help="Few-shot includes 5 representative Hinglish in-context examples",
    )

    enable_cache = st.sidebar.checkbox(
        "Enable SQLite Cache",
        value=base_cfg.cache.enabled,
        help="Cache responses to save API tokens and reduce latency",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("📖 [Project Specification & Docs](SPEC.md)")

    # Build active runtime configuration
    cfg = base_cfg.model_copy(deep=True)
    cfg.llm.provider = selected_provider
    cfg.prompt.mode = prompt_mode
    cfg.cache.enabled = enable_cache

    # Load sample inputs for dropdown
    sample_file = BASE_DIR / cfg.data.samples
    sample_options = ["-- Select an example --"]
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            sample_options.extend([line.strip() for line in f if line.strip()])

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(
        ["📝 Single Analysis", "📊 Batch Processing", "📈 Evaluation & Reports"]
    )

    # TAB 1: SINGLE ANALYSIS
    with tab1:
        st.markdown("### Analyze Hinglish Text")

        selected_sample = st.selectbox(
            "Load Example Sentence", sample_options, index=0
        )
        default_text = (
            selected_sample if selected_sample != "-- Select an example --" else ""
        )

        input_text = st.text_area(
            "Enter Hinglish Text",
            value=default_text,
            height=100,
            placeholder="e.g. yaar ye movie bakwaas thi but acting mast thi",
        )

        analyze_btn = st.button("🚀 Analyze Text", type="primary")

        if analyze_btn or (selected_sample != "-- Select an example --" and input_text):
            if not input_text.strip():
                st.warning("Please enter a non-empty text string.")
            else:
                with st.spinner("Analyzing Hinglish text with LLM pipeline..."):
                    try:
                        analyzer = get_analyzer(selected_provider, prompt_mode, enable_cache)
                        res = analyzer.analyze(input_text)

                        if res.error and "empty input" in res.error:
                            st.warning("Input is empty.")
                        else:
                            st.markdown("---")
                            st.markdown("#### 1. Token Language Tagging & Normalization")
                            st.caption(
                                "Color Legend: 🟧 HI (Hindi) | 🟦 EN (English) | 🟩 NE (Named Entity) | 🩶 UNIV (Independent) | 🟥 OTHER"
                            )
                            st.markdown(
                                render_token_chips(res.tokens), unsafe_allow_html=True
                            )

                            st.markdown("---")
                            col1, col2 = st.columns([1, 1])

                            with col1:
                                st.markdown("#### 2. Sentence Analysis")
                                st.markdown(f"**Original Text:** `{res.text}`")
                                st.markdown(
                                    f"**Normalized Text:** `{res.normalized_text}`"
                                )

                                s_color = SENTIMENT_COLORS.get(
                                    res.sentiment or "neutral", "#6c757d"
                                )
                                st.markdown(
                                    f'**Sentiment:** <span style="background-color: {s_color}; color: white; '
                                    f'padding: 3px 8px; border-radius: 4px; font-weight: bold;">'
                                    f'{(res.sentiment or "N/A").upper()}</span>',
                                    unsafe_allow_html=True,
                                )

                                if res.confidence is not None:
                                    st.progress(
                                        res.confidence,
                                        text=f"Confidence: {res.confidence:.0%}",
                                    )

                                st.markdown(f"**Emotion:** `{res.emotion or 'N/A'}`")
                                if res.sarcasm:
                                    st.error(
                                        "🎭 Sarcasm Detected! Intended meaning differs from literal words."
                                    )
                                else:
                                    st.info("Direct Intended Meaning (No Sarcasm)")

                                st.markdown("**English Translation:**")
                                st.success(f'"{res.translation_en or "N/A"}"')
                                st.markdown(
                                    f"**Rationale:** _{res.rationale or 'N/A'}_"
                                )

                            with col2:
                                st.markdown("#### 3. Code-Mixing Metrics")
                                m_col1, m_col2 = st.columns(2)
                                m_col1.metric(
                                    "CMI Score",
                                    f"{res.cmi:.1f}%",
                                    help="Code-Mixing Index (0 = monolingual, 100 = max mix)",
                                )
                                m_col2.metric(
                                    "Switch Points",
                                    res.switch_points,
                                    help="Language transitions between HI and EN",
                                )

                                st.markdown("**Language Distribution**")
                                dist_df = pd.DataFrame(
                                    list(res.lang_distribution.items()),
                                    columns=["Tag", "Count"],
                                )
                                st.bar_chart(dist_df.set_index("Tag"))

                            # Expander for Metadata & Raw JSON
                            with st.expander("🛠️ Metadata & Raw JSON Output"):
                                st.json({
                                    "meta": res.meta.model_dump(),
                                    "error": res.error,
                                    "full_result": res.model_dump(),
                                })

                    except PermanentLLMError as e:
                        st.error(f"❌ Configuration/Authentication Error: {e}")
                        st.info(
                            "Please verify that your API key is correctly configured in environment variables or Streamlit Secrets."
                        )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"❌ An unexpected error occurred: {e}")

    # TAB 2: BATCH PROCESSING
    with tab2:
        st.markdown("### Batch Processing")
        st.markdown(
            "Upload a CSV file containing a `text` column, or paste multiple texts line-by-line."
        )

        batch_input_mode = st.radio(
            "Batch Source", ["Paste Lines", "Upload CSV"], horizontal=True
        )

        batch_texts = []
        if batch_input_mode == "Paste Lines":
            pasted_lines = st.text_area("Paste Texts (one per line)", height=150)
            if pasted_lines:
                batch_texts = [
                    line.strip()
                    for line in pasted_lines.splitlines()
                    if line.strip()
                ]
        else:
            uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])
            if uploaded_file:
                try:
                    try:
                        b_df = pd.read_csv(uploaded_file, encoding="utf-8")
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        b_df = pd.read_csv(uploaded_file, encoding="latin-1")

                    # Normalize column names
                    col_map = {c: str(c).strip().lower() for c in b_df.columns}
                    b_df.rename(columns=col_map, inplace=True)

                    if "text" in b_df.columns:
                        batch_texts = list(b_df["text"].dropna().astype(str))
                        st.success(f"Loaded {len(batch_texts)} rows from CSV.")
                    else:
                        str_cols = b_df.select_dtypes(include=["object"]).columns
                        if len(str_cols) > 0:
                            target_col = str_cols[0]
                            batch_texts = list(b_df[target_col].dropna().astype(str))
                            st.warning(
                                f"No 'text' column found. Using '{target_col}' column instead."
                            )
                        else:
                            st.error("Uploaded CSV must contain a text column.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Failed to parse CSV file: {e}")

        if st.button("⚡ Run Batch Analysis", type="primary"):
            if not batch_texts:
                st.warning("No input texts provided for batch processing.")
            else:
                progress_bar = st.progress(0, text="Processing batch items...")
                try:
                    analyzer = get_analyzer(selected_provider, prompt_mode, enable_cache)

                    def update_progress(done, total):
                        try:
                            pct = min(max(done / total, 0.0), 1.0)
                            progress_bar.progress(
                                pct, text=f"Processed {done}/{total} items..."
                            )
                        except Exception:  # noqa: BLE001, S110
                            pass

                    results = analyzer.analyze_batch(
                        batch_texts, progress_cb=update_progress
                    )

                    table_rows = []
                    for r in results:
                        if r is not None:
                            table_rows.append({
                                "text": r.text,
                                "normalized_text": r.normalized_text,
                                "sentiment": r.sentiment,
                                "confidence": r.confidence,
                                "emotion": r.emotion,
                                "sarcasm": r.sarcasm,
                                "translation_en": r.translation_en,
                                "cmi": r.cmi,
                                "switch_points": r.switch_points,
                                "error": r.error,
                            })

                    res_df = pd.DataFrame(table_rows)
                    st.dataframe(res_df, use_container_width=True)

                    csv_data = res_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Batch Results CSV",
                        data=csv_data,
                        file_name="hinglishlens_batch_results.csv",
                        mime="text/csv",
                    )
                except PermanentLLMError as e:
                    st.error(f"❌ API Key Error: {e}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Batch processing error: {e}")

    # TAB 3: EVALUATION & REPORTS
    with tab3:
        st.markdown("### Evaluation Harness & Benchmark Results")

        report_file = BASE_DIR / "results" / "report.md"
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as f:
                report_content = f.read()
            st.markdown(report_content)

            st.markdown("---")
            st.markdown("#### Saved Confusion Matrices")
            confusion_imgs = list((BASE_DIR / "results").glob("confusion_*.png"))
            if confusion_imgs:
                cols = st.columns(min(len(confusion_imgs), 3))
                for idx, img_path in enumerate(sorted(confusion_imgs)):
                    col = cols[idx % len(cols)]
                    title_name = (
                        img_path.stem.replace("confusion_", "")
                        .replace("_", " ")
                        .title()
                    )
                    col.image(
                        str(img_path),
                        caption=f"{title_name} Confusion Matrix",
                        use_container_width=True,
                    )
        else:
            st.info("Evaluation report not generated yet.")
            st.markdown(
                "Run the evaluation CLI command to generate benchmarks:\n"
                "```bash\npython -m hinglishlens evaluate --modes zero_shot few_shot --split test\n```"
            )


if __name__ == "__main__":
    main()
