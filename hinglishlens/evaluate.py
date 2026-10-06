"""Evaluation harness and report generator for HinglishLens.

Evaluates TF-IDF baseline vs LLM zero-shot vs LLM few-shot on sentiment classification,
language identification (LID), and operational metrics.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from hinglishlens.analyzer import HinglishAnalyzer
from hinglishlens.baseline import predict_baseline, train_baseline
from hinglishlens.config import Config, load_config
from hinglishlens.llm_client import PermanentLLMError, get_client

logger = logging.getLogger(__name__)

SENTIMENT_CLASSES = ["positive", "negative", "neutral"]
LID_TAGS = ["HI", "EN", "NE", "UNIV", "OTHER"]


def save_confusion_matrix_plot(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Generate and save confusion matrix heatmap PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_sentiment_system(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    """Compute sentiment classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=SENTIMENT_CLASSES, average="macro")

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=SENTIMENT_CLASSES, zero_division=0
    )

    per_class = {}
    for idx, cls_name in enumerate(SENTIMENT_CLASSES):
        per_class[cls_name] = {
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
    }


def run_evaluation(
    config: Config | None = None,
    modes: list[str] | None = None,
    split: str = "test",
    limit: int | None = None,
) -> dict[str, Any]:
    """Run comprehensive evaluation across baseline and LLM modes.

    Args:
        config: Configuration object.
        modes: List of LLM prompt modes to evaluate (e.g. ['zero_shot', 'few_shot']).
        split: Dataset split to evaluate ('test' or 'train').
        limit: Optional maximum number of rows to evaluate (useful for quick runs).

    Returns:
        Dictionary of all evaluation metrics and metadata.
    """
    cfg = config or load_config()
    eval_modes = modes or ["zero_shot", "few_shot"]
    out_dir = Path(cfg.evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Datasets
    train_path = Path(cfg.data.train)
    split_path = Path(cfg.data.test) if split == "test" else train_path
    lid_path = Path(cfg.data.lid_gold)

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: '{train_path}'")
    if not split_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: '{split_path}'")

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(split_path)

    if limit and limit > 0:
        eval_df = eval_df.head(limit)

    eval_texts = list(eval_df["text"])
    eval_labels = list(eval_df["label"])

    metrics_results: dict[str, Any] = {
        "split": split,
        "eval_samples": len(eval_df),
        "sentiment": {},
        "lid": {},
        "operational": {},
    }

    # 2. Evaluate TF-IDF Baseline
    logger.info("Evaluating TF-IDF + LogisticRegression baseline on %s split...", split)
    pipeline = train_baseline(train_df, seed=cfg.evaluation.seed)
    baseline_preds = predict_baseline(pipeline, eval_texts)

    base_sentiment_metrics = evaluate_sentiment_system(eval_labels, baseline_preds)
    metrics_results["sentiment"]["baseline"] = base_sentiment_metrics

    # Save baseline predictions and confusion matrix
    pred_df = eval_df.copy()
    pred_df["predicted_label"] = baseline_preds
    pred_df.to_csv(out_dir / "predictions_baseline.csv", index=False)
    save_confusion_matrix_plot(
        eval_labels,
        baseline_preds,
        SENTIMENT_CLASSES,
        f"Baseline Sentiment Confusion Matrix ({split})",
        out_dir / "confusion_baseline.png",
    )

    # Check API key availability for LLM evaluation
    provider = cfg.llm.provider.lower()
    key_map = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    required_key_var = key_map.get(provider, "API_KEY")

    try:
        # Dry-run client initialization to check API key
        get_client(cfg)
        has_api_key = True
    except PermanentLLMError:
        has_api_key = False
        logger.warning("No API key available for provider '%s'. LLM evaluation will be recorded as TBD.", provider)

    # 3. Evaluate LLM Modes (zero_shot, few_shot)
    for mode in eval_modes:
        if not has_api_key:
            metrics_results["sentiment"][mode] = "TBD (No API key provided - run python -m hinglishlens evaluate with API key)"
            metrics_results["lid"][mode] = "TBD (No API key provided - run python -m hinglishlens evaluate with API key)"
            metrics_results["operational"][mode] = "TBD (No API key provided)"
            continue

        logger.info("Evaluating LLM mode '%s' on %s split...", mode, split)
        mode_cfg = cfg.model_copy(deep=True)
        mode_cfg.prompt.mode = mode
        analyzer = HinglishAnalyzer(config=mode_cfg)
        results = analyzer.analyze_batch(eval_texts)

        mode_preds = [r.sentiment if r.sentiment else "neutral" for r in results]
        mode_metrics = evaluate_sentiment_system(eval_labels, mode_preds)
        metrics_results["sentiment"][mode] = mode_metrics

        # Save predictions & confusion matrix
        m_pred_df = eval_df.copy()
        m_pred_df["predicted_label"] = mode_preds
        m_pred_df["confidence"] = [r.confidence for r in results]
        m_pred_df["sarcasm"] = [r.sarcasm for r in results]
        m_pred_df["translation_en"] = [r.translation_en for r in results]
        m_pred_df["error"] = [r.error for r in results]
        m_pred_df.to_csv(out_dir / f"predictions_{mode}.csv", index=False)

        save_confusion_matrix_plot(
            eval_labels,
            mode_preds,
            SENTIMENT_CLASSES,
            f"LLM {mode} Sentiment Confusion Matrix ({split})",
            out_dir / f"confusion_{mode}.png",
        )

        # Operational stats
        latencies = [r.meta.latency_ms for r in results]
        cached_cnt = sum(1 for r in results if r.meta.cached)
        repaired_cnt = sum(1 for r in results if r.meta.repaired)
        degraded_cnt = sum(1 for r in results if r.meta.degraded)
        in_toks = sum(r.meta.input_tokens or 0 for r in results)
        out_toks = sum(r.meta.output_tokens or 0 for r in results)

        metrics_results["operational"][mode] = {
            "total_calls": len(results),
            "cache_hits": cached_cnt,
            "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
            "total_input_tokens": in_toks,
            "total_output_tokens": out_toks,
            "repaired_count": repaired_cnt,
            "degraded_count": degraded_cnt,
        }

        # Evaluate Token-level Language ID on lid_gold.jsonl
        if lid_path.exists():
            logger.info("Evaluating Language ID on '%s' for mode '%s'...", lid_path, mode)
            lid_records = []
            with open(lid_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lid_records.append(json.loads(line.strip()))

            lid_texts = [r["text"] for r in lid_records]
            gold_tags_flat: list[str] = []
            pred_tags_flat: list[str] = []
            hi_baseline_flat: list[str] = []

            lid_results = analyzer.analyze_batch(lid_texts)
            for r, record in zip(lid_results, lid_records):
                gold_t = record["tags"]
                pred_t = [tok.lang for tok in r.tokens]
                # Ensure 1-to-1 matching length
                min_len = min(len(gold_t), len(pred_t))
                gold_tags_flat.extend(gold_t[:min_len])
                pred_tags_flat.extend(pred_t[:min_len])
                hi_baseline_flat.extend(["HI"] * min_len)

            lid_acc = accuracy_score(gold_tags_flat, pred_tags_flat)
            hi_base_acc = accuracy_score(gold_tags_flat, hi_baseline_flat)

            lid_prec, lid_rec, lid_f1, lid_supp = precision_recall_fscore_support(
                gold_tags_flat, pred_tags_flat, labels=LID_TAGS, zero_division=0
            )

            per_tag = {}
            for idx, tag_name in enumerate(LID_TAGS):
                per_tag[tag_name] = {
                    "precision": round(float(lid_prec[idx]), 4),
                    "recall": round(float(lid_rec[idx]), 4),
                    "f1": round(float(lid_f1[idx]), 4),
                    "support": int(lid_supp[idx]),
                }

            metrics_results["lid"][mode] = {
                "token_accuracy": round(float(lid_acc), 4),
                "hi_majority_baseline_accuracy": round(float(hi_base_acc), 4),
                "per_tag": per_tag,
            }

    # 4. Save metrics.json and report.md
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_results, f, indent=2)

    generate_markdown_report(metrics_results, out_dir / "report.md", required_key_var)

    return metrics_results


def generate_markdown_report(results: dict[str, Any], output_path: Path, key_var: str) -> None:
    """Generate human-readable report.md markdown table report."""
    split = results["split"]
    n_samples = results["eval_samples"]

    lines = [
        "# HinglishLens Evaluation Report",
        "",
        f"**Evaluation Split:** `{split}` ({n_samples} samples)",
        "",
        "## 1. Sentiment Classification Performance",
        "",
        "| System | Accuracy | Macro F1 | Positive F1 | Negative F1 | Neutral F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    # Baseline row
    base = results["sentiment"].get("baseline", {})
    b_acc = base.get("accuracy", "N/A")
    b_mf1 = base.get("macro_f1", "N/A")
    b_pos = base.get("per_class", {}).get("positive", {}).get("f1", "N/A")
    b_neg = base.get("per_class", {}).get("negative", {}).get("f1", "N/A")
    b_neu = base.get("per_class", {}).get("neutral", {}).get("f1", "N/A")
    lines.append(f"| **TF-IDF + LogisticRegression** | {b_acc} | {b_mf1} | {b_pos} | {b_neg} | {b_neu} |")

    # LLM rows
    for mode in ["zero_shot", "few_shot"]:
        m_data = results["sentiment"].get(mode)
        if isinstance(m_data, dict):
            acc = m_data.get("accuracy", "N/A")
            mf1 = m_data.get("macro_f1", "N/A")
            pos = m_data.get("per_class", {}).get("positive", {}).get("f1", "N/A")
            neg = m_data.get("per_class", {}).get("negative", {}).get("f1", "N/A")
            neu = m_data.get("per_class", {}).get("neutral", {}).get("f1", "N/A")
            lines.append(f"| **LLM ({mode})** | {acc} | {mf1} | {pos} | {neg} | {neu} |")
        else:
            lines.append(f"| **LLM ({mode})** | TBD | TBD | TBD | TBD | TBD |")

    lines.extend([
        "",
        "## 2. Token-Level Language Identification (LID)",
        "",
        "| System | Token Accuracy | HI F1 | EN F1 | NE F1 | UNIV F1 | OTHER F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for mode in ["zero_shot", "few_shot"]:
        l_data = results["lid"].get(mode)
        if isinstance(l_data, dict):
            acc = l_data.get("token_accuracy", "N/A")
            pt = l_data.get("per_tag", {})
            hi_f1 = pt.get("HI", {}).get("f1", "N/A")
            en_f1 = pt.get("EN", {}).get("f1", "N/A")
            ne_f1 = pt.get("NE", {}).get("f1", "N/A")
            u_f1 = pt.get("UNIV", {}).get("f1", "N/A")
            o_f1 = pt.get("OTHER", {}).get("f1", "N/A")
            lines.append(f"| **LLM ({mode})** | {acc} | {hi_f1} | {en_f1} | {ne_f1} | {u_f1} | {o_f1} |")
        else:
            lines.append(f"| **LLM ({mode})** | TBD | TBD | TBD | TBD | TBD | TBD |")

    lines.extend([
        "",
        "## 3. Operational & Cost Statistics",
        "",
        "| Mode | Calls | Cache Hits | Mean Latency (ms) | P95 Latency (ms) | Input Tokens | Output Tokens | Repaired | Degraded |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for mode in ["zero_shot", "few_shot"]:
        op = results["operational"].get(mode)
        if isinstance(op, dict):
            lines.append(
                f"| **{mode}** | {op['total_calls']} | {op['cache_hits']} | {op['mean_latency_ms']} | "
                f"{op['p95_latency_ms']} | {op['total_input_tokens']} | {op['total_output_tokens']} | "
                f"{op['repaired_count']} | {op['degraded_count']} |"
            )
        else:
            lines.append(f"| **{mode}** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")

    lines.extend([
        "",
        "---",
        f"**Note:** If LLM cells show `TBD`, please set your API key environment variable (`export {key_var}=...`) and execute:",
        "```bash",
        "python -m hinglishlens evaluate --modes zero_shot few_shot --split test",
        "```",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
