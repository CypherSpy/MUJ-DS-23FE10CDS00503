"""Command Line Interface (CLI) for HinglishLens."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from hinglishlens.analyzer import HinglishAnalyzer
from hinglishlens.config import load_config
from hinglishlens.evaluate import run_evaluation


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="hinglishlens",
        description="HinglishLens: Code-Mixed (Hinglish) Text Normalizer & Analyzer",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Command 1: analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze a single Hinglish text string")
    p_analyze.add_argument("text", type=str, help="Hinglish text string to analyze")
    p_analyze.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Prompt mode (default: few_shot)",
    )
    p_analyze.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output full AnalysisResult as formatted JSON",
    )

    # Command 2: batch
    p_batch = subparsers.add_parser("batch", help="Batch analyze text lines from a file")
    p_batch.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/sample_inputs.txt",
        help="Input text file path (one text per line)",
    )
    p_batch.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/batch.csv",
        help="Output CSV file path",
    )
    p_batch.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Prompt mode (default: few_shot)",
    )

    # Command 3: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run evaluation harness")
    p_eval.add_argument(
        "--modes",
        nargs="+",
        choices=["zero_shot", "few_shot"],
        default=["zero_shot", "few_shot"],
        help="LLM prompt modes to evaluate",
    )
    p_eval.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    p_eval.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to evaluate",
    )

    return parser


def run_analyze_cmd(args: argparse.Namespace) -> int:
    """Execute single text analyze command."""
    try:
        config = load_config()
        config.prompt.mode = args.mode
        analyzer = HinglishAnalyzer(config=config)
        result = analyzer.analyze(args.text)

        if args.json_output:
            print(result.model_dump_json(indent=2))
            return 0 if result.error is None else 1

        # Printable table output
        print("=" * 60)
        print(f"INPUT TEXT: {result.text}")
        print("=" * 60)
        print(f"{'INDEX':<6} | {'TOKEN':<15} | {'TAG':<6} | {'NORMALIZED'}")
        print("-" * 60)
        for tok in result.tokens:
            print(f"{tok.start:<6} | {tok.text:<15} | {tok.lang:<6} | {tok.normalized}")
        print("=" * 60)
        print(f"NORMALIZED TEXT: {result.normalized_text}")
        print(f"SENTIMENT:       {result.sentiment} (Confidence: {result.confidence})")
        print(f"EMOTION:         {result.emotion}")
        print(f"SARCASM:         {result.sarcasm}")
        print(f"TRANSLATION:     {result.translation_en}")
        print(f"RATIONALE:       {result.rationale}")
        print(f"CMI SCORE:       {result.cmi}%")
        print(f"SWITCH POINTS:   {result.switch_points}")
        print(f"LANG DIST:       {result.lang_distribution}")
        print(f"LATENCY:         {result.meta.latency_ms} ms (Cached: {result.meta.cached})")
        if result.error:
            print(f"WARNING/ERROR:   {result.error}")
            return 1
        return 0

    except Exception as e:  # noqa: BLE001
        print(f"Error executing analyze: {e}", file=sys.stderr)
        return 1


def run_batch_cmd(args: argparse.Namespace) -> int:
    """Execute batch file analyze command."""
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input file not found: '{args.input}'", file=sys.stderr)
        return 1

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Processing batch of {len(lines)} sentences from '{args.input}'...")
        config = load_config()
        config.prompt.mode = args.mode
        analyzer = HinglishAnalyzer(config=config)

        results = analyzer.analyze_batch(lines)

        rows = []
        for r in results:
            rows.append({
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

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"Batch analysis complete. Results saved to '{args.output}'.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error executing batch: {e}", file=sys.stderr)
        return 1


def run_evaluate_cmd(args: argparse.Namespace) -> int:
    """Execute evaluation harness command."""
    try:
        config = load_config()
        run_evaluation(
            config=config,
            modes=args.modes,
            split=args.split,
            limit=args.limit,
        )
        print("Evaluation finished. Outputs written to 'results/'.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Error executing evaluate: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return run_analyze_cmd(args)
    elif args.command == "batch":
        return run_batch_cmd(args)
    elif args.command == "evaluate":
        return run_evaluate_cmd(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
