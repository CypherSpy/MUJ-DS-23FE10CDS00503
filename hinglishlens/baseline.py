"""TF-IDF + Logistic Regression classical NLP baseline for sentiment classification."""

import re
from collections.abc import Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing and collapsing 3+ repeated characters to 2.

    Args:
        text: Input raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    text_lower = text.lower()
    # Collapse 3 or more repeated characters to 2 (e.g., 'sooooo' -> 'soo', 'mastttt' -> 'mastt')
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", text_lower)
    return cleaned.strip()


def train_baseline(train_df: pd.DataFrame, seed: int = 42) -> Pipeline:
    """Train TF-IDF + Logistic Regression baseline pipeline.

    Args:
        train_df: DataFrame with 'text' and 'label' columns.
        seed: Random seed for Logistic Regression reproducibility.

    Returns:
        Fitted scikit-learn Pipeline object.
    """
    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("DataFrame must contain 'text' and 'label' columns")

    X_train = [preprocess_text(str(t)) for t in train_df["text"]]
    y_train = list(train_df["label"])

    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def predict_baseline(pipeline: Pipeline, texts: Sequence[str]) -> list[str]:
    """Predict sentiment labels using trained baseline pipeline.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        texts: Sequence of input text strings.

    Returns:
        List of predicted label strings.
    """
    X_clean = [preprocess_text(str(t)) for t in texts]
    preds = pipeline.predict(X_clean)
    return list(preds)
