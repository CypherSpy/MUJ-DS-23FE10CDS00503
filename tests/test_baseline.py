"""Unit tests for baseline.py."""

import pandas as pd

from hinglishlens.baseline import predict_baseline, preprocess_text, train_baseline


def test_preprocess_text():
    assert preprocess_text("SOOOOO GOOD!!!!") == "soo good!!"
    assert preprocess_text("mastttttt") == "mastt"
    assert preprocess_text("") == ""


def test_train_and_predict_baseline():
    data = [
        {"text": "bohot hi accha hai", "label": "positive"},
        {"text": "sabse bekaar product", "label": "negative"},
        {"text": "meeting kal subah 10 bje hai", "label": "neutral"},
    ]
    df = pd.DataFrame(data)
    pipeline = train_baseline(df, seed=42)

    preds = predict_baseline(pipeline, ["accha product", "bekaar hai", "kal meeting hai"])
    assert len(preds) == 3
    assert isinstance(preds[0], str)
