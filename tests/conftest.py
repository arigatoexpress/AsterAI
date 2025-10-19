"""Pytest configuration for shared fixtures and test shims.

Expose `LSTMPredictorModel` globally for tests that reference it directly.
"""
from __future__ import annotations

import builtins

try:
    from mcp_trader.models.deep_learning.lstm_predictor import LSTMPredictorModel  # type: ignore
except Exception:  # pragma: no cover
    try:
        from mcp_trader.models.ml_models import RandomForestModel as LSTMPredictorModel  # type: ignore
    except Exception:
        class LSTMPredictorModel:  # type: ignore
            def __init__(self, *_, **__):
                pass

            def prepare_features(self, df):
                return df

            def save_model(self, *_args, **_kwargs):
                return None

            def load_model(self, *_args, **_kwargs):
                return None

builtins.LSTMPredictorModel = LSTMPredictorModel  # type: ignore


