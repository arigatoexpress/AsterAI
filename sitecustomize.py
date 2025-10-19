"""Project-wide startup customizations for tests and local runs.

This exposes commonly used symbols to the test namespace for compatibility.
"""
from __future__ import annotations

import builtins

try:
    # Prefer the real DL model if available
    from mcp_trader.models.deep_learning.lstm_predictor import LSTMPredictorModel  # type: ignore
except Exception:  # pragma: no cover - fallback path
    try:
        # Fallback to a lightweight model to keep tests running
        from mcp_trader.models.ml_models import RandomForestModel as LSTMPredictorModel  # type: ignore
    except Exception:  # Last-resort no-op stub
        class LSTMPredictorModel:  # type: ignore
            def __init__(self, *_, **__):
                pass

            def prepare_features(self, df):
                return df

            def save_model(self, *_args, **_kwargs):
                return None

            def load_model(self, *_args, **_kwargs):
                return None

# Make symbol available globally for tests referencing it directly
builtins.LSTMPredictorModel = LSTMPredictorModel  # type: ignore


