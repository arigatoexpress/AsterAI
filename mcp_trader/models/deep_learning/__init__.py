"""
Deep Learning models for price prediction.
Provides compatibility layer for PyTorch dependencies.
"""

import logging

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch is available")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using fallback implementations")

# Export models based on availability
if PYTORCH_AVAILABLE:
    from .lstm_predictor import (
        LSTMPredictorModel,
        TransformerPredictorModel,
        EnsembleDLPredictor
    )
else:
    # Fallback to traditional ML models
    from ..ml_models import (
        RandomForestModel as LSTMPredictorModel,
        XGBoostModel as TransformerPredictorModel,
        XGBoostModel as EnsembleDLPredictor
    )
    logger.warning("Using traditional ML models as fallback for deep learning")

__all__ = [
    'LSTMPredictorModel',
    'TransformerPredictorModel',
    'EnsembleDLPredictor',
    'PYTORCH_AVAILABLE'
]

