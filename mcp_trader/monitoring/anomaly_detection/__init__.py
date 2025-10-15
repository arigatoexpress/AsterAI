"""
Anomaly detection and self-healing system.
Provides compatibility layer for PyTorch-based autoencoders.
"""

import logging

logger = logging.getLogger(__name__)

# Check PyTorch availability for autoencoder
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available - autoencoder disabled")

# Always export main classes (they handle torch gracefully)
from .anomaly_detector import (
    EnsembleAnomalyDetector,
    SelfHealingSystem,
    AnomalyConfig,
    AnomalyResult
)

__all__ = [
    'EnsembleAnomalyDetector',
    'SelfHealingSystem',
    'AnomalyConfig',
    'AnomalyResult',
    'PYTORCH_AVAILABLE'
]


