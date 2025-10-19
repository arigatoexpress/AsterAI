"""
Rari Trade - Enterprise AI Trading Platform
Advanced algorithmic trading with ensemble AI models and comprehensive risk management.
"""

__version__ = "2.0.0"

# Avoid heavy imports at package import time (e.g., torch on systems without GPU wheels).
# Downstream modules should import the needed symbols directly from their submodules.

__all__ = [
    'config',
    'backtesting',
    'trading',
    'execution',
]

