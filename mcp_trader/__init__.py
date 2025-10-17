"""
Rari Trade - Enterprise AI Trading Platform
Advanced algorithmic trading with ensemble AI models and comprehensive risk management.
"""

__version__ = "2.0.0"

from .config import get_settings
from .execution.aster_client import AsterClient, AsterConfig
from .trading.autonomous_trader import AutonomousTrader

__all__ = [
    'get_settings',
    'AsterConfig',
    'AsterClient',
    'AutonomousTrader'
]

