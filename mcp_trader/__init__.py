"""
Rari Trade AI - Autonomous Aster DEX Trading Platform
Focused on maximizing profits through intelligent grid trading and risk management.
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

