from typing import Dict
import logging

from strategies.hft.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class CapitalScaler:
    """
    Automated capital scaling for live trading.
    Reduces risk as capital grows to preserve gains.
    Target: Compound $100 to $100k sustainably.
    """
    def __init__(self, risk_mgr: RiskManager, initial_capital: float = 100.0):
        self.risk_mgr = risk_mgr
        self.initial_capital = initial_capital
        self.milestones = {
            2.0: 0.75,   # At 2x capital, reduce risk to 75%
            5.0: 0.50,   # At 5x, reduce to 50%
            10.0: 0.30,  # At 10x, reduce to 30%
            50.0: 0.20,  # At 50x, reduce to 20%
            100.0: 0.10  # At 100x ($10k), ultra-conservative 10%
        }
        
    def check_scaling(self, current_capital: float):
        """Check if capital milestone reached and adjust risk."""
        multiplier = current_capital / self.initial_capital
        for threshold, risk_factor in sorted(self.milestones.items()):
            if multiplier >= threshold:
                new_max_position = self.risk_mgr.params.max_position_size * risk_factor
                self.risk_mgr.params.max_position_size = max(new_max_position, 0.01)  # Min 1%
                logger.info(f"💰 Capital milestone reached: {multiplier:.1f}x (${current_capital:.0f}). Risk reduced to {risk_factor:.0%}.")
                break
        else:
            logger.info(f"Capital: ${current_capital:.0f} ({multiplier:.1f}x initial). No scaling needed.")
            
    def get_status(self) -> Dict:
        """Get current scaling status."""
        current_capital = self.risk_mgr.capital
        multiplier = current_capital / self.initial_capital
        next_milestone = min([t for t in self.milestones if multiplier < t], default=None)
        return {
            'current_capital': current_capital,
            'multiplier': multiplier,
            'current_risk_factor': self.risk_mgr.params.max_position_size / 0.1,  # Relative to initial 10%
            'next_milestone': next_milestone,
            'target_100k': current_capital >= 100000
        }

# Integration example in live trader
# scaler = CapitalScaler(risk_mgr)
# After each trade: scaler.check_scaling(risk_mgr.capital)
