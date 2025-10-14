"""
Reinforcement Learning trading agents.
Provides compatibility layer for RL dependencies.
"""

import logging

logger = logging.getLogger(__name__)

# Check stable-baselines3 availability
try:
    import stable_baselines3
    import gymnasium
    RL_AVAILABLE = True
    logger.info("Reinforcement Learning libraries available")
except ImportError:
    RL_AVAILABLE = False
    logger.warning("RL libraries not available - using rule-based fallback")

# Export based on availability
if RL_AVAILABLE:
    from .trading_agents import (
        RLTradingAgent,
        EnsembleRLAgent,
        TradingEnvironment
    )
else:
    # Provide mock classes for compatibility
    class RLTradingAgent:
        def __init__(self, *args, **kwargs):
            logger.warning("RLTradingAgent not available - using fallback")
            self.is_trained = False

        def train(self, *args, **kwargs):
            return self

        def predict(self, *args, **kwargs):
            return np.array([0.0]), {}

    class EnsembleRLAgent(RLTradingAgent):
        pass

    class TradingEnvironment:
        pass

__all__ = [
    'RLTradingAgent',
    'EnsembleRLAgent',
    'TradingEnvironment',
    'RL_AVAILABLE'
]

