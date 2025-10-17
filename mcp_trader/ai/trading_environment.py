"""
Advanced Trading Environment for Reinforcement Learning

Custom Gym environment optimized for cryptocurrency perpetual futures trading:
- Multi-asset portfolio management
- Realistic market microstructure
- Commission and slippage modeling
- Risk management integration
- High-frequency trading support
- Self-improving reward functions

Designed for volatile bull market downturns with HFT capabilities.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

from mcp_trader.ai.ml_training_data_structure import MLTrainingSample

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Configuration for trading environment"""

    # Portfolio settings
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # Max position as % of portfolio
    max_leverage: float = 20.0
    maintenance_margin: float = 0.05  # 5% maintenance margin

    # Trading costs
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0005  # 0.05%
    slippage_model: str = 'realistic'  # 'none', 'fixed', 'realistic'

    # Market settings
    symbols: List[str] = field(default_factory=lambda: ['BTC', 'ETH'])
    base_currency: str = 'USDT'

    # Time settings
    episode_length_hours: int = 24
    step_size_minutes: int = 60  # 1 hour steps

    # Reward function
    reward_function: str = 'sharpe_ratio'  # 'pnl', 'sharpe_ratio', 'sortino', 'calmar'
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

    # Risk management
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    stop_loss_threshold: float = 0.03  # 3% stop loss
    take_profit_threshold: float = 0.05  # 5% take profit

    # HFT settings
    enable_hft: bool = True
    min_order_size: float = 0.001
    max_orders_per_step: int = 5

    # Market microstructure
    enable_market_microstructure: bool = True
    order_book_depth: int = 10
    vpin_enabled: bool = True

    # Self-improving features
    adaptive_reward: bool = True
    curriculum_learning: bool = True
    difficulty_levels: List[str] = field(default_factory=lambda: ['easy', 'medium', 'hard', 'expert'])


@dataclass
class Position:
    """Trading position representation"""

    symbol: str
    side: str  # 'long' or 'short'
    size: float  # Contract size
    entry_price: float
    entry_time: datetime
    leverage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.size * self.leverage
        else:  # short
            return (self.entry_price - current_price) * self.size * self.leverage

    @property
    def liquidation_price(self) -> float:
        """Calculate liquidation price"""
        if self.side == 'long':
            return self.entry_price * (1 - 1/self.leverage + self.maintenance_margin)
        else:  # short
            return self.entry_price * (1 + 1/self.leverage - self.maintenance_margin)


class AdvancedTradingEnvironment(gym.Env):
    """
    Advanced Trading Environment for Reinforcement Learning

    Features:
    - Multi-asset portfolio management
    - Realistic slippage and commissions
    - Risk management with margin calls
    - High-frequency trading capabilities
    - Market microstructure modeling
    - Self-improving reward functions
    """

    def __init__(self, config: TradingConfig = None, training_data: Dict[str, List[MLTrainingSample]] = None):
        super().__init__()

        self.config = config or TradingConfig()
        self.training_data = training_data or {}

        # Initialize market data
        self.market_data = {}
        self.current_step = 0
        self.current_time = None
        self.episode_start_time = None

        # Portfolio state
        self.balance = self.config.initial_balance
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []

        # Risk management
        self.peak_balance = self.config.initial_balance
        self.max_drawdown = 0.0
        self.daily_pnl = []
        self.monthly_returns = []

        # Action and observation spaces
        self._define_spaces()

        # Market microstructure
        self.order_book = {}
        self.pending_orders = []
        self.market_orders = []

        # Self-improving features
        self.reward_history = []
        self.performance_metrics = {}
        self.difficulty_level = 'medium'

        logger.info("Advanced Trading Environment initialized")

    def _define_spaces(self):
        """Define Gym action and observation spaces"""

        # Action space: [action_type, symbol_idx, size, price_limit]
        # action_type: 0=hold, 1=buy_market, 2=sell_market, 3=buy_limit, 4=sell_limit, 5=close_position
        self.action_space = gym.spaces.MultiDiscrete([
            6,  # action types
            len(self.config.symbols),  # symbols
            11,  # position sizes (0.0, 0.1, 0.2, ..., 1.0)
            21   # price limits (-10% to +10% in 1% increments)
        ])

        # Observation space
        # Features per symbol + portfolio state + market state
        features_per_symbol = 50  # Technical indicators, price data, etc.
        portfolio_features = 20   # Balance, positions, P&L, etc.
        market_features = 10      # Market-wide indicators

        total_features = (features_per_symbol * len(self.config.symbols) +
                         portfolio_features + market_features)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""

        super().reset(seed=seed)

        # Reset portfolio
        self.balance = self.config.initial_balance
        self.positions.clear()
        self.trade_history.clear()
        self.portfolio_history.clear()
        self.pending_orders.clear()
        self.market_orders.clear()

        # Reset risk management
        self.peak_balance = self.config.initial_balance
        self.max_drawdown = 0.0

        # Reset market data
        self.current_step = 0
        self.episode_start_time = datetime.now()

        # Get initial observation
        observation = self._get_observation()

        # Initialize info
        info = {
            'balance': self.balance,
            'positions': len(self.positions),
            'unrealized_pnl': self._calculate_unrealized_pnl(),
            'max_drawdown': self.max_drawdown
        }

        return observation, info

    def step(self, action):
        """Execute one step in the environment"""

        # Parse action
        action_type, symbol_idx, size_pct, price_limit_pct = action

        symbol = self.config.symbols[symbol_idx]
        size_pct = size_pct / 10.0  # Convert to 0.0-1.0 range
        price_limit_pct = (price_limit_pct - 10) / 100.0  # Convert to -10% to +10%

        # Execute action
        reward = 0.0
        executed = False

        if action_type == 0:  # Hold
            pass
        elif action_type in [1, 2, 3, 4]:  # Trading actions
            reward, executed = self._execute_trade(action_type, symbol, size_pct, price_limit_pct)
        elif action_type == 5:  # Close position
            reward, executed = self._close_position(symbol)

        # Update market state
        self._update_market_state()

        # Check risk limits
        self._check_risk_limits()

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        if reward == 0.0:  # If no specific reward from action
            reward = self._calculate_reward()

        # Check if episode is done
        terminated = self._is_episode_done()
        truncated = False

        # Update info
        info = {
            'balance': self.balance,
            'positions': len(self.positions),
            'unrealized_pnl': self._calculate_unrealized_pnl(),
            'max_drawdown': self.max_drawdown,
            'executed': executed,
            'current_time': self.current_time,
            'reward_components': self._get_reward_components()
        }

        # Record portfolio state
        self.portfolio_history.append({
            'timestamp': self.current_time,
            'balance': self.balance,
            'unrealized_pnl': info['unrealized_pnl'],
            'total_value': self.balance + info['unrealized_pnl'],
            'max_drawdown': self.max_drawdown,
            'positions_count': len(self.positions)
        })

        self.current_step += 1

        return observation, reward, terminated, truncated, info

    def _execute_trade(self, action_type: int, symbol: str, size_pct: float,
                      price_limit_pct: float) -> Tuple[float, bool]:
        """Execute a trading action"""

        if symbol not in self.market_data:
            return 0.0, False

        current_price = self.market_data[symbol].get('close', 0)
        if current_price <= 0:
            return 0.0, False

        # Determine trade type and direction
        if action_type == 1:  # Buy market
            side = 'long'
            order_type = 'market'
        elif action_type == 2:  # Sell market
            side = 'short'
            order_type = 'market'
        elif action_type == 3:  # Buy limit
            side = 'long'
            order_type = 'limit'
        elif action_type == 4:  # Sell limit
            side = 'short'
            order_type = 'limit'

        # Calculate order size
        max_position_value = self.balance * self.config.max_position_size
        order_value = max_position_value * size_pct
        order_size = order_value / current_price

        # Check existing position
        existing_position = self.positions.get(symbol)

        # For simplicity, we'll only allow one position per symbol
        if existing_position and existing_position.side != side:
            # Close existing position first
            self._close_position(symbol)

        if existing_position and existing_position.side == side:
            # Increase position size
            total_size = existing_position.size + order_size
            avg_price = ((existing_position.entry_price * existing_position.size) +
                        (current_price * order_size)) / total_size
            existing_position.size = total_size
            existing_position.entry_price = avg_price
        else:
            # Create new position
            position = Position(
                symbol=symbol,
                side=side,
                size=order_size,
                entry_price=current_price,
                entry_time=self.current_time or datetime.now(),
                leverage=self.config.max_leverage
            )
            self.positions[symbol] = position

        # Calculate trading costs
        commission = order_value * self.config.taker_fee

        # Apply slippage
        slippage = self._calculate_slippage(symbol, order_size, order_type)
        effective_price = current_price * (1 + slippage if side == 'long' else 1 - slippage)

        # Update balance
        if side == 'long':
            self.balance -= order_value + commission
        else:
            self.balance += order_value - commission  # Short selling credits

        # Record trade
        trade = {
            'timestamp': self.current_time,
            'symbol': symbol,
            'side': side,
            'size': order_size,
            'price': effective_price,
            'value': order_value,
            'commission': commission,
            'slippage': slippage,
            'order_type': order_type
        }
        self.trade_history.append(trade)

        # Calculate immediate reward (execution quality)
        reward = self._calculate_execution_reward(trade)

        return reward, True

    def _close_position(self, symbol: str) -> Tuple[float, bool]:
        """Close position for a symbol"""

        if symbol not in self.positions:
            return 0.0, False

        position = self.positions[symbol]
        current_price = self.market_data.get(symbol, {}).get('close', position.entry_price)

        # Calculate P&L
        if position.side == 'long':
            pnl = (current_price - position.entry_price) * position.size * position.leverage
        else:
            pnl = (position.entry_price - current_price) * position.size * position.leverage

        # Calculate trading costs
        position_value = position.size * current_price
        commission = position_value * self.config.taker_fee

        # Update balance
        self.balance += pnl - commission

        # Record closing trade
        trade = {
            'timestamp': self.current_time,
            'symbol': symbol,
            'side': 'close',
            'size': position.size,
            'price': current_price,
            'pnl': pnl,
            'commission': commission,
            'position': position.__dict__
        }
        self.trade_history.append(trade)

        # Remove position
        del self.positions[symbol]

        # Calculate reward based on trade outcome
        reward = self._calculate_closing_reward(trade)

        return reward, True

    def _calculate_slippage(self, symbol: str, order_size: float, order_type: str) -> float:
        """Calculate realistic slippage"""

        if self.config.slippage_model == 'none':
            return 0.0

        if self.config.slippage_model == 'fixed':
            return 0.001  # 0.1% fixed slippage

        # Realistic slippage based on order size and market conditions
        base_slippage = 0.0005  # 0.05% base

        # Size impact
        size_impact = min(order_size / 1000, 0.01)  # Max 1% for large orders

        # Volatility impact
        volatility = self.market_data.get(symbol, {}).get('returns_volatility', 0.02)
        vol_impact = volatility * 0.1

        # Market microstructure impact
        microstructure_impact = 0.0002  # Base market impact

        total_slippage = base_slippage + size_impact + vol_impact + microstructure_impact

        return min(total_slippage, 0.05)  # Cap at 5%

    def _calculate_reward(self) -> float:
        """Calculate step reward based on configured reward function"""

        if self.config.reward_function == 'pnl':
            return self._calculate_pnl_reward()

        elif self.config.reward_function == 'sharpe_ratio':
            return self._calculate_sharpe_reward()

        elif self.config.reward_function == 'sortino':
            return self._calculate_sortino_reward()

        elif self.config.reward_function == 'calmar':
            return self._calculate_calmar_reward()

        else:
            return self._calculate_pnl_reward()  # Default

    def _calculate_pnl_reward(self) -> float:
        """Calculate reward based on P&L"""

        unrealized_pnl = self._calculate_unrealized_pnl()
        portfolio_value = self.balance + unrealized_pnl

        # Normalize by initial balance
        reward = (portfolio_value - self.config.initial_balance) / self.config.initial_balance

        # Scale to reasonable range
        return reward * 100  # Scale up for better learning

    def _calculate_sharpe_reward(self) -> float:
        """Calculate reward based on Sharpe ratio"""

        if len(self.portfolio_history) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_value = self.portfolio_history[i-1]['total_value']
            curr_value = self.portfolio_history[i]['total_value']
            ret = (curr_value - prev_value) / prev_value
            returns.append(ret)

        if not returns or np.std(returns) == 0:
            return 0.0

        # Annualize Sharpe ratio (assuming daily returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (avg_return - self.config.risk_free_rate/365) / std_return * np.sqrt(365)

        return sharpe

    def _calculate_sortino_reward(self) -> float:
        """Calculate reward based on Sortino ratio (downside risk only)"""

        if len(self.portfolio_history) < 2:
            return 0.0

        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_value = self.portfolio_history[i-1]['total_value']
            curr_value = self.portfolio_history[i]['total_value']
            ret = (curr_value - prev_value) / prev_value
            returns.append(ret)

        if not returns:
            return 0.0

        # Calculate downside deviation
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return np.mean(returns) * 100  # No downside risk

        downside_deviation = np.std(negative_returns)

        if downside_deviation == 0:
            return 0.0

        # Sortino ratio
        avg_return = np.mean(returns)
        sortino = (avg_return - self.config.risk_free_rate/365) / downside_deviation * np.sqrt(365)

        return sortino

    def _calculate_calmar_reward(self) -> float:
        """Calculate reward based on Calmar ratio (return vs max drawdown)"""

        if not self.portfolio_history:
            return 0.0

        # Calculate current drawdown
        current_value = self.portfolio_history[-1]['total_value']
        current_drawdown = (self.peak_balance - current_value) / self.peak_balance

        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        if self.max_drawdown == 0:
            return 0.0

        # Calculate total return
        total_return = (current_value - self.config.initial_balance) / self.config.initial_balance

        # Annualize (assuming episode represents one year equivalent)
        annualized_return = total_return  # Simplified

        # Calmar ratio
        calmar = annualized_return / self.max_drawdown

        return calmar

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L"""

        total_pnl = 0.0

        for symbol, position in self.positions.items():
            current_price = self.market_data.get(symbol, {}).get('close', position.entry_price)
            total_pnl += position.unrealized_pnl(current_price)

        return total_pnl

    def _check_risk_limits(self):
        """Check and enforce risk management limits"""

        # Check drawdown limit
        unrealized_pnl = self._calculate_unrealized_pnl()
        total_value = self.balance + unrealized_pnl

        if total_value > self.peak_balance:
            self.peak_balance = total_value

        current_drawdown = (self.peak_balance - total_value) / self.peak_balance

        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        if current_drawdown > self.config.max_drawdown_limit:
            logger.warning(f"Drawdown limit exceeded: {current_drawdown:.2%}")
            # Force close all positions
            for symbol in list(self.positions.keys()):
                self._close_position(symbol)

    def _is_episode_done(self) -> bool:
        """Check if episode should end"""

        # Time limit
        if self.current_step >= (self.config.episode_length_hours * 60 // self.config.step_size_minutes):
            return True

        # Balance depletion
        if self.balance <= self.config.initial_balance * 0.1:  # 90% loss
            return True

        # Extreme drawdown
        if self.max_drawdown > self.config.max_drawdown_limit * 2:  # Double the limit
            return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""

        features = []

        # Symbol-specific features
        for symbol in self.config.symbols:
            symbol_features = self._get_symbol_features(symbol)
            features.extend(symbol_features)

        # Portfolio features
        portfolio_features = self._get_portfolio_features()
        features.extend(portfolio_features)

        # Market-wide features
        market_features = self._get_market_features()
        features.extend(market_features)

        return np.array(features, dtype=np.float32)

    def _get_symbol_features(self, symbol: str) -> List[float]:
        """Get features for a specific symbol"""

        market_data = self.market_data.get(symbol, {})

        features = [
            market_data.get('close', 0),
            market_data.get('volume', 0),
            market_data.get('returns', 0),
            market_data.get('rsi', 50),
            market_data.get('macd', 0),
            market_data.get('bb_upper', 0),
            market_data.get('bb_lower', 0),
            # Add more technical indicators...
        ]

        # Pad or truncate to fixed size
        while len(features) < 50:  # features_per_symbol
            features.append(0.0)

        return features[:50]

    def _get_portfolio_features(self) -> List[float]:
        """Get portfolio-level features"""

        unrealized_pnl = self._calculate_unrealized_pnl()
        total_value = self.balance + unrealized_pnl

        features = [
            self.balance,
            unrealized_pnl,
            total_value,
            len(self.positions),
            self.max_drawdown,
            # Position details...
        ]

        # Pad to fixed size
        while len(features) < 20:  # portfolio_features
            features.append(0.0)

        return features[:20]

    def _get_market_features(self) -> List[float]:
        """Get market-wide features"""

        features = [
            # Market sentiment, volatility, etc.
            0.0,  # Placeholder
        ]

        # Pad to fixed size
        while len(features) < 10:  # market_features
            features.append(0.0)

        return features[:10]

    def _update_market_state(self):
        """Update market data for next step"""

        # This would integrate with real-time data feeds
        # For now, use training data progression
        pass

    def _calculate_execution_reward(self, trade: Dict[str, Any]) -> float:
        """Calculate reward based on trade execution quality"""

        # Reward for successful execution
        base_reward = 0.1

        # Penalize high slippage
        slippage_penalty = abs(trade.get('slippage', 0)) * 10

        # Penalize high commissions
        commission_penalty = trade.get('commission', 0) / trade.get('value', 1) * 5

        return base_reward - slippage_penalty - commission_penalty

    def _calculate_closing_reward(self, trade: Dict[str, Any]) -> float:
        """Calculate reward for position closing"""

        pnl = trade.get('pnl', 0)

        # Reward profitable trades
        if pnl > 0:
            reward = pnl / self.config.initial_balance * 10
        else:
            reward = pnl / self.config.initial_balance * 20  # Larger penalty for losses

        return reward

    def _get_reward_components(self) -> Dict[str, float]:
        """Get detailed reward components for analysis"""

        unrealized_pnl = self._calculate_unrealized_pnl()
        total_value = self.balance + unrealized_pnl

        return {
            'unrealized_pnl': unrealized_pnl,
            'total_value': total_value,
            'balance': self.balance,
            'positions_count': len(self.positions),
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_reward(),
            'sortino_ratio': self._calculate_sortino_reward(),
            'calmar_ratio': self._calculate_calmar_reward()
        }

    def render(self, mode='human'):
        """Render environment state"""

        if mode == 'human':
            print(f"Balance: ${self.balance:.2f}")
            print(f"Unrealized P&L: ${self._calculate_unrealized_pnl():.2f}")
            print(f"Positions: {len(self.positions)}")
            print(f"Max Drawdown: {self.max_drawdown:.2%}")
            print(f"Current Step: {self.current_step}")

    def close(self):
        """Clean up environment"""

        self.positions.clear()
        self.trade_history.clear()
        self.portfolio_history.clear()
        self.pending_orders.clear()
        self.market_orders.clear()


# Convenience functions
def create_trading_environment(config: TradingConfig = None,
                              training_data: Dict[str, List[MLTrainingSample]] = None) -> AdvancedTradingEnvironment:
    """Create a trading environment instance"""
    return AdvancedTradingEnvironment(config, training_data)


def create_hft_environment(symbols: List[str] = None) -> AdvancedTradingEnvironment:
    """Create environment optimized for high-frequency trading"""

    config = TradingConfig(
        symbols=symbols or ['BTC', 'ETH'],
        step_size_minutes=1,  # 1-minute bars for HFT
        enable_hft=True,
        max_orders_per_step=10,
        enable_market_microstructure=True,
        order_book_depth=20,
        slippage_model='realistic',
        maker_fee=0.0001,  # Lower fees for HFT
        taker_fee=0.0003
    )

    return AdvancedTradingEnvironment(config)


def create_volatility_environment() -> AdvancedTradingEnvironment:
    """Create environment optimized for volatile bull market downturns"""

    config = TradingConfig(
        reward_function='sortino',  # Focus on downside risk
        max_drawdown_limit=0.10,  # Tighter risk control
        stop_loss_threshold=0.02,  # 2% stop loss
        take_profit_threshold=0.03,  # 3% take profit
        adaptive_reward=True,
        difficulty_levels=['hard', 'expert']
    )

    return AdvancedTradingEnvironment(config)
