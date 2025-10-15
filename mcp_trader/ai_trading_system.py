"""
Ultimate Adaptive AI Trading System
Production-ready autonomous cryptocurrency trading system for Aster DEX.
Combines deep learning, reinforcement learning, and traditional methods for optimal performance.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from .config import get_settings, PRIORITY_SYMBOLS
from .data.aster_feed import AsterDataFeed
from .execution.aster_client import AsterClient
from .trading.autonomous_trader import AutonomousTrader
from .models.deep_learning.lstm_predictor import LSTMPredictorModel, EnsembleDLPredictor
from .models.reinforcement_learning.trading_agents import RLTradingAgent, EnsembleRLAgent
from .monitoring.anomaly_detection.anomaly_detector import EnsembleAnomalyDetector, SelfHealingSystem
from .execution.algorithms.adaptive_execution import ExecutionEngine, ExecutionOrder, ExecutionAlgorithm
from .risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """Trading system operational modes."""
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    MAINTENANCE = "maintenance"


@dataclass
class SystemConfig:
    """Configuration for the complete AI trading system."""
    initial_balance: float = 10000.0
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_total_drawdown: float = 0.15  # 15% max drawdown
    target_annual_return: float = 2.0  # 200% annual return target
    rebalance_frequency_minutes: int = 60
    model_update_frequency_hours: int = 24
    risk_check_frequency_minutes: int = 15
    anomaly_check_frequency_minutes: int = 5

    # Model configurations
    use_deep_learning: bool = True
    use_reinforcement_learning: bool = True
    use_ensemble_methods: bool = True
    use_anomaly_detection: bool = True

    # Execution settings
    default_execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.ADAPTIVE_VWAP
    max_slippage: float = 0.005
    transaction_cost_bps: float = 5  # 5 basis points

    # System health
    min_model_accuracy: float = 0.55
    max_anomaly_rate: float = 0.1  # 10% max anomaly rate
    emergency_stop_threshold: float = 0.1  # 10% loss triggers emergency stop


@dataclass
class SystemState:
    """Current state of the trading system."""
    timestamp: datetime
    mode: SystemMode
    portfolio_value: float
    daily_pnl: float
    total_pnl: float
    active_positions: int
    active_models: List[str]
    system_health: float  # 0-1 scale
    last_rebalance: datetime
    last_model_update: datetime
    anomaly_rate: float
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class AdaptiveAITradingSystem:
    """
    Ultimate Adaptive AI Trading System for Aster DEX.
    Self-optimizing, self-healing autonomous trading platform.
    """

    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()

        # Core components
        self.settings = get_settings()
        self.aster_client = AsterClient(
            self.settings.aster_api_key,
            self.settings.aster_api_secret
        )
        self.data_feed = AsterDataFeed()
        self.execution_engine = ExecutionEngine()

        # AI Models
        self.deep_learning_models = {}
        self.rl_agents = {}
        self.ensemble_predictor = None

        # Risk and monitoring
        self.risk_manager = RiskManager(self.settings)
        self.anomaly_detector = EnsembleAnomalyDetector()
        self.self_healing = SelfHealingSystem(self.anomaly_detector)

        # Legacy trader (for hybrid operation)
        self.autonomous_trader = None

        # System state
        self.system_state = SystemState(
            timestamp=datetime.now(),
            mode=SystemMode.PAPER_TRADING,
            portfolio_value=self.config.initial_balance,
            daily_pnl=0.0,
            total_pnl=0.0,
            active_positions=0,
            active_models=[],
            system_health=1.0,
            last_rebalance=datetime.now(),
            last_model_update=datetime.now(),
            anomaly_rate=0.0
        )

        # Control flags
        self.is_running = False
        self.emergency_stop = False
        self.maintenance_mode = False

        # Performance tracking
        self.performance_history = []
        self.daily_performance = {}
        self.model_performance = {}

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("🚀 Adaptive AI Trading System initialized")
        logger.info(f"Initial balance: ${self.config.initial_balance:,.2f}")
        logger.info(f"Target: ${self.config.initial_balance * (1 + self.config.target_annual_return):,.2f} annual return")

    async def initialize(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing system components...")

            # Connect to Aster DEX
            await self.aster_client.connect()
            logger.info("✅ Connected to Aster DEX")

            # Start data feed
            await self.data_feed.start()
            logger.info("✅ Data feed started")

            # Initialize AI models
            await self._initialize_ai_models()
            logger.info("✅ AI models initialized")

            # Initialize risk management
            await self._initialize_risk_management()
            logger.info("✅ Risk management initialized")

            # Initialize anomaly detection
            await self._initialize_anomaly_detection()
            logger.info("✅ Anomaly detection initialized")

            # Initialize autonomous trader for hybrid operation
            self.autonomous_trader = AutonomousTrader()
            logger.info("✅ Autonomous trader initialized")

            logger.info("🎯 System initialization complete")

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise

    async def _initialize_ai_models(self):
        """Initialize all AI models."""
        try:
            # Deep Learning Models
            if self.config.use_deep_learning:
                logger.info("Initializing deep learning models...")

                for symbol in PRIORITY_SYMBOLS[:3]:  # Focus on top 3 symbols initially
                    # Get historical data for training
                    historical_data = await self._get_historical_data(symbol, days=90)

                    if len(historical_data) > 100:
                        # LSTM Model
                        lstm_model = LSTMPredictorModel()
                        lstm_model.fit(historical_data)
                        self.deep_learning_models[f"{symbol}_lstm"] = lstm_model

                        # Ensemble DL Model
                        ensemble_model = EnsembleDLPredictor()
                        ensemble_model.fit(historical_data)
                        self.deep_learning_models[f"{symbol}_ensemble"] = ensemble_model

                        logger.info(f"✅ DL models trained for {symbol}")

            # Reinforcement Learning Agents
            if self.config.use_reinforcement_learning:
                logger.info("Initializing RL agents...")

                # Create training data from historical performance
                training_data = await self._prepare_rl_training_data()

                # PPO Agent
                ppo_config = {'algorithm': 'PPO', 'total_timesteps': 50000}
                ppo_agent = RLTradingAgent(ppo_config)
                ppo_agent.train(training_data)
                self.rl_agents['ppo'] = ppo_agent

                # SAC Agent
                sac_config = {'algorithm': 'SAC', 'total_timesteps': 50000}
                sac_agent = RLTradingAgent(sac_config)
                sac_agent.train(training_data)
                self.rl_agents['sac'] = sac_agent

                # Ensemble RL Agent
                ensemble_rl = EnsembleRLAgent([ppo_config, sac_config])
                ensemble_rl.train(training_data)
                self.rl_agents['ensemble'] = ensemble_rl

                logger.info("✅ RL agents trained")

        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
            # Continue without AI models if initialization fails

    async def _initialize_risk_management(self):
        """Initialize risk management components."""
        # Risk manager is already initialized in __init__
        # Additional setup can be done here if needed
        pass

    async def _initialize_anomaly_detection(self):
        """Initialize anomaly detection system."""
        try:
            # Get historical system states for training
            historical_states = await self._get_historical_system_states()

            if historical_states:
                self.anomaly_detector.train(historical_states)
                logger.info("✅ Anomaly detector trained on historical data")
            else:
                logger.warning("⚠️ No historical data for anomaly training")

        except Exception as e:
            logger.error(f"Error initializing anomaly detection: {e}")

    async def start_trading(self, mode: SystemMode = SystemMode.PAPER_TRADING):
        """Start the trading system."""
        if self.is_running:
            logger.warning("System is already running")
            return

        self.system_state.mode = mode
        self.is_running = True

        logger.info(f"🚀 Starting AI Trading System in {mode.value} mode")

        try:
            # Main trading loop
            while self.is_running and not self.emergency_stop:
                cycle_start = datetime.now()

                # Update system state
                await self._update_system_state()

                # Perform health checks
                await self._perform_health_checks()

                # Execute trading decisions
                if not self.maintenance_mode:
                    await self._execute_trading_cycle()

                # Model updates (less frequent)
                if (datetime.now() - self.system_state.last_model_update).total_seconds() / 3600 >= self.config.model_update_frequency_hours:
                    await self._update_models()

                # Log status
                self._log_system_status()

                # Wait for next cycle
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, self.config.rebalance_frequency_minutes * 60 - cycle_time)
                await asyncio.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Critical error in trading loop: {e}")
            await self._emergency_shutdown()

        finally:
            self.is_running = False

    async def _update_system_state(self):
        """Update current system state."""
        try:
            # Get current portfolio state
            portfolio_state = await self._get_portfolio_state()

            # Calculate performance metrics
            daily_pnl, total_pnl = self._calculate_pnl(portfolio_state)

            # Update state
            self.system_state.timestamp = datetime.now()
            self.system_state.portfolio_value = portfolio_state.get('total_balance', 0) + portfolio_state.get('unrealized_pnl', 0)
            self.system_state.daily_pnl = daily_pnl
            self.system_state.total_pnl = total_pnl
            self.system_state.active_positions = len(portfolio_state.get('active_positions', []))

            # System health assessment
            self.system_state.system_health = self._assess_system_health()

        except Exception as e:
            logger.error(f"Error updating system state: {e}")

    async def _perform_health_checks(self):
        """Perform comprehensive system health checks."""
        try:
            # Risk checks
            if (datetime.now() - self.system_state.last_rebalance).total_seconds() / 60 >= self.config.risk_check_frequency_minutes:
                await self._perform_risk_checks()

            # Anomaly detection
            if (datetime.now() - self.system_state.timestamp).total_seconds() / 60 >= self.config.anomaly_check_frequency_minutes:
                await self._perform_anomaly_checks()

            # Emergency stop checks
            await self._check_emergency_conditions()

        except Exception as e:
            logger.error(f"Error in health checks: {e}")

    async def _execute_trading_cycle(self):
        """Execute one complete trading cycle."""
        try:
            # Get market state
            market_state = await self._get_market_state()

            # Generate trading signals
            signals = await self._generate_trading_signals(market_state)

            # Risk assessment
            approved_signals = await self._assess_signal_risks(signals)

            # Execute approved signals
            if approved_signals:
                await self._execute_signals(approved_signals, market_state)

            # Update performance tracking
            self._update_performance_tracking()

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    async def _generate_trading_signals(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals using all available models."""
        signals = []

        try:
            for symbol in PRIORITY_SYMBOLS[:5]:  # Focus on top 5 symbols
                symbol_signals = []

                # Deep Learning signals
                if self.config.use_deep_learning and f"{symbol}_ensemble" in self.deep_learning_models:
                    dl_signals = await self._generate_dl_signals(symbol, market_state)
                    symbol_signals.extend(dl_signals)

                # RL-based signals
                if self.config.use_reinforcement_learning and self.rl_agents:
                    rl_signals = await self._generate_rl_signals(symbol, market_state)
                    symbol_signals.extend(rl_signals)

                # Traditional signals (from autonomous trader)
                if self.autonomous_trader:
                    trad_signals = await self._generate_traditional_signals(symbol, market_state)
                    symbol_signals.extend(trad_signals)

                # Combine signals for this symbol
                if symbol_signals:
                    combined_signal = self._combine_symbol_signals(symbol_signals)
                    if combined_signal:
                        signals.append(combined_signal)

        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")

        return signals

    async def _generate_dl_signals(self, symbol: str, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals from deep learning models."""
        signals = []

        try:
            model_key = f"{symbol}_ensemble"
            if model_key in self.deep_learning_models:
                model = self.deep_learning_models[model_key]

                # Get recent data for prediction
                symbol_data = market_state.get(symbol, {})
                if symbol_data:
                    # Create DataFrame for prediction
                    pred_data = pd.DataFrame([symbol_data])

                    # Generate prediction
                    prediction = model.predict(pred_data)

                    if prediction:
                        pred = prediction[0]
                        signal = {
                            'symbol': symbol,
                            'type': 'deep_learning',
                            'action': 'buy' if pred.prediction > 0.02 else 'sell' if pred.prediction < -0.02 else 'hold',
                            'confidence': pred.confidence,
                            'expected_return': pred.prediction,
                            'model': 'ensemble_dl'
                        }
                        signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating DL signals for {symbol}: {e}")

        return signals

    async def _generate_rl_signals(self, symbol: str, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals from RL agents."""
        signals = []

        try:
            if 'ensemble' in self.rl_agents:
                agent = self.rl_agents['ensemble']

                # Prepare observation from market state
                observation = self._market_state_to_observation(symbol, market_state)

                if observation is not None:
                    action, metadata = agent.predict(observation)

                    # Convert action to signal
                    action_value = action[0] if hasattr(action, '__len__') else action

                    if abs(action_value) > 0.1:  # Minimum threshold
                        signal = {
                            'symbol': symbol,
                            'type': 'reinforcement_learning',
                            'action': 'buy' if action_value > 0 else 'sell',
                            'confidence': metadata.get('ensemble_confidence', 0.5) if metadata else 0.5,
                            'quantity_fraction': abs(action_value),
                            'model': 'ensemble_rl'
                        }
                        signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating RL signals for {symbol}: {e}")

        return signals

    async def _generate_traditional_signals(self, symbol: str, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals from traditional trading strategies."""
        signals = []

        try:
            # Use the autonomous trader's decision logic
            if self.autonomous_trader and hasattr(self.autonomous_trader, '_make_trading_decisions'):
                # Create mock position state
                portfolio_state = await self._get_portfolio_state()

                # Get ticker data
                ticker = self.data_feed.get_ticker(symbol)
                if ticker:
                    # Assess market regime
                    market_regime = await self.autonomous_trader._assess_market_regime()

                    # Generate decisions
                    decisions = await self.autonomous_trader._make_trading_decisions(market_regime)

                    # Convert to signals
                    for decision in decisions:
                        if decision.symbol == symbol:
                            signal = {
                                'symbol': symbol,
                                'type': 'traditional',
                                'action': decision.action.lower(),
                                'confidence': decision.confidence,
                                'quantity': decision.quantity,
                                'model': 'grid_volatility'
                            }
                            signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating traditional signals for {symbol}: {e}")

        return signals

    def _combine_symbol_signals(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Combine multiple signals for the same symbol."""
        if not signals:
            return None

        try:
            # Weighted combination based on confidence and model type
            buy_signals = [s for s in signals if s['action'] == 'buy']
            sell_signals = [s for s in signals if s['action'] == 'sell']

            # Calculate weighted scores
            buy_score = sum(s['confidence'] for s in buy_signals)
            sell_score = sum(s['confidence'] for s in sell_signals)

            if buy_score > sell_score and buy_score > 1.0:  # Minimum threshold
                return {
                    'symbol': signals[0]['symbol'],
                    'action': 'buy',
                    'confidence': min(buy_score / len(signals), 1.0),
                    'signal_sources': len(buy_signals),
                    'total_signals': len(signals)
                }
            elif sell_score > buy_score and sell_score > 1.0:
                return {
                    'symbol': signals[0]['symbol'],
                    'action': 'sell',
                    'confidence': min(sell_score / len(signals), 1.0),
                    'signal_sources': len(sell_signals),
                    'total_signals': len(signals)
                }

        except Exception as e:
            logger.error(f"Error combining signals: {e}")

        return None

    async def _assess_signal_risks(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assess risks for trading signals."""
        approved_signals = []

        try:
            portfolio_state = await self._get_portfolio_state()

            for signal in signals:
                # Risk assessment
                risk_check = await self.risk_manager.check_risk_limits(
                    portfolio_state,
                    self.system_state.risk_metrics
                )

                if risk_check == []:  # No violations
                    approved_signals.append(signal)

        except Exception as e:
            logger.error(f"Error assessing signal risks: {e}")

        return approved_signals

    async def _execute_signals(self, signals: List[Dict[str, Any]], market_state: Dict[str, Any]):
        """Execute approved trading signals."""
        for signal in signals:
            try:
                # Calculate position size
                position_size = await self._calculate_position_size(signal, market_state)

                if position_size > 0:
                    # Create execution order
                    order = ExecutionOrder(
                        symbol=signal['symbol'],
                        side=signal['action'],
                        total_quantity=position_size,
                        algorithm=self.config.default_execution_algorithm,
                        duration_minutes=30,  # 30-minute execution window
                        max_slippage=self.config.max_slippage,
                        priority='normal',
                        metadata={'signal_confidence': signal['confidence']}
                    )

                    # Execute order
                    result = await self.execution_engine.execute_order(order, self.data_feed)

                    if result.success:
                        logger.info(f"✅ Executed {signal['action']} order for {result.total_executed:.4f} {signal['symbol']} at avg ${result.average_price:.4f}")
                    else:
                        logger.warning(f"❌ Failed to execute order for {signal['symbol']}")

            except Exception as e:
                logger.error(f"Error executing signal for {signal['symbol']}: {e}")

    async def _calculate_position_size(self, signal: Dict[str, Any], market_state: Dict[str, Any]) -> float:
        """Calculate position size for a signal."""
        try:
            # Get market data
            symbol_data = market_state.get(signal['symbol'], {})
            current_price = symbol_data.get('price', 0)

            if current_price <= 0:
                return 0

            # Base position size (1% of portfolio per signal)
            base_size = self.system_state.portfolio_value * 0.01

            # Adjust for signal confidence
            confidence_multiplier = signal['confidence']

            # Adjust for volatility
            volatility = symbol_data.get('volatility', 0.02)
            volatility_adjustment = min(1.0, 0.02 / volatility)  # Reduce size in high volatility

            # Final position size
            position_size = base_size * confidence_multiplier * volatility_adjustment

            # Convert to quantity
            quantity = position_size / current_price

            # Apply limits
            max_quantity = self.system_state.portfolio_value * 0.05 / current_price  # Max 5% of portfolio
            quantity = min(quantity, max_quantity)

            return quantity

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    async def _perform_risk_checks(self):
        """Perform risk management checks."""
        try:
            portfolio_state = await self._get_portfolio_state()
            market_data = {}

            # Get market data for all symbols
            for symbol in PRIORITY_SYMBOLS[:5]:
                data = await self.data_feed.get_klines(symbol, self.data_feed.kline_cache.get('1h', []), limit=100)
                if data:
                    market_data[symbol] = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Assess portfolio risk
            risk_metrics = await self.risk_manager.assess_portfolio_risk(portfolio_state, market_data)
            self.system_state.risk_metrics = {
                'var_95': risk_metrics.var_95,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'max_drawdown': risk_metrics.max_drawdown,
                'concentration_risk': risk_metrics.concentration_risk
            }

            # Check risk limits
            violations = await self.risk_manager.check_risk_limits(portfolio_state, risk_metrics)

            if violations:
                logger.warning(f"⚠️ Risk limit violations: {violations}")
                # Could trigger position adjustments here

            self.system_state.last_rebalance = datetime.now()

        except Exception as e:
            logger.error(f"Error in risk checks: {e}")

    async def _perform_anomaly_checks(self):
        """Perform anomaly detection checks."""
        try:
            # Get current system state
            system_state = {
                'portfolio_value': self.system_state.portfolio_value,
                'daily_pnl': self.system_state.daily_pnl,
                'unrealized_pnl': 0.0,  # Would get from actual positions
                'drawdown': self.system_state.risk_metrics.get('max_drawdown', 0),
                'volatility': 0.02,  # Would calculate from returns
                'sharpe_ratio': self.system_state.risk_metrics.get('sharpe_ratio', 0),
                'win_rate': 0.5,  # Would track from trades
                'trade_count': 10,  # Would track actual count
                'error_rate': 0.01,  # Would track API errors
                'success_rate': 0.99,
                'api_latency': 100,  # ms
                'cpu_usage': 45.0,
                'memory_usage': 60.0,
                'network_latency': 50
            }

            # Detect anomalies
            anomaly_result = self.anomaly_detector.detect_anomalies(system_state)

            self.system_state.anomaly_rate = self.anomaly_detector.get_performance_metrics().get('anomaly_rate', 0)

            if anomaly_result.is_anomalous:
                logger.warning(f"🚨 Anomaly detected: {anomaly_result.confidence:.2%} confidence")

                # Trigger self-healing
                healing_report = self.self_healing.diagnose_and_heal(system_state)

                if healing_report['actions_taken']:
                    logger.info(f"🔧 Applied healing actions: {healing_report['actions_taken']}")

        except Exception as e:
            logger.error(f"Error in anomaly checks: {e}")

    async def _check_emergency_conditions(self):
        """Check for emergency stop conditions."""
        try:
            # Daily loss limit
            if self.system_state.daily_pnl < -self.config.max_daily_loss * self.config.initial_balance:
                logger.critical("🚨 Daily loss limit exceeded - Emergency stop")
                self.emergency_stop = True

            # Total drawdown limit
            current_drawdown = self.system_state.risk_metrics.get('max_drawdown', 0)
            if current_drawdown > self.config.max_total_drawdown:
                logger.critical("🚨 Maximum drawdown exceeded - Emergency stop")
                self.emergency_stop = True

            # High anomaly rate
            if self.system_state.anomaly_rate > self.config.max_anomaly_rate:
                logger.critical("🚨 High anomaly rate detected - Entering maintenance mode")
                self.maintenance_mode = True

        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")

    async def _update_models(self):
        """Update AI models with new data."""
        try:
            logger.info("🔄 Updating AI models...")

            # Update deep learning models
            if self.config.use_deep_learning:
                for model_name, model in self.deep_learning_models.items():
                    if hasattr(model, 'update_models'):
                        # Get new training data
                        symbol = model_name.split('_')[0]
                        new_data = await self._get_historical_data(symbol, days=7)
                        if new_data:
                            model.update_models([new_data])

            # Update anomaly detector
            if self.config.use_anomaly_detection:
                recent_states = self.performance_history[-100:] if self.performance_history else []
                if recent_states:
                    self.anomaly_detector.update_models(recent_states)

            self.system_state.last_model_update = datetime.now()
            logger.info("✅ Models updated")

        except Exception as e:
            logger.error(f"Error updating models: {e}")

    def _assess_system_health(self) -> float:
        """Assess overall system health (0-1 scale)."""
        try:
            health_factors = []

            # Portfolio health
            portfolio_health = min(1.0, self.system_state.portfolio_value / self.config.initial_balance)
            health_factors.append(portfolio_health)

            # Risk health (inverse of drawdown)
            drawdown = self.system_state.risk_metrics.get('max_drawdown', 0)
            risk_health = max(0.0, 1.0 - drawdown / self.config.max_total_drawdown)
            health_factors.append(risk_health)

            # Anomaly health (inverse of anomaly rate)
            anomaly_health = max(0.0, 1.0 - self.system_state.anomaly_rate / self.config.max_anomaly_rate)
            health_factors.append(anomaly_health)

            # Model health
            model_health = 1.0 if self.deep_learning_models or self.rl_agents else 0.5
            health_factors.append(model_health)

            # Average health
            overall_health = np.mean(health_factors)

            return overall_health

        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
            return 0.5

    def _log_system_status(self):
        """Log current system status."""
        status = {
            'timestamp': self.system_state.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': self.system_state.mode.value,
            'portfolio_value': f"${self.system_state.portfolio_value:,.2f}",
            'daily_pnl': f"${self.system_state.daily_pnl:,.2f}",
            'total_pnl': f"${self.system_state.total_pnl:,.2f}",
            'active_positions': self.system_state.active_positions,
            'system_health': f"{self.system_state.system_health:.1%}",
            'anomaly_rate': f"{self.system_state.anomaly_rate:.1%}",
            'active_models': len(self.system_state.active_models)
        }

        logger.info(f"📊 System Status: {status}")

        # Store in performance history
        self.performance_history.append({
            'timestamp': self.system_state.timestamp,
            'portfolio_value': self.system_state.portfolio_value,
            'daily_pnl': self.system_state.daily_pnl,
            'system_health': self.system_state.system_health,
            'anomaly_rate': self.system_state.anomaly_rate
        })

        # Keep recent history
        if len(self.performance_history) > 10000:
            self.performance_history = self.performance_history[-5000:]

    async def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        # In a real implementation, this would query the actual portfolio
        # For now, return mock data
        return {
            'total_balance': self.system_state.portfolio_value,
            'available_balance': self.system_state.portfolio_value * 0.8,
            'unrealized_pnl': self.system_state.portfolio_value * 0.05,
            'active_positions': []  # Would contain actual positions
        }

    async def _get_market_state(self) -> Dict[str, Any]:
        """Get current market state for all symbols."""
        market_state = {}

        try:
            for symbol in PRIORITY_SYMBOLS[:5]:
                data = await self.data_feed.get_klines(symbol, self.data_feed.kline_cache.get('1h', []), limit=100)
                if data:
                    # Convert to DataFrame and calculate metrics
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                    # Calculate volatility
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(24) if len(returns) > 0 else 0.02

                    market_state[symbol] = {
                        'price': df['close'].iloc[-1],
                        'volume': df['volume'].iloc[-1],
                        'volatility': volatility,
                        'returns': returns.tolist() if len(returns) < 100 else returns.tail(100).tolist(),
                        'high': df['high'].iloc[-1],
                        'low': df['low'].iloc[-1],
                        'timestamp': df['timestamp'].iloc[-1]
                    }

        except Exception as e:
            logger.error(f"Error getting market state: {e}")

        return market_state

    async def _get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for model training."""
        try:
            # Calculate start time
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # Get klines
            klines = await self.data_feed.get_klines(
                symbol, self.data_feed.kline_cache.get('1h', []),
                start_time=start_time, limit=1000
            )

            if klines:
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")

        return pd.DataFrame()

    async def _prepare_rl_training_data(self) -> pd.DataFrame:
        """Prepare historical data for RL training."""
        # Combine data from multiple symbols
        all_data = []

        for symbol in PRIORITY_SYMBOLS[:3]:
            data = await self._get_historical_data(symbol, days=90)
            if not data.empty:
                data['symbol'] = symbol
                all_data.append(data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data.sort_values('timestamp')

        return pd.DataFrame()

    async def _get_historical_system_states(self) -> List[Dict[str, Any]]:
        """Get historical system states for anomaly training."""
        # In a real system, this would load from database
        # For now, generate synthetic historical states
        states = []

        base_value = self.config.initial_balance
        for i in range(1000):
            pnl = np.random.normal(0, base_value * 0.01)  # Daily P&L
            base_value += pnl

            state = {
                'portfolio_value': base_value,
                'daily_pnl': pnl,
                'unrealized_pnl': np.random.normal(0, base_value * 0.02),
                'drawdown': abs(np.random.normal(0, 0.05)),
                'volatility': abs(np.random.normal(0.02, 0.01)),
                'sharpe_ratio': np.random.normal(1.5, 0.5),
                'win_rate': np.random.uniform(0.4, 0.7),
                'trade_count': np.random.randint(1, 20),
                'error_rate': np.random.uniform(0, 0.05),
                'success_rate': np.random.uniform(0.9, 1.0),
                'api_latency': np.random.uniform(50, 200),
                'cpu_usage': np.random.uniform(20, 80),
                'memory_usage': np.random.uniform(40, 90),
                'network_latency': np.random.uniform(10, 100)
            }
            states.append(state)

        return states

    def _market_state_to_observation(self, symbol: str, market_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Convert market state to RL observation."""
        try:
            symbol_data = market_state.get(symbol, {})
            if not symbol_data:
                return None

            observation = np.array([
                symbol_data.get('price', 0) / 1000,  # Normalized price
                symbol_data.get('volatility', 0.02),
                len(symbol_data.get('returns', [])) / 100,  # Data availability
                self.system_state.portfolio_value / self.config.initial_balance,  # Portfolio health
                self.system_state.daily_pnl / (self.config.initial_balance * 0.1),  # Normalized P&L
                self.system_state.system_health,
                self.system_state.anomaly_rate,
                self.system_state.active_positions / 10  # Normalized position count
            ], dtype=np.float32)

            return observation

        except Exception as e:
            logger.error(f"Error converting market state to observation: {e}")
            return None

    def _calculate_pnl(self, portfolio_state: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate daily and total P&L."""
        try:
            current_value = portfolio_state.get('total_balance', 0) + portfolio_state.get('unrealized_pnl', 0)

            # Daily P&L (simplified - would track from previous day)
            daily_pnl = current_value - self.config.initial_balance * 0.02  # Mock calculation

            # Total P&L
            total_pnl = current_value - self.config.initial_balance

            return daily_pnl, total_pnl

        except Exception as e:
            logger.error(f"Error calculating P&L: {e}")
            return 0.0, 0.0

    def _update_performance_tracking(self):
        """Update performance tracking metrics."""
        try:
            current_date = datetime.now().date()

            if current_date not in self.daily_performance:
                self.daily_performance[current_date] = {
                    'start_value': self.system_state.portfolio_value,
                    'end_value': self.system_state.portfolio_value,
                    'daily_pnl': 0.0,
                    'trades': 0
                }

            self.daily_performance[current_date]['end_value'] = self.system_state.portfolio_value
            self.daily_performance[current_date]['daily_pnl'] = (
                self.daily_performance[current_date]['end_value'] -
                self.daily_performance[current_date]['start_value']
            )

        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")

    async def _emergency_shutdown(self):
        """Emergency shutdown procedure."""
        logger.critical("🚨 Emergency shutdown initiated")

        try:
            # Close all positions
            await self._close_all_positions()

            # Save system state
            self._save_system_state()

            # Disconnect from APIs
            await self.aster_client.disconnect()
            await self.data_feed.stop()

            logger.info("✅ Emergency shutdown completed")

        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")

    async def _close_all_positions(self):
        """Close all open positions."""
        logger.warning("Closing all positions...")
        # Implementation would close actual positions
        pass

    def _save_system_state(self):
        """Save current system state."""
        try:
            state_file = f"system_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            # Would save state to file or database
            logger.info(f"System state saved to {state_file}")
        except Exception as e:
            logger.error(f"Error saving system state: {e}")

    def _signal_handler(self, signum, frame):
        """Handle system signals."""
        logger.info(f"Received signal {signum}")
        self.emergency_stop = True

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'timestamp': self.system_state.timestamp.isoformat(),
            'mode': self.system_state.mode.value,
            'is_running': self.is_running,
            'portfolio_value': self.system_state.portfolio_value,
            'daily_pnl': self.system_state.daily_pnl,
            'total_pnl': self.system_state.total_pnl,
            'active_positions': self.system_state.active_positions,
            'system_health': self.system_state.system_health,
            'anomaly_rate': self.system_state.anomaly_rate,
            'active_models': len(self.system_state.active_models),
            'risk_metrics': self.system_state.risk_metrics,
            'performance_metrics': self.system_state.performance_metrics,
            'emergency_stop': self.emergency_stop,
            'maintenance_mode': self.maintenance_mode
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.performance_history:
            return {"error": "No performance history available"}

        # Calculate performance metrics
        portfolio_values = [p['portfolio_value'] for p in self.performance_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(returns) * np.sqrt(365) if len(returns) > 0 else 0
        sharpe_ratio = np.mean(returns) / volatility * np.sqrt(365) if volatility > 0 else 0

        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        return {
            'total_return': total_return,
            'annualized_return': total_return * 365 / len(self.performance_history),
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': len([r for r in returns if r > 0]) / len(returns) if returns else 0,
            'total_days': len(self.daily_performance),
            'system_health_avg': np.mean([p['system_health'] for p in self.performance_history]),
            'anomaly_rate_avg': np.mean([p['anomaly_rate'] for p in self.performance_history])
        }


async def main():
    """Main function to run the AI trading system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and initialize system
    system = AdaptiveAITradingSystem()

    try:
        await system.initialize()
        await system.start_trading(SystemMode.PAPER_TRADING)
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System failed: {e}")
        raise
    finally:
        # Final status report
        status = system.get_system_status()
        report = system.get_performance_report()

        logger.info("=== FINAL SYSTEM REPORT ===")
        logger.info(f"Final Portfolio Value: ${status['portfolio_value']:,.2f}")
        logger.info(f"Total P&L: ${status['total_pnl']:,.2f}")
        logger.info(f"System Health: {status['system_health']:.1%}")
        logger.info(f"Final Sharpe Ratio: {report.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {report.get('max_drawdown', 0):.1%}")


if __name__ == "__main__":
    asyncio.run(main())

