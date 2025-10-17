#!/usr/bin/env python3
"""
Autonomous MCP-Integrated Master Agent
Orchestrates data, strategy, risk, and execution sub-agents for self-improving trading.

Features:
- MCP integration for external tool access
- Multi-agent architecture with specialized sub-agents
- Self-learning and parameter optimization
- Real-time performance monitoring
- Automatic strategy selection and weighting
- Risk management and position sizing
- Emergency stop and circuit breaker mechanisms
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Local imports
from mcp_trader.config import get_settings, PRIORITY_SYMBOLS
from mcp_trader.ai.adaptive_trading_agent import AdaptiveTradingAgent, AdaptiveAgentConfig
from mcp_trader.strategies.market_making import MarketMakingStrategy, MarketMakingConfig
from mcp_trader.strategies.funding_arbitrage import FundingArbitrageStrategy, FundingArbConfig
from mcp_trader.strategies.degen_trading import DegenTradingStrategy, DegenConfig
from mcp_trader.risk.risk_manager import RiskManager
from mcp_trader.execution.aster_client import AsterClient
from autonomous_data_pipeline import AutonomousDataPipeline, AutonomousDataConfig

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of sub-agents in the system."""
    DATA = "data"
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"
    MONITORING = "monitoring"


@dataclass
class AgentPerformance:
    """Performance metrics for a sub-agent."""
    agent_type: AgentType
    agent_name: str
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    performance_score: float = 0.0


@dataclass
class TradingDecision:
    """Trading decision from strategy agent."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    strategy_name: str
    confidence: float
    risk_score: float
    expected_return: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MasterAgentConfig:
    """Configuration for the master agent."""
    # Agent settings
    max_concurrent_positions: int = 3
    max_position_size_usd: float = 100.0
    min_position_size_usd: float = 1.0
    
    # Risk management
    max_daily_loss_pct: float = 0.10  # 10%
    max_portfolio_risk_pct: float = 0.20  # 20%
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    
    # Learning and adaptation
    learning_rate: float = 0.01
    adaptation_window_hours: int = 24
    performance_evaluation_hours: int = 168  # 1 week
    
    # Strategy weights (will be learned)
    initial_strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'market_making': 0.3,
        'funding_arbitrage': 0.3,
        'degen_trading': 0.2,
        'adaptive_agent': 0.2
    })
    
    # MCP settings
    enable_mcp: bool = True
    mcp_tools: List[str] = field(default_factory=lambda: [
        'web_search',
        'data_analysis',
        'risk_calculation',
        'performance_analysis'
    ])


class DataAgent:
    """Data collection and validation sub-agent."""
    
    def __init__(self, config: MasterAgentConfig):
        self.config = config
        self.data_pipeline = AutonomousDataPipeline()
        self.performance = AgentPerformance(AgentType.DATA, "data_agent")
        self.is_running = False
        
    async def initialize(self):
        """Initialize the data agent."""
        try:
            await self.data_pipeline.initialize()
            self.is_running = True
            logger.info("✅ Data agent initialized")
        except Exception as e:
            logger.error(f"❌ Data agent initialization failed: {e}")
            raise
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get latest market data for symbols."""
        try:
            start_time = datetime.now()
            
            # Get data from pipeline
            raw_data = await self.data_pipeline.collect_market_data()
            
            # Filter for requested symbols
            filtered_data = {symbol: data for symbol, data in raw_data.items() 
                           if symbol in symbols}
            
            # Update performance metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.performance.avg_latency_ms = (
                (self.performance.avg_latency_ms + latency) / 2
            )
            
            if filtered_data:
                self.performance.success_rate = 1.0
            else:
                self.performance.success_rate = 0.0
                self.performance.error_count += 1
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Data agent error: {e}")
            self.performance.error_count += 1
            self.performance.success_rate = 0.0
            return {}
    
    async def validate_data_quality(self, data: Dict[str, Any]) -> bool:
        """Validate data quality."""
        try:
            if not data:
                return False
            
            # Check data freshness
            for symbol, market_data in data.items():
                timestamp = market_data.get('timestamp')
                if timestamp:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    age_minutes = (datetime.now() - timestamp).total_seconds() / 60
                    if age_minutes > 10:  # Data older than 10 minutes
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False


class StrategyAgent:
    """Strategy selection and signal generation sub-agent."""
    
    def __init__(self, config: MasterAgentConfig):
        self.config = config
        self.strategies = {}
        self.strategy_weights = config.initial_strategy_weights.copy()
        self.performance = AgentPerformance(AgentType.STRATEGY, "strategy_agent")
        self.performance_history = []
        
        # Initialize strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all available strategies."""
        try:
            # Market Making Strategy
            mm_config = MarketMakingConfig(
                min_order_size_usd=self.config.min_position_size_usd,
                max_order_size_usd=self.config.max_position_size_usd,
                risk_per_trade_pct=1.0
            )
            self.strategies['market_making'] = MarketMakingStrategy(mm_config)
            
            # Funding Arbitrage Strategy
            fa_config = FundingArbConfig(
                min_position_size_usd=self.config.min_position_size_usd,
                max_position_size_usd=self.config.max_position_size_usd,
                min_funding_rate_pct=0.3
            )
            self.strategies['funding_arbitrage'] = FundingArbitrageStrategy(fa_config)
            
            # Degen Trading Strategy
            degen_config = DegenConfig(
                max_position_size_pct=0.15,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct
            )
            self.strategies['degen_trading'] = DegenTradingStrategy(degen_config)
            
            # Adaptive Trading Agent
            adaptive_config = AdaptiveAgentConfig(
                initial_balance=10000.0,
                max_allocation_per_trade=0.1,
                risk_tolerance=0.15
            )
            self.strategies['adaptive_agent'] = AdaptiveTradingAgent(adaptive_config)
            
            logger.info(f"✅ Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"❌ Strategy initialization failed: {e}")
            raise
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingDecision]:
        """Generate trading signals from all strategies."""
        try:
            start_time = datetime.now()
            all_decisions = []
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Get strategy weight
                    weight = self.strategy_weights.get(strategy_name, 0.0)
                    if weight <= 0:
                        continue
                    
                    # Generate signals from strategy
                    if hasattr(strategy, 'generate_signals'):
                        decisions = await strategy.generate_signals(market_data)
                    elif hasattr(strategy, 'execute_strategy'):
                        # For strategies that need different interface
                        decisions = await self._adapt_strategy_interface(strategy, market_data)
                    else:
                        continue
                    
                    # Apply strategy weight and add metadata
                    for decision in decisions:
                        if isinstance(decision, dict):
                            decision = TradingDecision(
                                symbol=decision.get('symbol', ''),
                                side=decision.get('side', 'buy'),
                                quantity=decision.get('quantity', 0.0),
                                price=decision.get('price', 0.0),
                                strategy_name=strategy_name,
                                confidence=decision.get('confidence', 0.5),
                                risk_score=decision.get('risk_score', 0.5),
                                expected_return=decision.get('expected_return', 0.0),
                                metadata=decision.get('metadata', {})
                            )
                        
                        # Apply weight
                        decision.quantity *= weight
                        decision.confidence *= weight
                        
                        all_decisions.append(decision)
                
                except Exception as e:
                    logger.error(f"Error in strategy {strategy_name}: {e}")
                    self.performance.error_count += 1
            
            # Update performance metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.performance.avg_latency_ms = (
                (self.performance.avg_latency_ms + latency) / 2
            )
            
            if all_decisions:
                self.performance.success_rate = 1.0
            else:
                self.performance.success_rate = 0.0
            
            return all_decisions
            
        except Exception as e:
            logger.error(f"Strategy agent error: {e}")
            self.performance.error_count += 1
            self.performance.success_rate = 0.0
            return []
    
    async def _adapt_strategy_interface(self, strategy, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adapt different strategy interfaces to common format."""
        try:
            decisions = []
            
            for symbol, data in market_data.items():
                if hasattr(strategy, 'execute_strategy'):
                    # For market making and other strategies
                    result = await strategy.execute_strategy(
                        symbol=symbol,
                        market_data=data,
                        orderbook=data.get('orderbook', {}),
                        capital_available=1000.0  # Default capital
                    )
                    
                    if result and isinstance(result, list):
                        for order in result:
                            if isinstance(order, dict):
                                decisions.append({
                                    'symbol': symbol,
                                    'side': order.get('side', 'buy'),
                                    'quantity': order.get('quantity', 0.0),
                                    'price': order.get('price', data.get('price', 0.0)),
                                    'confidence': 0.7,  # Default confidence
                                    'risk_score': 0.3,  # Default risk
                                    'expected_return': 0.02,  # Default 2%
                                    'metadata': {'strategy_type': 'market_making'}
                                })
            
            return decisions
            
        except Exception as e:
            logger.error(f"Strategy interface adaptation error: {e}")
            return []
    
    def update_strategy_weights(self, performance_data: Dict[str, float]):
        """Update strategy weights based on performance."""
        try:
            total_performance = sum(performance_data.values())
            if total_performance == 0:
                return
            
            # Normalize performance scores
            normalized_scores = {
                name: score / total_performance 
                for name, score in performance_data.items()
            }
            
            # Update weights using exponential moving average
            alpha = self.config.learning_rate
            for strategy_name in self.strategy_weights:
                if strategy_name in normalized_scores:
                    current_weight = self.strategy_weights[strategy_name]
                    target_weight = normalized_scores[strategy_name]
                    new_weight = alpha * target_weight + (1 - alpha) * current_weight
                    self.strategy_weights[strategy_name] = max(0.1, min(0.8, new_weight))
            
            # Renormalize weights
            total_weight = sum(self.strategy_weights.values())
            for strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] /= total_weight
            
            logger.info(f"Updated strategy weights: {self.strategy_weights}")
            
        except Exception as e:
            logger.error(f"Error updating strategy weights: {e}")


class RiskAgent:
    """Risk management and position sizing sub-agent."""
    
    def __init__(self, config: MasterAgentConfig):
        self.config = config
        self.risk_manager = RiskManager(get_settings())
        self.performance = AgentPerformance(AgentType.RISK, "risk_agent")
        self.portfolio_value = 10000.0  # Starting value
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 10000.0
        
    async def assess_risk(self, decision: TradingDecision, 
                         current_positions: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """Assess risk for a trading decision."""
        try:
            start_time = datetime.now()
            
            # Check daily loss limit
            if self.daily_pnl < -self.config.max_daily_loss_pct * self.portfolio_value:
                return False, 0.0, "Daily loss limit exceeded"
            
            # Check portfolio risk
            total_exposure = sum(pos.get('value_usd', 0) for pos in current_positions)
            if total_exposure > self.config.max_portfolio_risk_pct * self.portfolio_value:
                return False, 0.0, "Portfolio risk limit exceeded"
            
            # Check position size
            position_value = decision.quantity * decision.price
            if position_value > self.config.max_position_size_usd:
                # Scale down position
                decision.quantity = self.config.max_position_size_usd / decision.price
                position_value = self.config.max_position_size_usd
            
            if position_value < self.config.min_position_size_usd:
                return False, 0.0, "Position too small"
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(decision, current_positions)
            
            # Check risk threshold
            if risk_score > 0.7:  # High risk threshold
                return False, risk_score, "Risk score too high"
            
            # Update performance metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.performance.avg_latency_ms = (
                (self.performance.avg_latency_ms + latency) / 2
            )
            
            self.performance.success_rate = 1.0
            
            return True, risk_score, "Risk assessment passed"
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            self.performance.error_count += 1
            self.performance.success_rate = 0.0
            return False, 1.0, f"Risk assessment error: {str(e)}"
    
    def _calculate_risk_score(self, decision: TradingDecision, 
                            current_positions: List[Dict[str, Any]]) -> float:
        """Calculate risk score for a decision."""
        try:
            risk_score = 0.0
            
            # Base risk from decision
            risk_score += decision.risk_score * 0.3
            
            # Position size risk
            position_value = decision.quantity * decision.price
            size_risk = min(position_value / self.portfolio_value, 1.0)
            risk_score += size_risk * 0.2
            
            # Concentration risk
            symbol_exposure = sum(
                pos.get('value_usd', 0) for pos in current_positions 
                if pos.get('symbol') == decision.symbol
            )
            concentration_risk = symbol_exposure / self.portfolio_value
            risk_score += concentration_risk * 0.2
            
            # Market risk (volatility)
            market_risk = 0.1  # Default market risk
            risk_score += market_risk * 0.3
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Risk score calculation error: {e}")
            return 1.0  # Maximum risk on error
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value and calculate metrics."""
        try:
            self.portfolio_value = new_value
            
            # Update peak value and drawdown
            if new_value > self.peak_value:
                self.peak_value = new_value
                self.max_drawdown = 0.0
            else:
                current_drawdown = (self.peak_value - new_value) / self.peak_value
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
        except Exception as e:
            logger.error(f"Portfolio value update error: {e}")


class ExecutionAgent:
    """Order execution and position management sub-agent."""
    
    def __init__(self, config: MasterAgentConfig):
        self.config = config
        self.aster_client = AsterClient(
            get_settings().aster_api_key,
            get_settings().aster_api_secret
        )
        self.performance = AgentPerformance(AgentType.EXECUTION, "execution_agent")
        self.active_positions = []
        self.order_history = []
        
    async def initialize(self):
        """Initialize the execution agent."""
        try:
            await self.aster_client.initialize()
            self.performance.success_rate = 1.0
            logger.info("✅ Execution agent initialized")
        except Exception as e:
            logger.error(f"❌ Execution agent initialization failed: {e}")
            self.performance.error_count += 1
            self.performance.success_rate = 0.0
            raise
    
    async def execute_decision(self, decision: TradingDecision) -> bool:
        """Execute a trading decision."""
        try:
            start_time = datetime.now()
            
            # Create order request
            from mcp_trader.execution.aster_client import OrderRequest
            order_request = OrderRequest(
                symbol=decision.symbol,
                side=decision.side,
                order_type='limit',
                quantity=decision.quantity,
                price=decision.price
            )
            
            # Place order
            order_response = await self.aster_client.place_order(order_request)
            
            if order_response and order_response.status == 'FILLED':
                # Add to active positions
                position = {
                    'symbol': decision.symbol,
                    'side': decision.side,
                    'quantity': decision.quantity,
                    'price': decision.price,
                    'value_usd': decision.quantity * decision.price,
                    'timestamp': datetime.now(),
                    'strategy_name': decision.strategy_name,
                    'stop_loss': decision.stop_loss,
                    'take_profit': decision.take_profit
                }
                self.active_positions.append(position)
                
                # Add to order history
                self.order_history.append({
                    'timestamp': datetime.now(),
                    'symbol': decision.symbol,
                    'side': decision.side,
                    'quantity': decision.quantity,
                    'price': decision.price,
                    'order_id': order_response.order_id,
                    'status': order_response.status
                })
                
                # Update performance metrics
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.performance.avg_latency_ms = (
                    (self.performance.avg_latency_ms + latency) / 2
                )
                
                self.performance.success_rate = 1.0
                logger.info(f"✅ Executed {decision.side} order: {decision.quantity} {decision.symbol} @ {decision.price}")
                return True
            else:
                logger.warning(f"⚠️ Order not filled: {order_response}")
                self.performance.error_count += 1
                return False
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            self.performance.error_count += 1
            self.performance.success_rate = 0.0
            return False
    
    async def manage_positions(self) -> List[Dict[str, Any]]:
        """Manage existing positions (stop loss, take profit)."""
        try:
            positions_to_close = []
            
            for position in self.active_positions:
                # Get current price
                ticker = await self.aster_client.get_24hr_ticker(position['symbol'])
                current_price = float(ticker.get('lastPrice', position['price']))
                
                # Check stop loss
                if position['stop_loss'] and current_price <= position['stop_loss']:
                    positions_to_close.append({
                        'position': position,
                        'reason': 'stop_loss',
                        'current_price': current_price
                    })
                
                # Check take profit
                elif position['take_profit'] and current_price >= position['take_profit']:
                    positions_to_close.append({
                        'position': position,
                        'reason': 'take_profit',
                        'current_price': current_price
                    })
            
            return positions_to_close
            
        except Exception as e:
            logger.error(f"Position management error: {e}")
            return []


class AutonomousMCPAgent:
    """
    Master autonomous agent that orchestrates all sub-agents.
    
    Features:
    - Multi-agent architecture
    - MCP integration for external tools
    - Self-learning and adaptation
    - Real-time performance monitoring
    - Automatic strategy optimization
    - Risk management and position sizing
    """
    
    def __init__(self, config: MasterAgentConfig = None):
        self.config = config or MasterAgentConfig()
        self.settings = get_settings()
        
        # Initialize sub-agents
        self.data_agent = DataAgent(self.config)
        self.strategy_agent = StrategyAgent(self.config)
        self.risk_agent = RiskAgent(self.config)
        self.execution_agent = ExecutionAgent(self.config)
        
        # MCP client
        self.mcp_client = None
        self.mcp_session = None
        
        # System state
        self.is_running = False
        self.emergency_stop = False
        self.last_update = datetime.now()
        
        # Performance tracking
        self.performance_history = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        logger.info("🚀 Autonomous MCP Agent initialized")
    
    async def initialize(self):
        """Initialize the master agent and all sub-agents."""
        try:
            logger.info("🔄 Initializing master agent...")
            
            # Initialize sub-agents
            await self.data_agent.initialize()
            await self.execution_agent.initialize()
            
            # Initialize MCP if enabled
            if self.config.enable_mcp:
                await self._initialize_mcp()
            
            self.is_running = True
            logger.info("✅ Master agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Master agent initialization failed: {e}")
            raise
    
    async def _initialize_mcp(self):
        """Initialize MCP client for external tool access."""
        try:
            # Initialize MCP client
            server_params = StdioServerParameters(
                command="python",
                args=["-m", "mcp.server.example"]
            )
            
            self.mcp_session = ClientSession(server_params)
            await self.mcp_session.initialize()
            
            logger.info("✅ MCP client initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ MCP initialization failed: {e}")
            self.config.enable_mcp = False
    
    async def run_trading_cycle(self):
        """Run one complete trading cycle."""
        try:
            cycle_start = datetime.now()
            logger.info("🔄 Starting trading cycle...")
            
            # 1. Get market data
            market_data = await self.data_agent.get_market_data(PRIORITY_SYMBOLS)
            if not market_data:
                logger.warning("⚠️ No market data available")
                return
            
            # Validate data quality
            if not await self.data_agent.validate_data_quality(market_data):
                logger.warning("⚠️ Data quality validation failed")
                return
            
            # 2. Generate trading signals
            decisions = await self.strategy_agent.generate_signals(market_data)
            if not decisions:
                logger.info("ℹ️ No trading signals generated")
                return
            
            # 3. Risk assessment and execution
            executed_trades = 0
            for decision in decisions:
                try:
                    # Risk assessment
                    approved, risk_score, reason = await self.risk_agent.assess_risk(
                        decision, self.execution_agent.active_positions
                    )
                    
                    if not approved:
                        logger.info(f"❌ Trade rejected: {reason}")
                        continue
                    
                    # Execute trade
                    success = await self.execution_agent.execute_decision(decision)
                    if success:
                        executed_trades += 1
                        logger.info(f"✅ Trade executed: {decision.symbol} {decision.side}")
                    
                except Exception as e:
                    logger.error(f"Error processing decision: {e}")
            
            # 4. Manage existing positions
            positions_to_close = await self.execution_agent.manage_positions()
            for close_info in positions_to_close:
                logger.info(f"🔄 Closing position: {close_info['reason']}")
                # Implement position closing logic here
            
            # 5. Update performance metrics
            await self._update_performance_metrics()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"✅ Trading cycle completed in {cycle_duration:.2f}s - {executed_trades} trades executed")
            
        except Exception as e:
            logger.error(f"❌ Trading cycle failed: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics and strategy weights."""
        try:
            # Calculate current performance
            current_value = sum(
                pos['quantity'] * pos['price'] 
                for pos in self.execution_agent.active_positions
            )
            
            # Update risk agent
            self.risk_agent.update_portfolio_value(current_value)
            
            # Calculate strategy performance
            strategy_performance = {}
            for strategy_name in self.strategy_agent.strategy_weights:
                # Simple performance calculation based on recent trades
                recent_trades = [
                    trade for trade in self.execution_agent.order_history
                    if trade.get('strategy_name') == strategy_name
                    and trade['timestamp'] > datetime.now() - timedelta(hours=24)
                ]
                
                if recent_trades:
                    # Calculate win rate and average return
                    wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
                    total_trades = len(recent_trades)
                    win_rate = wins / total_trades if total_trades > 0 else 0
                    
                    strategy_performance[strategy_name] = win_rate
                else:
                    strategy_performance[strategy_name] = 0.5  # Neutral performance
            
            # Update strategy weights
            self.strategy_agent.update_strategy_weights(strategy_performance)
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': current_value,
                'strategy_weights': self.strategy_agent.strategy_weights.copy(),
                'strategy_performance': strategy_performance.copy(),
                'active_positions': len(self.execution_agent.active_positions)
            })
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")
    
    async def run_autonomous_trading(self):
        """Run autonomous trading continuously."""
        logger.info("🚀 Starting autonomous trading...")
        self.is_running = True
        
        try:
            while self.is_running and not self.emergency_stop:
                await self.run_trading_cycle()
                
                # Wait before next cycle
                await asyncio.sleep(60)  # 1 minute between cycles
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading error: {e}")
        finally:
            self.is_running = False
            logger.info("🛑 Autonomous trading stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop,
            'last_update': self.last_update.isoformat(),
            'portfolio_value': self.risk_agent.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'active_positions': len(self.execution_agent.active_positions),
            'strategy_weights': self.strategy_agent.strategy_weights,
            'agent_performance': {
                'data': {
                    'success_rate': self.data_agent.performance.success_rate,
                    'avg_latency_ms': self.data_agent.performance.avg_latency_ms,
                    'error_count': self.data_agent.performance.error_count
                },
                'strategy': {
                    'success_rate': self.strategy_agent.performance.success_rate,
                    'avg_latency_ms': self.strategy_agent.performance.avg_latency_ms,
                    'error_count': self.strategy_agent.performance.error_count
                },
                'risk': {
                    'success_rate': self.risk_agent.performance.success_rate,
                    'avg_latency_ms': self.risk_agent.performance.avg_latency_ms,
                    'error_count': self.risk_agent.performance.error_count
                },
                'execution': {
                    'success_rate': self.execution_agent.performance.success_rate,
                    'avg_latency_ms': self.execution_agent.performance.avg_latency_ms,
                    'error_count': self.execution_agent.performance.error_count
                }
            }
        }
    
    def emergency_stop_system(self):
        """Emergency stop the entire system."""
        logger.warning("🚨 EMERGENCY STOP triggered!")
        self.emergency_stop = True
        self.is_running = False
        
        # Close all positions if possible
        # This would need to be implemented based on the exchange API


async def main():
    """Main function to run the autonomous MCP agent."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = MasterAgentConfig(
        max_concurrent_positions=3,
        max_position_size_usd=100.0,
        min_position_size_usd=1.0,
        max_daily_loss_pct=0.10,
        max_portfolio_risk_pct=0.20,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        learning_rate=0.01,
        enable_mcp=True
    )
    
    # Create and run master agent
    agent = AutonomousMCPAgent(config)
    
    try:
        await agent.initialize()
        await agent.run_autonomous_trading()
    except Exception as e:
        logger.error(f"Master agent failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
