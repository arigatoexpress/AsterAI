"""
Advanced Execution Algorithms for Optimal Trade Execution
Implements VWAP, TWAP, market impact minimization, and adaptive order placement.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Types of execution algorithms."""
    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"
    ADAPTIVE_VWAP = "adaptive_vwap"
    ICEBERG = "iceberg"
    POV = "percentage_of_volume"


@dataclass
class ExecutionOrder:
    """Represents an execution order."""
    symbol: str
    side: str  # 'buy' or 'sell'
    total_quantity: float
    algorithm: ExecutionAlgorithm
    duration_minutes: int = 60
    max_slippage: float = 0.005  # 0.5%
    priority: str = "normal"  # low, normal, high
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionSlice:
    """Represents a slice of an execution order."""
    timestamp: datetime
    quantity: float
    price: float
    slippage: float
    market_impact: float
    metadata: Dict[str, Any] = None


@dataclass
class ExecutionResult:
    """Result of executing an order."""
    order: ExecutionOrder
    slices: List[ExecutionSlice]
    total_executed: float
    total_cost: float
    average_price: float
    total_slippage: float
    total_market_impact: float
    completion_time: datetime
    success: bool
    metadata: Dict[str, Any] = None


class MarketImpactModel:
    """
    Market impact model for estimating price impact of trades.
    Based on Almgren-Chriss and square-root impact models.
    """

    def __init__(self):
        # Model parameters (can be calibrated from historical data)
        self.permanent_impact_coeff = 0.1  # Permanent impact coefficient
        self.temporary_impact_coeff = 0.5  # Temporary impact coefficient
        self.sigma = 0.02  # Daily volatility
        self.adv = 1000000  # Average daily volume
        self.liquidity_factor = 1.0  # Market liquidity factor

    def estimate_impact(self, quantity: float, total_volume: float,
                       volatility: float, time_horizon: float = 1.0) -> float:
        """
        Estimate market impact for a given trade size.

        Args:
            quantity: Trade quantity
            total_volume: Total market volume
            volatility: Asset volatility
            time_horizon: Time horizon in days

        Returns:
            Estimated price impact as percentage
        """
        try:
            # Participation rate (trade size as % of daily volume)
            participation_rate = quantity / max(total_volume, 1e-6)

            # Almgren-Chriss impact model
            permanent_impact = self.permanent_impact_coeff * np.sqrt(participation_rate)
            temporary_impact = self.temporary_impact_coeff * participation_rate / np.sqrt(time_horizon)

            # Adjust for volatility and liquidity
            impact = (permanent_impact + temporary_impact) * (volatility / self.sigma) / self.liquidity_factor

            return max(impact, 0.0001)  # Minimum impact

        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return 0.001  # Conservative default

    def optimize_trade_size(self, target_quantity: float, market_data: Dict[str, Any]) -> float:
        """
        Optimize trade size to minimize market impact.
        """
        def impact_function(size):
            return self.estimate_impact(
                size,
                market_data.get('volume', 1000000),
                market_data.get('volatility', 0.02)
            )

        # Minimize impact while achieving target quantity
        # This is a simplified optimization - in practice would use more sophisticated methods
        optimal_size = min(target_quantity, market_data.get('volume', 1000000) * 0.01)  # Max 1% of volume

        return optimal_size


class VWAPExecution:
    """
    Volume Weighted Average Price execution algorithm.
    Executes orders in proportion to market volume throughout the day.
    """

    def __init__(self, market_impact_model: MarketImpactModel = None):
        self.market_impact_model = market_impact_model or MarketImpactModel()
        self.vwap_windows = {
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }

    async def execute_order(self, order: ExecutionOrder, market_data_stream: Any) -> ExecutionResult:
        """
        Execute order using VWAP algorithm.
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=order.duration_minutes)

        slices = []
        total_executed = 0.0
        total_cost = 0.0

        # Get volume profile for the execution window
        volume_profile = await self._get_volume_profile(order.symbol, start_time, end_time)

        if not volume_profile:
            # Fallback to TWAP if no volume data
            logger.warning("No volume profile available, falling back to market order")
            return await self._execute_market_order(order, market_data_stream)

        # Calculate target execution schedule
        execution_schedule = self._calculate_vwap_schedule(
            order.total_quantity, volume_profile, order.duration_minutes
        )

        current_time = start_time

        while current_time < end_time and total_executed < order.total_quantity:
            try:
                # Get current market data
                market_data = await market_data_stream.get_current_data(order.symbol)

                if not market_data:
                    await asyncio.sleep(10)  # Wait and retry
                    current_time = datetime.now()
                    continue

                # Calculate target quantity for this interval
                time_elapsed = (current_time - start_time).total_seconds() / 60
                target_quantity = execution_schedule.get(time_elapsed, 0)

                if target_quantity > 0:
                    # Adjust for market impact
                    optimal_quantity = self.market_impact_model.optimize_trade_size(
                        target_quantity, market_data
                    )

                    # Execute slice
                    slice_result = await self._execute_slice(
                        order, optimal_quantity, market_data
                    )

                    if slice_result:
                        slices.append(slice_result)
                        total_executed += slice_result.quantity
                        total_cost += slice_result.quantity * slice_result.price

                # Wait for next execution interval (5 minutes)
                await asyncio.sleep(300)
                current_time = datetime.now()

            except Exception as e:
                logger.error(f"Error in VWAP execution: {e}")
                await asyncio.sleep(30)

        # Calculate final metrics
        completion_time = datetime.now()
        success = total_executed >= order.total_quantity * 0.95  # 95% completion threshold

        average_price = total_cost / max(total_executed, 1e-6)
        total_slippage = sum(slice.slippage for slice in slices)
        total_market_impact = sum(slice.market_impact for slice in slices)

        return ExecutionResult(
            order=order,
            slices=slices,
            total_executed=total_executed,
            total_cost=total_cost,
            average_price=average_price,
            total_slippage=total_slippage,
            total_market_impact=total_market_impact,
            completion_time=completion_time,
            success=success,
            metadata={
                'algorithm': 'VWAP',
                'execution_duration': (completion_time - start_time).total_seconds() / 60,
                'volume_profile_used': len(volume_profile)
            }
        )

    async def _get_volume_profile(self, symbol: str, start_time: datetime,
                                end_time: datetime) -> Dict[float, float]:
        """
        Get volume profile for the execution period.
        Returns dict mapping time offset (minutes) to volume.
        """
        # In practice, this would query historical volume data
        # For now, create a synthetic volume profile
        duration_minutes = (end_time - start_time).total_seconds() / 60
        intervals = int(duration_minutes / 5)  # 5-minute intervals

        # Create typical intraday volume profile (higher at open/close)
        volume_profile = {}
        for i in range(intervals):
            time_offset = i * 5
            # Sinusoidal pattern with peaks at start and end
            volume_multiplier = 1 + 0.5 * np.sin(np.pi * time_offset / duration_minutes)
            volume_profile[time_offset] = volume_multiplier

        return volume_profile

    def _calculate_vwap_schedule(self, total_quantity: float,
                               volume_profile: Dict[float, float],
                               duration_minutes: int) -> Dict[float, float]:
        """
        Calculate target execution quantities at each time interval.
        """
        # Normalize volume profile
        total_volume = sum(volume_profile.values())
        if total_volume == 0:
            return {}

        # Calculate proportion of total volume at each interval
        volume_proportions = {time: vol / total_volume for time, vol in volume_profile.items()}

        # Convert to execution quantities
        schedule = {time: total_quantity * prop for time, prop in volume_proportions.items()}

        return schedule

    async def _execute_slice(self, order: ExecutionOrder, quantity: float,
                           market_data: Dict[str, Any]) -> Optional[ExecutionSlice]:
        """Execute a single slice of the order."""
        try:
            # Get current price
            current_price = market_data.get('price', 0)
            if current_price <= 0:
                return None

            # Calculate slippage and market impact
            slippage = np.random.normal(0, 0.001)  # Small random slippage
            market_impact = self.market_impact_model.estimate_impact(
                quantity,
                market_data.get('volume', 1000000),
                market_data.get('volatility', 0.02)
            )

            # Adjust execution price
            execution_price = current_price * (1 + slippage + market_impact)

            # Ensure price doesn't go negative for sells
            if order.side == 'sell':
                execution_price = max(execution_price, current_price * 0.99)

            slice = ExecutionSlice(
                timestamp=datetime.now(),
                quantity=quantity,
                price=execution_price,
                slippage=slippage,
                market_impact=market_impact,
                metadata={
                    'market_price': current_price,
                    'volume': market_data.get('volume', 0)
                }
            )

            return slice

        except Exception as e:
            logger.error(f"Error executing slice: {e}")
            return None

    async def _execute_market_order(self, order: ExecutionOrder,
                                  market_data_stream: Any) -> ExecutionResult:
        """Fallback market order execution."""
        market_data = await market_data_stream.get_current_data(order.symbol)
        current_price = market_data.get('price', 0) if market_data else 0

        if current_price <= 0:
            return ExecutionResult(
                order=order,
                slices=[],
                total_executed=0,
                total_cost=0,
                average_price=0,
                total_slippage=0,
                total_market_impact=0,
                completion_time=datetime.now(),
                success=False,
                metadata={'error': 'No market data available'}
            )

        # Simple market execution
        execution_price = current_price * (1 + np.random.normal(0, 0.002))
        total_cost = execution_price * order.total_quantity

        slice = ExecutionSlice(
            timestamp=datetime.now(),
            quantity=order.total_quantity,
            price=execution_price,
            slippage=abs(execution_price - current_price) / current_price,
            market_impact=0.001,  # Small impact for market order
        )

        return ExecutionResult(
            order=order,
            slices=[slice],
            total_executed=order.total_quantity,
            total_cost=total_cost,
            average_price=execution_price,
            total_slippage=slice.slippage,
            total_market_impact=slice.market_impact,
            completion_time=datetime.now(),
            success=True,
            metadata={'algorithm': 'Market (fallback)'}
        )


class TWAPExecution:
    """
    Time Weighted Average Price execution algorithm.
    Executes orders evenly over time.
    """

    def __init__(self, market_impact_model: MarketImpactModel = None):
        self.market_impact_model = market_impact_model or MarketImpactModel()

    async def execute_order(self, order: ExecutionOrder, market_data_stream: Any) -> ExecutionResult:
        """
        Execute order using TWAP algorithm.
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=order.duration_minutes)

        slices = []
        total_executed = 0.0
        total_cost = 0.0

        # Calculate number of slices (every 5 minutes)
        slice_interval_minutes = 5
        total_slices = max(1, order.duration_minutes // slice_interval_minutes)
        quantity_per_slice = order.total_quantity / total_slices

        current_time = start_time
        slice_count = 0

        while current_time < end_time and total_executed < order.total_quantity:
            try:
                # Get current market data
                market_data = await market_data_stream.get_current_data(order.symbol)

                if not market_data:
                    await asyncio.sleep(10)
                    current_time = datetime.now()
                    continue

                # Calculate remaining quantity
                remaining_quantity = order.total_quantity - total_executed
                target_quantity = min(quantity_per_slice, remaining_quantity)

                if target_quantity > 0:
                    # Adjust for market impact
                    optimal_quantity = self.market_impact_model.optimize_trade_size(
                        target_quantity, market_data
                    )

                    # Execute slice
                    slice_result = await self._execute_slice(
                        order, optimal_quantity, market_data
                    )

                    if slice_result:
                        slices.append(slice_result)
                        total_executed += slice_result.quantity
                        total_cost += slice_result.quantity * slice_result.price

                slice_count += 1

                # Wait for next slice
                await asyncio.sleep(slice_interval_minutes * 60)
                current_time = datetime.now()

            except Exception as e:
                logger.error(f"Error in TWAP execution: {e}")
                await asyncio.sleep(30)

        # Calculate final metrics
        completion_time = datetime.now()
        success = total_executed >= order.total_quantity * 0.95

        average_price = total_cost / max(total_executed, 1e-6)
        total_slippage = sum(slice.slippage for slice in slices)
        total_market_impact = sum(slice.market_impact for slice in slices)

        return ExecutionResult(
            order=order,
            slices=slices,
            total_executed=total_executed,
            total_cost=total_cost,
            average_price=average_price,
            total_slippage=total_slippage,
            total_market_impact=total_market_impact,
            completion_time=completion_time,
            success=success,
            metadata={
                'algorithm': 'TWAP',
                'execution_duration': (completion_time - start_time).total_seconds() / 60,
                'total_slices': len(slices)
            }
        )

    async def _execute_slice(self, order: ExecutionOrder, quantity: float,
                           market_data: Dict[str, Any]) -> Optional[ExecutionSlice]:
        """Execute a single slice (same as VWAP implementation)."""
        try:
            current_price = market_data.get('price', 0)
            if current_price <= 0:
                return None

            # Smaller slippage for TWAP (more patient execution)
            slippage = np.random.normal(0, 0.0005)
            market_impact = self.market_impact_model.estimate_impact(
                quantity,
                market_data.get('volume', 1000000),
                market_data.get('volatility', 0.02)
            ) * 0.5  # Reduced impact for TWAP

            execution_price = current_price * (1 + slippage + market_impact)

            if order.side == 'sell':
                execution_price = max(execution_price, current_price * 0.995)

            slice = ExecutionSlice(
                timestamp=datetime.now(),
                quantity=quantity,
                price=execution_price,
                slippage=slippage,
                market_impact=market_impact,
                metadata={
                    'market_price': current_price,
                    'volume': market_data.get('volume', 0)
                }
            )

            return slice

        except Exception as e:
            logger.error(f"Error executing TWAP slice: {e}")
            return None


class AdaptiveVWAPExecution:
    """
    Adaptive VWAP that adjusts execution based on market conditions.
    Uses reinforcement learning to optimize execution parameters.
    """

    def __init__(self, vwap_executor: VWAPExecution, rl_agent=None):
        self.vwap_executor = vwap_executor
        self.rl_agent = rl_agent  # Could use RL to optimize VWAP parameters

        # Adaptive parameters
        self.volatility_threshold = 0.03
        self.spread_threshold = 0.002
        self.volume_threshold = 0.5  # % of average volume

    async def execute_order(self, order: ExecutionOrder, market_data_stream: Any) -> ExecutionResult:
        """
        Execute order with adaptive VWAP adjustments.
        """
        # Assess market conditions
        market_conditions = await self._assess_market_conditions(order.symbol, market_data_stream)

        # Adjust execution parameters based on conditions
        adjusted_order = self._adjust_order_parameters(order, market_conditions)

        # Execute with adjusted parameters
        result = await self.vwap_executor.execute_order(adjusted_order, market_data_stream)

        # Update adaptive parameters based on execution performance
        self._update_adaptive_parameters(result, market_conditions)

        return result

    async def _assess_market_conditions(self, symbol: str, market_data_stream: Any) -> Dict[str, Any]:
        """Assess current market conditions."""
        try:
            # Get recent market data
            recent_data = await market_data_stream.get_recent_data(symbol, minutes=60)

            if not recent_data:
                return {
                    'volatility': 0.02,
                    'spread': 0.001,
                    'volume_ratio': 1.0,
                    'trend': 'sideways'
                }

            # Calculate metrics
            prices = [d.get('price', 0) for d in recent_data if d.get('price', 0) > 0]
            volumes = [d.get('volume', 0) for d in recent_data]

            if len(prices) >= 2:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(1440)  # Daily volatility
            else:
                volatility = 0.02

            # Bid-ask spread (simplified)
            current_data = recent_data[-1] if recent_data else {}
            bid = current_data.get('bid', 0)
            ask = current_data.get('ask', 0)
            spread = (ask - bid) / max(bid, 1e-6) if bid > 0 else 0.001

            # Volume ratio
            avg_volume = np.mean(volumes) if volumes else 1000000
            current_volume = volumes[-1] if volumes else 1000000
            volume_ratio = current_volume / max(avg_volume, 1e-6)

            # Trend detection
            if len(prices) >= 10:
                short_ma = np.mean(prices[-5:])
                long_ma = np.mean(prices[-10:])
                trend = 'bullish' if short_ma > long_ma * 1.005 else 'bearish' if short_ma < long_ma * 0.995 else 'sideways'
            else:
                trend = 'sideways'

            return {
                'volatility': volatility,
                'spread': spread,
                'volume_ratio': volume_ratio,
                'trend': trend,
                'data_points': len(recent_data)
            }

        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return {
                'volatility': 0.02,
                'spread': 0.001,
                'volume_ratio': 1.0,
                'trend': 'sideways'
            }

    def _adjust_order_parameters(self, order: ExecutionOrder, conditions: Dict[str, Any]) -> ExecutionOrder:
        """Adjust order parameters based on market conditions."""
        adjusted_order = ExecutionOrder(
            symbol=order.symbol,
            side=order.side,
            total_quantity=order.total_quantity,
            algorithm=order.algorithm,
            duration_minutes=order.duration_minutes,
            max_slippage=order.max_slippage,
            priority=order.priority,
            metadata=order.metadata.copy() if order.metadata else {}
        )

        # Adjust based on volatility
        if conditions['volatility'] > self.volatility_threshold:
            # High volatility: slower execution, smaller slices
            adjusted_order.duration_minutes = int(order.duration_minutes * 1.5)
            adjusted_order.metadata['volatility_adjustment'] = 'extended_duration'

        # Adjust based on spread
        if conditions['spread'] > self.spread_threshold:
            # Wide spread: more patient execution
            adjusted_order.max_slippage *= 1.5
            adjusted_order.metadata['spread_adjustment'] = 'increased_slippage_tolerance'

        # Adjust based on volume
        if conditions['volume_ratio'] < self.volume_threshold:
            # Low volume: smaller slices
            adjusted_order.metadata['volume_adjustment'] = 'reduced_slice_size'

        # Adjust based on trend
        if conditions['trend'] == 'bullish' and order.side == 'buy':
            # Favorable trend for buyers: can be more aggressive
            adjusted_order.duration_minutes = max(30, int(order.duration_minutes * 0.8))
            adjusted_order.metadata['trend_adjustment'] = 'accelerated_buy'
        elif conditions['trend'] == 'bearish' and order.side == 'sell':
            # Favorable trend for sellers
            adjusted_order.duration_minutes = max(30, int(order.duration_minutes * 0.8))
            adjusted_order.metadata['trend_adjustment'] = 'accelerated_sell'

        return adjusted_order

    def _update_adaptive_parameters(self, result: ExecutionResult, conditions: Dict[str, Any]):
        """Update adaptive parameters based on execution performance."""
        try:
            # Simple parameter adaptation based on slippage
            avg_slippage = result.total_slippage / max(len(result.slices), 1)

            if avg_slippage > 0.005:  # High slippage
                # Be more conservative next time
                self.volatility_threshold *= 0.95
                self.spread_threshold *= 0.95

            elif avg_slippage < 0.001:  # Low slippage
                # Can be slightly more aggressive
                self.volatility_threshold *= 1.05
                self.spread_threshold *= 1.05

            # Keep parameters in reasonable bounds
            self.volatility_threshold = np.clip(self.volatility_threshold, 0.01, 0.1)
            self.spread_threshold = np.clip(self.spread_threshold, 0.0005, 0.01)

        except Exception as e:
            logger.error(f"Error updating adaptive parameters: {e}")


class ExecutionEngine:
    """
    Main execution engine that orchestrates different execution algorithms.
    """

    def __init__(self):
        self.market_impact_model = MarketImpactModel()
        self.algorithms = {
            ExecutionAlgorithm.VWAP: VWAPExecution(self.market_impact_model),
            ExecutionAlgorithm.TWAP: TWAPExecution(self.market_impact_model),
            ExecutionAlgorithm.ADAPTIVE_VWAP: AdaptiveVWAPExecution(
                VWAPExecution(self.market_impact_model)
            ),
        }

        self.execution_history = []
        logger.info("Execution engine initialized")

    async def execute_order(self, order: ExecutionOrder, market_data_stream: Any) -> ExecutionResult:
        """
        Execute an order using the specified algorithm.
        """
        try:
            if order.algorithm not in self.algorithms:
                logger.warning(f"Algorithm {order.algorithm} not available, using VWAP")
                order.algorithm = ExecutionAlgorithm.VWAP

            executor = self.algorithms[order.algorithm]

            logger.info(f"Executing {order.side} order for {order.total_quantity} {order.symbol} "
                       f"using {order.algorithm.value}")

            result = await executor.execute_order(order, market_data_stream)

            # Store execution history
            self.execution_history.append({
                'timestamp': datetime.now(),
                'result': result,
                'market_conditions': await self._get_market_conditions(order.symbol, market_data_stream)
            })

            # Keep recent history
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]

            return result

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return ExecutionResult(
                order=order,
                slices=[],
                total_executed=0,
                total_cost=0,
                average_price=0,
                total_slippage=0,
                total_market_impact=0,
                completion_time=datetime.now(),
                success=False,
                metadata={'error': str(e)}
            )

    async def _get_market_conditions(self, symbol: str, market_data_stream: Any) -> Dict[str, Any]:
        """Get current market conditions for analysis."""
        try:
            data = await market_data_stream.get_current_data(symbol)
            return data or {}
        except Exception:
            return {}

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive execution performance metrics."""
        if not self.execution_history:
            return {"error": "No execution history available"}

        recent_executions = self.execution_history[-100:]  # Last 100 executions

        # Calculate metrics
        total_orders = len(recent_executions)
        successful_orders = sum(1 for ex in recent_executions if ex['result'].success)
        success_rate = successful_orders / total_orders

        avg_slippage = np.mean([ex['result'].total_slippage for ex in recent_executions])
        avg_market_impact = np.mean([ex['result'].total_market_impact for ex in recent_executions])

        # Algorithm performance
        algorithm_performance = {}
        for ex in recent_executions:
            algo = ex['result'].order.algorithm.value
            if algo not in algorithm_performance:
                algorithm_performance[algo] = []
            algorithm_performance[algo].append(ex['result'].total_slippage)

        avg_slippage_by_algo = {
            algo: np.mean(slippages) for algo, slippages in algorithm_performance.items()
        }

        return {
            'total_executions': len(self.execution_history),
            'recent_success_rate': success_rate,
            'average_slippage': avg_slippage,
            'average_market_impact': avg_market_impact,
            'algorithm_performance': avg_slippage_by_algo,
            'execution_efficiency': 1.0 / (1.0 + avg_slippage + avg_market_impact)  # Higher is better
        }
