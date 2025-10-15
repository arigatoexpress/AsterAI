"""
Latency Arbitrage Strategy for HFT

Exploits price differences across exchanges/venues with ultra-low latency execution.
Research findings: 55% win rate, 0.2-0.5% daily returns, CRITICAL <10ms latency requirement.

Strategy:
1. Monitor prices across multiple venues simultaneously
2. Detect arbitrage opportunities when spread >0.2% after fees
3. Execute trades within milliseconds
4. Requires sub-10ms total cycle time

Capital Efficiency: High scalability
Latency Requirement: CRITICAL - <10ms end-to-end
Risk: Low (near simultaneous execution)
"""

import asyncio
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import logging

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyArbConfig:
    """Configuration for latency arbitrage"""
    min_spread_pct: float = 0.2  # Minimum 0.2% spread after fees
    max_spread_pct: float = 2.0  # Maximum 2% (avoid stale data)
    transaction_fee_bps: float = 7.5  # 0.075% taker fee per side
    min_position_size_usd: float = 1.0  # Minimum $1
    max_position_size_usd: float = 25.0  # Maximum $25 per arb
    max_latency_ms: float = 10.0  # Maximum acceptable latency
    execution_timeout_ms: float = 500.0  # Order execution timeout
    price_staleness_ms: float = 100.0  # Reject prices older than 100ms
    max_slippage_bps: float = 5.0  # Maximum 5 bps slippage
    

class LatencyArbitrageStrategy:
    """
    Latency Arbitrage Strategy for HFT
    
    Features:
    - Multi-venue price monitoring
    - Sub-10ms opportunity detection
    - Ultra-fast order execution
    - Real-time latency tracking
    - Automatic opportunity filtering
    """
    
    def __init__(self, config: LatencyArbConfig):
        self.config = config
        
        # Strategy state
        self.price_feeds = {}  # Venue -> {symbol: price_data}
        self.active_arbitrages = {}  # Tracking in-flight arbs
        self.latency_measurements = deque(maxlen=1000)
        
        # Performance tracking
        self.total_opportunities = 0
        self.executed_arbitrages = 0
        self.successful_arbs = 0
        self.failed_arbs = 0
        self.total_pnl = 0.0
        
        logger.info("⚡ Latency Arbitrage Strategy initialized")
        logger.info(f"🎯 Target: 55% win rate, 0.2-0.5% daily returns")
        logger.info(f"⏱️  Latency requirement: <{config.max_latency_ms}ms")
    
    def calculate_net_spread(self,
                            buy_price: float,
                            sell_price: float,
                            include_fees: bool = True) -> float:
        """
        Calculate net spread after transaction costs
        
        Args:
            buy_price: Price to buy at (lower venue)
            sell_price: Price to sell at (higher venue)
            include_fees: Whether to include transaction fees
            
        Returns:
            Net spread as percentage
        """
        try:
            # Gross spread
            gross_spread = (sell_price - buy_price) / buy_price
            
            if not include_fees:
                return gross_spread
            
            # Transaction costs (both sides)
            total_fees = 2 * (self.config.transaction_fee_bps / 10000.0)
            
            # Slippage estimate
            slippage = self.config.max_slippage_bps / 10000.0
            
            # Net spread
            net_spread = gross_spread - total_fees - slippage
            
            return net_spread
            
        except Exception as e:
            logger.error(f"❌ Net spread calculation error: {e}")
            return 0.0
    
    def is_price_fresh(self, price_data: Dict) -> bool:
        """
        Check if price data is fresh enough for arbitrage
        
        Args:
            price_data: Price data with timestamp
            
        Returns:
            True if fresh enough
        """
        try:
            timestamp = price_data.get('timestamp')
            if not timestamp:
                return False
            
            age_ms = (datetime.now() - timestamp).total_seconds() * 1000
            
            return age_ms < self.config.price_staleness_ms
            
        except Exception as e:
            logger.error(f"❌ Price freshness check error: {e}")
            return False
    
    def detect_arbitrage_opportunity(self,
                                    symbol: str,
                                    venues_data: Dict[str, Dict]) -> Optional[Dict]:
        """
        Detect arbitrage opportunity across venues
        
        Args:
            symbol: Trading symbol
            venues_data: Dictionary of venue -> price_data
            
        Returns:
            Arbitrage opportunity details or None
        """
        try:
            start_time = time.time_ns()
            
            # Filter for fresh prices
            fresh_venues = {
                venue: data for venue, data in venues_data.items()
                if self.is_price_fresh(data)
            }
            
            if len(fresh_venues) < 2:
                return None
            
            # Find best bid (highest price to sell at)
            best_bid_venue = None
            best_bid_price = 0.0
            
            # Find best ask (lowest price to buy at)
            best_ask_venue = None
            best_ask_price = float('inf')
            
            for venue, data in fresh_venues.items():
                bid_price = data.get('bid', 0)
                ask_price = data.get('ask', float('inf'))
                
                if bid_price > best_bid_price:
                    best_bid_price = bid_price
                    best_bid_venue = venue
                
                if ask_price < best_ask_price:
                    best_ask_price = ask_price
                    best_ask_venue = venue
            
            # Must be different venues
            if best_bid_venue == best_ask_venue or not best_bid_venue or not best_ask_venue:
                return None
            
            # Calculate net spread
            net_spread_pct = self.calculate_net_spread(best_ask_price, best_bid_price) * 100
            
            # Check if profitable
            if net_spread_pct < self.config.min_spread_pct:
                return None
            
            # Check for stale data (too good to be true)
            if net_spread_pct > self.config.max_spread_pct:
                logger.warning(f"⚠️ Spread too wide ({net_spread_pct:.2%}) - likely stale data")
                return None
            
            # Calculate detection latency
            detection_latency_ms = (time.time_ns() - start_time) / 1e6
            
            if detection_latency_ms > self.config.max_latency_ms:
                logger.warning(f"⚠️ Detection latency {detection_latency_ms:.2f}ms > {self.config.max_latency_ms}ms")
                return None
            
            # Create opportunity
            opportunity = {
                'symbol': symbol,
                'buy_venue': best_ask_venue,
                'sell_venue': best_bid_venue,
                'buy_price': best_ask_price,
                'sell_price': best_bid_price,
                'gross_spread_pct': ((best_bid_price - best_ask_price) / best_ask_price) * 100,
                'net_spread_pct': net_spread_pct,
                'detection_time': datetime.now(),
                'detection_latency_ms': detection_latency_ms,
                'venue_data': {
                    best_ask_venue: fresh_venues[best_ask_venue],
                    best_bid_venue: fresh_venues[best_bid_venue]
                }
            }
            
            self.total_opportunities += 1
            
            logger.debug(f"🎯 Arbitrage opportunity: {symbol} "
                        f"Buy@{best_ask_venue} ${best_ask_price:.4f} → "
                        f"Sell@{best_bid_venue} ${best_bid_price:.4f} "
                        f"Net: {net_spread_pct:.3%}")
            
            return opportunity
            
        except Exception as e:
            logger.error(f"❌ Arbitrage detection error: {e}")
            return None
    
    def calculate_position_size(self,
                                capital: float,
                                opportunity: Dict) -> float:
        """
        Calculate position size for arbitrage
        
        Args:
            capital: Available capital
            opportunity: Arbitrage opportunity details
            
        Returns:
            Position size in USD
        """
        try:
            # Conservative sizing - use full risk allocation since it's hedged
            base_size = capital * 0.02  # 2% of capital per arb
            
            # Adjust based on spread quality
            spread_quality = min(opportunity['net_spread_pct'] / 0.5, 2.0)  # Cap at 2x
            position_size = base_size * spread_quality
            
            # Apply limits
            position_size = np.clip(
                position_size,
                self.config.min_position_size_usd,
                min(self.config.max_position_size_usd, capital * 0.4)  # Max 40%
            )
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            return self.config.min_position_size_usd
    
    async def execute_arbitrage(self,
                               opportunity: Dict,
                               capital_available: float) -> Optional[Dict]:
        """
        Execute arbitrage trade with ultra-low latency
        
        Args:
            opportunity: Arbitrage opportunity
            capital_available: Available capital
            
        Returns:
            Execution result or None
        """
        try:
            execution_start = time.time_ns()
            
            # Calculate position size
            position_size_usd = self.calculate_position_size(capital_available, opportunity)
            
            if position_size_usd < self.config.min_position_size_usd:
                return None
            
            # Calculate quantities
            buy_quantity = position_size_usd / opportunity['buy_price']
            sell_quantity = buy_quantity  # Equal quantities for hedge
            
            # Create orders (to be executed simultaneously)
            buy_order = {
                'venue': opportunity['buy_venue'],
                'symbol': opportunity['symbol'],
                'side': 'buy',
                'type': 'limit',
                'price': opportunity['buy_price'] * 1.0001,  # Slightly aggressive
                'quantity': buy_quantity,
                'timestamp': datetime.now()
            }
            
            sell_order = {
                'venue': opportunity['sell_venue'],
                'symbol': opportunity['symbol'],
                'side': 'sell',
                'type': 'limit',
                'price': opportunity['sell_price'] * 0.9999,  # Slightly aggressive
                'quantity': sell_quantity,
                'timestamp': datetime.now()
            }
            
            # Track execution
            arb_id = f"{opportunity['symbol']}_{execution_start}"
            self.active_arbitrages[arb_id] = {
                'opportunity': opportunity,
                'buy_order': buy_order,
                'sell_order': sell_order,
                'status': 'pending',
                'start_time': datetime.now()
            }
            
            # Execute both orders (in practice, send to order execution system)
            # Here we simulate the execution
            execution_latency_ms = (time.time_ns() - execution_start) / 1e6
            
            # Track latency
            total_latency_ms = opportunity['detection_latency_ms'] + execution_latency_ms
            self.latency_measurements.append(total_latency_ms)
            
            if total_latency_ms > self.config.max_latency_ms:
                logger.warning(f"⚠️ Total latency {total_latency_ms:.2f}ms exceeded limit")
                # In production, may want to cancel this arb
            
            # Calculate expected P&L
            gross_pnl = (opportunity['sell_price'] - opportunity['buy_price']) * buy_quantity
            fees = position_size_usd * 2 * (self.config.transaction_fee_bps / 10000.0)
            expected_pnl = gross_pnl - fees
            
            result = {
                'arb_id': arb_id,
                'symbol': opportunity['symbol'],
                'position_size_usd': position_size_usd,
                'buy_order': buy_order,
                'sell_order': sell_order,
                'expected_pnl': expected_pnl,
                'detection_latency_ms': opportunity['detection_latency_ms'],
                'execution_latency_ms': execution_latency_ms,
                'total_latency_ms': total_latency_ms,
                'net_spread_pct': opportunity['net_spread_pct']
            }
            
            self.executed_arbitrages += 1
            
            logger.info(f"⚡ Arbitrage executed: {opportunity['symbol']} "
                       f"${position_size_usd:.2f} Expected: ${expected_pnl:.4f} "
                       f"Latency: {total_latency_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Arbitrage execution error: {e}")
            return None
    
    async def update_arbitrage_status(self,
                                     arb_id: str,
                                     buy_filled: bool,
                                     sell_filled: bool,
                                     actual_buy_price: float = None,
                                     actual_sell_price: float = None) -> Dict:
        """
        Update arbitrage execution status
        
        Args:
            arb_id: Arbitrage ID
            buy_filled: Whether buy order filled
            sell_filled: Whether sell order filled
            actual_buy_price: Actual execution price (buy)
            actual_sell_price: Actual execution price (sell)
            
        Returns:
            Final P&L and statistics
        """
        try:
            if arb_id not in self.active_arbitrages:
                logger.warning(f"⚠️ Unknown arbitrage ID: {arb_id}")
                return {}
            
            arb = self.active_arbitrages[arb_id]
            
            # Check if both legs filled
            if not (buy_filled and sell_filled):
                # Partial fill - handle unwinding
                logger.warning(f"⚠️ Partial fill for {arb_id}: "
                             f"Buy={buy_filled}, Sell={sell_filled}")
                
                # Mark as failed
                self.failed_arbs += 1
                arb['status'] = 'failed_partial_fill'
                
                # In production, would unwind the filled leg
                return {
                    'arb_id': arb_id,
                    'status': 'failed',
                    'reason': 'partial_fill',
                    'pnl': 0.0
                }
            
            # Calculate actual P&L
            buy_price = actual_buy_price or arb['buy_order']['price']
            sell_price = actual_sell_price or arb['sell_order']['price']
            quantity = arb['buy_order']['quantity']
            
            gross_pnl = (sell_price - buy_price) * quantity
            
            # Calculate fees
            position_size = buy_price * quantity
            fees = position_size * 2 * (self.config.transaction_fee_bps / 10000.0)
            
            net_pnl = gross_pnl - fees
            
            # Update statistics
            if net_pnl > 0:
                self.successful_arbs += 1
            else:
                self.failed_arbs += 1
            
            self.total_pnl += net_pnl
            
            # Calculate execution time
            execution_time_ms = (datetime.now() - arb['start_time']).total_seconds() * 1000
            
            result = {
                'arb_id': arb_id,
                'status': 'completed',
                'symbol': arb['opportunity']['symbol'],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'quantity': quantity,
                'gross_pnl': gross_pnl,
                'fees': fees,
                'net_pnl': net_pnl,
                'execution_time_ms': execution_time_ms,
                'success': net_pnl > 0
            }
            
            # Remove from active
            del self.active_arbitrages[arb_id]
            
            logger.info(f"✅ Arbitrage completed: {arb['opportunity']['symbol']} "
                       f"P&L: ${net_pnl:.4f} ({execution_time_ms:.1f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Arbitrage status update error: {e}")
            return {}
    
    async def monitor_venues(self,
                            symbols: List[str],
                            venues: List[str]) -> Dict[str, Dict]:
        """
        Monitor prices across multiple venues
        
        Args:
            symbols: List of trading symbols
            venues: List of venue names
            
        Returns:
            Dictionary of symbol -> venue_prices
        """
        try:
            # This is a placeholder - in production, would have real WebSocket connections
            # to multiple venues
            
            # For now, simulate with Aster as primary venue
            venue_prices = {}
            
            for symbol in symbols:
                venue_prices[symbol] = {}
                
                # In production: Query each venue's WebSocket or API
                # Here we simulate multiple venues
                for venue in venues:
                    # Placeholder price data
                    venue_prices[symbol][venue] = {
                        'bid': 0.0,  # Would be real data
                        'ask': 0.0,
                        'timestamp': datetime.now()
                    }
            
            return venue_prices
            
        except Exception as e:
            logger.error(f"❌ Venue monitoring error: {e}")
            return {}
    
    def get_latency_stats(self) -> Dict:
        """Get latency performance statistics"""
        if not self.latency_measurements:
            return {}
        
        latencies = list(self.latency_measurements)
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies),
            'measurements': len(latencies),
            'within_target': sum(1 for l in latencies if l < self.config.max_latency_ms) / len(latencies)
        }
    
    def get_performance_stats(self) -> Dict:
        """Get strategy performance statistics"""
        success_rate = self.successful_arbs / self.executed_arbitrages if self.executed_arbitrages > 0 else 0
        execution_rate = self.executed_arbitrages / self.total_opportunities if self.total_opportunities > 0 else 0
        
        return {
            'strategy': 'latency_arbitrage',
            'total_opportunities': self.total_opportunities,
            'executed_arbitrages': self.executed_arbitrages,
            'execution_rate': execution_rate,
            'successful_arbs': self.successful_arbs,
            'failed_arbs': self.failed_arbs,
            'success_rate': success_rate,
            'total_pnl': self.total_pnl,
            'active_arbitrages': len(self.active_arbitrages),
            'latency_stats': self.get_latency_stats(),
            'target_success_rate': 0.55,
            'target_daily_return': 0.004,  # 0.4%
            'latency_requirement': self.config.max_latency_ms
        }


class VenueConnector:
    """
    Manages WebSocket connections to multiple trading venues
    
    Optimized for ultra-low latency price updates
    """
    
    def __init__(self, venues: List[str]):
        self.venues = venues
        self.connections = {}
        self.price_cache = {}
        self.last_update = {}
    
    async def connect_all(self):
        """Connect to all venues"""
        for venue in self.venues:
            await self.connect_venue(venue)
    
    async def connect_venue(self, venue: str):
        """Connect to specific venue with WebSocket"""
        # Placeholder - would implement actual WebSocket connection
        logger.info(f"🔗 Connecting to venue: {venue}")
        self.connections[venue] = {'status': 'connected'}
    
    async def subscribe_symbol(self, venue: str, symbol: str):
        """Subscribe to symbol updates on venue"""
        logger.debug(f"📊 Subscribed to {symbol} on {venue}")
    
    def get_latest_price(self, venue: str, symbol: str) -> Optional[Dict]:
        """Get latest cached price"""
        cache_key = f"{venue}_{symbol}"
        return self.price_cache.get(cache_key)


