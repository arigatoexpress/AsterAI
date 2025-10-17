"""
MEV Protection and Advanced Slippage Management System

This module implements comprehensive MEV protection, slippage optimization,
and front-running prevention for the autonomous trading system.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import hashlib
import secrets
from concurrent.futures import ThreadPoolExecutor

from mcp_trader.execution.aster_client import AsterClient
from mcp_trader.risk.risk_manager import RiskManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MEVProtectionConfig:
    """Configuration for MEV protection"""
    max_slippage_pct: float = 0.1  # 0.1% maximum slippage
    min_liquidity_threshold: float = 10000  # Minimum liquidity required
    max_price_impact_pct: float = 0.05  # 0.05% maximum price impact
    mev_detection_window: int = 5  # seconds to detect MEV
    private_mempool_enabled: bool = True
    random_delay_range: Tuple[float, float] = (0.1, 0.5)  # Random delay in seconds
    max_retries: int = 3
    gas_price_multiplier: float = 1.1  # 10% above base gas price

@dataclass
class SlippageAnalysis:
    """Slippage analysis results"""
    expected_price: float
    actual_price: float
    slippage_pct: float
    price_impact_pct: float
    liquidity_depth: float
    market_volatility: float
    timestamp: datetime

@dataclass
class MEVThreat:
    """MEV threat detection"""
    threat_type: str  # 'frontrun', 'sandwich', 'backrun', 'arbitrage'
    confidence: float  # 0-1 confidence score
    estimated_profit: float  # Estimated MEV profit
    detection_time: datetime
    recommended_action: str  # 'delay', 'cancel', 'adjust', 'proceed'

class SlippageOptimizer:
    """Advanced slippage optimization and prediction"""
    
    def __init__(self, config: MEVProtectionConfig):
        self.config = config
        self.historical_slippage = []
        self.market_conditions = {}
        
    async def calculate_optimal_slippage(self, symbol: str, side: str, size: float, 
                                       current_price: float, order_book: Dict) -> float:
        """Calculate optimal slippage tolerance based on market conditions"""
        
        try:
            # Analyze order book depth
            liquidity_depth = self._analyze_liquidity_depth(order_book, side)
            
            # Calculate price impact
            price_impact = self._calculate_price_impact(size, order_book, side)
            
            # Get market volatility
            volatility = await self._get_market_volatility(symbol)
            
            # Calculate base slippage
            base_slippage = self._calculate_base_slippage(size, liquidity_depth, volatility)
            
            # Apply MEV protection adjustments
            mev_adjustment = self._calculate_mev_adjustment(symbol, side, size)
            
            # Calculate final slippage
            optimal_slippage = min(
                base_slippage + mev_adjustment,
                self.config.max_slippage_pct
            )
            
            # Ensure minimum slippage for small orders
            optimal_slippage = max(optimal_slippage, 0.01)  # 0.01% minimum
            
            logger.info(f"Optimal slippage for {symbol} {side} {size}: {optimal_slippage:.4f}%")
            
            return optimal_slippage
            
        except Exception as e:
            logger.error(f"Error calculating optimal slippage: {e}")
            return self.config.max_slippage_pct
    
    def _analyze_liquidity_depth(self, order_book: Dict, side: str) -> float:
        """Analyze order book liquidity depth"""
        try:
            if side == 'BUY':
                bids = order_book.get('bids', [])
                total_liquidity = sum(float(bid[1]) for bid in bids[:10])  # Top 10 levels
            else:
                asks = order_book.get('asks', [])
                total_liquidity = sum(float(ask[1]) for ask in asks[:10])  # Top 10 levels
            
            return total_liquidity
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity depth: {e}")
            return 0.0
    
    def _calculate_price_impact(self, size: float, order_book: Dict, side: str) -> float:
        """Calculate expected price impact for given order size"""
        try:
            if side == 'BUY':
                levels = order_book.get('asks', [])
            else:
                levels = order_book.get('bids', [])
            
            if not levels:
                return 0.0
            
            remaining_size = size
            total_cost = 0.0
            weighted_price = 0.0
            
            for level in levels:
                level_price = float(level[0])
                level_size = float(level[1])
                
                if remaining_size <= 0:
                    break
                
                size_to_take = min(remaining_size, level_size)
                total_cost += size_to_take * level_price
                remaining_size -= size_to_take
            
            if size > 0:
                avg_price = total_cost / size
                mid_price = (float(levels[0][0]) + float(levels[0][0])) / 2
                price_impact = abs(avg_price - mid_price) / mid_price
                return price_impact
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {e}")
            return 0.0
    
    async def _get_market_volatility(self, symbol: str) -> float:
        """Get current market volatility for symbol"""
        try:
            # This would typically fetch from your data pipeline
            # For now, return a simulated volatility
            return np.random.uniform(0.01, 0.05)  # 1-5% volatility
            
        except Exception as e:
            logger.error(f"Error getting market volatility: {e}")
            return 0.02  # Default 2% volatility
    
    def _calculate_base_slippage(self, size: float, liquidity_depth: float, volatility: float) -> float:
        """Calculate base slippage based on order size and market conditions"""
        try:
            # Size-based slippage (larger orders = more slippage)
            size_factor = min(size / liquidity_depth, 1.0) * 0.1  # Up to 0.1% for large orders
            
            # Volatility-based slippage
            volatility_factor = volatility * 0.5  # 50% of volatility
            
            # Liquidity-based slippage (lower liquidity = more slippage)
            liquidity_factor = max(0, (10000 - liquidity_depth) / 100000) * 0.05  # Up to 0.05%
            
            base_slippage = size_factor + volatility_factor + liquidity_factor
            
            return min(base_slippage, 0.5)  # Cap at 0.5%
            
        except Exception as e:
            logger.error(f"Error calculating base slippage: {e}")
            return 0.02  # Default 0.02%
    
    def _calculate_mev_adjustment(self, symbol: str, side: str, size: float) -> float:
        """Calculate additional slippage for MEV protection"""
        try:
            # MEV risk increases with order size
            mev_risk = min(size / 1000, 1.0) * 0.02  # Up to 0.02% for large orders
            
            # Time-based MEV risk (higher during volatile periods)
            current_hour = datetime.now().hour
            if current_hour in [9, 10, 14, 15, 20, 21]:  # High activity hours
                mev_risk *= 1.5
            
            return mev_risk
            
        except Exception as e:
            logger.error(f"Error calculating MEV adjustment: {e}")
            return 0.01  # Default 0.01%

class MEVDetector:
    """Advanced MEV threat detection and prevention"""
    
    def __init__(self, config: MEVProtectionConfig):
        self.config = config
        self.threat_history = []
        self.suspicious_patterns = {}
        
    async def detect_mev_threats(self, symbol: str, side: str, size: float, 
                               current_price: float, order_book: Dict) -> List[MEVThreat]:
        """Detect potential MEV threats before placing order"""
        
        threats = []
        
        try:
            # Check for front-running patterns
            frontrun_threat = await self._detect_frontrun_threat(symbol, side, size, order_book)
            if frontrun_threat:
                threats.append(frontrun_threat)
            
            # Check for sandwich attack patterns
            sandwich_threat = await self._detect_sandwich_threat(symbol, side, size, order_book)
            if sandwich_threat:
                threats.append(sandwich_threat)
            
            # Check for arbitrage opportunities
            arb_threat = await self._detect_arbitrage_threat(symbol, current_price)
            if arb_threat:
                threats.append(arb_threat)
            
            # Check for back-running patterns
            backrun_threat = await self._detect_backrun_threat(symbol, side, size, order_book)
            if backrun_threat:
                threats.append(backrun_threat)
            
            # Store threat history
            self.threat_history.extend(threats)
            if len(self.threat_history) > 1000:
                self.threat_history = self.threat_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error detecting MEV threats: {e}")
        
        return threats
    
    async def _detect_frontrun_threat(self, symbol: str, side: str, size: float, 
                                    order_book: Dict) -> Optional[MEVThreat]:
        """Detect potential front-running attacks"""
        try:
            # Check for unusual order book activity
            if side == 'BUY':
                asks = order_book.get('asks', [])
                if len(asks) < 5:
                    return None
                
                # Check for large orders at better prices
                best_ask = float(asks[0][0])
                second_best = float(asks[1][0]) if len(asks) > 1 else best_ask
                
                if second_best < best_ask * 0.999:  # 0.1% better price
                    confidence = min(size / 1000, 1.0)  # Higher confidence for larger orders
                    
                    return MEVThreat(
                        threat_type='frontrun',
                        confidence=confidence,
                        estimated_profit=abs(best_ask - second_best) * size,
                        detection_time=datetime.now(),
                        recommended_action='delay' if confidence > 0.7 else 'proceed'
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting frontrun threat: {e}")
            return None
    
    async def _detect_sandwich_threat(self, symbol: str, side: str, size: float, 
                                    order_book: Dict) -> Optional[MEVThreat]:
        """Detect potential sandwich attacks"""
        try:
            # Check for orders on both sides of the spread
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if len(bids) < 3 or len(asks) < 3:
                return None
            
            # Check for similar sized orders on both sides
            bid_sizes = [float(bid[1]) for bid in bids[:3]]
            ask_sizes = [float(ask[1]) for ask in asks[:3]]
            
            # Look for matching sizes (potential sandwich setup)
            for bid_size in bid_sizes:
                for ask_size in ask_sizes:
                    if abs(bid_size - ask_size) / max(bid_size, ask_size) < 0.1:  # Within 10%
                        if bid_size > size * 0.5:  # Larger than our order
                            confidence = min(size / 500, 1.0)
                            
                            return MEVThreat(
                                threat_type='sandwich',
                                confidence=confidence,
                                estimated_profit=abs(float(bids[0][0]) - float(asks[0][0])) * min(bid_size, ask_size),
                                detection_time=datetime.now(),
                                recommended_action='cancel' if confidence > 0.8 else 'delay'
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting sandwich threat: {e}")
            return None
    
    async def _detect_arbitrage_threat(self, symbol: str, current_price: float) -> Optional[MEVThreat]:
        """Detect potential arbitrage opportunities that could be exploited"""
        try:
            # This would typically check multiple exchanges
            # For now, simulate arbitrage detection
            if np.random.random() < 0.1:  # 10% chance of detecting arbitrage
                confidence = np.random.uniform(0.3, 0.8)
                estimated_profit = np.random.uniform(0.1, 2.0)
                
                return MEVThreat(
                    threat_type='arbitrage',
                    confidence=confidence,
                    estimated_profit=estimated_profit,
                    detection_time=datetime.now(),
                    recommended_action='delay' if confidence > 0.6 else 'proceed'
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage threat: {e}")
            return None
    
    async def _detect_backrun_threat(self, symbol: str, side: str, size: float, 
                                   order_book: Dict) -> Optional[MEVThreat]:
        """Detect potential back-running attacks"""
        try:
            # Check for orders that could profit from our trade
            if side == 'BUY':
                # Look for large sell orders that could benefit from price increase
                asks = order_book.get('asks', [])
                if len(asks) > 0:
                    large_ask = max(asks, key=lambda x: float(x[1]))
                    if float(large_ask[1]) > size * 2:  # Much larger than our order
                        confidence = min(size / 2000, 1.0)
                        
                        return MEVThreat(
                            threat_type='backrun',
                            confidence=confidence,
                            estimated_profit=float(large_ask[1]) * 0.001,  # 0.1% profit estimate
                            detection_time=datetime.now(),
                            recommended_action='delay' if confidence > 0.6 else 'proceed'
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting backrun threat: {e}")
            return None

class PrivateMempoolManager:
    """Manages private mempool and transaction privacy"""
    
    def __init__(self, config: MEVProtectionConfig):
        self.config = config
        self.private_orders = {}
        self.order_hashes = set()
        
    async def create_private_order(self, symbol: str, side: str, size: float, 
                                 price: float, slippage: float) -> Dict[str, Any]:
        """Create a private order with MEV protection"""
        
        try:
            # Generate unique order hash
            order_data = f"{symbol}{side}{size}{price}{slippage}{time.time()}"
            order_hash = hashlib.sha256(order_data.encode()).hexdigest()
            
            # Add random delay to prevent timing attacks
            delay = np.random.uniform(*self.config.random_delay_range)
            await asyncio.sleep(delay)
            
            # Create private order
            private_order = {
                'hash': order_hash,
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price,
                'slippage': slippage,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(minutes=5),
                'private': True,
                'mev_protected': True
            }
            
            # Store private order
            self.private_orders[order_hash] = private_order
            self.order_hashes.add(order_hash)
            
            logger.info(f"Created private order: {order_hash[:8]}...")
            
            return private_order
            
        except Exception as e:
            logger.error(f"Error creating private order: {e}")
            return {}
    
    async def execute_private_order(self, order_hash: str, aster_client: AsterClient) -> Dict[str, Any]:
        """Execute private order with additional MEV protection"""
        
        try:
            if order_hash not in self.private_orders:
                raise ValueError("Order not found in private mempool")
            
            order = self.private_orders[order_hash]
            
            # Check if order is still valid
            if datetime.now() > order['expires_at']:
                del self.private_orders[order_hash]
                raise ValueError("Order expired")
            
            # Execute with private mempool
            result = await aster_client.place_order(
                symbol=order['symbol'],
                side=order['side'],
                type='LIMIT',
                quantity=order['size'],
                price=order['price'],
                timeInForce='GTC',
                private=True  # Use private mempool
            )
            
            # Clean up
            if order_hash in self.private_orders:
                del self.private_orders[order_hash]
            
            logger.info(f"Executed private order: {order_hash[:8]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing private order: {e}")
            return {}

class MEVProtectionSystem:
    """Main MEV protection system that orchestrates all components"""
    
    def __init__(self, config: MEVProtectionConfig):
        self.config = config
        self.slippage_optimizer = SlippageOptimizer(config)
        self.mev_detector = MEVDetector(config)
        self.private_mempool = PrivateMempoolManager(config)
        self.protection_stats = {
            'orders_protected': 0,
            'mev_threats_detected': 0,
            'orders_cancelled': 0,
            'orders_delayed': 0,
            'total_slippage_saved': 0.0
        }
    
    async def protect_order(self, symbol: str, side: str, size: float, 
                          current_price: float, order_book: Dict, 
                          aster_client: AsterClient) -> Dict[str, Any]:
        """Protect order from MEV attacks and optimize slippage"""
        
        try:
            logger.info(f"Protecting order: {side} {size} {symbol}")
            
            # 1. Detect MEV threats
            threats = await self.mev_detector.detect_mev_threats(
                symbol, side, size, current_price, order_book
            )
            
            # 2. Analyze threats and decide action
            action = await self._analyze_threats(threats)
            
            if action == 'cancel':
                self.protection_stats['orders_cancelled'] += 1
                return {'status': 'cancelled', 'reason': 'MEV threat detected'}
            
            if action == 'delay':
                delay_time = np.random.uniform(1, 5)  # 1-5 second delay
                await asyncio.sleep(delay_time)
                self.protection_stats['orders_delayed'] += 1
            
            # 3. Calculate optimal slippage
            optimal_slippage = await self.slippage_optimizer.calculate_optimal_slippage(
                symbol, side, size, current_price, order_book
            )
            
            # 4. Create private order
            private_order = await self.private_mempool.create_private_order(
                symbol, side, size, current_price, optimal_slippage
            )
            
            # 5. Execute with protection
            result = await self.private_mempool.execute_private_order(
                private_order['hash'], aster_client
            )
            
            # 6. Update stats
            self.protection_stats['orders_protected'] += 1
            self.protection_stats['mev_threats_detected'] += len(threats)
            
            if 'slippage' in result:
                self.protection_stats['total_slippage_saved'] += result['slippage']
            
            logger.info(f"Order protected successfully: {result.get('orderId', 'N/A')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error protecting order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _analyze_threats(self, threats: List[MEVThreat]) -> str:
        """Analyze MEV threats and determine action"""
        
        if not threats:
            return 'proceed'
        
        # Count threat types
        threat_counts = {}
        for threat in threats:
            threat_counts[threat.threat_type] = threat_counts.get(threat.threat_type, 0) + 1
        
        # Check for high-confidence threats
        high_confidence_threats = [t for t in threats if t.confidence > 0.8]
        
        if high_confidence_threats:
            # Check for sandwich attacks (most dangerous)
            if any(t.threat_type == 'sandwich' for t in high_confidence_threats):
                return 'cancel'
            
            # Check for multiple threats
            if len(high_confidence_threats) > 1:
                return 'cancel'
            
            # Single high-confidence threat
            return 'delay'
        
        # Medium confidence threats
        medium_confidence_threats = [t for t in threats if 0.5 < t.confidence <= 0.8]
        
        if medium_confidence_threats:
            return 'delay'
        
        # Low confidence threats
        return 'proceed'
    
    def get_protection_stats(self) -> Dict[str, Any]:
        """Get MEV protection statistics"""
        return {
            **self.protection_stats,
            'threat_detection_rate': (
                self.protection_stats['mev_threats_detected'] / 
                max(self.protection_stats['orders_protected'], 1)
            ),
            'cancellation_rate': (
                self.protection_stats['orders_cancelled'] / 
                max(self.protection_stats['orders_protected'], 1)
            ),
            'delay_rate': (
                self.protection_stats['orders_delayed'] / 
                max(self.protection_stats['orders_protected'], 1)
            )
        }

# Example usage and testing
async def main():
    """Test the MEV protection system"""
    
    # Create configuration
    config = MEVProtectionConfig(
        max_slippage_pct=0.1,
        min_liquidity_threshold=10000,
        max_price_impact_pct=0.05,
        mev_detection_window=5,
        private_mempool_enabled=True,
        random_delay_range=(0.1, 0.5),
        max_retries=3
    )
    
    # Create MEV protection system
    mev_system = MEVProtectionSystem(config)
    
    # Simulate order book data
    order_book = {
        'bids': [['45000', '0.5'], ['44999', '1.0'], ['44998', '2.0']],
        'asks': [['45001', '0.5'], ['45002', '1.0'], ['45003', '2.0']]
    }
    
    # Test order protection
    result = await mev_system.protect_order(
        symbol='BTCUSDT',
        side='BUY',
        size=0.1,
        current_price=45000,
        order_book=order_book,
        aster_client=None  # Would be actual client in production
    )
    
    print("MEV Protection Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Print protection stats
    stats = mev_system.get_protection_stats()
    print("\nProtection Statistics:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
