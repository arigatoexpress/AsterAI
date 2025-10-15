"""
Ultra-Low Latency HFT Trading Agent for Aster DEX

Optimized for transforming $50 into $500k through high-frequency trading.
Utilizes RTX 5070Ti GPU acceleration for real-time processing.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from google.cloud import pubsub_v1

from ..config import get_settings
from ..execution.aster_client import AsterClient
from ..data.aster_feed import AsterDataFeed
from ..features.gpu_features import HFTFeatureEngine
from ..trading.types import PortfolioState, MarketState, Order, Position
from ..risk.risk_manager import RiskManager
from ..logging_utils import get_logger

logger = get_logger(__name__)


class HFTStrategy(Enum):
    """HFT Strategy Types"""
    STATISTICAL_ARBITRAGE = "stat_arb"
    MARKET_MAKING = "market_making"
    MOMENTUM_TRADING = "momentum"
    ORDER_FLOW = "order_flow"
    LATENCY_ARBITRAGE = "latency_arb"


@dataclass
class HFTAgentConfig:
    """Configuration for HFT Trading Agent"""
    initial_balance: float = 50.0
    max_position_size_usd: float = 25.0  # Max $25 per position for $50 account
    min_order_size_usd: float = 1.0     # Minimum $1 orders
    max_open_positions: int = 10        # Maximum concurrent positions
    target_profit_threshold: float = 0.001  # 0.1% profit target per trade
    stop_loss_threshold: float = 0.002    # 0.2% stop loss
    max_daily_loss: float = 10.0        # Max $10 daily loss
    trading_fee_rate: float = 0.0005    # 0.05% trading fee
    latency_threshold_ms: float = 1.0   # Max 1ms latency
    gpu_acceleration: bool = True       # Enable RTX 5070Ti acceleration


class HFTTradingAgent:
    """
    Ultra-Low Latency HFT Trading Agent

    Features:
    - RTX 5070Ti GPU acceleration
    - Sub-millisecond order execution
    - Real-time market making
    - Statistical arbitrage
    - Order flow analysis
    """

    def __init__(self, config: HFTAgentConfig):
        self.config = config
        self.aster_client: Optional[AsterClient] = None
        self.data_feed: Optional[AsterDataFeed] = None
        self.risk_manager: Optional[RiskManager] = None

        # Portfolio state
        self.portfolio_state = PortfolioState(
            timestamp=datetime.now(),
            total_balance=config.initial_balance,
            available_balance=config.initial_balance
        )

        # Active positions and orders
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.active_strategies: Dict[HFTStrategy, bool] = {
            strategy: True for strategy in HFTStrategy
        }

        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_volume = 0.0

        # GPU acceleration setup
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.gpu_acceleration else 'cpu')
        self.setup_gpu_optimization()

        # ML models for prediction
        self.price_predictor = None
        self.order_flow_analyzer = None
        self.arbitrage_detector = None

        # Market data cache
        self.market_data_cache = {}
        self.last_update = datetime.now()

        # Sentiment analysis
        self.sentiment_cache = {}  # Symbol -> sentiment score
        self.sentiment_subscriber = None
        self.sentiment_executor = ThreadPoolExecutor(max_workers=1)

        # Feature engineering with sentiment
        self.feature_engine = HFTFeatureEngine(self.device)

        logger.info(f"🧠 HFT Agent initialized with ${config.initial_balance} balance")
        logger.info(f"🎯 Target: $500k through HFT on Aster DEX")
        logger.info(f"⚡ GPU Acceleration: {self.device}")

    def setup_gpu_optimization(self):
        """Setup RTX 5070Ti GPU optimizations"""
        if self.device.type == 'cuda':
            # Enable TF32 for Ada Lovelace architecture
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable cuDNN benchmarking
            torch.backends.cudnn.benchmark = True

            # Set memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of 16GB VRAM

            # Enable async data loading
            torch.cuda.set_device(0)

            logger.info("🎮 RTX 5070Ti GPU optimizations enabled")
            logger.info(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    async def initialize(self):
        """Initialize HFT trading agent"""
        try:
            settings = get_settings()

            # Initialize Aster client
            self.aster_client = AsterClient(
                settings.aster_api_key,
                settings.aster_api_secret
            )

            # Initialize data feed with ultra-low latency
            self.data_feed = AsterDataFeed()
            await self.data_feed.initialize()

            # Initialize risk manager
            self.risk_manager = RiskManager(settings)

            # Initialize ML models
            await self.initialize_ml_models()

            # Connect to Aster DEX
            await self.aster_client.connect()
            logger.info("✅ Connected to Aster DEX")

            # Start real-time data streaming
            await self.data_feed.start_streaming()

            # Initialize sentiment analysis
            await self.initialize_sentiment_analysis()

            logger.info("🚀 HFT Agent fully initialized and ready for trading")

        except Exception as e:
            logger.error(f"❌ HFT Agent initialization failed: {e}")
            raise

    async def initialize_sentiment_analysis(self):
        """Initialize Pub/Sub sentiment analysis subscription"""
        try:
            settings = get_settings()

            # Initialize Pub/Sub subscriber
            self.sentiment_subscriber = pubsub_v1.SubscriberClient()
            project_id = settings.gcp_project_id or "hft-aster-trader"
            subscription_id = "hft-sentiment-sub"

            subscription_path = self.sentiment_subscriber.subscription_path(
                project_id, subscription_id
            )

            # Start sentiment subscriber in background thread
            self.sentiment_executor.submit(self._sentiment_subscriber_loop, subscription_path)

            logger.info("📊 Sentiment analysis initialized with Pub/Sub")

        except Exception as e:
            logger.warning(f"⚠️ Sentiment analysis initialization failed: {e} (continuing without sentiment)")

    def _sentiment_subscriber_loop(self, subscription_path):
        """Background thread for sentiment subscription"""
        try:
            def callback(message):
                try:
                    # Parse sentiment message
                    data = json.loads(message.data.decode('utf-8'))
                    symbol = data.get('symbol', 'market-sentiment')
                    sentiment_score = data.get('sentiment_score', 0.0)
                    timestamp = data.get('timestamp', time.time())

                    # Update sentiment cache
                    self.sentiment_cache[symbol] = {
                        'score': sentiment_score,
                        'timestamp': timestamp,
                        'source': data.get('source', 'unknown')
                    }

                    logger.debug(f"📊 Updated sentiment for {symbol}: {sentiment_score:.3f}")

                    message.ack()

                except Exception as e:
                    logger.error(f"❌ Error processing sentiment message: {e}")
                    message.nack()

            # Subscribe to topic
            streaming_pull_future = self.sentiment_subscriber.subscribe(
                subscription_path, callback=callback
            )

            logger.info("🎧 Listening for sentiment updates...")

            # Keep the subscriber running
            try:
                streaming_pull_future.result()
            except KeyboardInterrupt:
                streaming_pull_future.cancel()
                logger.info("🛑 Sentiment subscriber stopped")

        except Exception as e:
            logger.error(f"❌ Sentiment subscriber failed: {e}")

    def get_current_sentiment(self, symbol: str = 'market-sentiment') -> float:
        """
        Get current sentiment score for a symbol

        Args:
            symbol: Trading symbol (default: market-sentiment for general market)

        Returns:
            Sentiment score [-1, 1] or 0.0 if no recent data
        """
        try:
            if symbol not in self.sentiment_cache:
                return 0.0

            sentiment_data = self.sentiment_cache[symbol]
            timestamp = sentiment_data.get('timestamp', 0)
            score = sentiment_data.get('score', 0.0)

            # Check if data is recent (within last 30 minutes)
            current_time = time.time()
            if current_time - timestamp > 1800:  # 30 minutes
                logger.debug(f"⚠️ Sentiment data for {symbol} is stale")
                return 0.0

            return score

        except Exception as e:
            logger.error(f"❌ Error getting sentiment: {e}")
            return 0.0

    async def initialize_ml_models(self):
        """Initialize GPU-accelerated ML models for HFT"""
        try:
            # Price prediction model (Transformer-based)
            self.price_predictor = HFTPricePredictor().to(self.device)

            # Order flow analyzer (LSTM-based)
            self.order_flow_analyzer = HFTOrderFlowAnalyzer().to(self.device)

            # Arbitrage detector (CNN-based)
            self.arbitrage_detector = HFTArbitrageDetector().to(self.device)

            logger.info("🧠 ML models initialized on GPU")

        except Exception as e:
            logger.error(f"❌ ML model initialization failed: {e}")
            # Fallback to CPU if GPU fails
            self.device = torch.device('cpu')
            logger.warning("⚠️ Falling back to CPU processing")

    async def start_trading(self):
        """Start HFT trading operations"""
        logger.info("🔥 Starting HFT trading operations")

        # Main trading loop - ultra-fast cycle
        while True:
            try:
                # Update market data (sub-1ms target)
                await self.update_market_data()

                # Run HFT strategies
                await self.execute_hft_strategies()

                # Update portfolio
                await self.update_portfolio()

                # Risk management checks
                await self.risk_management_checks()

                # Brief pause for ultra-low latency operation
                await asyncio.sleep(0.001)  # 1ms cycles

            except Exception as e:
                logger.error(f"❌ HFT trading error: {e}")
                await asyncio.sleep(0.01)  # Brief pause on error

    async def update_market_data(self):
        """Update market data with ultra-low latency"""
        try:
            # Get real-time Aster market data
            aster_symbols = await self.data_feed.get_aster_symbols()
            market_updates = {}

            for symbol in aster_symbols[:50]:  # Focus on top 50 assets
                # Get order book and recent trades
                orderbook = await self.data_feed.get_orderbook(symbol, depth=10)
                recent_trades = await self.data_feed.get_recent_trades(symbol, limit=50)

                # Add sentiment score
                sentiment_score = self.get_current_sentiment(symbol)

                market_updates[symbol] = {
                    'orderbook': orderbook,
                    'trades': recent_trades,
                    'sentiment': sentiment_score,
                    'timestamp': datetime.now()
                }

            self.market_data_cache = market_updates
            self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"❌ Market data update failed: {e}")

    async def execute_hft_strategies(self):
        """Execute all active HFT strategies"""
        try:
            # Statistical Arbitrage
            if self.active_strategies[HFTStrategy.STATISTICAL_ARBITRAGE]:
                await self.execute_statistical_arbitrage()

            # Market Making
            if self.active_strategies[HFTStrategy.MARKET_MAKING]:
                await self.execute_market_making()

            # Momentum Trading
            if self.active_strategies[HFTStrategy.MOMENTUM_TRADING]:
                await self.execute_momentum_trading()

            # Order Flow Analysis
            if self.active_strategies[HFTStrategy.ORDER_FLOW]:
                await self.execute_order_flow_analysis()

            # Latency Arbitrage
            if self.active_strategies[HFTStrategy.LATENCY_ARBITRAGE]:
                await self.execute_latency_arbitrage()

        except Exception as e:
            logger.error(f"❌ Strategy execution failed: {e}")

    async def execute_statistical_arbitrage(self):
        """Execute statistical arbitrage between Aster assets"""
        try:
            # Get price data for correlated pairs
            price_data = await self.get_price_matrix()

            # Use GPU to detect arbitrage opportunities
            if self.device.type == 'cuda':
                opportunities = await self.detect_arbitrage_gpu(price_data)
            else:
                opportunities = self.detect_arbitrage_cpu(price_data)

            # Execute arbitrage trades
            for opp in opportunities:
                if self.can_execute_trade(opp['size'], opp['expected_pnl']):
                    await self.execute_arbitrage_trade(opp)

        except Exception as e:
            logger.error(f"❌ Statistical arbitrage failed: {e}")

    async def execute_market_making(self):
        """Provide liquidity through market making"""
        try:
            for symbol in self.market_data_cache.keys():
                orderbook = self.market_data_cache[symbol]['orderbook']

                # Calculate optimal bid/ask spread
                spread = self.calculate_optimal_spread(orderbook, symbol)

                # Place limit orders
                if spread > 0.0001:  # Minimum spread threshold
                    await self.place_market_making_orders(symbol, spread)

        except Exception as e:
            logger.error(f"❌ Market making failed: {e}")

    async def execute_momentum_trading(self):
        """Execute momentum-based trades"""
        try:
            for symbol in self.market_data_cache.keys():
                # Analyze recent price momentum
                momentum = self.calculate_momentum(symbol)

                if abs(momentum) > 0.001:  # Significant momentum threshold
                    direction = 1 if momentum > 0 else -1
                    await self.execute_momentum_trade(symbol, direction, momentum)

        except Exception as e:
            logger.error(f"❌ Momentum trading failed: {e}")

    async def get_price_matrix(self) -> np.ndarray:
        """Get price matrix for arbitrage detection"""
        symbols = list(self.market_data_cache.keys())
        prices = []

        for symbol in symbols:
            if 'orderbook' in self.market_data_cache[symbol]:
                mid_price = (
                    self.market_data_cache[symbol]['orderbook']['bids'][0][0] +
                    self.market_data_cache[symbol]['orderbook']['asks'][0][0]
                ) / 2
                prices.append(mid_price)
            else:
                prices.append(0.0)

        return np.array(prices)

    async def detect_arbitrage_gpu(self, price_data: np.ndarray) -> List[Dict]:
        """GPU-accelerated arbitrage detection"""
        # Convert to tensor and move to GPU
        price_tensor = torch.tensor(price_data, dtype=torch.float32).to(self.device)

        # Run arbitrage detection model
        with torch.no_grad():
            arbitrage_scores = self.arbitrage_detector(price_tensor.unsqueeze(0))

        # Find opportunities above threshold
        opportunities = []
        threshold = 0.001  # 0.1% arbitrage opportunity

        for i in range(len(price_data)):
            for j in range(i+1, len(price_data)):
                score = arbitrage_scores[0, i, j].item()
                if score > threshold:
                    opportunities.append({
                        'pair': (i, j),
                        'score': score,
                        'size': min(self.config.max_position_size_usd / price_data[i],
                                  self.config.max_position_size_usd / price_data[j]),
                        'expected_pnl': score * 100  # Rough PNL estimate
                    })

        return opportunities

    def detect_arbitrage_cpu(self, price_data: np.ndarray) -> List[Dict]:
        """CPU-based arbitrage detection fallback"""
        opportunities = []
        threshold = 0.001

        for i in range(len(price_data)):
            for j in range(i+1, len(price_data)):
                # Simple price difference check
                price_diff = abs(price_data[i] - price_data[j]) / min(price_data[i], price_data[j])
                if price_diff > threshold:
                    opportunities.append({
                        'pair': (i, j),
                        'score': price_diff,
                        'size': min(self.config.max_position_size_usd / price_data[i],
                                  self.config.max_position_size_usd / price_data[j]),
                        'expected_pnl': price_diff * 100
                    })

        return opportunities

    def can_execute_trade(self, size_usd: float, expected_pnl: float) -> bool:
        """Check if trade meets risk and performance criteria"""
        # Check position limits
        if len(self.positions) >= self.config.max_open_positions:
            return False

        # Check position size
        if size_usd > self.config.max_position_size_usd:
            return False

        # Check available balance
        if size_usd > self.portfolio_state.available_balance * 0.1:  # Max 10% of balance
            return False

        # Check daily loss limit
        if self.daily_pnl < -self.config.max_daily_loss:
            return False

        # Check expected PNL vs risk
        if expected_pnl < self.config.target_profit_threshold:
            return False

        return True

    async def execute_arbitrage_trade(self, opportunity: Dict):
        """Execute statistical arbitrage trade"""
        try:
            pair = opportunity['pair']
            symbols = list(self.market_data_cache.keys())
            symbol1, symbol2 = symbols[pair[0]], symbols[pair[1]]

            # Place arbitrage orders
            size1 = opportunity['size'] / self.market_data_cache[symbol1]['orderbook']['asks'][0][0]
            size2 = opportunity['size'] / self.market_data_cache[symbol2]['orderbook']['bids'][0][0]

            # Execute simultaneous orders
            order1 = await self.aster_client.place_limit_order(
                symbol1, 'sell', size1,
                self.market_data_cache[symbol1]['orderbook']['bids'][0][0] * 1.0001
            )

            order2 = await self.aster_client.place_limit_order(
                symbol2, 'buy', size2,
                self.market_data_cache[symbol2]['orderbook']['asks'][0][0] * 0.9999
            )

            logger.info(f"🔄 Arbitrage executed: {symbol1}/{symbol2} - Size: ${opportunity['size']:.2f}")

        except Exception as e:
            logger.error(f"❌ Arbitrage trade execution failed: {e}")

    def calculate_optimal_spread(self, orderbook: Dict, symbol: str) -> float:
        """Calculate optimal spread for market making"""
        try:
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2

            # Base spread on volatility and liquidity
            volatility = self.calculate_volatility(symbol)
            spread_multiplier = 1 + volatility * 10  # Increase spread with volatility

            optimal_spread = mid_price * 0.0002 * spread_multiplier  # Base 0.02% spread

            return min(optimal_spread, mid_price * 0.001)  # Max 0.1% spread

        except Exception:
            return 0.0001  # Default spread

    def calculate_momentum(self, symbol: str) -> float:
        """Calculate price momentum"""
        try:
            if 'trades' not in self.market_data_cache[symbol]:
                return 0.0

            trades = self.market_data_cache[symbol]['trades']
            if len(trades) < 10:
                return 0.0

            # Calculate momentum over last 10 trades
            prices = [trade['price'] for trade in trades[-10:]]
            momentum = (prices[-1] - prices[0]) / prices[0]

            return momentum

        except Exception:
            return 0.0

    def calculate_volatility(self, symbol: str) -> float:
        """Calculate price volatility"""
        try:
            if 'trades' not in self.market_data_cache[symbol]:
                return 0.01  # Default volatility

            trades = self.market_data_cache[symbol]['trades']
            if len(trades) < 20:
                return 0.01

            prices = [trade['price'] for trade in trades[-20:]]
            returns = np.diff(np.log(prices))
            volatility = np.std(returns)

            return volatility

        except Exception:
            return 0.01

    async def update_portfolio(self):
        """Update portfolio state"""
        try:
            if self.aster_client:
                account_data = await self.aster_client.get_account_info()
                self.portfolio_state = PortfolioState(
                    timestamp=datetime.now(),
                    total_balance=account_data['total_balance'],
                    available_balance=account_data['available_balance']
                )

                # Update positions
                self.positions = {}
                for pos_data in account_data.get('positions', []):
                    position = Position(
                        symbol=pos_data['symbol'],
                        size=pos_data['size'],
                        entry_price=pos_data['entry_price'],
                        current_price=pos_data['current_price'],
                        pnl=pos_data['pnl'],
                        timestamp=datetime.now()
                    )
                    self.positions[pos_data['symbol']] = position

        except Exception as e:
            logger.error(f"❌ Portfolio update failed: {e}")

    async def risk_management_checks(self):
        """Perform risk management checks"""
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.config.max_daily_loss:
                logger.warning("🚨 Daily loss limit reached - pausing trading")
                # Close all positions and stop trading
                await self.emergency_stop()

            # Check position concentration
            total_exposure = sum(abs(pos.size * pos.current_price) for pos in self.positions.values())
            if total_exposure > self.portfolio_state.total_balance * 0.5:  # Max 50% exposure
                logger.warning("🚨 High position concentration - reducing exposure")
                await self.reduce_exposure()

            # Reset daily P&L at midnight UTC
            now = datetime.now()
            if now.hour == 0 and now.minute == 0:
                self.daily_pnl = 0.0

        except Exception as e:
            logger.error(f"❌ Risk management check failed: {e}")

    async def emergency_stop(self):
        """Emergency stop - close all positions"""
        logger.warning("🚨 Emergency stop activated")

        for symbol, position in self.positions.items():
            try:
                # Close position at market
                await self.aster_client.place_market_order(
                    symbol, 'sell' if position.size > 0 else 'buy',
                    abs(position.size)
                )
                logger.info(f"📤 Closed position: {symbol}")
            except Exception as e:
                logger.error(f"❌ Failed to close position {symbol}: {e}")

        # Disable all strategies
        for strategy in self.active_strategies:
            self.active_strategies[strategy] = False

    async def reduce_exposure(self):
        """Reduce overall position exposure"""
        try:
            # Close 20% of positions with smallest P&L
            positions_by_pnl = sorted(
                self.positions.items(),
                key=lambda x: x[1].pnl
            )

            positions_to_close = positions_by_pnl[:len(positions_by_pnl)//5]

            for symbol, position in positions_to_close:
                await self.aster_client.place_market_order(
                    symbol, 'sell' if position.size > 0 else 'buy',
                    abs(position.size) * 0.5  # Close half position
                )
                logger.info(f"📤 Reduced position: {symbol}")

        except Exception as e:
            logger.error(f"❌ Exposure reduction failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            'total_balance': self.portfolio_state.total_balance,
            'available_balance': self.portfolio_state.available_balance,
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'open_positions': len(self.positions),
            'total_exposure': sum(abs(pos.size * pos.current_price) for pos in self.positions.values()),
            'active_strategies': [s.value for s in self.active_strategies if self.active_strategies[s]]
        }


# GPU-Accelerated ML Models for HFT

class HFTPricePredictor(nn.Module):
    """Transformer-based price prediction model"""

    def __init__(self, input_dim=50, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]

        # Apply transformer
        x = self.transformer(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Output projection
        x = self.dropout(x)
        return self.output_projection(x)


class HFTOrderFlowAnalyzer(nn.Module):
    """LSTM-based order flow analysis model"""

    def __init__(self, input_dim=20, hidden_dim=128, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, 3)  # Buy, Sell, Neutral signals

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)

        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last timestep
        final_features = attn_out[:, -1, :]

        return self.output_projection(final_features)


class HFTArbitrageDetector(nn.Module):
    """CNN-based arbitrage detection model"""

    def __init__(self, num_assets=50):
        super().__init__()
        self.num_assets = num_assets

        # Create correlation matrix processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((num_assets//4, num_assets//4))

        # Output arbitrage scores for each pair
        self.output_projection = nn.Linear(128 * (num_assets//4) * (num_assets//4), num_assets * num_assets)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, price_matrix):
        # price_matrix: (batch, num_assets) - convert to correlation matrix
        batch_size = price_matrix.shape[0]

        # Create correlation matrix
        normalized_prices = (price_matrix - price_matrix.mean(dim=1, keepdim=True)) / (price_matrix.std(dim=1, keepdim=True) + 1e-8)
        corr_matrix = torch.bmm(normalized_prices.unsqueeze(2), normalized_prices.unsqueeze(1))

        # Add channel dimension
        x = corr_matrix.unsqueeze(1)  # (batch, 1, num_assets, num_assets)

        # Apply convolutions
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten and project
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        arbitrage_scores = self.output_projection(x)

        # Reshape to (batch, num_assets, num_assets)
        return arbitrage_scores.view(batch_size, self.num_assets, self.num_assets)

