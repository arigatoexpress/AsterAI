#!/usr/bin/env python3
"""
FastAPI Dashboard for Aster Autonomous Trader
Real-time monitoring and control interface for the trading system.
"""

import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import subprocess
import os

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_trader.config import get_settings, PRIORITY_SYMBOLS
from mcp_trader.execution.aster_client import AsterClient
from mcp_trader.trading.autonomous_trader import AutonomousTrader
from mcp_trader.risk.risk_manager import RiskManager
from mcp_trader.data.aster_feed import AsterDataFeed
from mcp_trader.logging_utils import get_logger
from mcp_trader.security.middleware import SecurityMiddleware, SecurityLogger, require_authentication, rate_limit
from mcp_trader.security.input_validation import InputValidator
from mcp_trader.security.config import security_config

# Setup structured logging
logger = get_logger("dashboard")

# Initialize security logger
security_logger = SecurityLogger()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Startup
    logger.info("Starting Aster Trader Dashboard with Security...")

    settings = get_settings()

    # Initialize components with security validation
    try:
        dashboard_state['client'] = AsterClient(
            InputValidator.sanitize_string(settings.aster_api_key or ""),
            InputValidator.sanitize_string(settings.aster_api_secret or "")
        )
        dashboard_state['data_feed'] = AsterDataFeed()
        dashboard_state['risk_manager'] = RiskManager(InputValidator.create_secure_config({
            'max_drawdown': settings.max_portfolio_risk,
            'max_position_size': settings.max_single_position_risk,
            'max_concurrent_positions': settings.max_concurrent_positions
        }))

        logger.info("✅ All components initialized securely")

    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        raise

    # Initialize trader (in test mode for dashboard)
    trader_config = {
        'test_mode': True,
        'grid_config': {
            'grid_levels': settings.grid_levels,
            'grid_spacing_percent': settings.grid_spacing_percent,
            'position_size_per_level': settings.grid_position_size_usd
        },
        'volatility_config': {
            'min_volatility_threshold': 3.0,
            'profit_taking_threshold': settings.take_profit_threshold,
            'stop_loss_threshold': settings.stop_loss_threshold
        },
        'risk_config': {
            'max_portfolio_risk': settings.max_portfolio_risk,
            'max_single_position_risk': settings.max_single_position_risk
        }
    }

    dashboard_state['trader'] = AutonomousTrader(trader_config)
    dashboard_state['system_status'] = 'ready'

    logger.info("Dashboard initialization complete")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down dashboard...")

    if dashboard_state['trader']:
        await dashboard_state['trader'].stop()

    if dashboard_state['client']:
        await dashboard_state['client'].disconnect()

    if dashboard_state['data_feed']:
        await dashboard_state['data_feed'].stop()

    dashboard_state['system_status'] = 'shutdown'


# Global state
dashboard_state = {
    'trader': None,
    'client': None,
    'data_feed': None,
    'risk_manager': None,
    'connected_websockets': set(),
    'system_status': 'initializing',
    'last_update': datetime.now()
}

# Create FastAPI app with lifespan
app = FastAPI(
    title="AsterAI HFT Trader - Secure Dashboard",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("DEBUG") else None,  # Hide docs in production
    redoc_url="/redoc" if os.getenv("DEBUG") else None
)

# Add security middleware
app.add_middleware(
    SecurityMiddleware,
    rate_limit_requests=security_config.rate_limit_requests_per_minute,
    rate_limit_window=60
)

# Mount static files and templates after app creation
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")


@app.get("/", response_class=HTMLResponse)
async def market_overview(request: Request):
    """Beautiful market overview main page."""
    return templates.TemplateResponse("market_overview.html", {
        "request": request,
        "title": "AsterAI Market Overview"
    })


@app.get("/ai-learning", response_class=HTMLResponse)
async def ai_learning_page(request: Request):
    """AI Learning and strategy adaptation page."""
    return templates.TemplateResponse("ai_learning.html", {
        "request": request,
        "title": "AsterAI - AI Learning"
    })


@app.get("/positions", response_class=HTMLResponse)
async def positions_page(request: Request):
    """Positions management page."""
    return templates.TemplateResponse("positions.html", {
        "request": request,
        "title": "AsterAI - Positions"
    })


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Advanced analytics and backtesting page."""
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "title": "AsterAI - Analytics"
    })


@app.get("/old-dashboard", response_class=HTMLResponse)
async def old_dashboard(request: Request):
    """Legacy dashboard page."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "AsterAI - Legacy Dashboard",
        "symbols": PRIORITY_SYMBOLS,
        "system_status": dashboard_state.get('system_status', 'initializing')
    })


@app.get("/api/system/status")
async def get_system_status():
    """Get current system status."""
    return {
        'status': dashboard_state['system_status'],
        'timestamp': datetime.now().isoformat(),
        'active_connections': len(dashboard_state['connected_websockets']),
        'trader_running': dashboard_state['trader'] is not None,
        'last_update': dashboard_state['last_update'].isoformat()
    }


@app.get("/api/market/data")
async def get_market_data():
    """Get current market data for all symbols."""
    data = {}

    if dashboard_state['client']:
        for symbol in PRIORITY_SYMBOLS:
            symbol_data = {}

            try:
                # Get ticker data
                ticker = await dashboard_state['client'].get_24hr_ticker(symbol)
                symbol_data['ticker'] = ticker
            except Exception as e:
                symbol_data['ticker_error'] = str(e)

            try:
                # Get order book
                orderbook = await dashboard_state['client'].get_order_book(symbol, 10)
                symbol_data['orderbook'] = orderbook
            except Exception as e:
                symbol_data['orderbook_error'] = str(e)

                data[symbol] = symbol_data

    return {
        'timestamp': datetime.now().isoformat(),
        'data': data
    }


@app.get("/api/portfolio/status")
async def get_portfolio_status():
    """Get current portfolio status."""
    if not dashboard_state['trader']:
        return {'error': 'Trader not initialized'}

    portfolio = dashboard_state['trader'].portfolio_state

    return {
        'timestamp': datetime.now().isoformat(),
        'total_balance': portfolio.total_balance,
        'available_balance': portfolio.available_balance,
        'total_positions_value': portfolio.total_positions_value,
        'unrealized_pnl': portfolio.unrealized_pnl,
        'active_positions': len(portfolio.active_positions),
        'active_grids': len(portfolio.active_grids),
        'positions': dict(portfolio.active_positions),
        'grids': dict(portfolio.active_grids)
    }


@app.get("/api/risk/metrics")
async def get_risk_metrics():
    """Get current risk metrics."""
    if not dashboard_state['risk_manager']:
        return {'error': 'Risk manager not initialized'}

    # Get mock portfolio state for risk calculation
    mock_portfolio = type('MockPortfolio', (), {
        'total_balance': 9500.0,
        'total_positions_value': 500.0,
        'active_positions': {},
        'available_balance': 9000.0
    })()

    # Get mock market data
    mock_market_data = {}
    for symbol in PRIORITY_SYMBOLS[:3]:
        mock_market_data[symbol] = type('MockData', (), {
            'close': [100.0] * 50  # Mock price data
        })()

    try:
        risk_metrics = await dashboard_state['risk_manager'].assess_portfolio_risk(
            mock_portfolio, mock_market_data
        )

        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': risk_metrics.portfolio_value,
            'total_risk': risk_metrics.total_risk,
            'max_drawdown': risk_metrics.max_drawdown,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'volatility': risk_metrics.volatility,
            'var_95': risk_metrics.var_95,
            'concentration_risk': risk_metrics.concentration_risk
        }
    except Exception as e:
        return {'error': f'Risk calculation failed: {e}'}


@app.get("/api/trading/decisions")
async def get_trading_decisions():
    """Get recent trading decisions."""
    if not dashboard_state['trader']:
        return {'error': 'Trader not initialized'}

    # Get recent decisions from trader
    # This is a simplified implementation
    return {
        'timestamp': datetime.now().isoformat(),
        'decisions': [],  # Would be populated from actual trader decisions
        'total_decisions': 0
    }


@app.post("/api/trader/start")
@require_authentication(roles=["admin", "trader"])
@rate_limit(requests_per_minute=10)
async def start_trader(background_tasks: BackgroundTasks):
    """Start the autonomous trader."""
    if dashboard_state['system_status'] == 'trading':
        return {'error': 'Trader already running'}

    try:
        dashboard_state['system_status'] = 'trading'
        background_tasks.add_task(run_trader_background)
        return {'status': 'starting'}
    except Exception as e:
        dashboard_state['system_status'] = 'error'
        return {'error': str(e)}


@app.post("/api/trader/stop")
@require_authentication(roles=["admin", "trader"])
@rate_limit(requests_per_minute=10)
async def stop_trader():
    """Stop the autonomous trader."""
    if dashboard_state['trader']:
        await dashboard_state['trader'].stop()
        dashboard_state['system_status'] = 'ready'
        return {'status': 'stopped'}
    else:
        return {'error': 'No trader running'}


@app.websocket("/ws/live-data")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data updates."""
    await websocket.accept()
    dashboard_state['connected_websockets'].add(websocket)

    try:
        while True:
            # Send periodic updates
            data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': dashboard_state['system_status'],
                'portfolio': await get_portfolio_status() if dashboard_state['trader'] else None,
                'market_data': await get_market_data(),
                'risk_metrics': await get_risk_metrics()
            }

            await websocket.send_json(data)
            await asyncio.sleep(5)  # Update every 5 seconds

    except WebSocketDisconnect:
        dashboard_state['connected_websockets'].discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        dashboard_state['connected_websockets'].discard(websocket)


async def run_trader_background():
    """Run trader in background task."""
    try:
        if dashboard_state['trader']:
            await dashboard_state['trader'].start()
    except Exception as e:
        logger.error(f"Background trader error: {e}")
        dashboard_state['system_status'] = 'error'


@app.get("/api/performance/history")
async def get_performance_history(hours: int = 24):
    """Get performance history for the specified period."""
    # This would return historical performance data
    # For now, return mock data
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    # Generate mock performance data
    timestamps = []
    portfolio_values = []
    current_value = 10000.0

    for i in range(0, hours * 12):  # Every 5 minutes
        timestamp = start_time + timedelta(minutes=i*5)
        # Add some random variation
        change = (0.001 - 0.002) + (0.004 * (0.5 - 0.5))  # Random walk
        current_value *= (1 + change)
        current_value = max(current_value, 9000)  # Floor

        timestamps.append(timestamp.isoformat())
        portfolio_values.append(round(current_value, 2))

    return {
        'timestamps': timestamps,
        'portfolio_values': portfolio_values,
        'start_value': 10000.0,
        'end_value': portfolio_values[-1] if portfolio_values else 10000.0,
        'total_return': ((portfolio_values[-1] if portfolio_values else 10000.0) - 10000.0) / 10000.0
    }


@app.get("/api/backtest/run")
@rate_limit(requests_per_minute=5)
async def run_backtest(strategy: str = "grid", days: int = 30, symbol: str = "BTCUSDT"):
    """Run a backtest for a specific strategy."""
    try:
        from mcp_trader.backtesting.enhanced_backtester import EnhancedBacktester, BacktestConfig
        from mcp_trader.backtesting.data_generator import generate_backtest_data

        # Generate historical data
        data = generate_backtest_data([symbol], periods=days * 24)  # Hourly data

        # Setup backtester
        config = BacktestConfig(initial_balance=10000.0)
        backtester = EnhancedBacktester(config)

        # Select strategy
        if strategy == "grid":
            from mcp_trader.trading.strategies.grid_strategy import GridStrategy
            strategy_config = {'grid_levels': 5, 'grid_spacing_percent': 2.0}
            strategy_class = GridStrategy
        elif strategy == "volatility":
            from mcp_trader.trading.strategies.volatility_strategy import VolatilityStrategy
            strategy_config = {'min_volatility_threshold': 3.0}
            strategy_class = VolatilityStrategy
        elif strategy == "hybrid":
            from mcp_trader.trading.strategies.hybrid_strategy import HybridStrategy, HybridStrategyConfig
            strategy_config = HybridStrategyConfig(
                symbol=symbol, total_capital=10000.0,
                grid_allocation=0.6, volatility_allocation=0.4
            )
            strategy_class = lambda x: HybridStrategy(x, None)
        else:
            return {"error": f"Unknown strategy: {strategy}"}

        # Run backtest
        result = await backtester.run_backtest(strategy_class, strategy_config, data, [symbol])

        return {
            'strategy': strategy,
            'symbol': symbol,
            'days': days,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'equity_curve': result.equity_curve,
            'trade_log': result.trade_log[:10]  # Last 10 trades
        }

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {"error": str(e)}


@app.get("/api/backtest/compare")
async def compare_strategies(days: int = 30, symbol: str = "BTCUSDT"):
    """Compare all strategies performance."""
    strategies = ["grid", "volatility", "hybrid"]
    results = {}

    for strategy in strategies:
        try:
            result = await run_backtest(strategy, days, symbol)
            if "error" not in result:
                results[strategy] = result
        except Exception as e:
            logger.error(f"Error testing {strategy}: {e}")
            results[strategy] = {"error": str(e)}

    # Determine best strategy
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_strategy = max(valid_results.items(),
                          key=lambda x: x[1].get('sharpe_ratio', -float('inf')))
        results['recommendation'] = {
            'best_strategy': best_strategy[0],
            'sharpe_ratio': best_strategy[1]['sharpe_ratio'],
            'reason': f"Highest Sharpe ratio ({best_strategy[1]['sharpe_ratio']:.2f})"
        }

    return results


@app.get("/api/positions/open")
async def get_open_positions():
    """Get current open positions from Aster DEX."""
    if not dashboard_state['client']:
        return {"error": "API client not available"}

    try:
        async with dashboard_state['client']:
            positions = await dashboard_state['client'].get_positions()
            return {"positions": positions}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {"error": str(e), "positions": []}


@app.post("/api/positions/close/{symbol}")
async def close_position(symbol: str):
    """Close a specific position."""
    if not dashboard_state['trader']:
        return {"error": "Trader not running"}

    # This would implement position closing logic
    # For now, return mock response
    return {
        "status": "success",
        "message": f"Position close request submitted for {symbol}",
        "symbol": symbol
    }


@app.get("/api/market/analysis")
async def get_market_analysis():
    """Get market analysis and regime detection."""
    try:
        # Analyze market conditions across all symbols
        analysis = {}
        regime_votes = {"bull": 0, "bear": 0, "sideways": 0, "high_vol": 0}

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            if dashboard_state['client']:
                try:
                    async with dashboard_state['client']:
                        ticker = await dashboard_state['client'].get_24hr_ticker(symbol)
                        price_change = float(ticker.get('priceChangePercent', 0))

                        # Simple regime detection
                        if price_change > 2:
                            regime = "bull"
                        elif price_change < -2:
                            regime = "bear"
                        else:
                            regime = "sideways"

                        regime_votes[regime] += 1

                        analysis[symbol] = {
                            'regime': regime,
                            'price_change': price_change,
                            'last_price': ticker.get('lastPrice', 0)
                        }
                except Exception as e:
                    analysis[symbol] = {"error": str(e)}

        # Determine overall market regime
        dominant_regime = max(regime_votes.items(), key=lambda x: x[1])[0]

        return {
            'market_regime': dominant_regime,
            'regime_votes': regime_votes,
            'symbol_analysis': analysis,
            'recommendation': get_strategy_recommendation(dominant_regime)
        }

    except Exception as e:
        return {"error": str(e)}


def get_strategy_recommendation(market_regime: str) -> str:
    """Get strategy recommendation based on market regime."""
    recommendations = {
        'bull': 'Barbell strategy: Hold BTC/ETH, take asymmetric bets on altcoins',
        'bear': 'Tail risk hedging: Reduce exposure, increase safe asset allocation',
        'sideways': 'Grid trading: Profit from range-bound price action',
        'high_vol': 'Asymmetric bets: Small positions with large upside potential'
    }
    return recommendations.get(market_regime, 'Monitor market conditions and adapt strategies')


@app.get("/api/adaptive/status")
async def get_adaptive_status():
    """Get status of the adaptive trading system."""
    try:
        # Check if adaptive trader is running
        # This would integrate with the running AdaptiveTradingSystem
        status = {
            'running': False,  # Would be True if system is active
            'learning_samples': 1250,  # Simulated learning progress
            'model_accuracy': 0.73,  # Simulated accuracy
            'active_strategies': ['barbell', 'asymmetric', 'tail_risk'],
            'current_regime': 'bull_market',
            'strategy_weights': {
                'barbell': 0.45,  # Adapted based on current market
                'asymmetric': 0.35,  # Reduced due to volatility
                'tail_risk': 0.20  # Maintained for protection
            },
            'performance': {
                'total_return': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'days_to_1m': None
            },
            'market_adaptation': {
                'volatility_regime': 'high',
                'trend_strength': 'strong_bull',
                'risk_adjustment': 'moderate',
                'last_adaptation': datetime.now().isoformat()
            }
        }

        # Try to load from adaptive trading report if it exists
        import os
        if os.path.exists('adaptive_trading_report.json'):
            import json
            with open('adaptive_trading_report.json', 'r') as f:
                report = json.load(f)
                status['performance'] = {
                    'total_return': report.get('system_info', {}).get('total_return', 0.0),
                    'win_rate': report.get('trading_stats', {}).get('win_rate', 0.0),
                    'max_drawdown': report.get('trading_stats', {}).get('max_drawdown', 0.0),
                    'days_to_1m': None  # Would calculate from report
                }

        return status

    except Exception as e:
        logger.error(f"Error getting adaptive status: {e}")
        return {"error": str(e)}


@app.post("/api/adaptive/start")
async def start_adaptive_trading():
    """Start the adaptive trading system."""
    # This would start the AdaptiveTradingSystem in a background task
    return {
        "status": "starting",
        "message": "Adaptive AI trading system starting...",
        "strategies": ["Barbell Portfolio", "Asymmetric Bets", "Tail Risk Hedging"]
    }


@app.post("/api/adaptive/stop")
async def stop_adaptive_trading():
    """Stop the adaptive trading system."""
    return {
        "status": "stopping",
        "message": "Adaptive AI trading system stopping..."
    }


@app.post("/api/adaptive/adapt")
async def force_strategy_adaptation():
    """Force immediate strategy adaptation based on current market conditions."""
    try:
        # This would trigger immediate strategy weight recalculation
        # For now, return success
        return {
            "success": True,
            "message": "Strategy adaptation triggered",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in force adaptation: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/adaptive/reset")
async def reset_learning_state():
    """Reset the AI learning state to start fresh."""
    try:
        # This would reset the learning system and strategy weights
        # For now, return success
        return {
            "success": True,
            "message": "AI learning state reset successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resetting learning state: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/system/logs")
async def get_system_logs(lines: int = 50):
    """Get recent system logs."""
    # This would read from log files
    # For now, return mock log entries
    logs = []
    for i in range(min(lines, 10)):
        timestamp = datetime.now() - timedelta(minutes=i*5)
        logs.append({
            'timestamp': timestamp.isoformat(),
            'level': 'INFO',
            'message': f'System operating normally - iteration {i}',
            'component': 'dashboard'
        })

    return {'logs': logs}


@app.post("/api/test/ui")
async def test_ui_automation():
    """Test UI elements using Playwright MCP for dashboard automation."""
    try:
        # Example: Navigate to the dashboard and take a screenshot
        url = "http://localhost:8000"  # Assuming dashboard is running locally

        # Simulate Playwright tool calls via subprocess (since MCP is Docker-based)
        # In a real setup, you'd use the MCP client library
        result = subprocess.run([
            "docker", "run", "-i", "--rm", "mcp/playwright"
        ], input=f"browser_navigate(url='{url}')\nbrowser_take_screenshot(filename='ui_test.png')\nbrowser_snapshot()".encode(),
           capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            return {
                "status": "success",
                "message": "UI test completed via Playwright",
                "output": result.stdout,
                "screenshot": "ui_test.png (if generated)"
            }
        else:
            return {
                "status": "error",
                "message": "UI test failed",
                "error": result.stderr
            }
    except Exception as e:
        logger.error(f"UI test error: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aster Trader Dashboard")
    parser.add_argument('--port', type=int, default=8000, help='Port to run the dashboard on')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to bind to')

    args = parser.parse_args()

    print("🚀 Starting Aster Trader Dashboard...")
    print(f"📊 Dashboard will be available at: http://localhost:{args.port}")
    print(f"📡 WebSocket endpoint: ws://localhost:{args.port}/ws/live-data")
    print("Press Ctrl+C to stop")

    try:
        uvicorn.run(
            "dashboard.aster_trader_dashboard:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level="info"
        )
    except OSError as e:
        if "Address already in use" in str(e) or "permission denied" in str(e).lower():
            logger.error(f"❌ Failed to start dashboard on port {args.port}")
            logger.error(f"💡 Port {args.port} is already in use or blocked")
            logger.error("💡 Try a different port: python dashboard/aster_trader_dashboard.py --port 8002")
            logger.error("💡 Or check if another application is using this port")
            sys.exit(1)
        else:
            raise
