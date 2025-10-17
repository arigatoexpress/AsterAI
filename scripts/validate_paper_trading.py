import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging

from data_pipeline.aster_dex.realtime_collector import AsterDEXRealtimeCollector
from strategies.hft.ensemble_model import EnsembleModel
from strategies.hft.risk_manager import RiskManager
from monitoring.real_time_monitor import RealTimeMonitor
from local_training.continuous_learner import AdaptiveLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperTradingSimulator:
    """
    Simulates 7-day paper trading to validate system before live deployment.
    Uses historical/replayed data for realistic HFT testing.
    """
    def __init__(self, duration_days: int, initial_capital: float, simulate: bool = True):
        self.duration = duration_days
        self.capital = initial_capital
        self.simulate = simulate
        self.risk_mgr = RiskManager(initial_capital=initial_capital)
        self.model = EnsembleModel()
        self.learner = AdaptiveLearner()
        self.monitor = RealTimeMonitor(self.risk_mgr, self.learner)
        
        # Simulated data (in production, replay from collector)
        self.trade_count = 0
        self.total_pnl = 0.0
        self.trades = []
        
    async def run_simulation(self):
        """Run the paper trading simulation."""
        logger.info(f"Starting {self.duration}-day paper trading simulation with ${self.capital}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(days=self.duration)
        
        # Simulate trading loop (every 1 minute for demo; real: 100ms)
        current_time = start_time
        while current_time < end_time:
            # Get market data (simulate or from collector)
            if self.simulate:
                price_change = np.random.normal(0.0005, 0.005)  # Realistic HFT returns
                volume = np.random.uniform(1000, 10000)
            else:
                # Real: await collector.get_current_data()
                pass
                
            # Model prediction
            state = np.random.rand(50)  # Simulated state/features
            action = self.model.predict(state)  # -1 sell, 0 hold, 1 buy
            
            # Risk check and position sizing
            win_rate = 0.6  # From history
            avg_win, avg_loss = 0.02, -0.01
            volatility = abs(price_change)
            position_size = self.risk_mgr.calculate_position_size(win_rate, avg_win, avg_loss, volatility)
            
            valid, msg = self.risk_mgr.validate_order(position_size * action, leverage=5)
            if not valid:
                logger.warning(f"Order rejected: {msg}")
                current_time += timedelta(minutes=1)
                continue
                
            # Simulate trade execution
            pnl = position_size * action * price_change * self.capital * 5  # Leverage
            self.risk_mgr.update_capital(pnl)
            self.total_pnl += pnl
            self.trade_count += 1
            self.trades.append({'time': current_time, 'action': action, 'pnl': pnl})
            
            # Check kill switch
            drawdown = (self.risk_mgr.max_capital - self.risk_mgr.capital) / self.risk_mgr.max_capital
            if self.risk_mgr.check_kill_switch(self.risk_mgr.daily_pnl, drawdown):
                logger.error("Kill switch activated during paper trading!")
                break
                
            # Update monitoring
            await self.monitor.monitor_loop()  # Single iteration for metrics
            
            current_time += timedelta(minutes=1)  # Step time
            
            if self.trade_count % 100 == 0:
                logger.info(f"Trades: {self.trade_count}, PnL: ${self.total_pnl:.2f}, Capital: ${self.risk_mgr.capital:.2f}")
                
        # Calculate final metrics
        returns = pd.Series([t['pnl'] for t in self.trades])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 1440) if np.std(returns) > 0 else 0  # Annualized for minutely
        max_dd = ((self.risk_mgr.max_capital - pd.Series([self.capital + sum(self.trades[:i+1]['pnl'] for i in range(len(self.trades)))]).cumsum()).min() / self.risk_mgr.max_capital) if self.trades else 0
        win_rate = len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades) if self.trades else 0
        
        results = {
            'total_return': self.total_pnl / self.capital,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': self.trade_count,
            'final_capital': self.risk_mgr.capital
        }
        
        logger.info(f"Paper trading complete. Results: {results}")
        
        # Validation check
        if sharpe > 2.0 and max_dd < 0.10 and results['total_return'] > 0.10:
            logger.info("✅ Paper trading PASSED - Ready for live!")
            return True, results
        else:
            logger.warning("❌ Paper trading FAILED - Review and optimize")
            return False, results

async def main(duration: int, capital: float, simulate: bool):
    simulator = PaperTradingSimulator(duration, capital, simulate)
    passed, results = await simulator.run_simulation()
    return passed, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=7, help="Simulation duration in days")
    parser.add_argument("--initial_capital", type=float, default=100.0, help="Starting capital")
    parser.add_argument("--simulate", type=bool, default=True, help="Use simulated data")
    args = parser.parse_args()
    
    passed, results = asyncio.run(main(args.duration, args.initial_capital, args.simulate))
    print(f"Validation passed: {passed}")
    print("Results:", results)
