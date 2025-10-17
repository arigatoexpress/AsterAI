import time
import asyncio
from prometheus_client import start_http_server, Gauge, Counter
from typing import Dict
import logging
import numpy as np

from strategies.hft.risk_manager import RiskManager
from local_training.continuous_learner import AdaptiveLearner
from data_pipeline.aster_dex.realtime_collector import AsterDEXRealtimeCollector  # Assuming integration

# Prometheus metrics
LATENCY_GAUGE = Gauge('hft_latency_ms', 'Decision loop latency in ms')
THROUGHPUT_COUNTER = Counter('hft_trades_processed', 'Number of trades processed')
VPIN_GAUGE = Gauge('hft_vpin_score', 'Current VPIN toxicity score')
SHARPE_GAUGE = Gauge('hft_sharpe_ratio', 'Rolling Sharpe ratio')
DRAWDOWN_GAUGE = Gauge('hft_drawdown_pct', 'Current drawdown percentage')
ERROR_COUNTER = Counter('hft_errors_total', 'Total errors encountered', ['type'])

logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """
    Real-time monitoring for HFT system profitability.
    Tracks latency, throughput, VPIN, Sharpe, drawdown with Prometheus export.
    """
    def __init__(self, risk_mgr: RiskManager, learner: AdaptiveLearner):
        self.risk_mgr = risk_mgr
        self.learner = learner
        self.start_time = time.time()
        
        # Start Prometheus server
        start_http_server(8001)  # Expose /metrics at :8001/metrics
        
    async def monitor_loop(self):
        """Main monitoring loop - runs every 10s."""
        while True:
            try:
                # Measure latency (simulate decision loop)
                start = time.time()
                # Simulate core logic: VPIN check + model predict + risk validate
                vpin = 0.3  # From collector
                prediction = 0.1  # From ensemble/PPO
                valid, _ = self.risk_mgr.validate_order(prediction, 5)
                latency_ms = (time.time() - start) * 1000
                LATENCY_GAUGE.set(latency_ms)
                
                # Update throughput (simulate trades)
                trades_this_cycle = np.random.randint(10, 50)
                THROUGHPUT_COUNTER.inc(trades_this_cycle)
                
                # VPIN and performance metrics
                VPIN_GAUGE.set(vpin)
                metrics = self.learner.monitor.get_metrics()
                SHARPE_GAUGE.set(metrics.sharpe_ratio)
                DRAWDOWN_GAUGE.set(metrics.drawdown)
                
                # Check for alerts
                self._check_alerts(latency_ms, metrics)
                
                logger.info(f"Metrics updated: Latency={latency_ms:.2f}ms, VPIN={vpin:.2f}, Sharpe={metrics.sharpe_ratio:.2f}")
                
            except Exception as e:
                ERROR_COUNTER.labels(type='monitor').inc()
                logger.error(f"Monitoring error: {e}")
                
            await asyncio.sleep(10)  # 10s interval for real-time but not overwhelming
            
    def _check_alerts(self, latency: float, metrics):
        """Trigger alerts on thresholds."""
        if latency > 1.0:  # >1ms latency alert
            self._send_alert("High Latency", f"Decision loop: {latency:.2f}ms")
        if metrics.sharpe_ratio < 1.5:
            self._send_alert("Low Sharpe", f"Current Sharpe: {metrics.sharpe_ratio:.2f}")
        if metrics.drawdown > 0.05:
            self._send_alert("High Drawdown", f"Drawdown: {metrics.drawdown:.2%}")
            
    def _send_alert(self, title: str, message: str):
        """Send alert via Slack/Telegram (placeholder - integrate with actual notifier)."""
        # In production: Use slack-sdk or python-telegram-bot
        print(f"🚨 ALERT: {title} - {message}")
        # e.g., slack_client.chat_postMessage(channel="#trading-alerts", text=f"{title}: {message}")

# Example integration
if __name__ == "__main__":
    risk_mgr = RiskManager()
    learner = AdaptiveLearner()
    monitor = RealTimeMonitor(risk_mgr, learner)
    
    # Run monitoring loop
    asyncio.run(monitor.monitor_loop())
