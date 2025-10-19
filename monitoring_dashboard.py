"""
Real-time Monitoring Dashboard for Autonomous Trading System

This module provides a web-based dashboard for monitoring trading performance,
positions, and system health in real-time.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for the monitoring dashboard"""
    refresh_interval: int = 5  # seconds
    max_data_points: int = 1000
    alert_thresholds: Dict[str, float] = None
    api_endpoints: Dict[str, str] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'daily_loss_limit': 0.10,
                'max_drawdown': 0.15,
                'position_limit': 3,
                'error_rate': 0.05
            }
        
        if self.api_endpoints is None:
            self.api_endpoints = {
                'trading_agent': 'http://localhost:8001/status',
                'data_pipeline': 'http://localhost:8002/status',
                'risk_manager': 'http://localhost:8003/status'
            }

class TradingMetrics(BaseModel):
    """Trading metrics data model"""
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_balance: float
    timestamp: datetime

class PositionData(BaseModel):
    """Position data model"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    strategy: str

class SystemStatus(BaseModel):
    """System status data model"""
    trading_agent: str
    data_pipeline: str
    risk_manager: str
    emergency_stop: bool
    last_update: datetime
    errors: List[str]

class MonitoringDashboard:
    """Main monitoring dashboard class"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.metrics_history: List[TradingMetrics] = []
        self.positions_history: List[PositionData] = []
        self.system_status_history: List[SystemStatus] = []
        self.alerts: List[Dict[str, Any]] = []
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Aster Trading Dashboard", version="1.0.0")
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def dashboard():
            return HTMLResponse(content=self._generate_dashboard_html())
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            return self.metrics_history[-1] if self.metrics_history else None
        
        @self.app.get("/api/positions")
        async def get_positions():
            return self.positions_history[-1] if self.positions_history else None
        
        @self.app.get("/api/status")
        async def get_status():
            return self.system_status_history[-1] if self.system_status_history else None
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            return self.alerts
        
        @self.app.post("/api/emergency_stop")
        async def emergency_stop():
            # This would trigger emergency stop in the trading system
            return {"status": "Emergency stop triggered"}
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        logger.info("Starting monitoring dashboard...")
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Start monitoring loop in background
        asyncio.create_task(self._monitoring_loop())
        
        # Start server
        await server.serve()
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect data from all services
                await self._collect_metrics()
                await self._collect_positions()
                await self._collect_system_status()
                
                # Check for alerts
                await self._check_alerts()
                
                # Update dashboard
                await self._update_dashboard()
                
                # Wait for next update
                await asyncio.sleep(self.config.refresh_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.refresh_interval)
    
    async def _collect_metrics(self):
        """Collect trading metrics from trading agent"""
        try:
            # In production, this would call the actual trading agent API
            # For now, we'll simulate the data
            metrics = TradingMetrics(
                total_pnl=np.random.normal(0, 10),
                daily_pnl=np.random.normal(0, 5),
                unrealized_pnl=np.random.normal(0, 2),
                total_trades=np.random.randint(0, 100),
                winning_trades=np.random.randint(0, 50),
                losing_trades=np.random.randint(0, 50),
                win_rate=np.random.uniform(0.4, 0.8),
                profit_factor=np.random.uniform(0.8, 2.0),
                sharpe_ratio=np.random.uniform(0.5, 2.0),
                max_drawdown=np.random.uniform(0.0, 0.1),
                current_balance=100.0 + np.random.normal(0, 20),
                timestamp=datetime.now()
            )
            
            self.metrics_history.append(metrics)
            
            # Keep only recent data
            if len(self.metrics_history) > self.config.max_data_points:
                self.metrics_history = self.metrics_history[-self.config.max_data_points:]
                
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _collect_positions(self):
        """Collect position data from trading agent"""
        try:
            # In production, this would call the actual trading agent API
            # For now, we'll simulate the data
            if np.random.random() < 0.3:  # 30% chance of having positions
                position = PositionData(
                    symbol="BTCUSDT",
                    side="long" if np.random.random() < 0.5 else "short",
                    size=np.random.uniform(0.01, 0.1),
                    entry_price=np.random.uniform(40000, 50000),
                    current_price=np.random.uniform(40000, 50000),
                    unrealized_pnl=np.random.normal(0, 5),
                    stop_loss=np.random.uniform(35000, 45000),
                    take_profit=np.random.uniform(45000, 55000),
                    entry_time=datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
                    strategy="market_making"
                )
                
                self.positions_history.append(position)
                
                # Keep only recent data
                if len(self.positions_history) > self.config.max_data_points:
                    self.positions_history = self.positions_history[-self.config.max_data_points:]
                    
        except Exception as e:
            logger.error(f"Error collecting positions: {e}")
    
    async def _collect_system_status(self):
        """Collect system status from all services"""
        try:
            # In production, this would call the actual service APIs
            # For now, we'll simulate the data
            status = SystemStatus(
                trading_agent="running",
                data_pipeline="running",
                risk_manager="running",
                emergency_stop=False,
                last_update=datetime.now(),
                errors=[]
            )
            
            self.system_status_history.append(status)
            
            # Keep only recent data
            if len(self.system_status_history) > self.config.max_data_points:
                self.system_status_history = self.system_status_history[-self.config.max_data_points:]
                
        except Exception as e:
            logger.error(f"Error collecting system status: {e}")
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        if not self.metrics_history:
            return
        
        current_metrics = self.metrics_history[-1]
        
        # Check daily loss limit
        if current_metrics.daily_pnl < -self.config.alert_thresholds['daily_loss_limit'] * 100:
            await self._create_alert(
                "Daily Loss Limit Exceeded",
                f"Daily P&L: {current_metrics.daily_pnl:.2f}",
                "critical"
            )
        
        # Check max drawdown
        if current_metrics.max_drawdown > self.config.alert_thresholds['max_drawdown'] * 100:
            await self._create_alert(
                "Maximum Drawdown Exceeded",
                f"Max Drawdown: {current_metrics.max_drawdown:.2f}",
                "critical"
            )
        
        # Check position limit
        if len(self.positions_history) > self.config.alert_thresholds['position_limit']:
            await self._create_alert(
                "Position Limit Exceeded",
                f"Positions: {len(self.positions_history)}",
                "warning"
            )
    
    async def _create_alert(self, title: str, message: str, severity: str):
        """Create a new alert"""
        alert = {
            'id': len(self.alerts) + 1,
            'title': title,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now(),
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert: {title} - {message}")
    
    async def _update_dashboard(self):
        """Update the dashboard display"""
        # This would update the web interface
        pass
    
    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Aster Trading Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-card { 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 20px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    background: #f9f9f9;
                }
                .metric-value { font-size: 24px; font-weight: bold; }
                .metric-label { font-size: 14px; color: #666; }
                .positive { color: green; }
                .negative { color: red; }
                .alert { 
                    padding: 10px; 
                    margin: 5px 0; 
                    border-radius: 3px; 
                }
                .alert-critical { background: #ffebee; border-left: 4px solid #f44336; }
                .alert-warning { background: #fff3e0; border-left: 4px solid #ff9800; }
            </style>
        </head>
        <body>
            <h1>Aster Trading Dashboard</h1>
            
            <div id="metrics-container">
                <div class="metric-card">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value" id="total-pnl">$0.00</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Daily P&L</div>
                    <div class="metric-value" id="daily-pnl">$0.00</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value" id="win-rate">0%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Positions</div>
                    <div class="metric-value" id="positions">0</div>
                </div>
            </div>
            
            <div id="charts-container">
                <div id="pnl-chart" style="width: 100%; height: 400px;"></div>
                <div id="positions-chart" style="width: 100%; height: 300px;"></div>
            </div>
            
            <div id="alerts-container">
                <h3>Alerts</h3>
                <div id="alerts-list"></div>
            </div>
            
            <script>
                // Update dashboard every 5 seconds
                setInterval(updateDashboard, 5000);
                
                function updateDashboard() {
                    fetch('/api/metrics')
                        .then(response => response.json())
                        .then(data => {
                            if (data) {
                                updateMetrics(data);
                            }
                        });
                    
                    fetch('/api/positions')
                        .then(response => response.json())
                        .then(data => {
                            if (data) {
                                updatePositions(data);
                            }
                        });
                    
                    fetch('/api/alerts')
                        .then(response => response.json())
                        .then(data => {
                            updateAlerts(data);
                        });
                }
                
                function updateMetrics(metrics) {
                    document.getElementById('total-pnl').textContent = '$' + metrics.total_pnl.toFixed(2);
                    document.getElementById('daily-pnl').textContent = '$' + metrics.daily_pnl.toFixed(2);
                    document.getElementById('win-rate').textContent = (metrics.win_rate * 100).toFixed(1) + '%';
                    
                    // Color code based on positive/negative
                    const totalPnl = document.getElementById('total-pnl');
                    const dailyPnl = document.getElementById('daily-pnl');
                    
                    totalPnl.className = 'metric-value ' + (metrics.total_pnl >= 0 ? 'positive' : 'negative');
                    dailyPnl.className = 'metric-value ' + (metrics.daily_pnl >= 0 ? 'positive' : 'negative');
                }
                
                function updatePositions(positions) {
                    document.getElementById('positions').textContent = positions.length;
                }
                
                function updateAlerts(alerts) {
                    const alertsList = document.getElementById('alerts-list');
                    alertsList.innerHTML = '';
                    
                    alerts.forEach(alert => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = 'alert alert-' + alert.severity;
                        alertDiv.innerHTML = '<strong>' + alert.title + '</strong><br>' + alert.message;
                        alertsList.appendChild(alertDiv);
                    });
                }
                
                // Initial load
                updateDashboard();
            </script>
        </body>
        </html>
        """

class StreamlitDashboard:
    """Streamlit-based dashboard for more advanced visualization"""
    
    def __init__(self):
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Aster Trading Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the Streamlit dashboard"""
        st.title("📈 Aster Trading Dashboard")
        
        # Sidebar
        with st.sidebar:
            st.header("Controls")
            refresh_interval = st.slider("Refresh Interval (seconds)", 1, 60, 5)
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            
            if st.button("Emergency Stop"):
                st.error("Emergency stop triggered!")
        
        # Main content
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total P&L", "$1,234.56", "123.45")
        
        with col2:
            st.metric("Daily P&L", "$56.78", "12.34")
        
        with col3:
            st.metric("Win Rate", "67.8%", "2.3%")
        
        with col4:
            st.metric("Positions", "3", "1")
        
        # Charts
        st.subheader("Performance Charts")
        
        # P&L Chart
        chart_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H'),
            'pnl': np.cumsum(np.random.normal(0, 10, 744))
        })
        
        st.line_chart(chart_data.set_index('timestamp')['pnl'])
        
        # Positions Table
        st.subheader("Active Positions")
        positions_data = pd.DataFrame({
            'Symbol': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            'Side': ['Long', 'Short', 'Long'],
            'Size': [0.05, 0.1, 0.02],
            'Entry Price': [45000, 3200, 0.45],
            'Current Price': [46000, 3100, 0.47],
            'P&L': [50, -10, 4],
            'Strategy': ['Market Making', 'Funding Arb', 'DMark']
        })
        
        st.dataframe(positions_data, use_container_width=True)
        
        # Alerts
        st.subheader("Alerts")
        alert_data = pd.DataFrame({
            'Time': ['2024-01-31 14:30', '2024-01-31 14:25', '2024-01-31 14:20'],
            'Severity': ['Critical', 'Warning', 'Info'],
            'Message': ['Daily loss limit exceeded', 'Position limit reached', 'New trade executed']
        })
        
        st.dataframe(alert_data, use_container_width=True)
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

# Example usage
async def main():
    """Run the monitoring dashboard"""
    
    # Create configuration
    config = DashboardConfig(
        refresh_interval=5,
        max_data_points=1000
    )
    
    # Create dashboard
    dashboard = MonitoringDashboard(config)
    
    # Start monitoring
    await dashboard.start_monitoring()

if __name__ == "__main__":
    # Run Streamlit dashboard
    streamlit_dashboard = StreamlitDashboard()
    streamlit_dashboard.run()
    
    # Or run FastAPI dashboard
    # asyncio.run(main())
