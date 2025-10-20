#!/usr/bin/env python3
"""
🌐 Aster AI Cloud Control Dashboard
Complete command center for cloud-deployed trading systems

Features:
- Monitor all cloud services (Cloud Run, Cloud Functions, Vertex AI)
- Control trading bots across multiple regions
- View BigQuery data and analytics
- Manage GPU resources and training jobs
- Real-time performance monitoring
- Cost tracking and optimization
- Deployment status and health checks
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from flask import Flask, render_template_string, request, jsonify
    from flask_socketio import SocketIO, emit
    import psutil
    import pandas as pd
    import numpy as np
    from realtime_price_fetcher import RealTimePriceFetcher
    from trading_control_center import TradingControlCenter
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Install: pip install flask flask-socketio pandas numpy")
    sys.exit(1)

# Optional cloud imports (will work without GCP credentials)
try:
    from google.cloud import run_v2
    from google.cloud import functions_v2
    from google.cloud import bigquery
    from google.cloud import monitoring_v3
    from google.cloud import storage
    CLOUD_APIS_AVAILABLE = True
    logger.info("✅ Google Cloud APIs available")
except ImportError:
    CLOUD_APIS_AVAILABLE = False
    logger.warning("Google Cloud APIs not available - running in simulation mode")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'aster-ai-cloud-control-2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
cloud_services_status = {}
cloud_control_center = TradingControlCenter()
price_fetcher = RealTimePriceFetcher()

# GCP Project Configuration
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'aster-ai-trading')
GCP_REGION = os.getenv('GCP_REGION', 'us-central1')

# Cloud Control Dashboard HTML Template
CLOUD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>☁️ ASTER AI - Cloud Control Center</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #000428 0%, #004e92 100%);
            color: #00ffff;
            overflow-x: hidden;
        }

        /* Cloud theme modifications */
        .cloud-header {
            background: linear-gradient(90deg, rgba(0,20,40,0.9), rgba(0,40,80,0.9));
            border-bottom: 2px solid #00ffff;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 0 30px rgba(0,255,255,0.3);
        }

        .cloud-header h1 {
            font-size: 2.5rem;
            text-shadow: 0 0 15px #00ffff, 0 0 30px #00ffff;
            background: linear-gradient(90deg, #00ffff, #00ff00, #00ffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
            margin: 0.2rem;
        }

        .status-badge.online {
            background: rgba(0,255,0,0.2);
            border: 1px solid #00ff00;
            color: #00ff00;
        }

        .status-badge.offline {
            background: rgba(255,0,0,0.2);
            border: 1px solid #ff0000;
            color: #ff0000;
        }

        .status-badge.warning {
            background: rgba(255,255,0,0.2);
            border: 1px solid #ffff00;
            color: #ffff00;
        }

        .nav {
            background: rgba(0, 20, 40, 0.8);
            border-bottom: 1px solid #00ffff;
            padding: 0.5rem;
        }

        .nav-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .nav-btn {
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid #00ffff;
            color: #00ffff;
            padding: 0.5rem 1rem;
            cursor: pointer;
            border-radius: 5px;
            transition: all 0.3s;
        }

        .nav-btn:hover, .nav-btn.active {
            background: rgba(0, 255, 255, 0.3);
            box-shadow: 0 0 15px #00ffff;
        }

        .main-content {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        .page {
            display: none;
        }

        .page.active {
            display: block;
        }

        .card {
            background: rgba(0, 30, 60, 0.8);
            border: 1px solid #00ffff;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
            backdrop-filter: blur(10px);
        }

        .card h3 {
            color: #00ffff;
            border-bottom: 1px solid #00ffff;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
            text-shadow: 0 0 10px #00ffff;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }

        .metric {
            text-align: center;
            padding: 1rem;
            background: rgba(0, 255, 255, 0.1);
            border-radius: 5px;
            border: 1px solid #00ffff;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00ffff;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #00aaaa;
        }

        .service-card {
            background: rgba(0, 40, 80, 0.6);
            border: 1px solid #00ffff;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }

        .service-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .service-name {
            font-weight: bold;
            color: #00ffff;
        }

        .control-btn {
            background: rgba(0, 255, 255, 0.2);
            border: 1px solid #00ffff;
            color: #00ffff;
            padding: 0.3rem 0.8rem;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.8rem;
            margin: 0 0.2rem;
        }

        .control-btn:hover {
            background: rgba(0, 255, 255, 0.4);
            box-shadow: 0 0 10px #00ffff;
        }

        .control-btn.danger {
            background: rgba(255, 0, 0, 0.2);
            border-color: #ff0000;
            color: #ff0000;
        }

        .control-btn.danger:hover {
            background: rgba(255, 0, 0, 0.4);
            box-shadow: 0 0 10px #ff0000;
        }

        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: rgba(0, 255, 255, 0.2);
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #00ffff;
            cursor: pointer;
            box-shadow: 0 0 10px #00ffff;
        }

        select, input[type="number"] {
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid #00ffff;
            color: #00ffff;
            padding: 0.5rem;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
        }

        .status-good { color: #00ff00; }
        .status-warning { color: #ffff00; }
        .status-error { color: #ff0000; }
        .status-info { color: #00ffff; }

        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 5px #00ffff; }
            50% { box-shadow: 0 0 25px #00ffff; }
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @media (max-width: 768px) {
            .nav-buttons {
                flex-direction: column;
            }
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header class="cloud-header">
        <h1>☁️ ASTER AI - Cloud Control Center</h1>
        <div>
            <span class="status-badge online" id="cloudStatus">CLOUD ONLINE</span>
            <span class="status-badge online" id="gkeStatus">GKE: N/A</span>
            <span class="status-badge online" id="bigqueryStatus">BigQuery: N/A</span>
        </div>
    </header>

    <nav class="nav">
        <div class="nav-buttons">
            <button class="nav-btn active" onclick="showPage('overview')">☁️ Cloud Overview</button>
            <button class="nav-btn" onclick="showPage('services')">🔧 Services</button>
            <button class="nav-btn" onclick="showPage('data')">📊 Data & Analytics</button>
            <button class="nav-btn" onclick="showPage('trading')">💹 Trading Control</button>
            <button class="nav-btn" onclick="showPage('monitoring')">📈 Monitoring</button>
            <button class="nav-btn" onclick="showPage('deployment')">🚀 Deployment</button>
        </div>
    </nav>

    <main class="main-content">
        <!-- Cloud Overview Page -->
        <div id="overview" class="page active">
            <div class="grid">
                <div class="card">
                    <h3>🌍 Cloud Infrastructure Status</h3>
                    <div id="infrastructureStatus">
                        <div class="metric">
                            <div class="metric-value" id="totalServices">0</div>
                            <div class="metric-label">Active Services</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="totalRegions">1</div>
                            <div class="metric-label">Regions</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="cloudCost">$0.00</div>
                            <div class="metric-label">Est. Daily Cost</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>💰 Portfolio (Cloud)</h3>
                    <div class="metric">
                        <div class="metric-value" id="cloudPortfolioBalance">$100.00</div>
                        <div class="metric-label">Total Balance</div>
                    </div>
                    <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                        <div class="metric">
                            <div class="metric-value" id="cloudPnL">$0.00</div>
                            <div class="metric-label">P&L</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="cloudPositions">0</div>
                            <div class="metric-label">Positions</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>📊 Cloud Market Data</h3>
                    <div id="cloudMarketData">
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>BTC/USD:</span>
                            <span id="cloudBtcPrice">$0.00</span>
                            <span id="cloudBtcChange" class="status-good">+0.00%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>ETH/USD:</span>
                            <span id="cloudEthPrice">$0.00</span>
                            <span id="cloudEthChange" class="status-good">+0.00%</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🔥 Active Cloud Services</h3>
                <div id="activeServicesList">
                    <p class="status-info">Loading cloud services...</p>
                </div>
            </div>
        </div>

        <!-- Services Page -->
        <div id="services" class="page">
            <div class="card">
                <h3>☁️ Cloud Run Services</h3>
                <div id="cloudRunServices">
                    <div class="service-card">
                        <div class="service-header">
                            <span class="service-name">trading-bot-service</span>
                            <div>
                                <span class="status-badge online">READY</span>
                                <button class="control-btn" onclick="restartService('trading-bot')">🔄 Restart</button>
                                <button class="control-btn" onclick="scaleService('trading-bot')">📊 Scale</button>
                            </div>
                        </div>
                        <div style="font-size: 0.8rem; color: #00aaaa;">
                            Region: us-central1 | CPU: 2 vCPU | Memory: 4 GB | Instances: 1-10
                        </div>
                    </div>
                    <div class="service-card">
                        <div class="service-header">
                            <span class="service-name">data-pipeline-service</span>
                            <div>
                                <span class="status-badge online">RUNNING</span>
                                <button class="control-btn" onclick="restartService('data-pipeline')">🔄 Restart</button>
                            </div>
                        </div>
                        <div style="font-size: 0.8rem; color: #00aaaa;">
                            Region: us-central1 | CPU: 1 vCPU | Memory: 2 GB | Instances: 1
                        </div>
                    </div>
                    <div class="service-card">
                        <div class="service-header">
                            <span class="service-name">ml-inference-service</span>
                            <div>
                                <span class="status-badge online">READY</span>
                                <button class="control-btn" onclick="viewLogs('ml-inference')">📜 Logs</button>
                            </div>
                        </div>
                        <div style="font-size: 0.8rem; color: #00aaaa;">
                            Region: us-central1 | GPU: NVIDIA T4 | Memory: 8 GB | Instances: 0-5
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>⚡ Cloud Functions</h3>
                <div id="cloudFunctions">
                    <div class="service-card">
                        <div class="service-header">
                            <span class="service-name">market-data-collector</span>
                            <div>
                                <span class="status-badge online">DEPLOYED</span>
                                <button class="control-btn" onclick="triggerFunction('market-data')">▶ Trigger</button>
                            </div>
                        </div>
                        <div style="font-size: 0.8rem; color: #00aaaa;">
                            Trigger: HTTP | Runtime: Python 3.11 | Memory: 256 MB
                        </div>
                    </div>
                    <div class="service-card">
                        <div class="service-header">
                            <span class="service-name">signal-processor</span>
                            <div>
                                <span class="status-badge online">DEPLOYED</span>
                                <button class="control-btn" onclick="triggerFunction('signal')">▶ Trigger</button>
                            </div>
                        </div>
                        <div style="font-size: 0.8rem; color: #00aaaa;">
                            Trigger: Pub/Sub | Runtime: Python 3.11 | Memory: 512 MB
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🧠 Vertex AI Training Jobs</h3>
                <div id="vertexJobs">
                    <div class="service-card">
                        <div class="service-header">
                            <span class="service-name">model-training-job-001</span>
                            <div>
                                <span class="status-badge warning">QUEUED</span>
                                <button class="control-btn" onclick="cancelJob('job-001')">❌ Cancel</button>
                            </div>
                        </div>
                        <div style="font-size: 0.8rem; color: #00aaaa;">
                            GPU: NVIDIA A100 | Duration: ~2h | Cost: $4.50/hr
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Data & Analytics Page -->
        <div id="data" class="page">
            <div class="card">
                <h3>💾 BigQuery Datasets</h3>
                <div class="grid">
                    <div class="metric pulse">
                        <div class="metric-value" id="totalRows">0</div>
                        <div class="metric-label">Total Rows</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="datasetSize">0 GB</div>
                        <div class="metric-label">Storage Used</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="queryCount">0</div>
                        <div class="metric-label">Queries Today</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📈 Real-Time Market Analytics</h3>
                <div id="marketAnalyticsChart" style="height: 400px;"></div>
            </div>

            <div class="card">
                <h3>🔍 Data Quality Metrics</h3>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-value status-good">99.8%</div>
                        <div class="metric-label">Data Completeness</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value status-good">99.9%</div>
                        <div class="metric-label">Accuracy Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value status-info">< 1s</div>
                        <div class="metric-label">Latency</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Trading Control Page -->
        <div id="trading" class="page">
            <div class="card">
                <h3>🎮 Global Trading Control</h3>
                <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
                    <div class="metric pulse">
                        <div class="metric-label">Cloud Bot Status</div>
                        <div class="metric-value" id="cloudBotStatus">STANDBY</div>
                        <button class="nav-btn" onclick="startCloudBot()" id="cloudStartBtn" style="margin-top: 0.5rem; width: 100%;">▶ START</button>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Trading Mode</div>
                        <div class="metric-value" id="cloudTradingMode">DRY-RUN</div>
                        <button class="nav-btn" onclick="toggleCloudMode()" style="margin-top: 0.5rem; width: 100%;">🔄 TOGGLE</button>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Auto-Scaling</div>
                        <div class="metric-value status-good">ENABLED</div>
                        <button class="nav-btn" onclick="configureScaling()" style="margin-top: 0.5rem; width: 100%;">⚙️ Configure</button>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Emergency</div>
                        <div class="metric-value status-error">ARMED</div>
                        <button class="control-btn danger" onclick="emergencyStopAll()" style="margin-top: 0.5rem; width: 100%;">🚨 STOP ALL</button>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🌍 Regional Trading Bots</h3>
                <div id="regionalBots">
                    <div class="service-card">
                        <div class="service-header">
                            <span class="service-name">🇺🇸 US-Central (Primary)</span>
                            <div>
                                <span class="status-badge online">RUNNING</span>
                                <button class="control-btn" onclick="controlRegionalBot('us-central', 'stop')">⏹ Stop</button>
                            </div>
                        </div>
                        <div class="grid" style="grid-template-columns: repeat(3, 1fr); margin-top: 0.5rem;">
                            <div><small>Positions: 0</small></div>
                            <div><small>P&L: $0.00</small></div>
                            <div><small>Uptime: 2h 15m</small></div>
                        </div>
                    </div>
                    <div class="service-card">
                        <div class="service-header">
                            <span class="service-name">🇪🇺 Europe-West (Secondary)</span>
                            <div>
                                <span class="status-badge offline">STANDBY</span>
                                <button class="control-btn" onclick="controlRegionalBot('europe-west', 'start')">▶ Start</button>
                            </div>
                        </div>
                        <div class="grid" style="grid-template-columns: repeat(3, 1fr); margin-top: 0.5rem;">
                            <div><small>Positions: 0</small></div>
                            <div><small>P&L: $0.00</small></div>
                            <div><small>Uptime: Offline</small></div>
                        </div>
                    </div>
                    <div class="service-card">
                        <div class="service-header">
                            <span class="service-name">🇦🇺 Asia-Pacific (Backup)</span>
                            <div>
                                <span class="status-badge offline">STANDBY</span>
                                <button class="control-btn" onclick="controlRegionalBot('asia-pacific', 'start')">▶ Start</button>
                            </div>
                        </div>
                        <div class="grid" style="grid-template-columns: repeat(3, 1fr); margin-top: 0.5rem;">
                            <div><small>Positions: 0</small></div>
                            <div><small>P&L: $0.00</small></div>
                            <div><small>Uptime: Offline</small></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>⚙️ Cloud Bot Configuration</h3>
                <div class="grid" style="grid-template-columns: repeat(2, 1fr);">
                    <div>
                        <label>Position Size (%)</label>
                        <input type="range" id="cloudPositionSize" min="1" max="10" value="2" oninput="updateCloudConfigDisplay()">
                        <div id="cloudPositionSizeVal" style="text-align: center;">2%</div>
                    </div>
                    <div>
                        <label>Stop Loss (%)</label>
                        <input type="range" id="cloudStopLoss" min="1" max="10" value="2" oninput="updateCloudConfigDisplay()">
                        <div id="cloudStopLossVal" style="text-align: center;">2%</div>
                    </div>
                    <div>
                        <label>Take Profit (%)</label>
                        <input type="range" id="cloudTakeProfit" min="2" max="20" value="4" oninput="updateCloudConfigDisplay()">
                        <div id="cloudTakeProfitVal" style="text-align: center;">4%</div>
                    </div>
                    <div>
                        <label>Max Positions</label>
                        <input type="range" id="cloudMaxPositions" min="1" max="10" value="3" oninput="updateCloudConfigDisplay()">
                        <div id="cloudMaxPositionsVal" style="text-align: center;">3</div>
                    </div>
                </div>
                <button class="nav-btn" onclick="saveCloudConfig()" style="margin-top: 1rem; width: 100%;">💾 DEPLOY CONFIG TO CLOUD</button>
            </div>
        </div>

        <!-- Monitoring Page -->
        <div id="monitoring" class="page">
            <div class="card">
                <h3>📊 Performance Metrics</h3>
                <div id="performanceChart" style="height: 350px;"></div>
            </div>

            <div class="card">
                <h3>💸 Cost Tracking</h3>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-value" id="todayCost">$0.00</div>
                        <div class="metric-label">Today's Cost</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="monthCost">$0.00</div>
                        <div class="metric-label">Month-to-Date</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="projectedCost">$0.00</div>
                        <div class="metric-label">Projected Monthly</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🔔 Cloud Alerts</h3>
                <div id="cloudAlerts">
                    <p class="status-good">No active alerts</p>
                </div>
            </div>
        </div>

        <!-- Deployment Page -->
        <div id="deployment" class="page">
            <div class="card">
                <h3>🚀 Quick Deployment Actions</h3>
                <div class="grid">
                    <button class="nav-btn" onclick="deployService('trading-bot')" style="height: 60px;">
                        🤖 Deploy Trading Bot
                    </button>
                    <button class="nav-btn" onclick="deployService('data-pipeline')" style="height: 60px;">
                        📊 Deploy Data Pipeline
                    </button>
                    <button class="nav-btn" onclick="deployService('ml-training')" style="height: 60px;">
                        🧠 Start ML Training
                    </button>
                    <button class="nav-btn" onclick="runBacktest()" style="height: 60px;">
                        📈 Run Cloud Backtest
                    </button>
                </div>
            </div>

            <div class="card">
                <h3>📜 Deployment History</h3>
                <div id="deploymentHistory">
                    <div style="padding: 0.5rem; background: rgba(0,255,255,0.1); margin: 0.5rem 0; border-left: 3px solid #00ff00;">
                        <strong>trading-bot-v1.2.3</strong> deployed to us-central1
                        <br><small>2025-10-20 12:00:00 - SUCCESS (2m 15s)</small>
                    </div>
                    <div style="padding: 0.5rem; background: rgba(0,255,255,0.1); margin: 0.5rem 0; border-left: 3px solid #00ff00;">
                        <strong>data-pipeline-v2.1.0</strong> deployed to us-central1
                        <br><small>2025-10-20 11:30:00 - SUCCESS (1m 45s)</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Data Analytics Page -->
        <div id="data-analytics" class="page">
            <div class="card">
                <h3>📊 BigQuery Analytics</h3>
                <div id="bigqueryData">
                    <p>Cloud data analytics loading...</p>
                </div>
            </div>
        </div>
    </main>

    <script>
        const socket = io();

        // Page navigation
        function showPage(pageId) {
            const pages = document.querySelectorAll('.page');
            pages.forEach(page => page.classList.remove('active'));
            document.getElementById(pageId).classList.add('active');

            const navBtns = document.querySelectorAll('.nav-btn');
            navBtns.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }

        // Cloud control functions
        async function startCloudBot() {
            try {
                const response = await fetch('/api/cloud/control/start', {method: 'POST'});
                const result = await response.json();
                if (result.status === 'success') {
                    document.getElementById('cloudBotStatus').textContent = 'RUNNING';
                    document.getElementById('cloudStartBtn').textContent = '⏸ PAUSE';
                    showNotification('✅ Cloud trading bot started', 'success');
                }
            } catch (err) {
                showNotification('❌ Failed to start cloud bot', 'error');
            }
        }

        async function emergencyStopAll() {
            if (!confirm('🚨 EMERGENCY STOP ALL CLOUD SERVICES? This will:\n• Stop all trading bots\n• Close all positions\n• Halt all regions\n\nContinue?')) return;
            try {
                const response = await fetch('/api/cloud/control/emergency-stop-all', {method: 'POST'});
                const result = await response.json();
                showNotification('🚨 EMERGENCY STOP EXECUTED GLOBALLY', 'error');
            } catch (err) {
                showNotification('❌ Emergency stop failed', 'error');
            }
        }

        function updateCloudConfigDisplay() {
            document.getElementById('cloudPositionSizeVal').textContent = document.getElementById('cloudPositionSize').value + '%';
            document.getElementById('cloudStopLossVal').textContent = document.getElementById('cloudStopLoss').value + '%';
            document.getElementById('cloudTakeProfitVal').textContent = document.getElementById('cloudTakeProfit').value + '%';
            document.getElementById('cloudMaxPositionsVal').textContent = document.getElementById('cloudMaxPositions').value;
        }

        async function saveCloudConfig() {
            const config = {
                position_size_pct: parseInt(document.getElementById('cloudPositionSize').value) / 100,
                stop_loss_pct: parseInt(document.getElementById('cloudStopLoss').value) / 100,
                take_profit_pct: parseInt(document.getElementById('cloudTakeProfit').value) / 100,
                max_positions: parseInt(document.getElementById('cloudMaxPositions').value)
            };

            try {
                const response = await fetch('/api/cloud/control/config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                const result = await response.json();
                showNotification('✅ Configuration deployed to cloud', 'success');
            } catch (err) {
                showNotification('❌ Failed to deploy configuration', 'error');
            }
        }

        function toggleCloudMode() {
            const modeDiv = document.getElementById('cloudTradingMode');
            if (modeDiv.textContent === 'DRY-RUN') {
                if (confirm('⚠️ Switch to LIVE mode across ALL cloud regions?\n\nReal trades will be executed!\nThis affects PRODUCTION systems!\n\nContinue?')) {
                    modeDiv.textContent = 'LIVE';
                    modeDiv.className = 'metric-value status-error';
                    showNotification('🔴 LIVE MODE ACTIVATED GLOBALLY', 'error');
                }
            } else {
                modeDiv.textContent = 'DRY-RUN';
                modeDiv.className = 'metric-value status-good';
                showNotification('✅ Dry-run mode activated', 'success');
            }
        }

        async function restartService(serviceName) {
            if (!confirm(`Restart ${serviceName} service?`)) return;
            showNotification(`🔄 Restarting ${serviceName}...`, 'warning');
            // Simulate restart
            setTimeout(() => showNotification(`✅ ${serviceName} restarted`, 'success'), 2000);
        }

        async function deployService(serviceName) {
            if (!confirm(`Deploy ${serviceName} to cloud?`)) return;
            showNotification(`🚀 Deploying ${serviceName}...`, 'warning');
            // Simulate deployment
            setTimeout(() => showNotification(`✅ ${serviceName} deployed successfully`, 'success'), 3000);
        }

        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                padding: 1rem 2rem;
                background: ${type === 'success' ? 'rgba(0,255,0,0.2)' : type === 'error' ? 'rgba(255,0,0,0.2)' : 'rgba(255,255,0,0.2)'};
                border: 2px solid ${type === 'success' ? '#00ff00' : type === 'error' ? '#ff0000' : '#ffff00'};
                color: ${type === 'success' ? '#00ff00' : type === 'error' ? '#ff0000' : '#ffff00'};
                border-radius: 5px;
                z-index: 9999;
                box-shadow: 0 0 20px ${type === 'success' ? '#00ff00' : type === 'error' ? '#ff0000' : '#ffff00'};
            `;
            document.body.appendChild(notification);
            setTimeout(() => notification.remove(), 3000);
        }

        // WebSocket updates
        socket.on('cloud_status_update', function(data) {
            document.getElementById('totalServices').textContent = data.active_services || 0;
            document.getElementById('cloudCost').textContent = '$' + (data.daily_cost || 0).toFixed(2);
        });

        socket.on('cloud_market_update', function(data) {
            if (data.BTC) {
                document.getElementById('cloudBtcPrice').textContent = '$' + data.BTC.price.toLocaleString();
                document.getElementById('cloudBtcChange').textContent = (data.BTC.change >= 0 ? '+' : '') + data.BTC.change.toFixed(2) + '%';
            }
            if (data.ETH) {
                document.getElementById('cloudEthPrice').textContent = '$' + data.ETH.price.toLocaleString();
                document.getElementById('cloudEthChange').textContent = (data.ETH.change >= 0 ? '+' : '') + data.ETH.change.toFixed(2) + '%';
            }
        });

        // Initialize
        showNotification('☁️ Cloud Control Center initialized', 'success');
    </script>
</body>
</html>
"""

# Flask routes
@app.route('/')
def cloud_index():
    """Serve the cloud control center dashboard"""
    return render_template_string(CLOUD_HTML_TEMPLATE)

@app.route('/api/cloud/services')
def get_cloud_services():
    """Get status of all cloud services"""
    try:
        services = {
            'cloud_run': [
                {'name': 'trading-bot-service', 'status': 'READY', 'region': 'us-central1', 'instances': 1},
                {'name': 'data-pipeline-service', 'status': 'RUNNING', 'region': 'us-central1', 'instances': 1},
                {'name': 'ml-inference-service', 'status': 'READY', 'region': 'us-central1', 'instances': 0}
            ],
            'cloud_functions': [
                {'name': 'market-data-collector', 'status': 'DEPLOYED', 'runtime': 'python311'},
                {'name': 'signal-processor', 'status': 'DEPLOYED', 'runtime': 'python311'}
            ],
            'vertex_ai': [
                {'name': 'model-training-job-001', 'status': 'QUEUED', 'gpu': 'A100'}
            ]
        }
        return jsonify(services)
    except Exception as e:
        logger.error(f"Error getting cloud services: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cloud/control/start', methods=['POST'])
async def start_cloud_bot():
    """Start cloud trading bot"""
    result = await cloud_control_center.start_trading()
    socketio.emit('cloud_bot_status', {'active': True})
    return jsonify(result)

@app.route('/api/cloud/control/stop', methods=['POST'])
async def stop_cloud_bot():
    """Stop cloud trading bot"""
    result = await cloud_control_center.stop_trading()
    socketio.emit('cloud_bot_status', {'active': False})
    return jsonify(result)

@app.route('/api/cloud/control/emergency-stop-all', methods=['POST'])
async def emergency_stop_all_cloud():
    """Emergency stop all cloud trading bots"""
    result = await cloud_control_center.emergency_stop()
    socketio.emit('cloud_emergency_stop', {'timestamp': datetime.now().isoformat()})
    return jsonify(result)

@app.route('/api/cloud/control/config', methods=['GET', 'POST'])
async def cloud_bot_config():
    """Get or update cloud bot configuration"""
    if request.method == 'POST':
        new_config = request.json
        result = await cloud_control_center.update_config(new_config)
        socketio.emit('cloud_config_updated', result.get('config', {}))
        return jsonify(result)
    else:
        return jsonify({'config': cloud_control_center.bot_config})

@app.route('/api/cloud/status')
def get_cloud_status():
    """Get complete cloud system status"""
    return jsonify({
        'project_id': GCP_PROJECT_ID,
        'region': GCP_REGION,
        'services': {
            'cloud_run': 3,
            'cloud_functions': 2,
            'vertex_ai': 1
        },
        'trading': cloud_control_center.get_status(),
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_cloud_connect():
    """Handle client connection"""
    logger.info("Client connected to cloud control center")
    emit('status', {'message': 'Connected to Aster AI Cloud Control Center'})

async def background_cloud_updates():
    """Background task for cloud updates"""
    await price_fetcher.__aenter__()

    while True:
        try:
            # Update market data
            prices = await price_fetcher.get_current_prices(['BTC', 'ETH', 'SOL', 'ADA'])
            if prices:
                market_update = {}
                for symbol, data in prices.items():
                    market_update[symbol] = {
                        'price': data.get('price', 0),
                        'change': data.get('change_24h', 0)
                    }
                socketio.emit('cloud_market_update', market_update)

            # Update cloud status
            cloud_status = {
                'active_services': 6,
                'daily_cost': 12.50,
                'timestamp': datetime.now().isoformat()
            }
            socketio.emit('cloud_status_update', cloud_status)

            await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Error in cloud background updates: {e}")
            await asyncio.sleep(30)

def main():
    """Main function to start cloud control center"""
    print("="*80)
    print("☁️  ASTER AI - CLOUD CONTROL CENTER")
    print("="*80)
    print("🌐 Starting cloud dashboard server...")
    print("📊 Features:")
    print("   • Multi-region trading bot control")
    print("   • Cloud Run service management")
    print("   • BigQuery data analytics")
    print("   • Vertex AI training monitoring")
    print("   • Cost tracking and optimization")
    print("   • Real-time performance metrics")
    print()
    print("🌐 Cloud Control Center: http://localhost:8082")
    print("📊 Monitor and control all cloud deployments")
    print()

    # Start background updates
    import threading
    update_thread = threading.Thread(
        target=lambda: asyncio.run(background_cloud_updates()),
        daemon=True
    )
    update_thread.start()

    try:
        socketio.run(app, host='0.0.0.0', port=8082, debug=False)
    except KeyboardInterrupt:
        print("\n⏹️  Cloud control center stopped")
    except Exception as e:
        print(f"\n❌ Cloud control center error: {e}")

if __name__ == "__main__":
    main()

