#!/usr/bin/env python3
"""
Integrate Dashboard Control with Cloud Trading Services

This script connects the dashboard control endpoints to the actual running
trading services (aster-trading-agent and aster-self-learning-trader).
"""

import os
import sys
import json
import requests
from typing import Dict, Any, Optional
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudServiceController:
    """Controller for cloud trading services"""

    def __init__(self):
        # GCP Project details
        self.project_id = "quant-ai-trader-credits"
        self.region = "us-central1"
        self.project_num = "880429861698"

        # Service URLs
        self.services = {
            "aster-trading-agent": f"https://aster-trading-agent-{self.project_num}.{self.region}.run.app",
            "aster-self-learning-trader": f"https://aster-self-learning-trader-{self.project_num}.{self.region}.run.app",
            "aster-enhanced-dashboard": f"https://aster-enhanced-dashboard-{self.project_num}.{self.region}.run.app"
        }

    def check_service_status(self, service_name: str) -> Dict[str, Any]:
        """Check status of a service"""
        try:
            url = f"{self.services[service_name]}/status"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def start_service(self, service_name: str) -> Dict[str, Any]:
        """Start a trading service"""
        try:
            url = f"{self.services[service_name]}/start"
            response = requests.post(url, timeout=30)
            if response.status_code == 200:
                return {"status": "success", "message": f"{service_name} started"}
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def stop_service(self, service_name: str) -> Dict[str, Any]:
        """Stop a trading service"""
        try:
            url = f"{self.services[service_name]}/stop"
            response = requests.post(url, timeout=30)
            if response.status_code == 200:
                return {"status": "success", "message": f"{service_name} stopped"}
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_service_performance(self, service_name: str) -> Dict[str, Any]:
        """Get performance data from a service"""
        try:
            url = f"{self.services[service_name]}/performance"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_service_positions(self, service_name: str) -> Dict[str, Any]:
        """Get positions data from a service"""
        try:
            url = f"{self.services[service_name]}/positions"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_combined_status(self) -> Dict[str, Any]:
        """Get combined status from all trading services"""
        trading_status = self.check_service_status("aster-trading-agent")
        self_learning_status = self.check_service_status("aster-self-learning-trader")

        # Determine overall status
        trading_running = trading_status.get("status") == "running"
        self_learning_running = self_learning_status.get("status") == "running"

        if trading_running or self_learning_running:
            overall_status = "running"
        else:
            overall_status = "stopped"

        return {
            "service": "Self-Learning Trading Bot",
            "version": "2.0.0",
            "status": overall_status,
            "features": [
                "Self-learning ML models",
                "Aggressive perpetual trading",
                "Multiple trading strategies",
                "Real-time adaptation",
                "Risk management"
            ],
            "endpoints": {
                "health": "/health",
                "status": "/status",
                "performance": "/performance",
                "positions": "/positions",
                "market_data": "/market-data",
                "strategy_weights": "/strategy-weights",
                "learning_status": "/learning-status",
                "start": "POST /start",
                "stop": "POST /stop",
                "manual_trade": "POST /manual-trade"
            },
            "services": {
                "aster_trading_agent": trading_status,
                "aster_self_learning_trader": self_learning_status
            }
        }

    def start_all_trading(self) -> Dict[str, Any]:
        """Start all trading services"""
        results = {}

        # Start main trading agent
        print("Starting aster-trading-agent...")
        results["aster_trading_agent"] = self.start_service("aster-trading-agent")

        # Start self-learning trader
        print("Starting aster-self-learning-trader...")
        results["aster_self_learning_trader"] = self.start_service("aster-self-learning-trader")

        return {
            "status": "completed",
            "results": results,
            "message": "Trading services start initiated"
        }

    def stop_all_trading(self) -> Dict[str, Any]:
        """Stop all trading services"""
        results = {}

        # Stop main trading agent
        print("Stopping aster-trading-agent...")
        results["aster_trading_agent"] = self.stop_service("aster-trading-agent")

        # Stop self-learning trader
        print("Stopping aster-self-learning-trader...")
        results["aster_self_learning_trader"] = self.stop_service("aster-self-learning-trader")

        return {
            "status": "completed",
            "results": results,
            "message": "Trading services stop initiated"
        }

    def get_combined_performance(self) -> Dict[str, Any]:
        """Get combined performance from all services"""
        trading_perf = self.get_service_performance("aster-trading-agent")
        self_learning_perf = self.get_service_performance("aster-self-learning-trader")

        # Combine performance data
        combined = {
            "trading_agent": trading_perf,
            "self_learning_trader": self_learning_perf,
            "combined_metrics": {}
        }

        # Calculate combined metrics if both have data
        if (trading_perf.get("status") != "error" and
            self_learning_perf.get("status") != "error"):

            # Simple combination - you can make this more sophisticated
            combined["combined_metrics"] = {
                "total_pnl": (trading_perf.get("total_pnl", 0) +
                            self_learning_perf.get("total_pnl", 0)),
                "total_trades": (trading_perf.get("total_trades", 0) +
                               self_learning_perf.get("total_trades", 0))
            }

        return combined

def main():
    """Main function for testing the integration"""
    print("🔗 Cloud Service Integration Test")
    print("=" * 50)

    controller = CloudServiceController()

    # Test combined status
    print("\n1. Testing Combined Status...")
    status = controller.get_combined_status()
    print("Status:", status["status"])
    print("Services:")
    for service, service_status in status["services"].items():
        print(f"  {service}: {service_status.get('status', 'unknown')}")

    # Test starting services
    print("\n2. Testing Start All Trading...")
    start_result = controller.start_all_trading()
    print("Start result:", start_result["status"])

    # Wait a moment and check status again
    print("\n3. Checking Status After Start...")
    import time
    time.sleep(3)
    status_after = controller.get_combined_status()
    print("Status after start:", status_after["status"])

    # Test performance
    print("\n4. Testing Combined Performance...")
    perf = controller.get_combined_performance()
    print("Performance data retrieved for", len(perf.get("combined_metrics", {})), "services")

    print("\n✅ Integration test completed!")

if __name__ == "__main__":
    main()
