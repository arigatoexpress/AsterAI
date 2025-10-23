#!/usr/bin/env python3
"""
Cloud Bot Startup Script
Starts and monitors the deployed trading bots on GCP.
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Optional
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudBotManager:
    """Manages cloud-deployed trading bots"""

    def __init__(self, project_id: str = "quant-ai-trader-credits", region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.base_url = f"https://{region}-run.googleapis.com"

        # Service URLs (using the actual deployed service names with project number)
        # Project number for quant-ai-trader-credits is 880429861698
        project_num = "880429861698"
        self.services = {
            "aster-trading-agent": f"https://aster-trading-agent-{project_num}.{region}.run.app",
            "aster-self-learning-trader": f"https://aster-self-learning-trader-{project_num}.{region}.run.app",
            "aster-enhanced-dashboard": f"https://aster-enhanced-dashboard-{project_num}.{region}.run.app"
        }

    def check_service_health(self, service_name: str) -> Dict:
        """Check health of a service"""
        try:
            url = f"{self.services[service_name]}/health"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to check {service_name} health: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def check_service_status(self, service_name: str) -> Dict:
        """Check status of a service"""
        try:
            url = f"{self.services[service_name]}/status"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to check {service_name} status: {e}")
            return {"status": "error", "error": str(e)}

    def start_service(self, service_name: str) -> Dict:
        """Start a trading service"""
        try:
            url = f"{self.services[service_name]}/start"
            response = requests.post(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to start {service_name}: {e}")
            return {"status": "error", "error": str(e)}

    def stop_service(self, service_name: str) -> Dict:
        """Stop a trading service"""
        try:
            url = f"{self.services[service_name]}/stop"
            response = requests.post(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to stop {service_name}: {e}")
            return {"status": "error", "error": str(e)}

    def get_service_logs(self, service_name: str, lines: int = 50) -> str:
        """Get recent logs from a service"""
        try:
            import subprocess
            cmd = [
                "gcloud", "logging", "read",
                f"resource.type=cloud_run_revision AND resource.labels.service_name={service_name}",
                f"--project={self.project_id}",
                f"--limit={lines}",
                "--format=value(textPayload)"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error getting logs: {result.stderr}"
        except Exception as e:
            return f"Failed to get logs: {e}"

    def check_all_services(self) -> Dict[str, Dict]:
        """Check health and status of all services"""
        results = {}
        for service_name in self.services.keys():
            logger.info(f"Checking {service_name}...")
            health = self.check_service_health(service_name)
            status = self.check_service_status(service_name)
            results[service_name] = {
                "health": health,
                "status": status,
                "url": self.services[service_name]
            }
        return results

    def start_all_trading_bots(self) -> Dict[str, Dict]:
        """Start all trading bots"""
        results = {}
        trading_services = ["aster-trading-agent", "aster-self-learning-trader"]

        for service_name in trading_services:
            logger.info(f"Starting {service_name}...")
            result = self.start_service(service_name)
            results[service_name] = result
            time.sleep(2)  # Brief pause between starts

        return results

    def get_system_overview(self) -> Dict:
        """Get comprehensive system overview"""
        services_status = self.check_all_services()

        return {
            "timestamp": time.time(),
            "project_id": self.project_id,
            "region": self.region,
            "services": services_status,
            "trading_active": any(
                service.get("status", {}).get("running", False)
                for service in services_status.values()
            )
        }

def print_service_status(service_name: str, info: Dict):
    """Print formatted service status"""
    print(f"\n🔍 {service_name.upper()}")
    print("-" * 50)
    print(f"URL: {info['url']}")
    print(f"Health: {info['health'].get('status', 'unknown')}")
    print(f"Status: {info['status'].get('status', 'unknown')}")

    if 'running' in info['status']:
        running_status = "🟢 RUNNING" if info['status']['running'] else "🔴 STOPPED"
        print(f"Trading: {running_status}")

    if 'error' in info['health']:
        print(f"❌ Error: {info['health']['error']}")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Cloud Bot Management Script")
    parser.add_argument("--action", choices=["status", "start", "stop", "logs", "overview"],
                       default="status", help="Action to perform")
    parser.add_argument("--service", help="Specific service to act on")
    parser.add_argument("--project", default="quant-ai-trader-credits", help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")

    args = parser.parse_args()

    manager = CloudBotManager(args.project, args.region)

    if args.action == "status":
        if args.service:
            info = {
                "url": manager.services[args.service],
                "health": manager.check_service_health(args.service),
                "status": manager.check_service_status(args.service)
            }
            print_service_status(args.service, info)
        else:
            results = manager.check_all_services()
            print("🚀 AsterAI Cloud Services Status")
            print("=" * 60)
            for service_name, info in results.items():
                print_service_status(service_name, info)

    elif args.action == "start":
        if args.service:
            result = manager.start_service(args.service)
            print(f"✅ Started {args.service}: {result}")
        else:
            results = manager.start_all_trading_bots()
            print("🚀 Starting All Trading Bots")
            print("=" * 40)
            for service_name, result in results.items():
                status = "✅ SUCCESS" if result.get("status") != "error" else "❌ FAILED"
                print(f"{service_name}: {status}")

    elif args.action == "stop":
        if args.service:
            result = manager.stop_service(args.service)
            print(f"🛑 Stopped {args.service}: {result}")
        else:
            print("Please specify a service to stop with --service")

    elif args.action == "logs":
        if args.service:
            logs = manager.get_service_logs(args.service)
            print(f"📋 Recent logs from {args.service}:")
            print("-" * 50)
            print(logs)
        else:
            print("Please specify a service with --service")

    elif args.action == "overview":
        overview = manager.get_system_overview()
        print("📊 System Overview")
        print("=" * 30)
        print(json.dumps(overview, indent=2, default=str))

if __name__ == "__main__":
    main()
