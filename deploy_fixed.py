#!/usr/bin/env python3
"""
Fixed AsterAI Deployment Script
Runs all services on different ports to avoid conflicts
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

def run_command(cmd, name, background=False):
    """Run a command and return the process."""
    print(f"Starting {name}...")
    try:
        if background:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            return process
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
            return result
    except Exception as e:
        print(f"Error starting {name}: {e}")
        return None

def kill_existing_processes():
    """Kill existing Python processes."""
    print("Stopping existing Python processes...")
    try:
        if sys.platform == "win32":
            subprocess.run("taskkill /f /im python.exe", shell=True, capture_output=True)
        else:
            subprocess.run("pkill -f python", shell=True, capture_output=True)
    except:
        pass

def main():
    print("=" * 60)
    print("AsterAI Fixed Deployment - No Port Conflicts")
    print("=" * 60)

    # Kill existing processes
    kill_existing_processes()
    time.sleep(2)

    processes = []

    try:
        # 1. Start Trading Analysis (background)
        print("\n[1/4] Starting Trading Analysis...")
        analysis_process = run_command(
            "python comprehensive_trading_analysis.py",
            "Trading Analysis",
            background=True
        )
        if analysis_process:
            processes.append(("Trading Analysis", analysis_process))
        time.sleep(3)

        # 2. Start Dashboard Server on port 8000
        print("\n[2/4] Starting Dashboard Server (port 8000)...")
        dashboard_process = run_command(
            "python dashboard_server.py",
            "Dashboard Server",
            background=True
        )
        if dashboard_process:
            processes.append(("Dashboard Server", dashboard_process))
        time.sleep(2)

        # 3. Start Trading Server on port 8001
        print("\n[3/4] Starting Trading Server (port 8001)...")
        trading_process = run_command(
            "python enhanced_trading_server.py",
            "Trading Server",
            background=True
        )
        if trading_process:
            processes.append(("Trading Server", trading_process))
        time.sleep(2)

        # 4. Start Self-Learning Trader on port 8002
        print("\n[4/4] Starting Self-Learning Trader (port 8002)...")
        self_learning_process = run_command(
            "python self_learning_trader.py",
            "Self-Learning Trader",
            background=True
        )
        if self_learning_process:
            processes.append(("Self-Learning Trader", self_learning_process))
        time.sleep(2)

        print("\n" + "=" * 60)
        print("DEPLOYMENT COMPLETE")
        print("=" * 60)

        # Check if processes are still running
        print("\nProcess Status:")
        for name, process in processes:
            if process.poll() is None:
                print(f"✅ {name}: Running (PID {process.pid})")
            else:
                print(f"❌ {name}: Stopped")

        print("\n🌐 Access URLs:")
        print("📊 Dashboard:    http://localhost:8000")
        print("🤖 Trading API:  http://localhost:8001")
        print("🎯 Cloud Bot:    https://aster-self-learning-trader-880429861698.us-central1.run.app")

        print("\n📁 Important Files:")
        print("📋 Deployment Report: DEPLOYMENT_SUMMARY.md")
        print("📊 Analysis Reports: trading_analysis_reports/")
        print("📈 Visualizations: trading_analysis_reports/trading_visualizations_*/")

        print("\n⚠️  Press Ctrl+C to stop all services")

        # Keep running and monitor
        try:
            while True:
                time.sleep(10)
                # Check if any process died
                dead_processes = []
                for name, process in processes:
                    if process.poll() is not None:
                        dead_processes.append(name)

                if dead_processes:
                    print(f"\n❌ These processes stopped: {', '.join(dead_processes)}")
                    break

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down services...")

    except Exception as e:
        print(f"\n❌ Deployment error: {e}")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        for name, process in processes:
            try:
                if process.poll() is None:
                    if sys.platform == "win32":
                        process.terminate()
                    else:
                        os.kill(process.pid, signal.SIGTERM)
                print(f"✅ {name}: Stopped")
            except:
                pass

        print("\n✅ All services stopped. Run this script again to restart.")

if __name__ == "__main__":
    main()
