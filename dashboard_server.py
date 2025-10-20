#!/usr/bin/env python3
"""
AsterAI Advanced RTX Trading Dashboard Server

Launches the advanced dashboard with real-time GPU data, live experiments,
and interactive trading controls.
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'flask', 'flask-socketio', 'psutil', 'plotly'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️  Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False

    return True

def main():
    """Launch the advanced dashboard server."""
    print("🚀 AsterAI Advanced RTX Trading Dashboard")
    print("=" * 60)
    print("🎮 GPU-Accelerated Real-Time Trading Interface")
    print("📊 Live Experiment Results & Performance Monitoring")
    print("⚡ WebSocket Live Data Streaming")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        return

    # Kill any existing processes on port 8081
    try:
        if sys.platform == "win32":
            subprocess.run("netstat -ano | findstr :8081 | findstr LISTENING", shell=True, capture_output=True)
            # Kill process if found
            result = subprocess.run("netstat -ano | findstr :8081 | findstr LISTENING", shell=True, capture_output=True, text=True)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[4]
                            try:
                                subprocess.run(f"taskkill /f /pid {pid}", shell=True, capture_output=True)
                                print(f"✅ Killed existing process on port 8081 (PID: {pid})")
                            except:
                                pass
        else:
            subprocess.run("pkill -f 'python.*advanced_dashboard_server'", shell=True, capture_output=True)
    except:
        pass

    print("\n🌐 Starting Advanced Dashboard Server...")
    print("📊 Features:")
    print("   • Real-time GPU performance monitoring")
    print("   • Live experiment results streaming")
    print("   • Interactive trading controls")
    print("   • Auto-updating charts and metrics")
    print("   • WebSocket live data connections")
    print()

    try:
        # Launch the advanced dashboard
        subprocess.run([sys.executable, 'advanced_dashboard_server.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting dashboard server: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

    print("\n✅ Dashboard server shutdown complete.")

if __name__ == "__main__":
    main()
