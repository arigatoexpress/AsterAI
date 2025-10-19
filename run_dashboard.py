#!/usr/bin/env python3
"""
Script to run the trading dashboard locally.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing dashboard requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "dashboard_requirements.txt"])

def run_dashboard():
    """Run the Streamlit dashboard."""
    print("Starting trading dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "trading_dashboard.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    try:
        install_requirements()
        run_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error: {e}")