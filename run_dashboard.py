#!/usr/bin/env python3
"""
Rari Trade AI Launcher
======================

This script ensures the correct environment is set up before launching
the Rari Trade AI dashboard.
"""

import sys
import os

def main():
    # Ensure the project root is in the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Set environment variables for Streamlit
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'

    # Import and run streamlit
    import subprocess

    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        os.path.join(project_root, 'dashboard', 'app.py'),
        '--server.headless', 'true',
        '--server.port', '8501'
    ]

    print("🚀 Launching Rari Trade AI Dashboard...")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[:3]}...")
    print(f"🌐 Dashboard will be available at: http://localhost:8501")
    print()

    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Rari Trade AI Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
