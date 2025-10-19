#!/usr/bin/env python3
"""
Test script to verify secrets are accessible
"""

import os
import requests
import json

def test_secrets():
    """Test if secrets are accessible"""
    print("Testing secrets access...")
    
    # Test the health endpoint first
    try:
        response = requests.get("https://aster-trading-agent-880429861698.us-central1.run.app/health", timeout=10)
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test a simple endpoint that shows environment variables
    try:
        response = requests.get("https://aster-trading-agent-880429861698.us-central1.run.app/", timeout=10)
        print(f"Root endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Root endpoint failed: {e}")

if __name__ == "__main__":
    test_secrets()
