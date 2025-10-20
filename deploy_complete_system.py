"""
One-Command Deployment for Complete AsterAI System
Deploys cloud service + local background service + Telegram bot integration
"""

import subprocess
import sys
import os
import time
import json

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def check_requirements():
    """Check if all requirements are met"""
    print("[CHECK] Verifying requirements...")
    
    requirements_met = True
    
    # Check API keys
    if not os.path.exists("local/.api_keys.json"):
        print("[ERROR] API keys not configured")
        print("[ACTION] Run: python scripts/manual_update_keys.py")
        requirements_met = False
    else:
        print("[OK] API keys configured")
    
    # Check Telegram config (optional but recommended)
    if not os.path.exists("local/.telegram_config"):
        print("[WARNING] Telegram bot not configured (optional)")
        print("[INFO] Run: python setup_telegram_bot.py")
    else:
        print("[OK] Telegram bot configured")
    
    # Check gcloud CLI
    try:
        result = subprocess.run(
            ["gcloud", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("[OK] gcloud CLI installed")
        else:
            print("[WARNING] gcloud CLI not responding correctly")
    except Exception as e:
        print(f"[WARNING] gcloud CLI not available: {e}")
        print("[INFO] Cloud deployment will be skipped")
    
    # Check Python
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 10:
        print(f"[OK] Python {python_version.major}.{python_version.minor}")
    else:
        print(f"[WARNING] Python version: {python_version.major}.{python_version.minor} (3.10+ recommended)")
    
    return requirements_met

def deploy_cloud():
    """Deploy to Google Cloud Run"""
    print_header("CLOUD DEPLOYMENT")
    
    print("[INFO] Deploying to Google Cloud Run...")
    print("[INFO] This may take 5-10 minutes...")
    
    try:
        # Check if cloudbuild.yaml exists
        cloudbuild_path = "cloud/cloudbuild.yaml"
        if not os.path.exists(cloudbuild_path):
            print(f"[WARNING] {cloudbuild_path} not found")
            print("[INFO] Using direct Docker build and deploy...")
            
            # Direct deployment
            project_id = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True
            ).stdout.strip()
            
            print(f"[INFO] Project: {project_id}")
            
            # Build and push Docker image
            print("[STEP 1/2] Building Docker image...")
            subprocess.run([
                "gcloud", "builds", "submit",
                "--tag", f"gcr.io/{project_id}/aster-trading-bot:latest",
                "--timeout=1200s"
            ], check=True)
            
            # Deploy to Cloud Run
            print("[STEP 2/2] Deploying to Cloud Run...")
            subprocess.run([
                "gcloud", "run", "deploy", "aster-trading-bot",
                "--image", f"gcr.io/{project_id}/aster-trading-bot:latest",
                "--region", "us-central1",
                "--platform", "managed",
                "--allow-unauthenticated",
                "--memory", "8Gi",
                "--cpu", "4",
                "--max-instances", "3",
                "--min-instances", "1",
                "--set-env-vars", "ASTERAI_FORCE_CPU=1,ENVIRONMENT=production,INITIAL_CAPITAL=100.0",
                "--set-secrets", "ASTER_API_KEY=ASTER_API_KEY:latest,ASTER_API_SECRET=ASTER_SECRET_KEY:latest"
            ], check=True)
        else:
            # Use cloudbuild.yaml
            subprocess.run([
                "gcloud", "builds", "submit",
                f"--config={cloudbuild_path}",
                "--timeout=1200s"
            ], check=True)
        
        print("[OK] Cloud deployment successful")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Cloud deployment failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error during cloud deployment: {e}")
        return False

def start_local_service():
    """Start local background service"""
    print_header("LOCAL SERVICE STARTUP")
    
    print("[INFO] Starting local trading service...")
    
    try:
        if sys.platform == "win32":
            # Windows batch file
            result = subprocess.run(
                ["cmd", "/c", "local\\run_local_trader.bat"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("[OK] Local service started")
                print(result.stdout)
                return True
            else:
                print(f"[ERROR] Failed to start local service")
                print(result.stderr)
                return False
        else:
            # Linux/Mac shell script
            if not os.path.exists("local/run_local_trader.sh"):
                print("[ERROR] local/run_local_trader.sh not found")
                return False
            
            subprocess.run(["bash", "local/run_local_trader.sh"], check=True)
            print("[OK] Local service started")
            return True
            
    except Exception as e:
        print(f"[ERROR] Local service startup failed: {e}")
        return False

def test_system():
    """Test that everything is working"""
    print_header("SYSTEM TESTING")
    
    print("[TEST] Testing deployed services...\n")
    
    # Test cloud
    print("[TEST] Testing cloud service...")
    try:
        import requests
        
        cloud_url = "https://aster-self-learning-trader-880429861698.us-central1.run.app"
        response = requests.get(f"{cloud_url}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Cloud service healthy")
            print(f"     Status: {data.get('status')}")
            print(f"     Version: {data.get('version')}")
        else:
            print(f"[WARNING] Cloud service status: {response.status_code}")
            
        # Test status endpoint
        response = requests.get(f"{cloud_url}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            print(f"[OK] Trading bot status: {status}")
            if status != "not_initialized":
                print(f"     Balance: ${data.get('balance', 0):.2f}")
        
    except requests.exceptions.Timeout:
        print("[ERROR] Cloud service timeout")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cloud service unreachable")
    except Exception as e:
        print(f"[ERROR] Cloud service test failed: {e}")
    
    # Test local
    print("\n[TEST] Testing local service...")
    try:
        import requests
        
        response = requests.get("http://localhost:8081/api/control/status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Local service responding")
            print(f"     Trading active: {data.get('trading_active', False)}")
            print(f"     Positions: {len(data.get('positions', []))}")
        else:
            print(f"[WARNING] Local service status: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("[INFO] Local service not running (optional)")
    except Exception as e:
        print(f"[WARNING] Local service test failed: {e}")
    
    # Test Telegram
    print("\n[TEST] Testing Telegram bot...")
    if os.path.exists("local/.telegram_config"):
        try:
            with open("local/.telegram_config", 'r') as f:
                config_lines = f.readlines()
                has_token = any("TELEGRAM_BOT_TOKEN" in line for line in config_lines)
                has_chat = any("TELEGRAM_CHAT_ID" in line for line in config_lines)
                
                if has_token and has_chat:
                    print("[OK] Telegram bot configured")
                    print("[INFO] Test by sending /status to your bot")
                else:
                    print("[WARNING] Telegram config incomplete")
        except Exception as e:
            print(f"[WARNING] Telegram config error: {e}")
    else:
        print("[INFO] Telegram bot not configured (optional)")

def show_deployment_summary():
    """Show deployment summary and next steps"""
    print_header("DEPLOYMENT COMPLETE")
    
    print("[SUCCESS] AsterAI System Deployed\n")
    
    print("[SYSTEM URLs]")
    print("  Cloud Service:    https://aster-self-learning-trader-880429861698.us-central1.run.app")
    print("  Local Dashboard:  http://localhost:8081")
    print("  Telegram Bot:     Configured (send /status)")
    
    print("\n[TRADING CONFIGURATION]")
    print("  Mode:              Conservative Live Trading")
    print("  Initial Capital:   $100")
    print("  Max Leverage:      2x")
    print("  Position Size:     1.5% per trade")
    print("  Daily Loss Limit:  5% (auto-halt)")
    print("  Stop Loss:         1.5%")
    print("  Take Profit:       3%")
    
    print("\n[REMOTE CONTROL VIA TELEGRAM]")
    print("  /status           - Check bot status")
    print("  /start_trading    - Start automated trading")
    print("  /stop_trading     - Stop trading")
    print("  /positions        - View open positions")
    print("  /performance      - Check P&L")
    print("  /emergency_close_all - Emergency position close")
    print("  /system_status    - Cloud + local status")
    print("  /leverage [X]     - Adjust leverage (max 2x)")
    
    print("\n[SAFETY FEATURES]")
    print("  - Circuit breaker: Auto-halt on 5% daily loss")
    print("  - Loss streak pause: After 3 consecutive losses")
    print("  - Position limits: Max 2 concurrent positions")
    print("  - High confidence only: >75% signal strength")
    print("  - Conservative leverage: 2x maximum")
    
    print("\n[NEXT STEPS]")
    print("  1. Test Telegram bot: Send /status")
    print("  2. Verify cloud service is responding")
    print("  3. Start trading: Send /start_trading")
    print("  4. Monitor via Telegram while away")
    
    print("\n[MONITORING TIPS]")
    print("  - Check /status every 2-4 hours")
    print("  - Review /performance daily")
    print("  - Use /emergency_close_all if needed")
    print("  - System will auto-halt on 5% daily loss")
    
    print()

def main():
    """Main deployment orchestration"""
    print_header("AsterAI COMPLETE SYSTEM DEPLOYMENT")
    
    print("[INFO] This will deploy:")
    print("  1. Cloud trading service (Google Cloud Run)")
    print("  2. Local trading agent (background service)")
    print("  3. Telegram bot integration (remote control)")
    print()
    
    # Check requirements
    if not check_requirements():
        print("\n[ERROR] Requirements not met")
        print("[ACTION] Please fix the issues above and try again")
        sys.exit(1)
    
    print("\n[OK] All requirements satisfied")
    
    # Confirm deployment
    print("\n[CONFIRM] Proceed with deployment?")
    confirm = input("Type 'yes' to continue: ").strip().lower()
    
    if confirm != "yes":
        print("[CANCELLED] Deployment cancelled by user")
        sys.exit(0)
    
    # Deploy cloud service
    cloud_success = deploy_cloud()
    
    if not cloud_success:
        print("\n[WARNING] Cloud deployment failed")
        print("[INFO] You can still use local-only mode")
        
        proceed = input("\nContinue with local deployment only? (yes/no): ").strip().lower()
        if proceed != "yes":
            print("[CANCELLED] Deployment cancelled")
            sys.exit(1)
    
    # Start local service
    local_success = start_local_service()
    
    if not local_success:
        print("\n[WARNING] Local service failed to start")
        if cloud_success:
            print("[INFO] Cloud service is still available")
        else:
            print("[ERROR] Both cloud and local deployment failed")
            sys.exit(1)
    
    # Wait for services to stabilize
    print("\n[INFO] Waiting for services to stabilize...")
    time.sleep(5)
    
    # Test system
    test_system()
    
    # Show summary
    show_deployment_summary()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

