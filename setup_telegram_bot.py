"""
Setup Script for Telegram Bot Integration
Run this to configure your bot and connect it to AsterAI
"""

import os
import requests
import sys
import json

def setup_telegram_bot():
    """Setup Telegram bot webhook and configuration"""
    
    print("="*60)
    print("AsterAI Telegram Bot Setup")
    print("="*60)
    print()
    
    # Step 1: Get bot token
    print("[STEP 1] Get Bot Token")
    print("1. Open Telegram and search for @BotFather")
    print("2. Send /newbot and follow instructions")
    print("3. Copy the bot token you receive")
    print()
    
    bot_token = input("Enter your Telegram bot token: ").strip()
    if not bot_token:
        print("[ERROR] Bot token required")
        sys.exit(1)
    
    # Step 2: Get chat ID
    print()
    print("[STEP 2] Get Your Chat ID")
    print("1. Send /start to your new bot in Telegram")
    print("2. Visit this URL in your browser:")
    print(f"   https://api.telegram.org/bot{bot_token}/getUpdates")
    print("3. Look for 'chat':{'id': YOUR_CHAT_ID}")
    print()
    
    chat_id = input("Enter your chat ID: ").strip()
    if not chat_id:
        print("[ERROR] Chat ID required")
        sys.exit(1)
    
    # Step 3: Choose deployment mode
    print()
    print("[STEP 3] Choose Deployment Mode")
    print("a) Cloud only (webhook to Cloud Run)")
    print("b) Local only (polling mode)")
    print("c) Hybrid (webhook to cloud, fallback to local)")
    print()
    
    mode = input("Select mode (a/b/c) [c]: ").strip().lower() or "c"
    
    webhook_url = None
    if mode in ["a", "c"]:
        print()
        webhook_url = input("Enter Cloud Run URL (e.g., https://your-service.run.app): ").strip()
        if not webhook_url:
            print("[WARNING] No webhook URL - using local polling mode")
            mode = "b"
    
    # Save configuration
    config_data = {
        "TELEGRAM_BOT_TOKEN": bot_token,
        "TELEGRAM_CHAT_ID": chat_id,
        "TELEGRAM_MODE": mode
    }
    
    if webhook_url:
        config_data["TELEGRAM_WEBHOOK_URL"] = webhook_url
    
    # Save to local config file
    os.makedirs("local", exist_ok=True)
    config_path = "local/.telegram_config"
    with open(config_path, "w") as f:
        for key, value in config_data.items():
            f.write(f"{key}={value}\n")
    
    print()
    print(f"[OK] Configuration saved to {config_path}")
    
    # Save as JSON for easy loading
    with open("local/telegram_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"[OK] Configuration also saved as JSON")
    
    # Set up webhook if cloud mode
    if webhook_url and mode in ["a", "c"]:
        print()
        print("[SETUP] Configuring Telegram webhook...")
        
        webhook_endpoint = f"{webhook_url}/webhook/{bot_token}"
        
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{bot_token}/setWebhook",
                json={"url": webhook_endpoint},
                timeout=10
            )
            
            result = response.json()
            if result.get("ok"):
                print(f"[OK] Webhook set to {webhook_endpoint}")
            else:
                print(f"[ERROR] Webhook setup failed: {result.get('description', 'Unknown error')}")
                print("[INFO] You can set it manually later")
        except Exception as e:
            print(f"[ERROR] Webhook request failed: {e}")
            print("[INFO] You can set it manually later")
    
    # Add to GCP secrets if in cloud mode
    if mode in ["a", "c"]:
        print()
        print("[GCP] Adding secrets to Google Cloud Secret Manager...")
        print("Run these commands:")
        print()
        print(f'echo -n "{bot_token}" | gcloud secrets create TELEGRAM_BOT_TOKEN --data-file=-')
        print(f'echo -n "{chat_id}" | gcloud secrets create TELEGRAM_CHAT_ID --data-file=-')
        print()
        
        add_secrets = input("Add secrets automatically? (y/n) [n]: ").strip().lower()
        if add_secrets == "y":
            try:
                # Create secrets
                import subprocess
                
                subprocess.run([
                    "bash", "-c",
                    f'echo -n "{bot_token}" | gcloud secrets create TELEGRAM_BOT_TOKEN --data-file=- 2>/dev/null || echo -n "{bot_token}" | gcloud secrets versions add TELEGRAM_BOT_TOKEN --data-file=-'
                ], check=True)
                
                subprocess.run([
                    "bash", "-c",
                    f'echo -n "{chat_id}" | gcloud secrets create TELEGRAM_CHAT_ID --data-file=- 2>/dev/null || echo -n "{chat_id}" | gcloud secrets versions add TELEGRAM_CHAT_ID --data-file=-'
                ], check=True)
                
                print("[OK] Secrets added to GCP")
            except Exception as e:
                print(f"[ERROR] Failed to add secrets: {e}")
                print("[INFO] Add them manually using the commands above")
    
    # Final instructions
    print()
    print("="*60)
    print("[OK] TELEGRAM BOT SETUP COMPLETE!")
    print("="*60)
    print()
    print("[NEXT STEPS]")
    print("1. Test your bot by sending: /status")
    print("2. Deploy cloud service with: python deploy_complete_system.py")
    print("3. Start trading with: /start_trading")
    print()
    print("[IMPORTANT]")
    print(f"Bot Token: {bot_token[:10]}... (saved securely)")
    print(f"Chat ID: {chat_id}")
    print(f"Mode: {mode}")
    print()

if __name__ == "__main__":
    try:
        setup_telegram_bot()
    except KeyboardInterrupt:
        print("\n[CANCELLED] Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        sys.exit(1)

