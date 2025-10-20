#!/usr/bin/env python3
"""
Manually update API keys by editing the JSON file
"""

import json
import os

def create_template():
    """Create a template file for manual editing"""
    print("📝 Creating API Keys Template")
    print("="*50)
    
    template_file = ".api_keys_EDIT_ME.json"
    
    template = {
        "aster_api_key": "PASTE_YOUR_API_KEY_HERE",
        "aster_secret_key": "PASTE_YOUR_SECRET_KEY_HERE",
        "alpha_vantage_key": "WEO6JTK3E9WFRGRE",
        "finnhub_key": "d3ndn01r01qo7510l2c0d3ndn01r01qo7510l2cg",
        "fred_api_key": "a5b90245298d19b19abb6777beea54e1",
        "newsapi_key": "d725036479da4a4185537696e40b04f1",
        "metals_api_key": "Not set"
    }
    
    with open(template_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✅ Created template file: {template_file}")
    print("\n📋 Instructions:")
    print("1. Open the file '.api_keys_EDIT_ME.json' in a text editor")
    print("2. Replace 'PASTE_YOUR_API_KEY_HERE' with your actual API key")
    print("3. Replace 'PASTE_YOUR_SECRET_KEY_HERE' with your actual secret")
    print("4. Save the file")
    print("5. Run this script again to apply the changes")
    
    print("\n⚠️ IMPORTANT: Keep your API secret secure!")
    
    # Open the file in default editor (Windows)
    if os.name == 'nt':
        os.system(f'notepad {template_file}')
    
    return template_file

def apply_manual_edits():
    """Apply the manually edited template"""
    template_file = ".api_keys_EDIT_ME.json"
    
    if not os.path.exists(template_file):
        print("❌ Template file not found. Creating it now...")
        create_template()
        return False
    
    # Load the edited template
    try:
        with open(template_file, 'r') as f:
            keys = json.load(f)
        
        api_key = keys.get('aster_api_key', '')
        api_secret = keys.get('aster_secret_key', '')
        
        # Check if user actually edited the file
        if api_key == "PASTE_YOUR_API_KEY_HERE" or api_secret == "PASTE_YOUR_SECRET_KEY_HERE":
            print("❌ Please edit the template file and replace the placeholder values!")
            print(f"   File: {template_file}")
            return False
        
        if not api_key or not api_secret:
            print("❌ API credentials cannot be empty!")
            return False
        
    # Save to actual API keys file in local directory
    with open('local/.api_keys.json', 'w') as f:
        json.dump(keys, f, indent=2)
        
        print(f"✅ API credentials updated successfully!")
        print(f"   API Key starts with: {api_key[:10]}...")
        
        # Remove template file
        os.remove(template_file)
        print(f"✅ Removed template file")
        
        return True
        
    except json.JSONDecodeError:
        print("❌ Error reading the template file. Make sure it's valid JSON!")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main function"""
    print("🔐 Manual API Key Update")
    print("="*50)
    
    # Check if template exists
    if os.path.exists(".api_keys_EDIT_ME.json"):
        print("Found existing template file. Checking for edits...")
        if apply_manual_edits():
            print("\n✅ All done! You can now run:")
            print("  python scripts/quick_api_test.py")
    else:
        create_template()
        print("\n📝 Please edit the template file and run this script again.")

if __name__ == "__main__":
    main()
