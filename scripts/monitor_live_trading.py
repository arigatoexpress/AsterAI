#!/usr/bin/env python3
"""
Live Trading Monitor
Real-time monitoring of paper trading and AI training progress
"""

import sys
from pathlib import Path
import time
import json
from datetime import datetime
import os

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def monitor_paper_trading():
    """Monitor paper trading results"""
    results_dir = Path("trading/paper_trading_results")
    
    if not results_dir.exists():
        return "⏳ Waiting for paper trading to start..."
    
    # Get latest result file
    result_files = sorted(results_dir.glob("paper_trading_*.json"))
    
    if not result_files:
        return "⏳ Paper trading initializing..."
    
    latest_file = result_files[-1]
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        status = f"""
📊 PAPER TRADING STATUS
{'='*60}
Start Time: {data.get('start_time', 'N/A')}
Initial Capital: ${data.get('initial_capital', 0):,.2f}
Current Capital: ${data.get('final_capital', 0):,.2f}
Total Return: {data.get('total_return_pct', 0):.2f}%

Trades:
  Total: {data.get('total_trades', 0)}
  Open Positions: {data.get('open_positions', 0)}
  Closed Trades: {len(data.get('closed_trades', []))}

Recent Trades:
"""
        
        # Show last 3 trades
        for trade in data.get('closed_trades', [])[-3:]:
            pnl_emoji = "🟢" if trade.get('pnl', 0) > 0 else "🔴"
            status += f"  {pnl_emoji} {trade.get('symbol', 'N/A')}: {trade.get('reason', 'N/A')} | P&L: ${trade.get('pnl', 0):.2f}\n"
        
        return status
        
    except Exception as e:
        return f"⚠️  Error reading paper trading results: {e}"

def monitor_ai_training():
    """Monitor AI training progress"""
    model_dir = Path("models/quick_trained")
    
    if not model_dir.exists():
        return "⏳ AI training not started yet..."
    
    metadata_file = model_dir / "metadata.json"
    
    if not metadata_file.exists():
        return "⏳ AI model training in progress..."
    
    try:
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        status = f"""
🤖 AI TRAINING STATUS
{'='*60}
Trained At: {data.get('trained_at', 'N/A')}
Device: {data.get('device', 'N/A')}
Accuracy: {data.get('accuracy', 0):.2%}
Features: {data.get('num_features', 0)}
Training Samples: {data.get('train_samples', 0):,}
Test Samples: {data.get('test_samples', 0):,}

Status: ✅ TRAINING COMPLETE
Model saved to: {model_dir}
"""
        return status
        
    except Exception as e:
        return f"⚠️  Error reading AI training status: {e}"

def monitor_data_collection():
    """Monitor data collection progress"""
    summary_file = Path("data/historical/ultimate_dataset/crypto/collection_summary.json")
    
    if not summary_file.exists():
        return "⏳ Data collection in progress..."
    
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        status = f"""
📥 DATA COLLECTION STATUS
{'='*60}
Total Assets: {data.get('total_assets', 0)}
Successful: {data.get('successful', 0)}
Failed: {data.get('failed', 0)}
Success Rate: {data.get('success_rate', 'N/A')}

Status: ✅ CRYPTO DATA COMPLETE
"""
        return status
        
    except Exception as e:
        return f"⏳ Data collection in progress..."

def main():
    """Main monitoring loop"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              Live Trading & Training Monitor                   ║
║                  Press Ctrl+C to exit                          ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        while True:
            clear_screen()
            
            print(f"""
╔════════════════════════════════════════════════════════════════╗
║              Live Trading & Training Monitor                   ║
║              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║
╚════════════════════════════════════════════════════════════════╝
            """)
            
            # Show paper trading status
            print(monitor_paper_trading())
            print()
            
            # Show AI training status
            print(monitor_ai_training())
            print()
            
            # Show data collection status
            print(monitor_data_collection())
            print()
            
            print("Refreshing in 10 seconds... (Ctrl+C to exit)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")

if __name__ == "__main__":
    main()

