#!/usr/bin/env python3
"""
Monitor data collection progress
Shows real-time statistics and estimated completion time
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import json

def get_directory_stats(directory: Path) -> dict:
    """Get statistics for a directory"""
    if not directory.exists():
        return {'files': 0, 'size_mb': 0, 'last_modified': None}
    
    files = list(directory.rglob('*.parquet')) + list(directory.rglob('*.json'))
    total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
    
    if files:
        last_modified = max(f.stat().st_mtime for f in files)
        last_modified = datetime.fromtimestamp(last_modified)
    else:
        last_modified = None
    
    return {
        'files': len(files),
        'size_mb': round(total_size, 2),
        'last_modified': last_modified
    }

def monitor_progress():
    """Monitor collection progress"""
    base_dir = Path("data/historical/ultimate_dataset")
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║           Data Collection Progress Monitor                     ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    start_time = datetime.now()
    
    while True:
        # Clear screen (works on Windows)
        print("\033[H\033[J", end="")
        
        print(f"Monitoring: {base_dir}")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed: {datetime.now() - start_time}\n")
        
        # Get stats for each subdirectory
        subdirs = {
            'crypto': base_dir / 'crypto',
            'traditional': base_dir / 'traditional',
            'alternative': base_dir / 'alternative',
            'aster': base_dir / 'aster'
        }
        
        total_files = 0
        total_size = 0
        
        print("📊 Collection Status:")
        print("=" * 60)
        
        for name, path in subdirs.items():
            stats = get_directory_stats(path)
            total_files += stats['files']
            total_size += stats['size_mb']
            
            status = "✓" if stats['files'] > 0 else "○"
            print(f"{status} {name:15s}: {stats['files']:4d} files, {stats['size_mb']:8.2f} MB", end="")
            
            if stats['last_modified']:
                age = datetime.now() - stats['last_modified']
                if age < timedelta(minutes=1):
                    print(f" [ACTIVE]")
                else:
                    print(f" [Last: {age.seconds//60}m ago]")
            else:
                print(" [Waiting]")
        
        print("=" * 60)
        print(f"Total: {total_files} files, {total_size:.2f} MB\n")
        
        # Check for summary files
        summary_files = [
            base_dir / "master_collection_summary.json",
            base_dir / "crypto" / "collection_summary.json",
            base_dir / "traditional" / "collection_summary.json",
            base_dir / "alternative" / "collection_summary.json"
        ]
        
        completed = []
        for summary in summary_files:
            if summary.exists():
                completed.append(summary.parent.name)
        
        if completed:
            print("✅ Completed sections:", ", ".join(completed))
        
        # Estimate completion
        if total_files > 0:
            # Rough estimate based on file count
            expected_files = 500  # Approximate total expected
            progress = min(100, (total_files / expected_files) * 100)
            
            print(f"\n📈 Estimated Progress: {progress:.1f}%")
            
            # Progress bar
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"[{bar}]")
            
            # ETA calculation
            if progress > 5:  # Only estimate after 5% complete
                elapsed = (datetime.now() - start_time).total_seconds()
                total_time = elapsed / (progress / 100)
                remaining = total_time - elapsed
                eta = datetime.now() + timedelta(seconds=remaining)
                print(f"\n⏱️  Estimated completion: {eta.strftime('%H:%M:%S')}")
        
        # Check if complete
        if (base_dir / "master_collection_summary.json").exists():
            print("\n\n✅ DATA COLLECTION COMPLETE!")
            
            # Read and display summary
            try:
                with open(base_dir / "master_collection_summary.json", 'r') as f:
                    summary = json.load(f)
                
                print("\nFinal Statistics:")
                print(f"  Total duration: {summary.get('total_duration', 'Unknown')}")
                
                if 'statistics' in summary:
                    stats = summary['statistics']
                    print(f"  Total files: {stats.get('total_files_created', 0)}")
                    print(f"  Total size: {stats.get('total_size_mb', 0):.2f} MB")
                
            except Exception as e:
                print(f"Error reading summary: {e}")
            
            break
        
        # Refresh every 10 seconds
        time.sleep(10)

if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")

