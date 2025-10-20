"""
Remove All Emojis from AsterAI Codebase
Fixes Windows PowerShell charmap codec errors
"""

import os
import re
from pathlib import Path
from typing import Dict, List

# Emoji to ASCII mapping
EMOJI_REPLACEMENTS = {
    # Status indicators
    '🚀': '[START]',
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    '🟢': '[ACTIVE]',
    '🔴': '[STOPPED]',
    '🟡': '[PENDING]',
    
    # Trading symbols
    '💰': '[MONEY]',
    '💵': '[VALUE]',
    '💲': '[PRICE]',
    '📊': '[DATA]',
    '📈': '[UP]',
    '📉': '[DOWN]',
    '🎯': '[TARGET]',
    '🔢': '[NUM]',
    
    # Actions
    '🔄': '[REFRESH]',
    '🔌': '[CONNECT]',
    '🛑': '[STOP]',
    '⏰': '[TIME]',
    '📅': '[DATE]',
    '🔔': '[ALERT]',
    '🚨': '[URGENT]',
    
    # Info
    '📝': '[NOTE]',
    '📍': '[INFO]',
    '🤖': '[BOT]',
    '🧠': '[AI]',
    '⚡': '[FAST]',
    '💻': '[PC]',
    '🌐': '[NET]',
    '🏗️': '[ARCH]',
    
    # Results
    '➖': '[FLAT]',
    '✔️': '[CHECK]',
    '❗': '[IMPORTANT]',
    
    # Additional common emojis
    '🎉': '[SUCCESS]',
    '🎊': '[CELEBRATE]',
    '👍': '[GOOD]',
    '👎': '[BAD]',
    '🔥': '[HOT]',
    '❄️': '[COLD]',
    '🌟': '[STAR]',
    '💡': '[IDEA]',
    '🎲': '[RANDOM]',
    '🎰': '[GAMBLE]',
    '📱': '[MOBILE]',
    '🖥️': '[DESKTOP]',
    '☁️': '[CLOUD]',
    '🔒': '[SECURE]',
    '🔓': '[OPEN]',
    '🔑': '[KEY]',
    '🛡️': '[SHIELD]',
    '⚙️': '[CONFIG]',
    '🔧': '[TOOL]',
    '🔨': '[BUILD]',
    '📦': '[PACKAGE]',
    '📂': '[FOLDER]',
    '📄': '[FILE]',
    '🗃️': '[ARCHIVE]',
    '🗂️': '[INDEX]',
    '🗄️': '[DATABASE]',
    '🧪': '[TEST]',
    '🔬': '[SCIENCE]',
    '🎓': '[LEARN]',
    '📚': '[DOCS]',
    '📖': '[READ]',
    '✏️': '[EDIT]',
    '✉️': '[MESSAGE]',
    '📮': '[SEND]',
    '📭': '[EMPTY]',
    '📬': '[FULL]',
    '📡': '[SIGNAL]',
    '📞': '[CALL]',
    '☎️': '[PHONE]',
    '📶': '[BARS]',
    '🌈': '[RAINBOW]',
    '⭐': '[STAR2]',
    '💪': '[STRONG]',
    '👌': '[PERFECT]',
    '🤝': '[HANDSHAKE]',
    '👋': '[WAVE]',
    '👀': '[EYES]',
    '🚗': '[CAR]',
    '🚀': '[ROCKET]',
    '✈️': '[PLANE]',
    '⏱️': '[TIMER]',
    '⏲️': '[CLOCK]',
    '⌛': '[HOURGLASS]',
    '⏳': '[SAND]',
    '🔋': '[BATTERY]',
    '🔌': '[PLUG]',
    '💾': '[SAVE]',
    '💿': '[DISC]',
    '🖨️': '[PRINT]',
}

def remove_emojis_from_file(filepath: Path) -> bool:
    """Remove emojis from a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        # Replace known emojis
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            if emoji in content:
                content = content.replace(emoji, replacement)
                changes_made = True
        
        # Remove any remaining emoji characters (Unicode ranges)
        # Emoji ranges: U+1F300 to U+1F9FF
        emoji_pattern = re.compile(
            "["
            "\U0001F300-\U0001F9FF"  # Misc Symbols and Pictographs, Emoticons, Transport
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F680-\U0001F6FF"  # Transport and Map
            "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        
        cleaned_content = emoji_pattern.sub('[EMOJI]', content)
        if cleaned_content != content:
            content = cleaned_content
            changes_made = True
        
        if changes_made:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")
        return False

def scan_and_fix_emojis(root_dir: str = ".") -> Dict[str, int]:
    """Scan all files and remove emojis"""
    stats = {
        "files_scanned": 0,
        "files_modified": 0,
        "python_files": 0,
        "markdown_files": 0,
        "other_files": 0
    }
    
    # File patterns to process
    patterns = ["**/*.py", "**/*.md", "**/*.html", "**/*.js", "**/*.json", "**/*.yaml", "**/*.yml"]
    
    # Directories to skip
    skip_dirs = {
        "__pycache__", "node_modules", ".git", "asterai_env", "asterai_venv",
        "asterai_env_fresh", "aster_trading_env", ".pytest_cache", "htmlcov"
    }
    
    files_to_process = []
    
    for pattern in patterns:
        for filepath in Path(root_dir).glob(pattern):
            # Skip if in excluded directory
            if any(skip_dir in filepath.parts for skip_dir in skip_dirs):
                continue
            
            files_to_process.append(filepath)
    
    print(f"[INFO] Found {len(files_to_process)} files to scan")
    
    for filepath in files_to_process:
        stats["files_scanned"] += 1
        
        if filepath.suffix == ".py":
            stats["python_files"] += 1
        elif filepath.suffix == ".md":
            stats["markdown_files"] += 1
        else:
            stats["other_files"] += 1
        
        # Process file
        if remove_emojis_from_file(filepath):
            stats["files_modified"] += 1
            print(f"[FIXED] {filepath}")
    
    return stats

def main():
    """Main function"""
    print("="*60)
    print("AsterAI Emoji Removal Tool")
    print("Fixing Windows PowerShell compatibility issues")
    print("="*60)
    print()
    
    print("[INFO] Scanning codebase for emojis...")
    stats = scan_and_fix_emojis()
    
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Files scanned:    {stats['files_scanned']}")
    print(f"Files modified:   {stats['files_modified']}")
    print(f"  Python files:   {stats['python_files']}")
    print(f"  Markdown files: {stats['markdown_files']}")
    print(f"  Other files:    {stats['other_files']}")
    print()
    
    if stats['files_modified'] > 0:
        print("[OK] Emoji removal complete!")
        print("[INFO] All emojis replaced with ASCII equivalents")
        print("[INFO] System is now Windows PowerShell compatible")
    else:
        print("[INFO] No emojis found - system already compatible")

if __name__ == "__main__":
    main()

