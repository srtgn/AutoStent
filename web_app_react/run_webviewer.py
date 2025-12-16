#!/usr/bin/env python3
"""
Launch script for 4C-webviewer.
Runs the official 4C visualizer as a web service.
"""

import subprocess
import sys
import os
from pathlib import Path

# Path to the 4C-webviewer
WEBVIEWER_DIR = Path(__file__).parent.parent / "vendor" / "4C-webviewer"


def main():
    """Launch the 4C-webviewer service."""
    
    if not WEBVIEWER_DIR.exists():
        print(f"ERROR: 4C-webviewer not found at {WEBVIEWER_DIR}")
        print("Please run: git clone https://github.com/4C-multiphysics/4C-webviewer vendor/4C-webviewer")
        sys.exit(1)
    
    print("="*60)
    print("4C-Webviewer Launcher")
    print("="*60)
    print(f"Webviewer directory: {WEBVIEWER_DIR}")
    print()
    
    # Check if fourc_webviewer is installed
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import fourc_webviewer"],
            capture_output=True,
            text=True,
            cwd=WEBVIEWER_DIR
        )
        if result.returncode != 0:
            print("Installing 4C-webviewer...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=WEBVIEWER_DIR,
                check=True
            )
    except Exception as e:
        print(f"Note: Could not check/install fourc_webviewer: {e}")
    
    print("Starting 4C-webviewer...")
    print("Access at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print()
    
    # Try to run the webviewer
    try:
        # Try the installed command first
        subprocess.run(
            ["fourc_webviewer"],
            cwd=WEBVIEWER_DIR,
            check=True
        )
    except FileNotFoundError:
        # Fall back to running main.py directly
        main_py = WEBVIEWER_DIR / "src" / "fourc_webviewer" / "__main__.py"
        if main_py.exists():
            subprocess.run(
                [sys.executable, str(main_py)],
                cwd=WEBVIEWER_DIR,
                check=True
            )
        else:
            print(f"ERROR: Could not find webviewer entry point")
            print("Please install manually:")
            print(f"  cd {WEBVIEWER_DIR}")
            print("  pip install -e .")
            print("  fourc_webviewer")
            sys.exit(1)


if __name__ == "__main__":
    main()
