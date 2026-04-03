#!/usr/bin/env python3
"""
Launcher for Weather Prediction System
Run this file to start the UI application
"""

import sys
import subprocess


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import numpy
        import pandas
        import matplotlib
        import pgmpy
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False


def main():
    print("=" * 60)
    print("🌤️  WEATHER PREDICTION SYSTEM LAUNCHER")
    print("=" * 60)

    if not check_dependencies():
        print("\nInstalling dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    print("\nStarting Weather Prediction UI...")
    print("(Close the window to exit)\n")

    # Run the UI
    import main_ui
    main_ui.main()


if __name__ == "__main__":
    main()