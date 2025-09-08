#!/usr/bin/env python3
"""
GAIKINDO Sales Analysis Platform - Main Entry Point
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run streamlit app from src folder
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/app.py", "--server.address", "0.0.0.0",
        "--server.port", "8501"
    ])

if __name__ == "__main__":
    main()