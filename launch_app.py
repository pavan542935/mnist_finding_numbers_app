#!/usr/bin/env python
"""
Simple Python launcher for the MNIST web app
This avoids environment conflicts by using subprocess
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 50)
    print("🔢 MNIST Digit Recognition Web App")
    print("=" * 50)
    print()
    
    # Get the project directory
    project_dir = Path(__file__).parent
    streamlit_app = project_dir / "streamlit_app.py"
    
    # Check if streamlit app exists
    if not streamlit_app.exists():
        print("❌ Error: streamlit_app.py not found!")
        return
    
    # Check if model exists
    model_path = project_dir / "mnist_model.h5"
    if not model_path.exists():
        print("⚠️ Warning: mnist_model.h5 not found!")
        print("Please run 'python ml_project.py' first to train the model.")
        return
    
    print("🚀 Starting Streamlit application...")
    print("📱 The app will open in your browser automatically.")
    print("🛑 Press Ctrl+C to stop the application")
    print()
    
    try:
        # Run streamlit using subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(streamlit_app),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], cwd=str(project_dir))
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()