"""
Create a desktop shortcut for the MNIST Web App
Run this script once to create a shortcut on your desktop
"""

import os
import winshell
from win32com.client import Dispatch
from pathlib import Path

def create_desktop_shortcut():
    """Create a desktop shortcut for the MNIST app"""
    
    # Get paths
    desktop = winshell.desktop()
    project_dir = Path(__file__).parent
    python_exe = project_dir / "mnist_env" / "Scripts" / "python.exe"
    launch_script = project_dir / "launch_app.py"
    
    # Create shortcut
    shell = Dispatch('WScript.Shell')
    shortcut_path = os.path.join(desktop, "MNIST Digit Recognition.lnk")
    shortcut = shell.CreateShortCut(shortcut_path)
    
    # Configure shortcut
    shortcut.Targetpath = str(python_exe)
    shortcut.Arguments = str(launch_script)
    shortcut.WorkingDirectory = str(project_dir)
    shortcut.Description = "MNIST Handwritten Digit Recognition Web App"
    shortcut.IconLocation = str(python_exe)  # Use Python icon
    
    # Save shortcut
    shortcut.save()
    
    print("✅ Desktop shortcut created successfully!")
    print(f"📍 Shortcut location: {shortcut_path}")
    print("🖱️ Double-click the shortcut to launch the app")

if __name__ == "__main__":
    try:
        create_desktop_shortcut()
    except ImportError:
        print("❌ Required packages not found. Installing...")
        print("Run: pip install pywin32 winshell")
    except Exception as e:
        print(f"❌ Error creating shortcut: {e}")
        print("You can manually run the app using the batch files or Python script.")