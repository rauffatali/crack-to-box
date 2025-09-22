#!/usr/bin/env python3
"""
Main entry point for the Image Annotation App Using Segmentation Masks.

This application provides two interfaces:
1. Tkinter - Desktop GUI application
2. Streamlit - Web-based interface

Usage:
    python main.py          # Interactive mode to choose interface
    python main.py --tk     # Launch Tkinter interface directly
    python main.py --st     # Launch Streamlit interface directly
"""

import argparse
import sys
import os

def launch_tkinter():
    """Launch the Tkinter interface"""
    try:
        from gui.tk.annotation_window import SegmentationAnnotator
        import tkinter as tk
        
        root = tk.Tk()
        app = SegmentationAnnotator(root)
        root.mainloop()
    except ImportError as e:
        print(f"Error launching Tkinter interface: {e}")
        print("Make sure all required dependencies are installed.")
        sys.exit(1)

def launch_streamlit():
    """Launch the Streamlit interface"""
    try:
        import subprocess
        
        # Launch streamlit with the app file
        streamlit_app_path = os.path.join(os.path.dirname(__file__), "gui", "streamlit_app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_app_path])
    except Exception as e:
        print(f"Error launching Streamlit interface: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
        sys.exit(1)

def interactive_choice():
    """Interactive mode to choose interface"""
    print("\n🖼️  Image Annotation App Using Segmentation Masks")
    print("=" * 50)
    print("\nChoose your preferred interface:")
    print("1. Tkinter GUI (Desktop Application)")
    print("2. Streamlit Web Interface")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == "1":
                print("Launching Tkinter interface...")
                launch_tkinter()
                break
            elif choice == "2":
                print("Launching Streamlit interface...")
                launch_streamlit()
                break
            elif choice == "3":
                print("Goodbye!")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Image Annotation App Using Segmentation Masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                    python main.py          # Interactive mode
                    python main.py --tk     # Launch Tkinter GUI
                    python main.py --st     # Launch Streamlit web interface
               """
    )
    
    parser.add_argument(
        "--tk", "--tkinter",
        action="store_true",
        help="Launch Tkinter desktop interface"
    )
    
    parser.add_argument(
        "--st", "--streamlit",
        action="store_true",
        help="Launch Streamlit web interface"
    )
    
    args = parser.parse_args()
    
    if args.tk and args.st:
        print("Error: Cannot specify both --tk and --st options.")
        sys.exit(1)
    
    if args.tk:
        launch_tkinter()
    elif args.st:
        launch_streamlit()
    else:
        interactive_choice()

if __name__ == "__main__":
    main() 