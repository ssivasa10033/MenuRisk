#!/usr/bin/env python3
"""
Open visualizations in default browser
Creates an HTML viewer and opens it automatically

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import os
import webbrowser
import sys
from pathlib import Path

def main():
    """Open visualizations in browser"""
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    html_file = script_dir / 'view_visualizations.html'
    charts_dir = script_dir / 'static' / 'charts'
    
    # Check if HTML file exists
    if not html_file.exists():
        print("Error: view_visualizations.html not found!")
        print("Please run visualize_results.py first to generate charts.")
        sys.exit(1)
    
    # Check if charts exist
    if not charts_dir.exists():
        print("Error: Charts directory not found!")
        print("Please run visualize_results.py first to generate charts.")
        sys.exit(1)
    
    chart_files = list(charts_dir.glob('*.png'))
    if not chart_files:
        print("Error: No chart files found!")
        print("Please run visualize_results.py first to generate charts.")
        sys.exit(1)
    
    # Convert to file:// URL
    html_url = f"file://{html_file}"
    
    print("=" * 60)
    print("Opening visualizations in browser...")
    print("=" * 60)
    print(f"HTML file: {html_file}")
    print(f"Charts found: {len(chart_files)}")
    print("=" * 60)
    
    # Open in default browser
    try:
        webbrowser.open(html_url)
        print("\n✓ Visualizations opened in your default browser!")
        print("\nIf the browser didn't open, manually open:")
        print(f"  {html_file}")
    except Exception as e:
        print(f"\n✗ Error opening browser: {e}")
        print(f"\nPlease manually open: {html_file}")


if __name__ == '__main__':
    main()

