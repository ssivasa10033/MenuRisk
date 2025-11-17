#!/usr/bin/env python3
"""
Open all chart PNG files in Preview (macOS default image viewer)
"""

import os
import subprocess
from pathlib import Path

def main():
    """Open all PNG charts in Preview"""
    charts_dir = Path('static/charts')
    
    if not charts_dir.exists():
        print("Error: Charts directory not found!")
        return
    
    png_files = list(charts_dir.glob('*.png'))
    
    if not png_files:
        print("Error: No PNG files found!")
        return
    
    print("=" * 60)
    print("Opening charts in Preview...")
    print("=" * 60)
    
    for png_file in sorted(png_files):
        print(f"Opening: {png_file.name}")
        # Use macOS 'open' command with Preview app
        subprocess.run(['open', '-a', 'Preview', str(png_file)])
    
    print(f"\n✓ Opened {len(png_files)} chart(s) in Preview")
    print("\nCharts opened:")
    for png_file in sorted(png_files):
        print(f"  • {png_file.name}")


if __name__ == '__main__':
    main()

