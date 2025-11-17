#!/usr/bin/env python3
"""
Run tests and generate visualizations
Combines unit testing with data visualization

Author: Seon Sivasathan
Institution: Computer Science @ Western University
"""

import sys
import subprocess
import os

def main():
    """Run tests and then generate visualizations"""
    print("=" * 60)
    print("Menu Price Optimizer - Tests & Visualizations")
    print("=" * 60)
    
    # Step 1: Run tests
    print("\n[Step 1/2] Running unit tests...")
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, 'test_menu_price_model.py'],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n⚠️  Tests failed. Visualizations will still be generated.")
        print("   Review test output above for details.\n")
    else:
        print("\n✓ All tests passed!\n")
    
    # Step 2: Generate visualizations
    print("[Step 2/2] Generating visualizations...")
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, 'visualize_results.py'],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✓ Visualizations generated successfully!")
        print(f"   Check the '{os.path.join(os.getcwd(), 'static', 'charts')}' directory")
    else:
        print("\n✗ Visualization generation failed.")
        print("   Make sure matplotlib and seaborn are installed:")
        print("   python3 -m pip install matplotlib seaborn")
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

