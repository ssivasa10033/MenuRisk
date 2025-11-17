#!/bin/bash
# Script to install dependencies and run tests

echo "Installing required packages..."
python3 -m pip install numpy pandas scikit-learn

echo ""
echo "Running tests..."
python3 test_menu_price_model.py

