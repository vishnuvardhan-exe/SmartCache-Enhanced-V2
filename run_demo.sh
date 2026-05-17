#!/bin/bash
# SmartCache Demo Launcher
# One-command demo startup

set -e

echo "🚀 SmartCache Demo Launcher"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from https://python.org"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎮 Starting SmartCache Demo..."
echo "   Open http://localhost:8501 in your browser"
echo "   Press Ctrl+C to stop"
echo ""

cd demo
streamlit run app.py
