#!/bin/bash
# Veo Web App - Local Development Startup Script

echo "🚀 Starting Veo Web App locally..."
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your API keys."
    echo "See .env.example for required variables."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads outputs temp

# Set environment variables for local development
export PORT=8000
export DATABASE_URL="sqlite:///./veo_web_app.db"

echo ""
echo "✅ Setup complete!"
echo "🌐 Starting server on http://localhost:8000"
echo "📝 Press Ctrl+C to stop"
echo ""

# Start the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
