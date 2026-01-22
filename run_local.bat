@echo off
REM Veo Web App - Local Development Startup Script (Windows)

echo 🚀 Starting Veo Web App locally...
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ❌ Error: .env file not found!
    echo Please create a .env file with your API keys.
    echo See .env.example for required variables.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs
if not exist "temp" mkdir temp

REM Set environment variables for local development
set PORT=8000
set DATABASE_URL=sqlite:///./veo_web_app.db

echo.
echo ✅ Setup complete!
echo 🌐 Starting server on http://localhost:8000
echo 📝 Press Ctrl+C to stop
echo.

REM Start the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
