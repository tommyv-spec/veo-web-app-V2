#!/usr/bin/env python3
"""
Test script to verify Veo Web App local setup
Run this before starting the server to check if everything is configured correctly
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.10+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def check_env_file():
    """Check if .env file exists and has API keys"""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        print("   Create one by copying .env.example:")
        print("   cp .env.example .env")
        return False
    
    # Check if it has at least one Gemini key
    with open(env_path) as f:
        content = f.read()
        if "GEMINI_API_KEY" not in content or "your_gemini_api_key_here" in content:
            print("❌ .env file exists but no valid Gemini API keys found")
            print("   Please add your Gemini API keys to .env")
            return False
    
    print("✅ .env file found with API keys")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    required = [
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "google-generativeai",
        "PIL",
        "sqlalchemy"
    ]
    
    missing = []
    for package in required:
        try:
            if package == "PIL":
                __import__("PIL")
            else:
                __import__(package.replace("-", "_"))
            print(f"✅ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} not installed")
    
    if missing:
        print("\n📦 Install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg installed: {version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("❌ FFmpeg not found")
    print("   Install FFmpeg:")
    print("   - Windows: Download from https://ffmpeg.org/download.html")
    print("   - Linux: sudo apt install ffmpeg")
    print("   - Mac: brew install ffmpeg")
    return False

def check_directories():
    """Check/create necessary directories"""
    dirs = ["uploads", "outputs", "temp"]
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"📁 Created directory: {d}")
        else:
            print(f"✅ Directory exists: {d}")
    return True

def test_api_keys():
    """Test if API keys are valid by trying to load config"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from config import get_gemini_keys_from_env
        keys = get_gemini_keys_from_env()
        
        if len(keys) == 0:
            print("❌ No Gemini API keys loaded from .env")
            return False
        
        print(f"✅ Loaded {len(keys)} Gemini API key(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error loading API keys: {e}")
        return False

def check_port():
    """Check if default port 8000 is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 8000))
        sock.close()
        print("✅ Port 8000 is available")
        return True
    except OSError:
        print("⚠️  Port 8000 is in use")
        print("   You can use a different port with: export PORT=8001")
        return True  # Not a critical error

def main():
    print("=" * 60)
    print("Veo Web App - Local Setup Test")
    print("=" * 60)
    print()
    
    checks = [
        ("Python version", check_python_version),
        (".env file", check_env_file),
        ("Dependencies", check_dependencies),
        ("FFmpeg", check_ffmpeg),
        ("Directories", check_directories),
        ("API keys", test_api_keys),
        ("Port availability", check_port)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n--- Checking {name} ---")
        results.append(check_func())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All checks passed! You're ready to start the server.")
        print("\nRun:")
        print("  - Linux/Mac: ./run_local.sh")
        print("  - Windows: run_local.bat")
        print("  - Manual: uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()
