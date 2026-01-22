# Veo Web App - Local Development Setup

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- FFmpeg installed and in PATH
- At least 2GB RAM available (4GB recommended)

### Installation Steps

#### Windows
1. Double-click `run_local.bat`
2. Wait for dependencies to install
3. Open browser to http://localhost:8000

#### Linux/Mac
1. Make script executable: `chmod +x run_local.sh`
2. Run: `./run_local.sh`
3. Open browser to http://localhost:8000

### Manual Setup (if scripts don't work)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create directories
mkdir -p uploads outputs temp

# 5. Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔑 Environment Variables

All API keys are loaded from the `.env` file in the root directory.

### Required Variables:
```env
# Gemini API Keys (at least 1 required)
GEMINI_API_KEY_1=your_key_here
GEMINI_API_KEY_2=your_key_here
# ... add more as needed

# Optional: OpenAI for prompt enhancement
OPENAI_API_KEY=your_key_here
```

### Optional Variables:
```env
# Server configuration
PORT=8000
HOST=0.0.0.0

# Database (defaults to SQLite)
DATABASE_URL=sqlite:///./veo_web_app.db

# File paths
UPLOADS_DIR=uploads
OUTPUTS_DIR=outputs
TEMP_DIR=temp
```

## 📁 Directory Structure

```
veo-web-app/
├── .env                    # Your API keys (NEVER commit this!)
├── run_local.sh           # Linux/Mac startup script
├── run_local.bat          # Windows startup script
├── main.py                # FastAPI application
├── worker.py              # Video generation worker
├── veo_generator.py       # Veo API integration
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── static/
│   └── index.html        # Frontend UI
├── uploads/              # Uploaded images (temporary)
├── outputs/              # Generated videos (temporary)
└── temp/                 # Temporary processing files

```

## 🎮 Usage

1. **Upload Images**: Click "Upload Images" and select 1-10 reference images
2. **Enter Dialogue**: Type your script (one line per video clip)
3. **Configure Settings**:
   - Language: Select target language
   - Duration: 8 or 16 seconds per clip
   - Aspect Ratio: 9:16 (vertical), 16:9 (horizontal)
   - AI Prompts: Enable for automatic prompt enhancement
4. **Storyboard Mode** (Optional): Drag images to assign to specific clips
5. **Start Generation**: Click "🚀 Start Generation"
6. **Review & Export**: Approve clips and export final video

## ⚙️ Performance Tips

### For 512MB RAM (Free Tier):
- Process 1 clip at a time (already optimized in config)
- Use 8-second clips instead of 16-second
- Limit to 8-10 clips per job

### For 2GB+ RAM:
Edit `config.py`:
```python
max_workers: int = 2  # Instead of 1
parallel_clips: int = 2  # Instead of 1
```

## 🔧 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt --upgrade
```

### "FFmpeg not found"
**Windows**: Download from https://ffmpeg.org/download.html
**Linux**: `sudo apt install ffmpeg`
**Mac**: `brew install ffmpeg`

### "Out of memory" during generation
- Reduce `parallel_clips` in config.py to 1
- Use 8-second clips instead of 16-second
- Process fewer clips per job

### Port already in use
Change port in run script or set environment variable:
```bash
export PORT=8001  # Linux/Mac
set PORT=8001     # Windows
```

## 🌐 Accessing from Other Devices

To access from other devices on your local network:

1. Find your local IP:
   - Windows: `ipconfig`
   - Linux/Mac: `ifconfig` or `ip addr`

2. Start with `--host 0.0.0.0` (already in scripts)

3. Access from other devices: `http://YOUR_IP:8000`

## 🔒 Security Notes

**IMPORTANT:**
- Never commit `.env` file to Git
- Keep API keys private
- Don't expose local server to internet without authentication
- The `.gitignore` file already excludes `.env`

## 📊 Database

The app uses SQLite by default (`veo_web_app.db`). To reset:
```bash
rm veo_web_app.db
# Database will be recreated on next startup
```

## 🐛 Debug Mode

For more verbose logging, start with:
```bash
# Linux/Mac
LOG_LEVEL=DEBUG ./run_local.sh

# Windows
set LOG_LEVEL=DEBUG
run_local.bat
```

## 📝 Development

The server runs with `--reload` flag, so any code changes will automatically restart the server.

### Key Files to Edit:
- `static/index.html` - Frontend UI
- `main.py` - API endpoints
- `worker.py` - Video generation logic
- `veo_generator.py` - Veo API integration
- `config.py` - Settings and configuration

## 🚀 Deploying to Production

See main README.md for deployment instructions to:
- Render.com
- Vercel
- Railway
- Docker

## 📄 License

See LICENSE file for details.
