# Veo 3.1 Video Generator - Web App

Generate AI videos with Google Veo 3.1 API. Upload images, add dialogue lines, get videos.

---

## Quick Start (Windows PowerShell)

```powershell
# 1. Extract and enter folder
cd veo_web_app

# 2. Create virtual environment
python -m venv venv

# 3. Activate venv
.\venv\Scripts\Activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install Google GenAI SDK (requires Rust)
pip install google-genai

# 6. Run the app
python main.py
```

Open: **http://localhost:8000**

---

## Setup Details

### Step 1: Virtual Environment

```powershell
# Create venv (one time)
python -m venv venv

# Activate venv (every time you open terminal)
.\venv\Scripts\Activate

# You should see (venv) in your prompt:
# (venv) PS C:\Users\...\veo_web_app>
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Install Google GenAI SDK

```powershell
pip install google-genai
```

**If it fails with Rust errors:**

1. Install Rust from https://rustup.rs/
2. **Close and reopen your terminal**
3. Verify: `rustc --version`
4. Try again: `pip install google-genai`

### Step 4: Configure API Keys

Your `.env` file should already have the keys. Verify:

```powershell
cat .env
```

Should show:
```
GEMINI_API_KEY_1=AIzaSy...
OPENAI_API_KEY=sk-proj-...
```

### Step 5: Run

```powershell
python main.py
```

---

## Testing

### Check SDK is installed
```powershell
python -c "from google import genai; print('SDK OK')"
```

### Check API keys are loaded
Open: http://localhost:8000/api/admin/keys

### Check server health
Open: http://localhost:8000/api/health

---

## Generate Your First Video

1. Open http://localhost:8000
2. **Upload 1 image** (any image of a person)
3. **Add dialogue**: `1|Ciao, questo è un test.`
4. **Settings**: Italian, 9:16, 720p, 8s
5. Click **🚀 Start**
6. Wait for generation (1-3 minutes per clip)
7. Review and approve the clip
8. Download!

---

## Features

| Feature | Description |
|---------|-------------|
| **Single Image Mode** | Upload 1 image → uses same frame for start/end |
| **Multi Image Mode** | Upload multiple → creates continuity between clips |
| **AI Prompts** | OpenAI optimizes visual prompts automatically |
| **Custom Prompts** | Turn off AI → write your own visual prompt |
| **Interpolation** | Smooth transitions (requires 8s duration) |
| **Review System** | Approve clips or request redo (max 3 attempts) |
| **API Key Rotation** | Multiple Gemini keys for rate limit handling |

---

## API Endpoints

### Admin
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/keys` | GET | Check API keys status |
| `/api/admin/keys/rotate` | POST | Switch to next Gemini key |
| `/api/admin/keys/reload` | POST | Reload keys from .env |

### Jobs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/jobs` | GET | List all jobs |
| `/api/jobs` | POST | Create new job |
| `/api/jobs/{id}` | GET | Get job details |
| `/api/jobs/{id}` | DELETE | Delete job |
| `/api/jobs/{id}/cancel` | POST | Cancel running job |
| `/api/jobs/{id}/clips` | GET | Get job clips |
| `/api/jobs/{id}/outputs` | GET | List output files |

### Clips
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/clips/{id}/approve` | POST | Approve clip |
| `/api/clips/{id}/redo` | POST | Request regeneration |

---

## File Structure

```
veo_web_app/
├── .env                 # API keys (your keys here)
├── .env.example         # Example config
├── main.py              # FastAPI server
├── config.py            # Configuration
├── models.py            # Database models
├── worker.py            # Background job processor
├── veo_generator.py     # Veo API integration
├── error_handler.py     # Error classification
├── requirements.txt     # Python dependencies
├── static/
│   └── index.html       # Frontend UI
├── uploads/             # Uploaded images (auto-created)
├── outputs/             # Generated videos (auto-created)
└── data/
    └── jobs.db          # SQLite database (auto-created)
```

---

## Troubleshooting

### "No module named 'google'"
```powershell
pip install google-genai
```

### "Rust not found" during install
1. Install Rust: https://rustup.rs/
2. Close terminal completely
3. Open new terminal
4. Activate venv: `.\venv\Scripts\Activate`
5. Retry: `pip install google-genai`

### "No Gemini API keys configured"
Check `.env` file exists and has keys:
```powershell
cat .env
```

### Server won't start
Make sure venv is activated:
```powershell
.\venv\Scripts\Activate
python main.py
```

### API returns 429 (rate limit)
- Add more Gemini keys to `.env`
- Or wait a few minutes

---

## Adding More API Keys

Edit `.env`:
```
GEMINI_API_KEY_1=AIzaSy...first-key...
GEMINI_API_KEY_2=AIzaSy...second-key...
GEMINI_API_KEY_3=AIzaSy...third-key...
```

Then reload without restart:
```powershell
curl -X POST http://localhost:8000/api/admin/keys/reload
```

---

## Common Commands

```powershell
# Activate venv
.\venv\Scripts\Activate

# Run server
python main.py

# Run with test data (for UI testing)
python test_mock_data.py

# Check installed packages
pip list

# Update a package
pip install --upgrade google-genai
```

---

## Support

- Veo API docs: https://ai.google.dev/
- Get Gemini keys: https://aistudio.google.com/apikey
- OpenAI keys: https://platform.openai.com/api-keys
