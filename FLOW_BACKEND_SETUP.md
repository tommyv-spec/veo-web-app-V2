# Flow Backend Setup Guide

This guide explains how to set up the **Flow backend** for users without API keys. The Flow backend uses browser automation (Playwright) to generate videos through Google Flow's web UI.

## Overview

The Veo Web App now supports two backends:

| Backend | When Used | Description |
|---------|-----------|-------------|
| **API** | User has valid Gemini API keys | Direct API calls (fast, reliable) |
| **Flow** | User has NO API keys | Browser automation via Flow UI |

Backend selection is **automatic** - users don't need to choose. The system routes to Flow only when no API keys are available.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Web Service   │────▶│  Redis/Valkey   │◀────│  Flow Worker    │
│   (main.py)     │     │    (Queue)      │     │ (flow_worker.py)│
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐                            ┌─────────────────┐
│   PostgreSQL    │                            │    Playwright   │
│   (Database)    │                            │    (Browser)    │
└─────────────────┘                            └─────────────────┘
         │                                               │
         └───────────────────┬───────────────────────────┘
                             ▼
                   ┌─────────────────┐
                   │   S3 / R2       │
                   │ (Object Storage)│
                   └─────────────────┘
```

## Prerequisites

### 1. Render Paid Plan
- **Web service**: Standard ($25/month)
- **Worker service**: Standard ($25/month) - Background workers require paid plan
- **Key-Value store**: Starter ($10/month) - For reliable queue persistence

### 2. Object Storage (S3/R2)
Since Render persistent disks cannot be shared between services, you need object storage for:
- User-uploaded frames
- Generated video outputs
- Flow authentication state

**Recommended**: Cloudflare R2 (cheaper, S3-compatible)

### 3. Google Account
A Google account for Flow authentication.

## Setup Steps

### Step 1: Create Object Storage Bucket

#### Cloudflare R2:
1. Go to Cloudflare Dashboard → R2
2. Create bucket (e.g., `veo-studio`)
3. Create API token with read/write access
4. Note: Endpoint URL, Access Key ID, Secret Access Key

#### AWS S3:
1. Create S3 bucket
2. Create IAM user with S3 access
3. Note credentials

### Step 2: Export Flow Authentication State

This must be done **locally** (not on server) as it requires browser GUI:

```bash
# Install dependencies
pip install playwright boto3
playwright install chromium

# Run auth export
cd veo-web-app-main
python -m backends.flow_backend --export-auth
```

This will:
1. Open a browser window
2. Navigate to Google Flow
3. Wait for you to log in
4. Save `flow_storage_state.json`
5. Optionally upload to S3/R2

### Step 3: Upload Auth State to S3/R2

If not auto-uploaded:
```bash
# Using AWS CLI (works with R2 too)
aws s3 cp flow_storage_state.json s3://veo-studio/flow/auth/default/storage_state.json \
    --endpoint-url YOUR_ENDPOINT
```

### Step 4: Configure Render Environment Variables

In Render Dashboard, set for **both** Web and Worker services:

```
# Object Storage
S3_ENDPOINT=https://xxx.r2.cloudflarestorage.com
S3_BUCKET=veo-studio
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_REGION=auto

# Flow Backend
FLOW_BACKEND_ENABLED=true
FLOW_STORAGE_STATE_URL=flow/auth/default/storage_state.json
```

### Step 5: Deploy with Updated render.yaml

Replace your `render.yaml` with `render.yaml.flow`:

```bash
cp render.yaml.flow render.yaml
git add .
git commit -m "Enable Flow backend"
git push
```

### Step 6: Run Database Migrations

The migrations should run automatically on startup, but you can also run manually:

```bash
python migrations/add_flow_fields.py
```

## Verification

### Check Backend Status

```bash
curl https://your-app.onrender.com/api/health

# Response includes:
{
  "backends": {
    "api": {"enabled": true, "available_keys": 5},
    "flow": {"enabled": true, "auth_configured": true, "queue_configured": true}
  }
}
```

### Check Queue Status

```bash
curl https://your-app.onrender.com/api/flow/queue/status

# Response:
{
  "pending": 0,
  "processing": 0,
  "failed": 0
}
```

## Operational Notes

### Authentication Expiration

Google sessions expire. When this happens:
1. Jobs will pause with `flow_needs_auth=true`
2. Admin receives alert (if configured)
3. Re-run auth export locally
4. Upload new storage state
5. Jobs automatically resume

### Monitoring

Watch for these log messages:
- `[FlowWorker] Claimed job: ...` - Job picked up
- `[Flow] Login required` - Auth expired (needs re-export)
- `[Flow] Downloaded: ...` - Clip completed

### Rate Limiting

Flow UI has its own rate limits. The worker processes 1 job at a time by default. Adjust with:
```
FLOW_MAX_CONCURRENT=1  # Keep at 1 unless you have multiple Google accounts
```

### Failure Recovery

Failed jobs are moved to `flow:failed` queue. To retry:
```python
# In Python shell on Render
from flow_worker import get_redis_client, FLOW_QUEUE_FAILED, FLOW_QUEUE_NAME

redis = get_redis_client()
while job := redis.rpoplpush(FLOW_QUEUE_FAILED, FLOW_QUEUE_NAME):
    print(f"Requeued: {job}")
```

## Cost Breakdown

| Service | Plan | Cost |
|---------|------|------|
| Web Service | Standard | $25/month |
| Flow Worker | Standard | $25/month |
| Key-Value | Starter | $10/month |
| PostgreSQL (optional) | Starter | $7/month |
| R2 Storage | Pay as you go | ~$0.015/GB/month |
| **Total** | | **~$60-67/month** |

## Troubleshooting

### "Login required" errors
- Auth state expired
- Re-run `--export-auth` locally
- Re-upload to S3

### Jobs stuck in "pending"
- Check worker is running in Render dashboard
- Check Redis connection: `KEYVALUE_URL` set correctly?
- Check worker logs

### Downloads failing
- Flow UI may have changed selectors
- Check `flow_backend.py` selectors match current UI
- Take screenshot on error (Playwright feature)

### Storage errors
- Verify S3 credentials
- Check bucket exists and is accessible
- Verify IAM permissions

## Security Considerations

1. **Auth State**: Contains sensitive cookies. Store encrypted or in secure bucket.
2. **S3 Credentials**: Use environment variables, never commit to git.
3. **Google Account**: Consider dedicated account for Flow automation.

## Files Added/Modified

| File | Description |
|------|-------------|
| `backends/__init__.py` | Backend module exports |
| `backends/selector.py` | Backend selection logic |
| `backends/storage.py` | S3/R2 object storage |
| `backends/flow_backend.py` | Playwright automation |
| `backends/routing.py` | Job routing helpers |
| `flow_worker.py` | Background worker service |
| `Dockerfile.flow` | Docker image for worker |
| `requirements-flow.txt` | Worker dependencies |
| `render.yaml.flow` | Updated Render config |
| `migrations/add_flow_fields.py` | Database migration |
| `models.py` | Added Flow fields |
