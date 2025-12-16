# Railway Deployment Guide

## Quick Deploy

1. **Push to GitHub** (if not already):
   ```bash
   git add .
   git commit -m "Add Railway deployment"
   git push origin main
   ```

2. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `4c` repository
   - Railway auto-detects `railway.toml`

3. **Wait for Build** (~5-10 minutes for first build)

4. **Get Your API URL**:
   - Railway assigns a URL like `your-app-xxx.railway.app`
   - Test: `curl https://your-app-xxx.railway.app/health`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Railway health check |
| `/simulate` | POST | Start async simulation |
| `/simulate/{id}` | GET | Get simulation status |
| `/simulate/sync` | POST | Run sync simulation |

## Example API Call

```bash
curl -X POST https://your-app.railway.app/simulate/sync \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "diameter": 10.0,
      "strut_thickness": 0.12,
      "num_struts": 12
    },
    "timeout": 60
  }'
```

## Update Frontend

In `web_app_react/index.html`, update the backend URL:

```javascript
const railwayUrl = 'https://your-app-xxx.railway.app';
```

## Troubleshooting

- **Build fails**: Check Docker build logs in Railway dashboard
- **Timeout**: Increase timeout in Railway settings → Environment
- **Out of Memory**: Upgrade to Pro plan for more RAM
