# Render Deployment Checklist

## Pre-deployment Checklist

### ✅ Code Quality
- [x] Type annotations added to all functions
- [x] Error handling implemented
- [x] Environment validation added
- [x] Syntax check passed for all Python files

### ✅ Deployment Files
- [x] `Procfile` - Gunicorn configuration
- [x] `requirements.txt` - All dependencies listed
- [x] `runtime.txt` - Python version specified
- [x] `.env.example` - Environment variables documented
- [x] `README.md` - Deployment instructions
- [x] `.gitignore` - Proper exclusions (keeping model files)

### 🔧 Required Environment Variables

Set these in your Render dashboard:

```bash
# Twitter API (Required)
TWITTER_BEARER_TOKEN=your_bearer_token
TWITTER_CONSUMER_KEY=your_consumer_key
TWITTER_CONSUMER_SECRET=your_consumer_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret

# Kore.ai Bot (Required)
KORE_AI_CLIENT_ID=your_client_id
KORE_AI_CLIENT_SECRET=your_client_secret
KORE_AI_BOT_ID=your_bot_id
KORE_AI_WEBHOOK_URL=your_webhook_url

# Email Service (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password

# Model Configuration
MODEL_PATH=./results/checkpoint-174152
```

## Deployment Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment with type fixes and configuration"
git push origin main
```

### 2. Create Render Web Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: twitter-sentiment-api
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 twitter_sentiment_alert_bot:app`

### 3. Set Environment Variables
In Render dashboard, add all required environment variables from the list above.

### 4. Deploy
Click "Create Web Service" - Render will automatically deploy your application.

## Post-deployment Testing

Test these endpoints once deployed:

```bash
# Health check
curl https://your-app.onrender.com/health

# Predict sentiment
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin is amazing!"}'

# Analyze recent tweets (requires Twitter API)
curl -X POST https://your-app.onrender.com/analyze_recent \
  -H "Content-Type: application/json" \
  -d '{"query": "Bitcoin -is:retweet lang:en", "max_results": 5}'
```

## Important Notes

- **Cold Start**: First request may take 30-60 seconds as the model loads
- **Model Size**: BERT model is ~400MB, ensure it's included in your repository
- **Memory**: Render free tier has 512MB RAM - should be sufficient for BERT
- **Timeout**: Gunicorn timeout set to 120 seconds for model loading

## Troubleshooting

### Model Loading Issues
- Ensure `results/checkpoint-174152/` directory contains all model files
- Check model path in environment variables
- Verify model files aren't corrupted

### API Key Issues
- Double-check all environment variables in Render dashboard
- Ensure Twitter API keys have correct permissions
- Test Kore.ai webhook URL is accessible

### Memory Issues
- Consider using smaller model if needed
- Optimize workers count (currently set to 1)
- Monitor memory usage in Render dashboard