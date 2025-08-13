# Twitter Sentiment Analysis API

A Flask-based API for real-time Twitter sentiment analysis using BERT, integrated with Kore.ai bot and email notifications.

## Features

- **Real-time Sentiment Analysis**: Analyze Twitter posts using fine-tuned BERT model
- **Kore.ai Integration**: Send alerts to Kore.ai chatbot
- **Email Notifications**: Send email alerts via SMTP
- **REST API**: Multiple endpoints for different use cases
- **Twitter Integration**: Fetch and analyze recent tweets

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Analyze sentiment of provided text
- `POST /webhook` - Webhook for receiving external data
- `POST /analyze_recent` - Fetch and analyze recent tweets

## Deployment on Render

### Prerequisites

1. Trained BERT model (checkpoint-174152)
2. Twitter API credentials
3. Kore.ai bot configuration
4. Email SMTP credentials

### Environment Variables

Set these environment variables in Render:

```
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_CONSUMER_KEY=your_twitter_consumer_key
TWITTER_CONSUMER_SECRET=your_twitter_consumer_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
KORE_AI_WEBHOOK_URL=https://bots.kore.ai/api/v1.1/public/webhook/your_id
KORE_AI_CLIENT_ID=your_kore_client_id
KORE_AI_CLIENT_SECRET=your_kore_client_secret
KORE_AI_BOT_ID=your_bot_id
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
MODEL_PATH=./results/checkpoint-174152
```

### Deployment Steps

1. **Push to GitHub**: Ensure all files including model checkpoint are in your repository
2. **Create Render Service**: Connect your GitHub repository to Render
3. **Configure Environment**: Add all required environment variables
4. **Deploy**: Render will automatically build and deploy your application

### Model Size Considerations

The BERT model files are large (~400MB). Consider:
- Using Git LFS for model files
- Or downloading models during startup
- Or using a smaller model variant

## Local Development

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your credentials
4. Run: `python twitter_sentiment_alert_bot.py`

## Usage Examples

### Analyze Text
```bash
curl -X POST http://your-render-url.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin is going to the moon!"}'
```

### Analyze Recent Tweets
```bash
curl -X POST http://your-render-url.com/analyze_recent \
  -H "Content-Type: application/json" \
  -d '{"query": "Bitcoin -is:retweet lang:en", "max_results": 10}'
```