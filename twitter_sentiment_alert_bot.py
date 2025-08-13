from flask import Flask, request, jsonify
import os
import requests
import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Environment validation
def validate_environment() -> None:
    """Validate that all required environment variables are set"""
    required_vars = [
        'TWITTER_BEARER_TOKEN',
        'KORE_AI_CLIENT_ID', 
        'KORE_AI_CLIENT_SECRET',
        'KORE_AI_BOT_ID'
    ]
    
    missing_vars: List[str] = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("Some features may not work properly.")
    else:
        print("✅ All required environment variables are set.")

# Validate environment on startup
validate_environment()

# === Twitter API Configuration ===
bearer_token: Optional[str] = os.getenv('TWITTER_BEARER_TOKEN')
consumer_key: Optional[str] = os.getenv('TWITTER_CONSUMER_KEY')
consumer_secret: Optional[str] = os.getenv('TWITTER_CONSUMER_SECRET')
access_token: Optional[str] = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_secret: Optional[str] = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# === Kore.ai Webhook Settings ===
kore_ai_webhook_url: Optional[str] = os.getenv('KORE_AI_WEBHOOK_URL')
kore_ai_client_id: Optional[str] = os.getenv('KORE_AI_CLIENT_ID')
kore_ai_client_secret: Optional[str] = os.getenv('KORE_AI_CLIENT_SECRET')
kore_ai_bot_id: Optional[str] = os.getenv('KORE_AI_BOT_ID')

# === Load Sentiment Analysis Model ===
script_dir: str = os.path.dirname(os.path.abspath(__file__))
default_model_path: str = os.path.join(script_dir, 'results', 'checkpoint-174152')
model_path: str = os.getenv('MODEL_PATH', default_model_path)

if not os.path.isabs(model_path):
    model_path = os.path.join(script_dir, model_path)

print(f"🔍 Looking for model at: {model_path}")

try:
    print(f"🔍 Loading tokenizer from: bert-base-uncased")
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"🔍 Loading model from: {model_path}")
    model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print(f"✅ Model loaded successfully from: {model_path}")
    print(f"✅ Tokenizer loaded from: bert-base-uncased")
except FileNotFoundError as e:
    print(f"❌ Model files not found at {model_path}")
    print(f"Error: {e}")
    print("Make sure the model checkpoint directory exists and contains required files.")
    raise
except Exception as e:
    print(f"❌ Could not load model from {model_path}")
    print(f"Error: {e}")
    print("Check if the model files are compatible and not corrupted.")
    raise

# === Predict Sentiment Function ===
def predict_sentiment(text: str) -> str:
    inputs: Dict[str, Any] = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits: torch.Tensor = outputs.logits
        predicted_class: int = torch.argmax(logits, dim=1).item()
        labels: List[str] = ['Negative', 'Neutral', 'Positive']
        return labels[predicted_class]

# === Send Alert to Kore.ai ===
def send_alert_to_kore(text: str, sentiment: str) -> bool:
    if not kore_ai_webhook_url or kore_ai_webhook_url == 'https://bots.kore.ai/api/v1.1/public/webhook/xxxxx':
        print(f"⚠️ Kore.ai not configured. Would send: {sentiment} for tweet: {text[:50]}...")
        return True
    
    # Send message to trigger dialog with context
    payload: Dict[str, Any] = {
        "message": {
            "text": f"New tweet alert: {sentiment} sentiment detected",
            "type": "text"
        },
        "from": {
            "id": "twitter_bot"
        },
        "context": {
            "tweet_text": text,
            "sentiment": sentiment,
            "timestamp": str(datetime.now().isoformat())
        }
    }
    
    try:
        # Use message API endpoint to send message to bot
        message_url: str = f"https://bots.kore.ai/api/v1.1/public/{kore_ai_bot_id}/messages"
        print(f"🔍 Sending to Kore.ai: {payload}")
        print(f"🔗 URL: {message_url}")
        response: requests.Response = requests.post(
            message_url,
            json=payload,
            headers={
                "X-Kore-Client-Id": kore_ai_client_id,
                "X-Kore-Client-Secret": kore_ai_client_secret,
                "Content-Type": "application/json"
            }
        )
        print(f"🔍 Kore.ai Response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            print(f"✅ Alert sent for tweet: {text[:50]}... | Sentiment: {sentiment}")
            return True
        else:
            print(f"❌ Failed to send alert. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error sending alert: {e}")
        return False

# === API Routes ===
@app.route('/health', methods=['GET'])
def health_check() -> Tuple[Dict[str, Any], int]:
    return jsonify({"status": "healthy", "model_loaded": True}), 200

@app.route('/predict', methods=['POST'])
def predict() -> Tuple[Dict[str, Any], int]:
    try:
        data: Optional[Dict[str, Any]] = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        text: str = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        sentiment: str = predict_sentiment(text)
        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[Dict[str, Any], int]:
    """This is your callback URL endpoint for receiving webhook data"""
    try:
        data: Optional[Dict[str, Any]] = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        print(f"📥 Received webhook data: {data}")
        
        # Handle different webhook formats
        tweets_to_process: List[Dict[str, Any]] = []
        
        # Twitter webhook format
        if 'tweet_create_events' in data:
            tweets_to_process = data['tweet_create_events']
        # Direct tweet format
        elif 'text' in data:
            tweets_to_process = [data]
        # BotKit SDK format
        elif 'message' in data and 'text' in data['message']:
            tweets_to_process = [{'text': data['message']['text']}]
        
        for tweet_data in tweets_to_process:
            text: str = tweet_data.get('text', '').strip()
            print(f"📝 Processing tweet: {text}")
            if text and not text.startswith('RT @'):  # Skip retweets
                sentiment: str = predict_sentiment(text)
                print(f"🧠 Sentiment analyzed: {sentiment}")
                print(f"📤 Calling send_alert_to_kore...")
                result: bool = send_alert_to_kore(text, sentiment)
                print(f"📧 Kore.ai call result: {result}")
                print(f"🎯 Processed: {text[:50]}... | Sentiment: {sentiment}")
            else:
                print(f"⏭️ Skipping tweet: {text[:30]}...")
        
        return jsonify({"status": "success", "processed": len(tweets_to_process)}), 200
    
    except Exception as e:
        print(f"❌ Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_recent', methods=['POST'])
def analyze_recent_tweets() -> Tuple[Dict[str, Any], int]:
    try:
        data: Dict[str, Any] = request.get_json() or {}
        query: str = data.get('query', 'Bitcoin -is:retweet lang:en')
        max_results: int = data.get('max_results', 5)
        
        client: tweepy.Client = tweepy.Client(bearer_token=bearer_token)
        response = client.search_recent_tweets(
            query=query, 
            tweet_fields=["text", "created_at"], 
            max_results=max_results
        )
        
        results: List[Dict[str, Any]] = []
        if response.data:
            for tweet in response.data:
                text: str = tweet.text.strip()
                if text:
                    sentiment: str = predict_sentiment(text)
                    send_alert_to_kore(text, sentiment)
                    results.append({
                        "text": text,
                        "sentiment": sentiment,
                        "created_at": str(tweet.created_at)
                    })
        
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port: int = int(os.getenv('PORT', '5000'))
    print("🚀 Starting Twitter Sentiment Analysis API...")
    print(f"📍 Webhook URL will be available on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
