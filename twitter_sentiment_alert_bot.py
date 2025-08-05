from flask import Flask, request, jsonify
import os
import requests
import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)

# === Twitter API Configuration ===
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# === Kore.ai Webhook Settings ===
kore_ai_webhook_url = os.getenv('KORE_AI_WEBHOOK_URL')
kore_ai_client_id = os.getenv('KORE_AI_CLIENT_ID')
kore_ai_client_secret = os.getenv('KORE_AI_CLIENT_SECRET')
kore_ai_bot_id = os.getenv('KORE_AI_BOT_ID')

# === Load Sentiment Analysis Model ===
script_dir = os.path.dirname(os.path.abspath(__file__))
default_model_path = os.path.join(script_dir, 'results', 'checkpoint-174152')
model_path = os.getenv('MODEL_PATH', default_model_path)

if not os.path.isabs(model_path):
    model_path = os.path.join(script_dir, model_path)

print(f"🔍 Looking for model at: {model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print(f"✅ Model loaded successfully from: {model_path}")
    print(f"✅ Tokenizer loaded from: bert-base-uncased")
except Exception as e:
    print(f"❌ Could not load model from {model_path}")
    print(f"Error: {e}")
    raise FileNotFoundError(f"Could not load model from {model_path}. Error: {e}")

# === Predict Sentiment Function ===
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        labels = ['Negative', 'Neutral', 'Positive']
        return labels[predicted_class]

# === Send Alert to Kore.ai ===
def send_alert_to_kore(text, sentiment):
    if not kore_ai_webhook_url or kore_ai_webhook_url == 'https://bots.kore.ai/api/v1.1/public/webhook/xxxxx':
        print(f"⚠️ Kore.ai not configured. Would send: {sentiment} for tweet: {text[:50]}...")
        return True
    
    # Send message to trigger dialog with context
    payload = {
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
        message_url = f"https://bots.kore.ai/api/v1.1/public/{kore_ai_bot_id}/messages"
        print(f"🔍 Sending to Kore.ai: {payload}")
        print(f"🔗 URL: {message_url}")
        response = requests.post(
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
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        sentiment = predict_sentiment(text)
        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """This is your callback URL endpoint for receiving webhook data"""
    try:
        data = request.get_json()
        print(f"📥 Received webhook data: {data}")
        
        # Handle different webhook formats
        tweets_to_process = []
        
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
            text = tweet_data.get('text', '').strip()
            print(f"📝 Processing tweet: {text}")
            if text and not text.startswith('RT @'):  # Skip retweets
                sentiment = predict_sentiment(text)
                print(f"🧠 Sentiment analyzed: {sentiment}")
                print(f"📤 Calling send_alert_to_kore...")
                result = send_alert_to_kore(text, sentiment)
                print(f"📧 Kore.ai call result: {result}")
                print(f"🎯 Processed: {text[:50]}... | Sentiment: {sentiment}")
            else:
                print(f"⏭️ Skipping tweet: {text[:30]}...")
        
        return jsonify({"status": "success", "processed": len(tweets_to_process)}), 200
    
    except Exception as e:
        print(f"❌ Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_recent', methods=['POST'])
def analyze_recent_tweets():
    try:
        data = request.get_json() or {}
        query = data.get('query', 'Bitcoin -is:retweet lang:en')
        max_results = data.get('max_results', 5)
        
        client = tweepy.Client(bearer_token=bearer_token)
        response = client.search_recent_tweets(
            query=query, 
            tweet_fields=["text", "created_at"], 
            max_results=max_results
        )
        
        results = []
        if response.data:
            for tweet in response.data:
                text = tweet.text.strip()
                if text:
                    sentiment = predict_sentiment(text)
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
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Twitter Sentiment Analysis API...")
    print(f"📍 Webhook URL will be: http://localhost:5000/webhook")
    print(f"📍 With ngrok: https://your-ngrok-url.ngrok.io/webhook")
    app.run(host='0.0.0.0', port=5000, debug=True)
