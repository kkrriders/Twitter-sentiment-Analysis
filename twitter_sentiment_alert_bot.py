import os
import json
import requests
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import HfApi, HfFolder, Repository

# Load environment variables from .env
load_dotenv()

# Twitter API credentials
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# Kore.ai credentials
KORE_BOT_ID = os.getenv("KORE_BOT_ID")
KORE_CLIENT_ID = os.getenv("KORE_CLIENT_ID")
KORE_CLIENT_SECRET = os.getenv("KORE_CLIENT_SECRET")
KORE_API_KEY = os.getenv("KORE_API_KEY")
KORE_AUTHORIZATION = os.getenv("KORE_AUTHORIZATION")


# Hugging Face credentials
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")  

# Authenticate Hugging Face
if HF_TOKEN:
    HfFolder.save_token(HF_TOKEN)


TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


MODEL_PATH = os.getenv("MODEL_PATH", "./results/checkpoint-87076")


if HF_REPO_ID and HF_TOKEN:
    MODEL_NAME = HF_REPO_ID
    print(f"[Model] Loading model from HuggingFace Hub: {HF_REPO_ID}")
    try:
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        print(f"[Model] ✅ Successfully loaded from HuggingFace Hub")
    except Exception as e:
        print(f"[Model] ❌ Failed to load from HuggingFace: {e}")
        print(f"[Model] 🔄 Falling back to local model: {MODEL_PATH}")
        MODEL_NAME = MODEL_PATH
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    MODEL_NAME = MODEL_PATH
    print(f"[Model] Loading model from local path: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

nlp_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def twitter_headers():
    return {"Authorization": f"Bearer {BEARER_TOKEN}"}

def process_tweet(text):
    sentiment = nlp_pipeline(text)[0]
    raw_label = sentiment["label"]
    score = sentiment["score"]
    
   
    label_mapping = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "POSITIVE", 
        "LABEL_2": "NEUTRAL"
    }
    
    label = label_mapping.get(raw_label, raw_label)

    print(f"[Tweet] {text}")
    print(f"[Sentiment] {label} ({score:.2f})")

   
    send_kore_alert(text, label, score)

def send_kore_alert(tweet_text, sentiment_label, sentiment_score):
    """Send sentiment data to Kore.ai webhook for processing and email sending"""
    webhook_url = os.getenv("KORE_AI_WEBHOOK_URL")
    
    if not webhook_url:
        print("[Kore.ai] Webhook URL not configured. Skipping alert.")
        return
    
    
    if "your_webhook_id" in webhook_url:
        print("[Kore.ai] Bot not published yet. Logging sentiment locally.")
        log_sentiment_locally(tweet_text, sentiment_label, sentiment_score)
        return
    
   
    emoji = "📊"  
    if sentiment_label.lower() in ["positive", "very positive"]:
        emoji = "🚀"
    elif sentiment_label.lower() in ["negative", "very negative"]:
        emoji = "🚨"
    
    payload = {
        "tweet_text": tweet_text,
        "sentiment": sentiment_label,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "timestamp": get_current_timestamp(),
        "emoji": emoji
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if KORE_AUTHORIZATION:
        headers["Authorization"] = KORE_AUTHORIZATION
    
    try:
        response = requests.post(webhook_url, headers=headers, json=payload)
        print(f"[Kore.ai] {emoji} {sentiment_label.upper()} sentiment sent to webhook. Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"[Kore.ai] Webhook response: {response.text}")
            
    except Exception as e:
        print(f"[Kore.ai] Error sending to webhook: {e}")

def log_sentiment_locally(tweet_text, sentiment_label, sentiment_score):
    """Log sentiment locally while webhook is not available"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
   
    emoji = "🚀" if sentiment_label == "POSITIVE" else "🚨" if sentiment_label == "NEGATIVE" else "📊"
    print(f"[Local Log] {emoji} {sentiment_label} ({sentiment_score:.2f}) - {tweet_text}")
    
 
    try:
        with open("sentiment_alerts.log", "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {sentiment_label} | {sentiment_score:.2f} | {tweet_text}\n")
    except Exception as e:
        print(f"[Local Log] Error writing to file: {e}")

def get_current_timestamp():
    """Get current timestamp in readable format"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

def search_bitcoin_tweets():
    """Search for recent Bitcoin tweets using Twitter API v2"""
    query = "bitcoin OR btc OR #bitcoin OR #btc -is:retweet lang:en"
    params = {
        "query": query,
        "max_results": 10,  # Stay within free tier limits
        "tweet.fields": "created_at,public_metrics"
    }
    
    try:
        response = requests.get(TWITTER_SEARCH_URL, headers=twitter_headers(), params=params)
        if response.status_code != 200:
            print(f"[Twitter] Error searching tweets: {response.status_code} - {response.text}")
            return []
        
        data = response.json()
        if "data" in data:
            print(f"[Twitter] Found {len(data['data'])} Bitcoin tweets")
            return data["data"]
        else:
            print("[Twitter] No tweets found")
            return []
            
    except Exception as e:
        print(f"[Twitter] Search error: {e}")
        return []

def start_periodic_search():
    """Run periodic Twitter search for Bitcoin tweets"""
    import time
    
    print("[System] Starting periodic Bitcoin tweet search...")
    print("[System] Searching every 30 minutes to stay within API limits")
    
    # Track processed tweets to avoid duplicates
    processed_tweets = set()
    
    while True:
        try:
            tweets = search_bitcoin_tweets()
            
            for tweet in tweets:
                tweet_id = tweet["id"]
                if tweet_id not in processed_tweets:
                    tweet_text = tweet["text"]
                    print(f"[New Tweet] Processing: {tweet_text[:100]}...")
                    process_tweet(tweet_text)
                    processed_tweets.add(tweet_id)
                    
                    # Keep only last 1000 processed IDs to manage memory
                    if len(processed_tweets) > 1000:
                        processed_tweets = set(list(processed_tweets)[-500:])
            
            # Wait 30 minutes between searches (Free tier: ~3 searches/day)
            print("[System] Waiting 30 minutes before next search...")
            time.sleep(1800)  # 30 minutes
            
        except KeyboardInterrupt:
            print("[System] Search stopped by user")
            break
        except Exception as e:
            print(f"[Error] Search error: {e}")
            print("[System] Retrying in 5 minutes...")
            time.sleep(300)  # Wait 5 minutes on error

def push_model_to_huggingface():
    """
    Pushes your fine-tuned model to the Hugging Face Hub.
    """
    if not HF_TOKEN or not HF_REPO_ID:
        print("[HF] Missing Hugging Face credentials. Skipping model push.")
        return

    try:
        repo = Repository(local_dir="hf_model_repo", clone_from=HF_REPO_ID, use_auth_token=HF_TOKEN)
        tokenizer.save_pretrained("hf_model_repo")
        model.save_pretrained("hf_model_repo")
        repo.push_to_hub(commit_message="Updated sentiment model")
        print("[HF] Model successfully pushed to Hugging Face Hub")
    except Exception as e:
        print(f"[HF] Error pushing model to Hugging Face: {e}")

if __name__ == "__main__":
    print("[System] Starting Bitcoin Sentiment Analysis Bot...")
    print(f"[System] Model: {MODEL_NAME}")
    print(f"[System] Webhook: {os.getenv('KORE_AI_WEBHOOK_URL', 'Not configured')}")
    
    try:
        
        if HF_TOKEN and HF_REPO_ID:
            push_model_to_huggingface()

        
        print("[System] Starting periodic Twitter search...")
        start_periodic_search()

    except KeyboardInterrupt:
        print("[System] Bot stopped by user.")
    except Exception as e:
        print(f"[Error] {e}")
