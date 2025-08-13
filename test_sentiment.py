import os
import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables
load_dotenv()

# === Twitter API Configuration ===
bearer_token: Optional[str] = os.getenv('TWITTER_BEARER_TOKEN')

# === Load Sentiment Analysis Model ===
script_dir: str = os.path.dirname(os.path.abspath(__file__))
default_model_path: str = os.path.join(script_dir, 'results', 'checkpoint-174152')
model_path: str = os.getenv('MODEL_PATH', default_model_path)

if not os.path.isabs(model_path):
    model_path = os.path.join(script_dir, model_path)

print(f"🔍 Looking for model at: {model_path}")

try:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print(f"✅ Model loaded successfully from: {model_path}")
    print(f"✅ Tokenizer loaded from: bert-base-uncased")
except Exception as e:
    print(f"❌ Could not load model from {model_path}")
    print(f"Error: {e}")
    exit(1)

# === Predict Sentiment Function ===
def predict_sentiment(text: str) -> str:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        labels = ['Negative', 'Neutral', 'Positive']
        return labels[predicted_class]

# === Fetch Tweets ===
def get_tweets(query: str, max_results: int = 5) -> List[tweepy.Tweet]:
    client: tweepy.Client = tweepy.Client(bearer_token=bearer_token)
    response = client.search_recent_tweets(query=query, tweet_fields=["text", "created_at"], max_results=max_results)
    return response.data if response.data else []

# === Main Driver ===
if __name__ == "__main__":
    query: str = "Bitcoin -is:retweet lang:en"
    tweets: List[tweepy.Tweet] = get_tweets(query=query, max_results=5)
    
    print(f"\n🎯 Analyzing {len(tweets)} tweets for query: '{query}'\n")
    print("=" * 80)
    
    for i, tweet in enumerate(tweets, 1):
        text: str = tweet.text.strip()
        if not text:
            continue
        
        sentiment: str = predict_sentiment(text)
        
        print(f"\n📱 Tweet #{i}:")
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Created: {tweet.created_at}")
        print("-" * 40)
    
    print(f"\n✅ Analysis complete! {len(tweets)} tweets processed.")