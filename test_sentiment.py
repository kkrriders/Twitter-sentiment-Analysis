import os
import tweepy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Twitter API Configuration ===
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

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
    exit(1)

# === Predict Sentiment Function ===
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        labels = ['Negative', 'Neutral', 'Positive']
        return labels[predicted_class]

# === Fetch Tweets ===
def get_tweets(query, max_results=5):
    client = tweepy.Client(bearer_token=bearer_token)
    response = client.search_recent_tweets(query=query, tweet_fields=["text", "created_at"], max_results=max_results)
    return response.data if response.data else []

# === Main Driver ===
if __name__ == "__main__":
    query = "Bitcoin -is:retweet lang:en"
    tweets = get_tweets(query=query, max_results=5)
    
    print(f"\n🎯 Analyzing {len(tweets)} tweets for query: '{query}'\n")
    print("=" * 80)
    
    for i, tweet in enumerate(tweets, 1):
        text = tweet.text.strip()
        if not text:
            continue
        
        sentiment = predict_sentiment(text)
        
        print(f"\n📱 Tweet #{i}:")
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Created: {tweet.created_at}")
        print("-" * 40)
    
    print(f"\n✅ Analysis complete! {len(tweets)} tweets processed.")