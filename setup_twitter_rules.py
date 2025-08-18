#!/usr/bin/env python3
"""
Script to set up Twitter streaming rules for Bitcoin sentiment analysis
"""
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
STREAM_RULES_URL = "https://api.twitter.com/2/tweets/search/stream/rules"

def get_headers():
    return {"Authorization": f"Bearer {BEARER_TOKEN}", "Content-Type": "application/json"}

def get_existing_rules():
    """Get existing streaming rules"""
    response = requests.get(STREAM_RULES_URL, headers=get_headers())
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting rules: {response.status_code} - {response.text}")
        return None

def delete_rules(rule_ids):
    """Delete existing rules"""
    if not rule_ids:
        return
    
    data = {"delete": {"ids": rule_ids}}
    response = requests.post(STREAM_RULES_URL, headers=get_headers(), json=data)
    if response.status_code == 200:
        print(f"✅ Deleted {len(rule_ids)} existing rules")
    else:
        print(f"❌ Error deleting rules: {response.status_code} - {response.text}")

def add_bitcoin_rules():
    """Add Bitcoin-related streaming rules"""
    rules = [
        {"value": "bitcoin OR btc OR Bitcoin OR BTC", "tag": "bitcoin_general"},
        {"value": "#bitcoin OR #Bitcoin OR #BTC OR #btc", "tag": "bitcoin_hashtags"},
        {"value": "bitcoin price OR btc price OR bitcoin news", "tag": "bitcoin_news"}
    ]
    
    data = {"add": rules}
    response = requests.post(STREAM_RULES_URL, headers=get_headers(), json=data)
    
    if response.status_code == 201:
        print("✅ Bitcoin streaming rules added successfully!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Error adding rules: {response.status_code} - {response.text}")

def main():
    if not BEARER_TOKEN:
        print("❌ TWITTER_BEARER_TOKEN not found in environment variables")
        return
    
    print("🔍 Checking existing Twitter streaming rules...")
    existing = get_existing_rules()
    
    if existing and "data" in existing:
        rule_ids = [rule["id"] for rule in existing["data"]]
        print(f"Found {len(rule_ids)} existing rules. Deleting...")
        delete_rules(rule_ids)
    else:
        print("No existing rules found.")
    
    print("➕ Adding Bitcoin sentiment analysis rules...")
    add_bitcoin_rules()
    
    print("✅ Setup complete! Your bot should now start receiving Bitcoin tweets.")

if __name__ == "__main__":
    main()