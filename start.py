#!/usr/bin/env python3
"""
Startup script for Twitter Sentiment Analysis API
Handles model initialization and server startup
"""

import os
import sys
from pathlib import Path

def check_model_exists():
    """Check if the required model files exist"""
    script_dir = Path(__file__).parent
    model_path = script_dir / 'results' / 'checkpoint-174152'
    
    required_files = ['config.json', 'pytorch_model.bin']
    
    for file_name in required_files:
        if not (model_path / file_name).exists():
            print(f"❌ Missing required model file: {file_name}")
            return False
    
    print(f"✅ Model files found at: {model_path}")
    return True

def main():
    """Main startup function"""
    print("🚀 Initializing Twitter Sentiment Analysis API...")
    
    # Check if model exists
    if not check_model_exists():
        print("❌ Model files not found. Please ensure the model is trained and saved.")
        sys.exit(1)
    
    # Import and run the Flask app
    try:
        from twitter_sentiment_alert_bot import app
        port = int(os.getenv('PORT', 5000))
        print(f"📍 Starting server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()