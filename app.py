import threading
import os
from flask import Flask, jsonify
from twitter_sentiment_alert_bot import start_stream, MODEL_NAME

app = Flask(__name__)

# Global variable to track bot status
bot_status = {"status": "starting", "model": MODEL_NAME}

@app.route('/')
def health_check():
    return jsonify({
        "status": "Twitter Sentiment Bot is running",
        "model": bot_status["model"],
        "bot_status": bot_status["status"]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

def run_bot():
    """Run the Twitter bot in a separate thread"""
    try:
        bot_status["status"] = "running"
        start_stream()
    except Exception as e:
        bot_status["status"] = f"error: {str(e)}"
        print(f"Bot error: {e}")

if __name__ == '__main__':
    # Start the Twitter bot in a background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Start Flask web server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)