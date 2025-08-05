from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route('/send-email', methods=['POST'])
def send_email():
    try:
        data = request.get_json()
        
        # Extract email parameters
        to_email = data.get('to_email')
        subject = data.get('subject', 'Sentiment Analysis Alert')
        tweet_text = data.get('tweet_text', '')
        sentiment = data.get('sentiment', '')
        
        # Email content
        body = f"""
        Sentiment Analysis Alert
        
        Tweet: {tweet_text}
        Sentiment: {sentiment}
        Timestamp: {data.get('timestamp', 'Not provided')}
        
        This is an automated alert from your Twitter Sentiment Analysis Bot.
        """
        
        # Email configuration
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        
        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, message.as_string())
        
        return jsonify({
            "status": "success",
            "message": "Email sent successfully",
            "to": to_email,
            "sentiment": sentiment
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "email"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)