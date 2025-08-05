from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load trained model from checkpoint
model_path = "./results/checkpoint-174152"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# Class labels (adjust based on training dataset)
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        sentiment = label_map[predicted_class]
    return sentiment

# Example usage
if __name__ == "__main__":
    sample_text = input("Enter tweet or sentence to analyze sentiment:\n> ")
    result = predict_sentiment(sample_text)
    print(f"\nPredicted Sentiment: {result}")
