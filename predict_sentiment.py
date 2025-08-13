from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Any

# Load tokenizer from base model
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load trained model from checkpoint
model_path: str = "./results/checkpoint-174152"
model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# Class labels (adjust based on training dataset)
label_map: Dict[int, str] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Prediction function
def predict_sentiment(text: str) -> str:
    inputs: Dict[str, Any] = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits: torch.Tensor = outputs.logits
        predicted_class: int = torch.argmax(logits, dim=1).item()
        sentiment: str = label_map[predicted_class]
    return sentiment

# Example usage
if __name__ == "__main__":
    sample_text: str = input("Enter tweet or sentence to analyze sentiment:\n> ")
    result: str = predict_sentiment(sample_text)
    print(f"\nPredicted Sentiment: {result}")
