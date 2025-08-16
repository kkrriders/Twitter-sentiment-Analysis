import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
from typing import Dict, Any, List

# Load your dataset
df: pd.DataFrame = pd.read_csv("C:\\Users\\karti\\Twitter sentiment Analysis\\Bitcoin_tweets_after_june_2022.csv")

# Auto-label if 'label' column not present
if 'label' not in df.columns:
    def improved_assign_label(text: Any) -> int:
        """Enhanced sentiment labeling with more comprehensive keywords"""
        text = str(text).lower()
        
        # Positive indicators
        positive_words = [
            'moon', 'bullish', 'pump', 'surge', 'rally', 'gains', 'profit', 'win', 
            'success', 'boom', 'soar', 'rocket', 'lambo', 'diamond', 'hodl',
            'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'like',
            'up', 'rise', 'increase', 'bull', 'green', 'buy', 'strong'
        ]
        
        # Negative indicators  
        negative_words = [
            'crash', 'dump', 'bear', 'bearish', 'drop', 'fall', 'down', 'loss',
            'terrible', 'awful', 'bad', 'hate', 'fear', 'panic', 'sell', 'red',
            'dead', 'rip', 'broke', 'bankruptcy', 'scam', 'fraud', 'ponzi'
        ]
        
        # Emojis
        positive_emojis = ['🚀', '🌙', '💎', '🔥', '💪', '🎉', '📈', '💰', '🤑']
        negative_emojis = ['😱', '💀', '😭', '📉', '😰', '🤮', '💸', '⚰️']
        
        # Count positive and negative indicators
        pos_score = sum(1 for word in positive_words if word in text)
        neg_score = sum(1 for word in negative_words if word in text)
        
        # Check emojis
        for emoji in positive_emojis:
            if emoji in text:
                pos_score += 1
        for emoji in negative_emojis:
            if emoji in text:
                neg_score += 1
        
        # Decision logic
        if pos_score > neg_score:
            return 1  # Positive
        elif neg_score > pos_score:
            return 0  # Negative
        else:
            return 2  # Neutral
    
    df['label'] = df['text'].apply(improved_assign_label)

# Only keep required columns
df = df[['text', 'label']]

# Show label distribution
print("🏷️  IMPROVED LABEL DISTRIBUTION:")
label_counts = df['label'].value_counts().sort_index()
total = len(df)
for label, count in label_counts.items():
    sentiment = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}[label]
    percentage = (count/total)*100
    print(f"   {sentiment}: {count} ({percentage:.1f}%)")
print(f"   Total: {total} tweets\n")

# Tokenizer
tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization with padding and truncation
def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128  # You can adjust this if needed
    )

# Convert to HuggingFace Dataset
dataset: Dataset = Dataset.from_pandas(df)

# Tokenize
tokenized_dataset: Dataset = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch and rename label column
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Train-test split
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset: Dataset = split_dataset["train"]
eval_dataset: Dataset = split_dataset["test"]

# Load model
model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Training arguments
training_args: TrainingArguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
)

# Trainer
trainer: Trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")