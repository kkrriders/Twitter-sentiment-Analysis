import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

# Load your dataset
df = pd.read_csv("C:\\Users\\karti\\Twitter sentiment Analysis\\Bitcoin_tweets_after_june_2022.csv")

# Auto-label if 'label' column not present
if 'label' not in df.columns:
    def assign_label(text):
        text = str(text).lower()
        if 'good' in text or 'love' in text or 'great' in text:
            return 1  # Positive
        elif 'bad' in text or 'hate' in text or 'terrible' in text:
            return 0  # Negative
        else:
            return 2  # Neutral
    df['label'] = df['text'].apply(assign_label)

# Only keep required columns
df = df[['text', 'label']]

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization with padding and truncation
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128  # You can adjust this if needed
    )

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Tokenize
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch and rename label column
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Train-test split
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Training arguments
training_args = TrainingArguments(
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
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")