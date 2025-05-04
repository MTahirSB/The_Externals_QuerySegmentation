# train_model.py
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    DistilBertConfig,
    Trainer,
    TrainingArguments
)
import transformers, inspect
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class QueryDataset(Dataset):
    def __init__(self, tokenizer, texts, tags, max_length=128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.unique_tags = list(set(tag for doc in tags for tag in doc))
        self.tag2id = {tag: id for id, tag in enumerate(self.unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        labels = []
        word_ids = encoding.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(self.tag2id[tags[word_idx]])
            else:
                labels.append(-100)
            previous_word_idx = word_idx
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Load and prepare data
df = pd.read_csv("data/bioes_sequences.csv")
df['tokens'] = df['tokens'].apply(eval)
df['bioes_tags'] = df['bioes_tags'].apply(eval)

# Split data
train_texts, val_texts, train_tags, val_tags = train_test_split(
    df['tokens'].tolist(),
    df['bioes_tags'].tolist(),
    test_size=0.2
)

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Create datasets
train_dataset = QueryDataset(tokenizer, train_texts, train_tags)
val_dataset = QueryDataset(tokenizer, val_texts, val_tags)

# Create config AFTER datasets
config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(train_dataset.unique_tags),
    id2label=train_dataset.id2tag,
    label2id=train_dataset.tag2id
)

# Initialize model
model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    config=config
)

print("transformers version:", transformers.__version__)
print("transformers module file:", transformers.__file__)
print("TrainingArguments signature:", inspect.signature(transformers.TrainingArguments.__init__))

# Training setup
training_args = TrainingArguments(
    output_dir="./models/results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",  # ✅ Works in v4.0+
    save_strategy="epoch",
    learning_rate=2e-5,
    logging_dir="./models/logs",
)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [train_dataset.id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [train_dataset.id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    flat_predictions = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]
    return classification_report(flat_labels, flat_predictions, output_dict=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("./models/saved_model")
tokenizer.save_pretrained("./models/saved_model")
