# The_Externals_QuerySegmentation
A deep learning model for query segmentation into brand, attributes, and product type.

# Query Segmentation

A deep learning model that segments search queries into three components: **Brand**, **Attributes**, and **Product Type**, using BIOES tagging and transformers.

## 🚀 Features

- BIOES sequence tagging using HuggingFace Transformers
- Evaluation metrics: Top‑1 Accuracy, Precision, Recall, F1
- Leaderboard-ready predictions
- Qualitative review tools

## 📂 Project Structure


## 📊 Metrics (Latest Run)

- **Top‑1 Accuracy:** %
- **Brand Precision:** %
- **Attribute Recall:** 85.3%

## 🧪 Quickstart

```bash
pip install -r requirements.txt

# Train model
python src/train_model.py

# Evaluate model
python src/evaluate_model.py \
  --test-data data/bioes_sequences.csv \
  --model-dir models/saved_model


