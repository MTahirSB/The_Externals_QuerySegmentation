# The_Externals_QuerySegmentation
# QuerySegmentationProject

Segment e‑commerce queries into **Brand**, **Attributes**, and **Product Type** using a BIOES‑tagged DistilBERT model.


## 🚀 Features

- BIOES sequence tagging
- Evaluation metrics: Top‑1 Accuracy, Precision, Recall, F1
- Leaderboard-ready predictions
- Qualitative review tools

## 📂 Project Structure

The_Externals_QuerySegmentation/
├── data/
│ ├── Brand_Inference.csv # raw test queries
│ ├── Queries.csv # leaderboard history
│ └── bioes_sequences.csv # processed BIOES sequences
├── src/
│ ├── train_model.py # training script
│ ├── evaluate_model.py # evaluation & metrics
│ └── BIOES_sequence.py # preprocessing & tagging logic
├── models/
│ └── saved_model/ # fine‑tuned model + tokenizer
│ └── results/ # evaluation outputs
├── requirements.txt
└── README.md

## 📊 Metrics (Latest Run)

- **Top‑1 Accuracy:  %
- **Brand Precision:  %
- **Attribute Recall:** 85.3%

## 🧪 Quickstart

git clone https://github.com/MTahirSB/The_Externals_QuerySegmentation
cd The_Externals_QuerySegmentation
pip install -r requirements.txt

# Generating BIOES Sequences

python src/BIOES_sequence.py \

This produces data/bioes_sequences.csv

# Train model

python src/train_model.py

Reads data/bioes_sequences.csv.

Fine‑tunes DistilBERT

Saves model & tokenizer under models/saved_model/

# Evaluation

python src/evaluate_model.py \

metrics.json (Top‑1 accuracy, precision & recall)

predictions.csv (detailed per‑query)

leaderboard_submission.csv (pipe‑delimited attributes)

## Approach

| Step              | Description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| **Preprocessing** | Clean and tokenize queries; parse attribute strings; apply BIOES tags.     |
| **Modeling**      | Fine‑tune `DistilBertForTokenClassification` on token‑level BIOES labels.  |
| **Inference**     | Tag unseen queries, extract entities by BIOES decode.                      |
| **Evaluation**    | Compute Top‑1 Accuracy, micro‑precision & recall for each segment.         |





