import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

# Import functions from your BIOES sequence module
from BIOES_sequence import process_chunk, extract_entities  # you will add extract_entities helper


def extract_entities(tokens, tags):  # if not present, include this implementation
    """
    Given a list of tokens and their BIOES tags, returns a dict of extracted entities by type.
    """
    entities = {'Brand': [], 'Attribute': [], 'ProductType': []}
    curr = []
    curr_type = None
    for token, tag in zip(tokens, tags):
        if tag.startswith('S-'):
            typ = tag.split('-', 1)[1]
            entities[typ].append(token)
        elif tag.startswith('B-'):
            curr_type = tag.split('-', 1)[1]
            curr = [token]
        elif tag.startswith('I-') and curr_type:
            curr.append(token)
        elif tag.startswith('E-') and curr_type:
            curr.append(token)
            entities[curr_type].append(" ".join(curr))
            curr_type = None
            curr = []
        else:
            curr_type = None
            curr = []
    return entities


def compute_top1_accuracy(df):
    matches = (
        (df['brand'] == df['brand_pred']) &
        (df['product_type'] == df['product_type_pred']) &
        (df['attributes'].apply(lambda x: set(x)) == df['attributes_pred'].apply(lambda x: set(x)))
    )
    return matches.mean()


def compute_precision_recall(df, key):
    # Flatten for multi-label attributes
    y_true, y_pred = [], []
    if key == 'attributes':
        for true_list, pred_list in zip(df['attributes'], df['attributes_pred']):
            labels = set(true_list) | set(pred_list)
            for lab in labels:
                y_true.append(1 if lab in true_list else 0)
                y_pred.append(1 if lab in pred_list else 0)
    else:
        y_true = df[key].tolist()
        y_pred = df[f"{key}_pred"].tolist()
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    return precision, recall


def main():
    parser = argparse.ArgumentParser(description="Evaluate Query Segmentation Model")
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test CSV with columns: query, brand, attributes, product_type')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to saved model directory (with config and tokenizer)')
    parser.add_argument('--output-dir', type=str, default='models/saved_model/results',
                        help='Directory to write predictions and metrics')
    parser.add_argument('--chunk-size', type=int, default=10000)
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    test_df = pd.read_csv(args.test_data)
    # Ensure attributes column is list
    test_df['attributes'] = test_df['attributes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Generate predictions using process_chunk
    preds = process_chunk(test_df, process_pool=None if args.no_parallel else None)
    pred_df = pd.DataFrame(preds)

    # Parse tokens and tags
    pred_df['tokens'] = pred_df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    pred_df['bioes_tags'] = pred_df['bioes_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Extract predicted entities
    ent = pred_df.apply(lambda row: extract_entities(row['tokens'], row['bioes_tags']), axis=1)
    pred_df['brand_pred'] = ent.apply(lambda e: e['Brand'][0] if e['Brand'] else '')
    pred_df['attributes_pred'] = ent.apply(lambda e: e['Attribute'])
    pred_df['product_type_pred'] = ent.apply(lambda e: e['ProductType'][0] if e['ProductType'] else '')

    # Combine with ground truth
    df = test_df.copy()
    df = df.join(pred_df[['brand_pred', 'attributes_pred', 'product_type_pred']])

    # Compute metrics
    top1 = compute_top1_accuracy(df)
    brand_prec, brand_rec = compute_precision_recall(df, 'brand')
    attr_prec, attr_rec = compute_precision_recall(df, 'attributes')
    pt_prec, pt_rec = compute_precision_recall(df, 'product_type')

    metrics = {
        'top1_accuracy': top1,
        'brand_precision': brand_prec, 'brand_recall': brand_rec,
        'attribute_precision': attr_prec, 'attribute_recall': attr_rec,
        'product_type_precision': pt_prec, 'product_type_recall': pt_rec
    }

    # Write metrics
    with open(Path(args.output_dir) / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Write detailed predictions for qualitative analysis
    df.to_csv(Path(args.output_dir) / 'predictions.csv', index=False)

    # Prepare leaderboard submission
    lb = df[['query', 'brand_pred', 'attributes_pred', 'product_type_pred']].copy()
    lb['attributes'] = lb['attributes_pred'].apply(lambda lst: "|".join(lst))
    lb = lb.rename(columns={'brand_pred': 'brand', 'product_type_pred': 'product_type'})
    lb.to_csv(Path(args.output_dir) / 'leaderboard_submission.csv', index=False)

    print("Evaluation complete. Metrics:")
    for k, v in metrics.items(): print(f"{k}: {v:.4f}")
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
