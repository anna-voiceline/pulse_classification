#!/usr/bin/env python
"""
Simplified prediction script for insight classifier.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import logging

# Add project root to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import InsightDataset
from src.models.classifier import InsightClassifier
from src.utils.set_up import parse_output_args

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    args = parse_output_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Load model checkpoint
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Get model configuration
    config = checkpoint.get('config', {})
    input_dim = config.get('input_dim', 1536)
    hidden_dim = config.get('hidden_dim', 512)

    # Create model
    model = InsightClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Get threshold
    threshold = args.threshold
    if threshold is None:
        threshold = checkpoint.get('threshold', 0.5)
    logger.info(f"Using classification threshold: {threshold}")

    # Load embeddings
    logger.info(f"Loading embeddings from {args.embeddings}")
    embeddings = np.load(args.embeddings)

    # Load texts if provided
    texts = None
    if args.texts:
        logger.info(f"Loading texts from {args.texts}")
        texts = np.load(args.texts, allow_pickle=True)
        assert len(texts) == len(embeddings), "Number of texts must match number of embeddings"

    # Create dummy labels (not used for prediction)
    dummy_labels = np.zeros(len(embeddings))

    # Create dataset and dataloader
    dataset = InsightDataset(embeddings, dummy_labels, texts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Make predictions
    logger.info("Making predictions...")
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)

            # Get predictions
            outputs = model(embeddings)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)

    # Create results dataframe
    results = pd.DataFrame({'probability': all_probs, 'prediction': all_preds})

    # Add texts if available
    if texts is not None:
        results['text'] = texts

    # Sort by probability
    results = results.sort_values('probability', ascending=False)

    # Save predictions
    logger.info(f"Saving predictions to {args.output}")
    results.to_csv(args.output, index=False)

    # Print summary
    n_important = results['prediction'].sum()
    logger.info(f"Total insights: {len(results)}")
    logger.info(f"Important insights: {n_important} ({n_important/len(results)*100:.1f}%)")

    # Print top important insights
    if n_important > 0 and texts is not None:
        important = results[results['prediction'] == 1]
        logger.info("\nTop 5 important insights:")
        for i, (_, row) in enumerate(important.head(5).iterrows()):
            logger.info(f"{i+1}. {row['text']} (confidence: {row['probability']:.4f})")

if __name__ == "__main__":
    main()