import json
import os

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from utils.plots import plot_confusion_matrix

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, loss_fn, optimizer,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                epochs=10, patience=5, threshold=0.5, beta=2.0):
    """
    Train the insight classifier model.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function (e.g., BCEWithLogitsLoss with pos_weight)
        optimizer: PyTorch optimizer
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of epochs to train for
        patience: Patience for early stopping
        threshold: Classification threshold
        beta: Beta value for F-beta score (higher values emphasize recall)

    Returns:
        Trained model and best validation metrics
    """
    # Move model to device
    model.to(device)

    # Track best metrics and model state
    # best_recall = 0
    best_f_beta = 0
    best_model_state = None
    patience_counter = 0

    # Lists to store metrics for each epoch
    train_losses = []
    val_metrics = []

    logger.info(f"Starting training for {epochs} epochs on {device}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            # Get batch data
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(embeddings)

            # Compute loss and backpropagate
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                # Get batch data
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)

                # Forward pass
                outputs = model(embeddings)
                probs = torch.sigmoid(outputs)
                preds = (probs >= threshold).float()

                # Store for metrics calculation
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        precision, recall, f_beta, _ = precision_recall_fscore_support(
            all_labels, all_preds, beta=beta, average='binary'
        )

        # Calculate TP, TN, FP, FN
        tp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))
        tn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))
        fp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
        fn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))

        # Add AUC if we have both classes
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0

        metrics = {
            'train_loss': avg_train_loss,
            'precision': precision,
            'recall': recall,
            'f_beta': f_beta,
            'auc': auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        val_metrics.append(metrics)

        # Log metrics
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Precision: {precision:.4f}, Recall: {recall:.4f}, F{beta}: {f_beta:.4f}, AUC: {auc:.4f}")
        logger.info(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

        # Check for improvement (based on recall)
        if f_beta > best_f_beta:
        # if recall > best_recall:
            # best_recall = recall
            best_f_beta = f_beta
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            # logger.info(f"  Improved recall ({recall:.4f})")
            logger.info(f"  Improved f beta ({f_beta:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} epochs")

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # logger.info(f"Loaded best model with recall: {best_recall:.4f}")
        logger.info(f"Loaded best model with f_beta: {best_f_beta:.4f}")

    return model, val_metrics


def find_optimal_threshold(model, val_loader, device, thresholds=None, beta=2.0):
    """
    Find the optimal classification threshold to maximize recall.

    Args:
        model: Trained model
        val_loader: DataLoader for validation data
        device: Device to use
        thresholds: List of thresholds to try (default: 0.1 to 0.9 in 0.1 increments)
        beta: Beta value for F-beta score

    Returns:
        Dictionary with optimal threshold and corresponding metrics
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    model.eval()

    # Get all probabilities and labels
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Finding optimal threshold"):
            embeddings = batch['embedding'].to(device)
            labels = batch['label']

            outputs = model(embeddings)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Try different thresholds
    results = []

    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)

        precision, recall, f_beta, _ = precision_recall_fscore_support(
            all_labels, preds, beta=beta, average='binary'
        )

        # Calculate TP, TN, FP, FN
        tp = np.sum((preds == 1) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f_beta': f_beta,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        })

        logger.info(f"Threshold: {threshold:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F{beta}: {f_beta:.4f}")
        logger.info(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

    # Find threshold with highest recall while maintaining minimum precision
    min_precision = 0.2  # Minimum acceptable precision
    valid_results = [r for r in results if r['precision'] >= min_precision]

    if valid_results:
        # Sort by recall (primary) and then F-beta (secondary)
        optimal = sorted(valid_results, key=lambda x: (x['recall'], x['f_beta']), reverse=True)[0]
    else:
        # If no threshold meets the precision requirement, take the one with highest F-beta
        optimal = sorted(results, key=lambda x: x['f_beta'], reverse=True)[0]

    logger.info(f"Optimal threshold: {optimal['threshold']:.2f}")
    logger.info(f"  Precision: {optimal['precision']:.4f}")
    logger.info(f"  Recall: {optimal['recall']:.4f}")
    logger.info(f"  F{beta}: {optimal['f_beta']:.4f}")
    logger.info(f"  TP: {optimal['true_positives']} | TN: {optimal['true_negatives']} | FP: {optimal['false_positives']} | FN: {optimal['false_negatives']}")

    return optimal


def evaluate_model(args, model, test_loader, device, threshold=0.5, beta=2.0, visualize_cm=False, output_dir='results',
                   save_detailed_results=True):
    """
    Evaluate the model on test data.

    Args:
        args: Arguments parsed from the command line.
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to use
        threshold: Classification threshold
        beta: Beta value for F-beta score
        visualize_cm: Whether to generate and save confusion matrix visualization
        output_dir: Directory to save visualizations and results
        save_detailed_results: Whether to save detailed results (text, probs, labels) for TP, FN, FP

    Returns:
        Dictionary with evaluation metrics and predictions
    """
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_samples = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            embeddings = batch['embedding'].to(device)
            labels = batch['label']

            outputs = model(embeddings)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())


            # Store all sample details
            for i in range(len(labels)):
                sample = {
                    'text': batch["text"][i] if 'text' in batch else f"Sample_{i}",
                    'probability': float(probs[i]),
                    'prediction': int(preds[i]),
                    'true_label': int(labels[i].item())
                }
                all_samples.append(sample)

    # Calculate metrics
    precision, recall, f_beta, _ = precision_recall_fscore_support(
        all_labels, all_preds, beta=beta, average='binary'
    )

    # Calculate AUC if we have both classes
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0

    # Calculate TP, TN, FP, FN
    tp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))
    tn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))
    fp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
    fn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))

    metrics = {
        'precision': precision,
        'recall': recall,
        'f_beta': f_beta,
        'auc': auc,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        results = {
            'args': vars(args),
            'threshold': threshold,
            'test_metrics': metrics,
        }
        json.dump(results, f, indent=4)

    # Log metrics
    logger.info("Test set metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F{beta}: {f_beta:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

    if save_detailed_results:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create a list of dictionaries for DataFrame
        results_data = []

        for sample in all_samples:
            # Determine the category (TP, FP, FN, TN)
            if sample['true_label'] == 1 and sample['prediction'] == 1:
                category = 'TP'
            elif sample['true_label'] == 1 and sample['prediction'] == 0:
                category = 'FN'
            elif sample['true_label'] == 0 and sample['prediction'] == 1:
                category = 'FP'
            else:  # true_label == 0 and prediction == 0
                category = 'TN'

            # Add category to the sample
            sample_with_category = {
                'text': sample['text'],
                'probability': sample['probability'],
                'prediction': sample['prediction'],
                'true_label': sample['true_label'],
                'category': category
            }

            results_data.append(sample_with_category)

        # Create DataFrame
        results_df = pd.DataFrame(results_data)

        # Save to CSV
        csv_path = os.path.join(output_dir, 'detailed_results.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8')

        logger.info(f"Detailed results saved to CSV: {csv_path}")

    if visualize_cm:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate confusion matrix plot
        plot_path = os.path.join(output_dir, 'confusion_matrix.png')

        logger.info(f"Generating confusion matrix visualization at {plot_path}")

        # Plot and save standard confusion matrix
        cm, _ = plot_confusion_matrix(
            all_labels,
            all_preds,
            classes=['Negative', 'Positive'],
            normalize=False,
            title=f'Confusion Matrix (threshold={threshold:.2f})',
            save_path=plot_path,
            show=False  # Don't show during training
        )

        logger.info(f"Confusion matrix visualization saved to {output_dir}")

# Return predictions and metrics
    result = {
        'metrics': metrics,
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'true_labels': np.array(all_labels)
    }

    return result