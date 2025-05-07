#!/usr/bin/env python
"""
Simplified model loading utilities for classification models.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from src.models.classifier import MultiClassEmbeddingClassifier


def load_model(model_path, logger=None):
    """
    Simple and robust model loader that finds and uses experiment_args.json.

    Args:
        model_path: Path to the model file (.pt)
        logger: Optional logger for messages

    Returns:
        model: Loaded model
        class_mapping: Dictionary mapping class indices to class names
    """
    log_fn = logger.info if logger else print

    # Verify model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Get the model directory
    model_dir = os.path.dirname(model_path)
    log_fn(f"Looking for configuration files in: {model_dir}")

    # Look for experiment_args.json in the model directory
    experiment_args_path = os.path.join(model_dir, "experiment_args.json")
    config_path = os.path.join(model_dir, "document_type_classifier_config.json")

    # Load experiment args if available
    model_args = {}
    if os.path.exists(experiment_args_path):
        log_fn(f"Found experiment_args.json: {experiment_args_path}")
        with open(experiment_args_path, 'r') as f:
            experiment_args = json.load(f)

        # Extract relevant model parameters
        model_args = {
            'input_dim': 1536,  # Default for OpenAI embeddings
            'hidden_dim': experiment_args.get('hidden_dim', 1024),
            'dropout_rate': experiment_args.get('dropout', 0.5),
            'use_second_layer': experiment_args.get('use_second_layer', True)
        }
        log_fn(f"Using architecture from experiment_args.json: {model_args}")
    else:
        log_fn("No experiment_args.json found, will try to infer parameters from model")

    # Load class mapping from config file
    class_mapping = {}
    if os.path.exists(config_path):
        log_fn(f"Found config file: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
            class_mapping = config.get('class_mapping', {})
            model_args['num_classes'] = config.get('num_classes', 3)
        log_fn(f"Using class mapping from config: {class_mapping}")
    else:
        log_fn("No config file found, using default class mapping")
        model_args['num_classes'] = 3
        class_mapping = {"0": "call", "1": "visit", "2": "other"}

    # Create model with appropriate architecture
    log_fn(f"Creating model with parameters: {model_args}")
    model = MultiClassEmbeddingClassifier(**model_args)

    # Load the weights
    log_fn(f"Loading weights from: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict)
    log_fn("Model loaded successfully")

    return model, class_mapping


def predict_with_model(model, embeddings, batch_size=32, device='cpu', logger=None):
    """
    Run prediction using the model's built-in predict method.

    Args:
        model: Loaded model with predict method
        embeddings: Numpy array of embeddings
        batch_size: Batch size for prediction
        device: Compute device
        logger: Optional logger

    Returns:
        class_indices: Predicted class indices
        probabilities: Predicted probabilities for each class
    """
    log_fn = logger.info if logger else print

    # Move model to device
    model = model.to(device)
    model.eval()

    # Convert embeddings to tensor if needed
    if not isinstance(embeddings, torch.Tensor):
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    else:
        embeddings_tensor = embeddings

    # Process in batches
    num_samples = embeddings_tensor.shape[0]
    log_fn(f"Running prediction on {num_samples} samples with batch size {batch_size}")

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch = embeddings_tensor[i:batch_end].to(device)

            # Use the model's built-in predict method
            probs = model.predict(batch)
            preds = model.predict_classes(batch)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    # Concatenate results
    probabilities = np.vstack(all_probs) if all_probs else np.array([])
    class_indices = np.concatenate(all_preds) if all_preds else np.array([])

    return class_indices, probabilities


def format_results(class_indices, probabilities, texts=None, true_labels=None, class_mapping=None):
    """
    Format prediction results into a DataFrame, optionally including true labels.

    Args:
        class_indices: Predicted class indices
        probabilities: Predicted probabilities for each class
        texts: Optional text samples corresponding to embeddings
        true_labels: Optional array of true label indices
        class_mapping: Dictionary mapping class indices to names

    Returns:
        DataFrame with formatted results
    """
    # Create default class mapping if none provided
    if class_mapping is None:
        num_classes = probabilities.shape[1]
        class_mapping = {str(i): f"class_{i}" for i in range(num_classes)}

    # Create results DataFrame
    results = pd.DataFrame()

    # Add texts if available
    if texts is not None:
        results['text'] = texts

    # Add predicted class and name
    results['predicted_class_idx'] = class_indices
    results['predicted_class'] = [
        class_mapping.get(str(int(idx)), str(idx))
        for idx in class_indices
    ]

    # Add true labels if available
    if true_labels is not None:
        results['true_label_idx'] = true_labels
        results['true_label'] = [
            class_mapping.get(str(int(idx)), str(idx))
            for idx in true_labels
        ]

        # Add a column to indicate if the prediction is correct
        results['is_correct'] = results['predicted_class_idx'] == results['true_label_idx']

    # Add probabilities for each class
    for i in range(probabilities.shape[1]):
        class_key = str(i)
        class_name = class_mapping.get(class_key, f"class_{i}")
        results[f"prob_{class_name}"] = probabilities[:, i]

    # Add confidence (maximum probability)
    results['confidence'] = np.max(probabilities, axis=1)

    return results


def save_results_with_analysis(results, output_path, class_mapping, threshold=0.0, logger=None):
    """
    Filter results by confidence threshold, save to CSV, and create error analysis files.

    Args:
        results: DataFrame with prediction results
        output_path: Path to save the CSV file
        class_mapping: Dictionary mapping class indices to class names
        threshold: Minimum confidence threshold (0-1)
        logger: Optional logger

    Returns:
        dict: Dictionary containing all generated DataFrame paths
    """
    log_fn = logger.info if logger else print

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # Filter by threshold if needed
    if threshold:
        filtered_results = results[results['confidence'] >= threshold]
        log_fn(f"Filtered from {len(results)} to {len(filtered_results)} samples with threshold {threshold}")
    else:
        filtered_results = results

    # Save main results
    filtered_results.to_csv(output_path, index=False)
    log_fn(f"Saved predictions to {output_path}")

    # If we have true labels, create error analysis files
    error_analysis_files = {}

    if 'true_label_idx' in results.columns:
        base_output_path = os.path.splitext(output_path)[0]

        # 1. Save correct predictions
        correct_predictions = filtered_results[filtered_results['is_correct']]
        correct_path = f"{base_output_path}_correct.csv"
        correct_predictions.to_csv(correct_path, index=False)
        log_fn(f"Saved {len(correct_predictions)} correct predictions to {correct_path}")
        error_analysis_files['correct'] = correct_path

        # 2. Save incorrect predictions
        incorrect_predictions = filtered_results[~filtered_results['is_correct']]
        incorrect_path = f"{base_output_path}_incorrect.csv"
        incorrect_predictions.to_csv(incorrect_path, index=False)
        log_fn(f"Saved {len(incorrect_predictions)} incorrect predictions to {incorrect_path}")
        error_analysis_files['incorrect'] = incorrect_path

        # 3. Create separate files for each class that was predicted incorrectly
        for true_idx, true_name in class_mapping.items():
            # Get samples where true label was this class but prediction was wrong
            class_errors = incorrect_predictions[incorrect_predictions['true_label_idx'] == int(true_idx)]

            if len(class_errors) > 0:
                class_error_path = f"{base_output_path}_errors_{true_name}.csv"
                class_errors.to_csv(class_error_path, index=False)
                log_fn(f"Saved {len(class_errors)} misclassified {true_name} samples to {class_error_path}")
                error_analysis_files[f'errors_{true_name}'] = class_error_path

        # 4. Create a confusion matrix as CSV
        confusion_matrix = pd.crosstab(
            filtered_results['true_label'],
            filtered_results['predicted_class'],
            rownames=['True'],
            colnames=['Predicted']
        )
        confusion_path = f"{base_output_path}_confusion.csv"
        confusion_matrix.to_csv(confusion_path)
        log_fn(f"Saved confusion matrix to {confusion_path}")
        error_analysis_files['confusion_matrix'] = confusion_path

        # 5. Create a detailed error analysis file
        if 'text' in filtered_results.columns:
            # For each misclassified example, add details about why it might have been misclassified
            detailed_errors = incorrect_predictions.copy()

            # Add columns with probability difference and margin
            for idx, row in detailed_errors.iterrows():
                true_idx = row['true_label_idx']
                pred_idx = row['predicted_class_idx']
                true_prob = row[f"prob_{class_mapping.get(str(int(true_idx)), 'unknown')}"]
                pred_prob = row[f"prob_{class_mapping.get(str(int(pred_idx)), 'unknown')}"]
                detailed_errors.loc[idx, 'true_class_prob'] = true_prob
                detailed_errors.loc[idx, 'prob_difference'] = pred_prob - true_prob
                detailed_errors.loc[idx, 'prob_margin'] = pred_prob / true_prob if true_prob > 0 else float('inf')

            # Sort by probability margin (most confident errors first)
            detailed_errors = detailed_errors.sort_values('prob_margin', ascending=False)

            detailed_path = f"{base_output_path}_detailed_errors.csv"
            detailed_errors.to_csv(detailed_path, index=False)
            log_fn(f"Saved detailed error analysis to {detailed_path}")
            error_analysis_files['detailed_errors'] = detailed_path

    return {
        'main_results': output_path,
        **error_analysis_files
    }


def evaluate_with_model(model, embeddings, labels, texts=None, batch_size=32,
                        device='cpu', class_mapping=None, logger=None):
    """
    Evaluate the model using its built-in evaluate method.

    Args:
        model: The loaded classifier model
        embeddings: Embeddings to evaluate
        labels: True labels for the embeddings
        texts: Optional text samples
        batch_size: Batch size for processing
        device: Compute device
        class_mapping: Mapping of class indices to names
        logger: Optional logger

    Returns:
        Evaluation results dictionary
    """
    log_fn = logger.info if logger else print

    # Move model to device
    model = model.to(device)

    # Create a simple data loader from numpy arrays
    from torch.utils.data import TensorDataset, DataLoader

    # Convert to tensors if needed
    if not isinstance(embeddings, torch.Tensor):
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    else:
        embeddings_tensor = embeddings

    if not isinstance(labels, torch.Tensor):
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    else:
        labels_tensor = labels

    # Create dataset and loader
    dataset = TensorDataset(embeddings_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare data loader in the format expected by the evaluate method
    class CustomLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for emb, lbl in self.loader:
                yield {'embedding': emb, 'label': lbl}

    # Get class names list from mapping
    if class_mapping:
        num_classes = max([int(k) for k in class_mapping.keys()]) + 1
        class_names = [class_mapping.get(str(i), f"class_{i}") for i in range(num_classes)]
    else:
        class_names = None

    log_fn(f"Evaluating model on {len(embeddings)} samples")

    # Use the model's built-in evaluate method
    results = model.evaluate(CustomLoader(loader), device=device, class_names=class_names)

    return results


def find_labels_file(embeddings_path, logger=None):
    """
    Find the corresponding labels file for embeddings.

    Args:
        embeddings_path: Path to the embeddings file
        logger: Optional logger

    Returns:
        Path to labels file if found, None otherwise
    """
    log_fn = logger.info if logger else print

    # Try different possible label file paths
    possible_paths = [
        # Standard test labels path
        os.path.join(os.path.dirname(embeddings_path), "test_labels.npy"),

        # Replace 'embeddings' with 'labels' in path
        embeddings_path.replace("embeddings", "labels"),

        # Replace '_emb' with '_labels' in filename
        embeddings_path.replace("_emb", "_labels"),

        # Same directory with '_labels' added before extension
        os.path.splitext(embeddings_path)[0] + "_labels.npy"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            log_fn(f"Found labels file at {path}")
            return path

    log_fn("No labels file found")
    return None