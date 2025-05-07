#!/usr/bin/env python
"""
Simplified training script for call,visit,unknown actions classifier.
"""
import json
import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.data import load_data, create_data_loaders
from src.models.classifier import (
    InsightClassifier,
    ModelWithSigmoid,
    get_pos_weight,
    MultiClassEmbeddingClassifier,
    )
from src.training.trainer import train_model, find_optimal_threshold, evaluate_model
from src.utils.plots import plot_class_distribution, plot_confusion_matrix, plot_training_history
from src.utils.set_up import parse_input_args, setup_logging, setup_environment, create_experiment_output_dir, export_test_dataset
from src.utils.save_models import save_model_artifacts


def main():
    logger = setup_logging()
    args = parse_input_args()

    env = setup_environment(args, logger)
    device = env["device"]
    output_dir = env["output_dir"]

    # Create directories if needed
    output_dir = create_experiment_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info("Loading datasets...")
    embeddings_paths = {
        'call': args.call_embs,
        'visit': args.visit_embs,
        'other': args.other_embs
    }
    texts_paths = {
        'call': args.call_texts,
        'visit': args.visit_texts,
        'other': args.other_texts
    }

    train_dataset, val_dataset, test_dataset = load_data(
        embeddings_paths=embeddings_paths,
        texts_paths=texts_paths,
        normalize=args.normalize,
        standardize=args.standardize,
        clean_other=args.clean_other,
        other_category=args.other_category,
        logger=logger  # Pass the logger from main
    )

    # Export test dataset for later use with prediction script
    if hasattr(args, 'export_test_data') and args.export_test_data:
        logger.info("Exporting test dataset for later use...")
        # Use the project root as the base directory for export
        export_base_dir = os.path.dirname(os.path.dirname(args.call_embs))
        export_paths = export_test_dataset(
            test_dataset=test_dataset,
            output_base_dir=export_base_dir,
            logger=logger
        )
        logger.info(f"Test dataset exported to: {export_paths['embeddings_path']}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        use_weighted_sampler=True
    )

    # Create model
    logger.info("Creating model...")
    model = MultiClassEmbeddingClassifier(
        input_dim=train_dataset.embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=len(embeddings_paths),
        dropout_rate=args.dropout,
        use_second_layer=args.use_second_layer,
    )

    # Train model using the fit method
    logger.info("Starting training...")
    results = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience if hasattr(args, 'patience') else 10,
        device=device,
        logger=logger
    )

    # Extract best model state
    best_model_state = results['best_model_state']

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch {results['best_epoch']}")

    # Class names for visualization
    class_names = list(embeddings_paths.keys())

    # Generate visualizations
    if args.visualize:
        # Get training history in the proper format
        val_metrics = model.get_training_history_for_plotting()

        # Plot training history
        history_plot_path = os.path.join(output_dir, 'training_history.png')
        plot_training_history(
            val_metrics,
            figsize=(12, 6),
            save_path=history_plot_path,
            show=args.show_plots
        )
        logger.info(f"Training history plot saved to {history_plot_path}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = model.evaluate(
        data_loader=test_loader,
        device=device,
        class_names=class_names
    )

    # Log results
    logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Macro F1 score: {test_results['metrics']['macro_f1']:.4f}")
    for cls, f1 in test_results['metrics']['class_f1'].items():
        logger.info(f"  {cls} F1 score: {f1:.4f}")

    # Generate confusion matrix visualization
    if args.visualize:
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            y_true=test_results['targets'],
            y_pred=test_results['predictions'],
            classes=class_names,
            normalize=True,
            figsize=(12, 6),
            save_path=cm_path,
            show=args.show_plots
        )
        logger.info(f"Confusion matrix saved to {cm_path}")

        # Plot class distribution of test set
        dist_path = os.path.join(output_dir, 'class_distribution.png')
        plot_class_distribution(
            test_dataset.labels,
            class_names=class_names,
            title="Test Set Class Distribution",
            save_path=dist_path,
            show=args.show_plots
        )
        logger.info(f"Class distribution plot saved to {dist_path}")

    # Save all model artifacts
    artifact_paths = save_model_artifacts(
        model=model,
        output_dir=output_dir,
        embeddings_paths=embeddings_paths,
        test_results=test_results,
        device=device,
        logger=logger
    )

    logger.info("Training complete!")
    return test_results['metrics']['accuracy']


if __name__ == "__main__":
    main()