#!/usr/bin/env python
"""
Prediction script for document type classification with error analysis.
"""

import os
import torch
import numpy as np
from src.utils.set_up import setup_logging, parse_output_args
from src.utils.predict_utils import (
    load_model,
    predict_with_model,
    format_results,
    save_results_with_analysis,
    find_labels_file
)


def main():
    """Run the prediction pipeline with comprehensive error analysis."""
    # Parse arguments
    args = parse_output_args()

    # Setup logging
    logger = setup_logging()
    logger.info(f"Running prediction with model: {args.model_path}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        model, class_mapping = load_model(args.model_path, logger=logger)

        # Load embeddings
        logger.info(f"Loading embeddings from {args.embeddings}")
        embeddings = np.load(args.embeddings)
        logger.info(f"Loaded embeddings with shape {embeddings.shape}")

        # Load texts if provided
        texts = None
        if args.texts:
            logger.info(f"Loading texts from {args.texts}")
            try:
                texts = np.load(args.texts, allow_pickle=True)
                logger.info(f"Loaded {len(texts)} texts")
            except Exception as e:
                logger.warning(f"Failed to load texts: {e}")

        # Load true labels if provided or try to find them automatically
        true_labels = None
        if args.labels:
            # Explicit path provided
            logger.info(f"Loading true labels from {args.labels}")
            try:
                true_labels = np.load(args.labels)
                logger.info(f"Loaded {len(true_labels)} true labels")
            except Exception as e:
                logger.warning(f"Failed to load true labels from specified path: {e}")
        else:
            # Try to find labels automatically
            labels_path = find_labels_file(args.embeddings, logger)
            if labels_path:
                try:
                    true_labels = np.load(labels_path)
                    logger.info(f"Loaded {len(true_labels)} true labels from auto-detected path")
                except Exception as e:
                    logger.warning(f"Failed to load auto-detected true labels: {e}")

        # Verify labels match embeddings if available
        if true_labels is not None and len(true_labels) != len(embeddings):
            logger.warning(
                f"Number of true labels ({len(true_labels)}) doesn't match number of embeddings ({len(embeddings)})")
            if len(true_labels) > len(embeddings):
                logger.warning("Truncating true labels to match embeddings")
                true_labels = true_labels[:len(embeddings)]
            else:
                logger.warning("Can't use true labels due to length mismatch")
                true_labels = None

        # Run prediction using the model's built-in methods
        logger.info("Running prediction using model's built-in methods...")
        class_indices, probabilities = predict_with_model(
            model,
            embeddings,
            batch_size=args.batch_size,
            device=device,
            logger=logger
        )

        # Verify probabilities sum to 1
        logger.info("Verifying probabilities:")
        prob_sums = np.sum(probabilities, axis=1)
        logger.info(f"Min probability sum: {prob_sums.min()}, Max: {prob_sums.max()}")
        if not np.allclose(prob_sums, 1.0, rtol=1e-5):
            logger.warning("WARNING: Some probability rows don't sum to 1.0")

        # Format results including true labels if available
        logger.info("Formatting results...")
        results = format_results(
            class_indices,
            probabilities,
            texts=texts,
            true_labels=true_labels,
            class_mapping=class_mapping
        )

        # Save results with error analysis
        logger.info(f"Saving results to {args.output} with threshold {args.threshold}")
        output_files = save_results_with_analysis(
            results,
            args.output,
            class_mapping,
            threshold=args.threshold,
            logger=logger
        )

        # Print distribution
        class_counts = results['predicted_class'].value_counts()
        logger.info(f"Prediction distribution: {dict(class_counts)}")

        # If we have true labels, print accuracy
        if true_labels is not None:
            accuracy = (class_indices == true_labels).mean()
            logger.info(f"Overall accuracy: {accuracy:.4f}")

            # Print accuracy per class
            for class_idx, class_name in class_mapping.items():
                class_mask = true_labels == int(class_idx)
                if class_mask.sum() > 0:
                    class_acc = (class_indices[class_mask] == true_labels[class_mask]).mean()
                    logger.info(f"Accuracy for class {class_name}: {class_acc:.4f} ({class_mask.sum()} samples)")

        # Print examples if texts are available
        if texts is not None and args.show_top > 0:
            logger.info(f"\nTop {args.show_top} examples per class:")
            for class_name in class_mapping.values():
                examples = results[results['predicted_class'] == class_name].sort_values(
                    'confidence', ascending=False
                ).head(args.show_top)

                if len(examples) > 0:
                    logger.info(f"\n{class_name.upper()} examples:")
                    for i, (_, row) in enumerate(examples.iterrows()):
                        # Include true label in output if available
                        true_label_str = ""
                        if 'true_label' in row:
                            true_label_str = f" (true: {row['true_label']})"

                        logger.info(f"  {i + 1}. '{row['text']}' (conf: {row['confidence']:.4f}){true_label_str}")

            # If we have true labels, also show some misclassified examples
            if true_labels is not None:
                misclassified = results[results['predicted_class_idx'] != results['true_label_idx']]
                if len(misclassified) > 0:
                    logger.info(f"\nTop {min(args.show_top, len(misclassified))} misclassified examples:")
                    # Sort by confidence (most confident errors first)
                    for i, (_, row) in enumerate(
                            misclassified.sort_values('confidence', ascending=False).head(args.show_top).iterrows()):
                        logger.info(
                            f"  {i + 1}. '{row['text']}' (pred: {row['predicted_class']} [{row['confidence']:.4f}], true: {row['true_label']})")

        logger.info("Prediction completed successfully")
        logger.info(f"Output files saved to: {', '.join(output_files.values())}")

        return results

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    main()