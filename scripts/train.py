#!/usr/bin/env python
"""
Simplified training script for insight classifier.
"""
import json
import os
import argparse
import torch
import torch.optim as optim
import logging
from datetime import datetime

# Add project root to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_data, create_data_loaders
from models.classifier import InsightClassifier, ModelWithSigmoid, get_pos_weight
from training.trainer import train_model, find_optimal_threshold, evaluate_model
from utils.plots import plot_training_history

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train insight classifier")

    # Data paths
    parser.add_argument("--pos-emb", type=str, default="data/embeddings/positives.npy",
                        help="Path to positive embeddings")
    parser.add_argument("--neg-emb", type=str, default="data/embeddings/negatives.npy",
                        help="Path to negative embeddings")
    parser.add_argument("--pos-texts", type=str, default=None,
                        help="Path to positive texts")
    parser.add_argument("--neg-texts", type=str, default=None,
                        help="Path to negative texts")

    # Preprocessing options
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="Apply L2 normalization to embeddings")
    parser.add_argument("--standardize", action="store_true", default=False,
                        help="Standardize embeddings (zero mean, unit variance)")

    # Training parameters
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--pos-weight-mult", type=float, default=2.0,
                        help="Multiplier for positive class weight")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="Beta for F-beta score (higher = more recall focus)")

    # Output parameters
    # parser.add_argument("--model-path", type=str, default="models/classifier.pt",
    #                     help="Path to save model")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save model, results and visualizations")

    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations (confusion matrix, training curves)")
    parser.add_argument("--show-plots", action="store_true",
                        help="Show plots during training (may require display)")

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")

    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create directories if needed
    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_data(
        positives_path=args.pos_emb,
        negatives_path=args.neg_emb,
        positive_texts_path=args.pos_texts,
        negative_texts_path=args.neg_texts,
        test_size=0.2,
        val_size=0.1,
        normalize=args.normalize,
        standardize=args.standardize,
    )

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
    model = InsightClassifier(
        input_dim=train_dataset.embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout
    )

    # Calculate class weights for loss function
    train_pos = torch.sum(train_dataset.labels).item()
    train_neg = len(train_dataset) - train_pos

    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"  Positive samples: {train_pos} ({train_pos/len(train_dataset)*100:.2f}%)")
    logger.info(f"  Negative samples: {train_neg} ({train_neg/len(train_dataset)*100:.2f}%)")

    pos_weight = get_pos_weight(train_pos, train_neg, args.pos_weight_mult)
    logger.info(f"Using positive class weight: {pos_weight.item():.2f}")

    # Define loss function with class weighting
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Train model
    logger.info("Starting training...")
    model, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=10,
        threshold=0.5,
        beta=args.beta
    )

    if args.visualize and val_metrics:
        # Extract training losses from val_metrics
        train_losses = [m.get('train_loss', 0) for m in val_metrics]

        # Plot training history
        history_plot_path = os.path.join(output_dir, 'training_history.png')
        plot_training_history(
            train_losses,
            val_metrics,
            save_path=history_plot_path,
            show=args.show_plots
        )
        logger.info(f"Training history plot saved to {history_plot_path}")

    # Find optimal threshold
    logger.info("Finding optimal threshold...")
    optimal = find_optimal_threshold(
        model=model,
        val_loader=val_loader,
        device=device,
        beta=args.beta
    )

    # Evaluate on test set using optimal threshold
    logger.info("Evaluating on test set...")
    test_results = evaluate_model(
        args=args,
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=optimal['threshold'],
        beta=args.beta,
        visualize_cm=args.visualize,
        output_dir=output_dir,
    )

    # Save model
    torch_model_path = os.path.join(output_dir, 'pulse_urgency_model.pt')
    logger.info(f"Saving model to {torch_model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': optimal['threshold'],
        'config': {
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim
        },
        'test_metrics': test_results['metrics']
    }, torch_model_path)

    # Save as ONNX
    onnx_model_path = os.path.join(output_dir, 'pulse_urgency_model.onnx')
    logger.info(f"Exporting model to ONNX format: {onnx_model_path}")

    wrapped_model = ModelWithSigmoid(model)
    dummy_input = torch.randn(1, model.input_dim, device=device)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )

    # Save model configs
    model_configs = {
        'threshold': optimal['threshold'],
    }
    with open(os.path.join(output_dir, 'pulse_urgency_model_config.json'), 'w') as f:
        json.dump(model_configs, f)

    logger.info("Training complete!")

if __name__ == "__main__":
    main()