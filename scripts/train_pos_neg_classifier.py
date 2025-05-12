#!/usr/bin/env python
"""
Simplified training script for insight classifier.
"""
import json
import os
import torch
import torch.optim as optim
import logging
from datetime import datetime

# Add project root to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data, create_data_loaders
from src.models.classifier import InsightClassifier, ModelWithSigmoid, get_pos_weight
from src.training.trainer import train_model, find_optimal_threshold, evaluate_model
from src.utils.plots import plot_training_history
from src.utils.set_up import parse_input_args

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    args = parse_input_args()

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

    # Create dictionaries for embeddings and texts paths
    embeddings_paths = {
        'pos': args.pos_emb,
        'neg': args.neg_emb,
    }

    texts_paths = {
        'pos': args.call_texts,
        'neg': args.visit_texts,
    }

    # Call the load_data function with the dictionaries
    train_dataset, val_dataset, test_dataset = load_data(
        embeddings_paths=embeddings_paths,
        texts_paths=texts_paths,
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