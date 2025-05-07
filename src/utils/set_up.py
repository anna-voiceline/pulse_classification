import torch
import logging
import os
from datetime import datetime
import argparse
import json
import numpy as np

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_input_args():
    parser = argparse.ArgumentParser(description="Train insight classifier")

    # Data pos-neg paths
    parser.add_argument("--pos-emb", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/embeddings/positives.npy"),
                        help="Path to positive embeddings")
    parser.add_argument("--neg-emb", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/embeddings/negatives.npy"),
                        help="Path to negative embeddings")
    parser.add_argument("--pos-texts", type=str, default=None,
                        help="Path to positive texts")
    parser.add_argument("--neg-texts", type=str, default=None,
                        help="Path to negative texts")

    # Data call-visit-unknown paths
    parser.add_argument("--call-embs", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/embeddings/callLog_emb.npy"),
                        help="Path to call embeddings")
    parser.add_argument("--visit-embs", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/embeddings/visit_emb.npy"),
                        help="Path to visit embeddings")
    parser.add_argument("--other-embs", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/embeddings/note_emb.npy"),
                        help="Path to other embeddings")

    parser.add_argument("--call-texts", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/texts/callLog_text.npy"),
                        help="Path to call texts")
    parser.add_argument("--visit-texts", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/texts/visit_text.npy"),
                        help="Path to visit texts")
    parser.add_argument("--other-texts", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/texts/note_text.npy"),
                        help="Path to other texts")

    # Data cleaning options
    parser.add_argument("--clean-other", action="store_true", default=False,
                        help="Clean and filter the 'other' category based on similarity")
    parser.add_argument("--other-category", type=str, default="other",
                        help="Name of the 'other' category in embeddings_paths")

    # Export test dataset
    parser.add_argument("--export-test-data", action="store_true", default=False,
                        help="Export test dataset to NPY files for later use with prediction script")

    # Preprocessing options
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="Apply L2 normalization to embeddings")
    parser.add_argument("--standardize", action="store_true", default=False,
                        help="Standardize embeddings (zero mean, unit variance)")

    # Model architecture parameters
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="First hidden layer dimension")
    parser.add_argument("--use-second-layer", action="store_true", default=False,
                        help="Add a second hidden layer with size hidden_dim/2")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")

    # Training parameters
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
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results"),
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


def parse_output_args():
    parser = argparse.ArgumentParser(description="Predict action types from embeddings")

    # Paths
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to embeddings to classify (.npy file)")
    parser.add_argument("--texts", type=str, default=None,
                        help="Path to texts (optional, .npy file)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to true labels (optional, .npy file)")
    parser.add_argument("--output", type=str, default="action_predictions.csv",
                        help="Path to save predictions")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to model config (if not specified, will look for it next to model)")

    # Parameters
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum threshold for inclusion in results (use model's threshold if not specified)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for prediction")
    parser.add_argument("--save-errors", action="store_true", default=False,
                        help="Save separate files for error analysis")

    # Miscellaneous
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--show-top", type=int, default=5,
                        help="Number of top predictions to show for each class")

    return parser.parse_args()


def setup_logging(log_level=logging.INFO, name=None):
    """
    Sets up logging with consistent formatting.

    Args:
        log_level: The logging level (default: logging.INFO)
        name: The logger name (default: __name__ of the calling module)

    Returns:
        logger: Configured logger instance
    """
    # Configure basic logging format
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get logger for the specified name or the calling module
    logger_name = name if name is not None else __name__
    logger = logging.getLogger(logger_name)

    return logger

# Setup environment
def setup_environment(args, logger):
    """
    Sets up the training environment with seed, directories, and device configuration.

    Args:
        logger: Logger instance
        args: The parsed command line arguments

    Returns:
        dict: A dictionary containing the device and output directory information
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create timestamped output directory
    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Using device: {device}")

    return {
        "device": device,
        "output_dir": output_dir
    }


def create_experiment_output_dir(args):
    """
    Create a descriptive experiment output directory based on parameters.

    Args:
        args: Parsed command line arguments

    Returns:
        Path to the created output directory
    """
    # Create a descriptive folder name based on key parameters
    folder_name = (
        f"lr_{args.lr}_"
        f"bs_{args.batch_size}_"
        f"hd_{args.hidden_dim}_"
        f"dr_{args.dropout}_"
        f"wd_{args.weight_decay}"
    )

    # Add second layer info if used
    if hasattr(args, 'use_second_layer') and args.use_second_layer:
        second_layer_size = args.hidden_dim // 2
        folder_name += f"_2layer_{second_layer_size}"

    # Add preprocessing flags
    if args.normalize:
        folder_name += "_norm"
    if args.standardize:
        folder_name += "_std"

    # Add cleaning info if enabled
    if hasattr(args, 'clean_other') and args.clean_other:
        folder_name += "_cleaned"

    # Add timestamp for uniqueness while maintaining readability
    timestamp = datetime.now().strftime("%m%d_%H%M")
    folder_name += f"_{timestamp}"

    # Create the full path
    output_dir = os.path.join(args.output_dir, folder_name)

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the command line arguments to a file for reference
    args_path = os.path.join(output_dir, 'experiment_args.json')
    with open(args_path, 'w') as f:
        # Convert args to dictionary and save as JSON
        args_dict = vars(args)
        # Handle non-serializable objects
        for key, value in args_dict.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                args_dict[key] = str(value)
        json.dump(args_dict, f, indent=2)

    return output_dir


def export_test_dataset(test_dataset, output_base_dir=None, logger=None):
    """
    Export the test dataset to separate NPY files for embeddings and texts.

    Args:
        test_dataset: The test dataset (InsightDataset instance)
        output_base_dir: Base directory for output (default: use default paths)
        logger: Optional logger for logging messages

    Returns:
        dict: Paths to the saved embeddings and texts files
    """
    # Define log function
    log_fn = logger.info if logger else print

    # Default paths if not specified
    if output_base_dir is None:
        output_base_dir = os.path.join(os.getcwd(), "data")

    # Create embeddings and texts directories
    embeddings_dir = os.path.join(output_base_dir, "embeddings")
    texts_dir = os.path.join(output_base_dir, "texts")

    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)

    # Define output file paths
    embeddings_path = os.path.join(embeddings_dir, "test_embeddings.npy")
    texts_path = os.path.join(texts_dir, "test_texts.npy")
    labels_path = os.path.join(embeddings_dir, "test_labels.npy")

    # Save embeddings
    log_fn(f"Saving test embeddings ({len(test_dataset.embeddings)} samples) to {embeddings_path}")
    np.save(embeddings_path, test_dataset.embeddings)

    # Save labels
    log_fn(f"Saving test labels to {labels_path}")
    np.save(labels_path, test_dataset.labels)

    # Save texts if available
    if hasattr(test_dataset, 'texts') and test_dataset.texts is not None:
        log_fn(f"Saving test texts to {texts_path}")
        np.save(texts_path, test_dataset.texts)
    else:
        log_fn("No texts available in test dataset, skipping text export")
        texts_path = None

    # Also save a small sample as text files for easy inspection
    sample_size = min(10, len(test_dataset))
    embeddings_sample_path = os.path.join(embeddings_dir, "test_embeddings_sample.txt")

    with open(embeddings_sample_path, 'w') as f:
        f.write(f"Sample of {sample_size} test embeddings (shape: {test_dataset.embeddings.shape}):\n\n")
        for i in range(sample_size):
            embedding = test_dataset.embeddings[i]
            # Print first 10 dimensions and last 10 dimensions
            first_dims = ', '.join([f"{x:.4f}" for x in embedding[:10]])
            last_dims = ', '.join([f"{x:.4f}" for x in embedding[-10:]])
            f.write(f"Sample {i} (label: {test_dataset.labels[i]}):\n")
            f.write(f"  First 10 dims: [{first_dims}]\n")
            f.write(f"  Last 10 dims: [{last_dims}]\n")
            if hasattr(test_dataset, 'texts') and test_dataset.texts is not None:
                f.write(f"  Text: {test_dataset.texts[i][:200]}...\n")
            f.write("\n")

    log_fn(f"Saved sample inspection to {embeddings_sample_path}")

    # Create a small dataframe sample and save as CSV for easy viewing
    try:
        import pandas as pd

        df_data = {"label": test_dataset.labels[:sample_size]}

        if hasattr(test_dataset, 'texts') and test_dataset.texts is not None:
            df_data["text"] = [t[:200] + "..." if len(t) > 200 else t
                               for t in test_dataset.texts[:sample_size]]

        # Add first 5 embedding dimensions
        for i in range(5):
            df_data[f"emb_dim_{i}"] = [e[i] for e in test_dataset.embeddings[:sample_size]]

        df = pd.DataFrame(df_data)
        csv_path = os.path.join(output_base_dir, "test_dataset_sample.csv")
        df.to_csv(csv_path, index=False)
        log_fn(f"Saved sample CSV to {csv_path}")
    except ImportError:
        log_fn("Pandas not available, skipping CSV export")

    return {
        "embeddings_path": embeddings_path,
        "texts_path": texts_path,
        "labels_path": labels_path,
        "sample_path": embeddings_sample_path
    }