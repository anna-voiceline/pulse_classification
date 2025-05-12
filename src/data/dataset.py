import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from src.data.preprocessing import normalize_embeddings, standardize_embeddings
from src.data.data_prep import classify_and_save

class InsightDataset(Dataset):
    """
    Dataset for loading and processing insight embeddings.

    Attributes:
        embeddings: Tensor of insight embeddings
        labels: Tensor of binary labels (1 for important, 0 for not important)
        texts: Optional array of original text strings
    """

    def __init__(self, embeddings, labels, texts=None):
        """
        Initialize the dataset with embeddings and labels.

        Args:
            embeddings: Numpy array of embeddings with shape (n_samples, embedding_dim)
            labels: Numpy array of binary labels with shape (n_samples,)
            texts: Optional numpy array or list of original text strings
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.texts = texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'embedding': self.embeddings[idx],
            'label': self.labels[idx]
        }

        if self.texts is not None:
            item['text'] = self.texts[idx]

        return item


def load_data(embeddings_paths: dict[str, str],
              texts_paths: dict[str, str],
              test_size=0.2,
              val_size=0.1,
              normalize=False,
              standardize=False,
              random_state=42,
              label_mapping=None,
              clean_other=False,
              other_category='other',
              logger=None):
    """
    Load and split embeddings data into train, validation, and test sets with optional cleaning.
    Supports multi-class classification with any number of classes.

    Args:
        embeddings_paths: Dict mapping class labels to embedding file paths
        texts_paths: Dict mapping class labels to text file paths
        test_size: Proportion of data to use for the test set
        val_size: Proportion of training data to use for validation
        normalize: Whether to apply L2 normalization to embeddings
        standardize: Whether to standardize embeddings using training data stats
        random_state: Random seed for reproducibility
        label_mapping: Dict mapping original class labels to numeric indices
        clean_other: Whether to clean and filter the 'other' category data
        other_category: The key in embeddings_paths that corresponds to the 'other' category
        logger: Logger instance for logging messages (if None, no logging is performed)

    Returns:
        train_dataset, val_dataset, test_dataset: Three InsightDataset objects with numeric labels
    """
    # Initialize lists to store data
    all_embeddings = []
    all_labels = []
    all_texts = [] if texts_paths else None

    # Create a mapping from original class labels to their enumerated index values
    if label_mapping is None:
        # If no mapping provided, create a default one (enumerate the keys)
        label_mapping = {label: idx for idx, label in enumerate(embeddings_paths.keys())}

    # Keep track of the mapping for metadata
    class_to_idx = label_mapping.copy()

    # Process each class
    for class_label, embedding_path in embeddings_paths.items():
        # Clean the 'other' category if requested
        if clean_other and class_label == other_category and class_label in texts_paths:
            logger.info(f"Cleaning '{other_category}' category using data_prep module...")

            # Use the classify_and_save function from data_prep module
            # This will create clean and dirty versions of the data
            temp_output_dir = os.path.join(os.path.dirname(embedding_path), "temp_clean")
            os.makedirs(temp_output_dir, exist_ok=True)

            result = classify_and_save(
                texts_path=texts_paths[class_label],
                embeddings_path=embedding_path,
                output_dir=temp_output_dir,
                save_with_suffix=""
            )

            # Use the clean version for further processing
            if result and result['clean_count'] > 0:
                logger.info(f"Using cleaned version of '{other_category}' with {result['clean_count']} samples")
                # Update paths to use the clean versions
                embedding_path = result['clean_embeddings_path']
                texts_paths[class_label] = result['clean_texts_path']
            else:
                logger.warning(f"Cleaning produced no usable samples, using original '{other_category}' data")

        # Load embeddings for this class
        class_embeddings = np.load(embedding_path)

        # Apply L2 normalization if requested
        if normalize:
            class_embeddings = normalize_embeddings(class_embeddings)

        # Map the class label to its numeric index
        numeric_label = label_mapping.get(class_label, class_label)  # Use mapping or original if not in mapping

        # Create labels for this class
        class_labels = np.full(len(class_embeddings), numeric_label)

        # Append to our master lists
        all_embeddings.append(class_embeddings)
        all_labels.append(class_labels)

        # Handle text data if provided
        if texts_paths and class_label in texts_paths:
            class_texts = np.load(texts_paths[class_label], allow_pickle=True)

            # Verify lengths match
            assert len(class_texts) == len(
                class_embeddings), f"Texts and embeddings for class {class_label} must have same length"

            if all_texts is not None:
                all_texts.append(class_texts)

    # Combine all data
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)
    if all_texts is not None and len(all_texts) > 0:
        all_texts = np.concatenate(all_texts)

    # Create index arrays for tracking data through splits
    indices = np.arange(len(all_labels))

    # Split into train+val and test using indices
    train_val_indices, test_indices, y_train_val, y_test = train_test_split(
        indices, all_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=all_labels  # Preserve class distribution
    )

    # Split train into train and validation using indices
    train_indices, val_indices, y_train, y_val = train_test_split(
        train_val_indices, y_train_val,
        test_size=val_size / (1 - test_size),  # Adjust val_size to be relative to train_val
        random_state=random_state,
        stratify=y_train_val  # Preserve class distribution
    )

    # Use indices to get the corresponding data
    X_train = all_embeddings[train_indices]
    X_val = all_embeddings[val_indices]
    X_test = all_embeddings[test_indices]

    # Standardize embeddings if requested
    if standardize:
        X_train, X_val, X_test = standardize_embeddings(
            X_train, X_val, X_test
        )

    # Get corresponding text data
    train_texts, val_texts, test_texts = None, None, None
    if all_texts is not None:
        train_texts = all_texts[train_indices]
        val_texts = all_texts[val_indices]
        test_texts = all_texts[test_indices]

    # Create datasets with text data and metadata
    train_dataset = InsightDataset(embeddings=X_train, labels=y_train, texts=train_texts)
    val_dataset = InsightDataset(embeddings=X_val, labels=y_val, texts=val_texts)
    test_dataset = InsightDataset(embeddings=X_test, labels=y_test, texts=test_texts)

    # Log class distribution information
    class_counts = {}
    for label in np.unique(all_labels):
        class_name = next((k for k, v in label_mapping.items() if v == label), str(label))
        class_counts[class_name] = (all_labels == label).sum()

    logger.info(f"Class distribution after processing: {class_counts}")

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset,
                        batch_size=32, use_weighted_sampler=True):
    """
    Create DataLoaders for the datasets with optional weighted sampling to handle class imbalance.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for training
        use_weighted_sampler: Whether to use weighted random sampling for training

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Regular loaders for validation and test
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # For training, we may want to use weighted sampling
    if use_weighted_sampler:
        # Calculate class weights (inverse frequency)
        labels = train_dataset.labels.numpy()
        class_counts = np.bincount(labels.astype(int))
        class_weights = 1.0 / class_counts

        # Assign weights to samples
        sample_weights = class_weights[labels.astype(int)]

        # Create a sampler with replacement (important for imbalanced data)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

    return train_loader, val_loader, test_loader
