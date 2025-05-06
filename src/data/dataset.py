import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from data.preprocessing import normalize_embeddings, standardize_embeddings


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


def load_data(positives_path='data/embeddings/positives.npy',
              negatives_path='data/embeddings/negatives.npy',
              positive_texts_path=None,
              negative_texts_path=None,
              test_size=0.2,
              val_size=0.1,
              normalize=False,
              standardize=False,
              random_state=42):
    """
    Load and split the embeddings data into train, validation, and test sets.

    Args:
        positives_path: Path to the positive embeddings
        negatives_path: Path to the negative embeddings
        positive_texts_path: Optional path to positive text data
        negative_texts_path: Optional path to negative text data
        test_size: Proportion of data to use for the test set
        val_size: Proportion of training data to use for validation
        normalize: Whether to apply L2 normalization to embeddings
        standardize: Whether to standardize embeddings using training data stats
        random_state: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset, test_dataset: Three InsightDataset objects
    """
    # Load embeddings
    positive_embeddings = np.load(positives_path)
    negative_embeddings = np.load(negatives_path)

    # Apply L2 normalization if requested
    if normalize:
        positive_embeddings = normalize_embeddings(positive_embeddings)
        negative_embeddings = normalize_embeddings(negative_embeddings)

    # Create labels
    positive_labels = np.ones(len(positive_embeddings))
    negative_labels = np.zeros(len(negative_embeddings))

    # Combine embeddings data
    all_embeddings = np.vstack((positive_embeddings, negative_embeddings))
    all_labels = np.concatenate((positive_labels, negative_labels))

    # Load text data if provided
    all_texts = None
    if positive_texts_path is not None and negative_texts_path is not None:
        positive_texts = np.load(positive_texts_path, allow_pickle=True)
        negative_texts = np.load(negative_texts_path, allow_pickle=True)

        # Verify lengths match
        assert len(positive_texts) == len(positive_embeddings), "Positive texts and embeddings must have same length"
        assert len(negative_texts) == len(negative_embeddings), "Negative texts and embeddings must have same length"

        # Combine text data
        all_texts = np.concatenate((positive_texts, negative_texts))

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

    # Create datasets with text data
    train_dataset = InsightDataset(X_train, y_train, texts=train_texts)
    val_dataset = InsightDataset(X_val, y_val, texts=val_texts)
    test_dataset = InsightDataset(X_test, y_test, texts=test_texts)

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
