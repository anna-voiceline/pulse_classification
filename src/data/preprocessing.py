import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit length.

    Args:
        embeddings: Numpy array of embeddings with shape (n_samples, embedding_dim)

    Returns:
        Normalized embeddings
    """
    # L2 normalize each embedding vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def standardize_embeddings(train_embeddings, val_embeddings=None, test_embeddings=None):
    """
    Standardize embeddings using mean and standard deviation from training data.

    Args:
        train_embeddings: Training embeddings
        val_embeddings: Optional validation embeddings
        test_embeddings: Optional test embeddings

    Returns:
        Standardized embeddings (train, val, test) with the same scaler applied
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)

    results = [train_scaled]

    if val_embeddings is not None:
        val_scaled = scaler.transform(val_embeddings)
        results.append(val_scaled)

    if test_embeddings is not None:
        test_scaled = scaler.transform(test_embeddings)
        results.append(test_scaled)

    return results if len(results) > 1 else results[0]

