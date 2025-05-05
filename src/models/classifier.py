import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InsightClassifier(nn.Module):
    """
    Simplified neural network classifier for important insights based on text embeddings.
    """

    def __init__(self, input_dim=1536, hidden_dim=512, dropout_rate=0.3):
        """
        Initialize the classifier model.

        Args:
            input_dim: Dimension of input embeddings (default: 1536 for Azure OpenAI embeddings)
            hidden_dim: Size of the hidden layer
            dropout_rate: Dropout probability for regularization
        """
        super(InsightClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Simple two-layer network with dropout
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Raw logits (not passed through sigmoid)
        """
        return self.model(x).squeeze(-1)

    def predict_proba(self, x):
        """
        Get probability predictions.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Probability scores between 0 and 1
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return probs

    def predict(self, x, threshold=0.5):
        """
        Get binary predictions.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            threshold: Decision threshold (lower values increase recall)

        Returns:
            Binary predictions (0 or 1)
        """
        return (self.predict_proba(x) >= threshold).float()


class ModelWithSigmoid(torch.nn.Module):
    """
    Wraps a model with a sigmoid activation function.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return torch.sigmoid(self.base_model(x))


def get_pos_weight(num_positive, num_negative, weight_multiplier=2.0):
    """
    Calculate positive class weight for BCEWithLogitsLoss.

    Args:
        num_positive: Number of positive examples
        num_negative: Number of negative examples
        weight_multiplier: Additional weighting factor to further increase recall

    Returns:
        Tensor containing the positive class weight
    """
    pos_weight = (num_negative / num_positive) * weight_multiplier
    return torch.tensor([pos_weight])


