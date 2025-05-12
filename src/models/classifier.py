import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
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


class MultiClassEmbeddingClassifier(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, num_classes=3, dropout_rate=0.3, use_second_layer=True):
        """
        Initialize the multi-class classifier model with an optional second hidden layer.

        Args:
            input_dim: Dimension of input embeddings (default: 1536 for Azure OpenAI embeddings)
            hidden_dim: Size of the first hidden layer
            num_classes: Number of output classes (default: 3 for call, visit, other)
            dropout_rate: Dropout probability for regularization
            use_second_layer: Whether to use a second hidden layer
        """
        super(MultiClassEmbeddingClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_second_layer = use_second_layer

        # Calculate second layer size as 1/2 of the first layer
        second_hidden_dim = hidden_dim // 2

        # Build model architecture
        if use_second_layer:
            self.model = nn.Sequential(
                # First layer
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                # Second layer
                nn.Linear(hidden_dim, second_hidden_dim),
                nn.BatchNorm1d(second_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                # Output layer
                nn.Linear(second_hidden_dim, num_classes)
            )
        else:
            # Original single hidden layer architecture
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        return self.model(x)

    def predict(self, x):
        """
        Predict class probabilities.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Probability tensor of shape [batch_size, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def predict_classes(self, x):
        """
        Predict class indices.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Predicted class indices of shape [batch_size]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-5):
        """
        Configure optimizer.

        Args:
            lr: Learning rate
            weight_decay: L2 regularization term

        Returns:
            Configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, batch, optimizer):
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing 'embedding', 'label', and 'text'
            optimizer: The optimizer to use

        Returns:
            Loss value
        """
        # Extract embeddings and labels from batch dictionary
        embeddings = batch['embedding']
        labels = batch['label'].long()  # Convert to long for CrossEntropyLoss

        optimizer.zero_grad()

        # Forward pass
        logits = self(embeddings)
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()

    def validation_step(self, batch):
        """
        Perform a single validation step.

        Args:
            batch: Dictionary containing 'embedding', 'label', and 'text'

        Returns:
            Dictionary with loss, accuracy, and predictions
        """
        # Extract embeddings and labels from batch dictionary
        embeddings = batch['embedding']
        labels = batch['label'].long()  # Convert to long for CrossEntropyLoss

        with torch.no_grad():
            # Forward pass
            logits = self(embeddings)
            loss = F.cross_entropy(logits, labels)

            # Calculate accuracy
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'predictions': preds.cpu().numpy(),
            'targets': labels.cpu().numpy()
        }

    def fit(self, train_loader, val_loader=None, epochs=10, lr=1e-3, weight_decay=1e-5,
            patience=5, device=None, logger=None):
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization term
            patience: Early stopping patience (0 to disable)
            device: Device to train on (will use CUDA if available)
            logger: Optional logger for logging messages

        Returns:
            Dictionary containing:
            - Training history
            - Best model state dictionary
            - Information about best epoch
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Logging function (uses logger if provided, otherwise print)
        log_fn = logger.info if logger else print

        self.to(device)
        optimizer = self.configure_optimizers(lr, weight_decay)

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []

            for batch in train_loader:
                # Move batch to device
                batch_on_device = {
                    'embedding': batch['embedding'].to(device),
                    'label': batch['label'].long().to(device),  # Ensure long type
                    # No need to move text to device
                }

                loss = self.training_step(batch_on_device, optimizer)
                train_losses.append(loss)

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            if val_loader:
                self.eval()
                val_metrics = self.evaluate(val_loader, device)

                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])

                log_fn(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - "
                       f"Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")

                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_model_state = self.state_dict().copy()
                    best_epoch = epoch + 1
                    patience_counter = 0
                    log_fn(f"New best model saved at epoch {epoch + 1}")
                else:
                    patience_counter += 1

                # Early stopping
                if patience > 0 and patience_counter >= patience:
                    log_fn(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                log_fn(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Store history for later use
        self._last_fit_history = history

        # Return history along with best model state and info
        return {
            'history': history,
            'best_model_state': best_model_state,
            'best_epoch': best_epoch
        }

    def evaluate(self, data_loader, device=None, class_names=None):
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader for evaluation data
            device: Device to evaluate on
            class_names: List of class names for metrics reporting

        Returns:
            Dictionary with evaluation metrics
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        self.eval()

        all_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch_on_device = {
                    'embedding': batch['embedding'].to(device),
                    'label': batch['label'].long().to(device)  # Ensure long type
                }

                results = self.validation_step(batch_on_device)
                all_losses.append(results['loss'])
                all_predictions.extend(results['predictions'])
                all_targets.extend(results['targets'])

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        accuracy = np.mean(all_predictions == all_targets)
        avg_loss = np.mean(all_losses)

        # Generate classification report
        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]

        try:
            report = classification_report(
                all_targets,
                all_predictions,
                target_names=class_names,
                output_dict=True
            )
        except Exception as e:
            # Fallback if there are issues with classification_report
            report = {
                'accuracy': accuracy,
                'macro avg': {'f1-score': 0},
                'weighted avg': {'f1-score': 0}
            }
            for cls in class_names:
                report[cls] = {'f1-score': 0}

        # Prepare detailed metrics
        detailed_metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'class_f1': {cls: report[cls]['f1-score'] for cls in class_names}
        }

        # Return structure compatible with both evaluate method users and save functions
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'report': report,
            'metrics': detailed_metrics  # Add this key for compatibility with save functions
        }

    def get_training_history_for_plotting(self, history=None):
        """
        Convert the training history to a format suitable for visualization.

        Args:
            history: Dictionary containing training history (default: use the history from the last fit)
                    Must contain 'train_loss' and may contain 'val_loss' and 'val_accuracy'

        Returns:
            List of dictionaries with 'epoch', 'train_loss', 'val_loss', 'val_accuracy' keys
        """
        # If no history provided, check if we have a stored history from the last fit
        if history is None:
            if not hasattr(self, '_last_fit_history'):
                raise ValueError("No training history available. Please provide history or call fit() first.")
            history = self._last_fit_history

        # Convert history to val_metrics format for visualization
        val_metrics = []
        for i in range(len(history['train_loss'])):
            metrics_dict = {
                'epoch': i + 1,
                'train_loss': history['train_loss'][i]
            }

            # Add validation metrics if available
            if 'val_loss' in history and i < len(history['val_loss']):
                metrics_dict['val_loss'] = history['val_loss'][i]

            if 'val_accuracy' in history and i < len(history['val_accuracy']):
                metrics_dict['val_accuracy'] = history['val_accuracy'][i]

            val_metrics.append(metrics_dict)

        return val_metrics