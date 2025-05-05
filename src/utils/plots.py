import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# import seaborn as sns
import os

def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues,
                          figsize=(8, 6), save_path=None, show=True):
    """
    Generate and plot a confusion matrix from true and predicted labels.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        classes: List of class names (default: ['Negative', 'Positive'])
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Color map for the plot
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (default: None, don't save)
        show: Whether to display the plot (plt.show())

    Returns:
        cm: The confusion matrix
        fig: The matplotlib figure
    """
    if classes is None:
        classes = ['Negative', 'Positive']

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # In case of binary classification with only one class present in the predictions
    if cm.shape == (1, 1):
        if np.all(y_pred == 1):  # All predictions are positive
            cm = np.array([[0, 0], [0, cm[0, 0]]])
        else:  # All predictions are negative
            cm = np.array([[cm[0, 0], 0], [0, 0]])

    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set axis labels and ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Add precision, recall and F1 as text
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    plt.figtext(0.5, 0.01, f'Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}',
                ha='center', fontsize=10)

    fig.tight_layout()

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    # Show plot if requested
    if show:
        plt.show()

    return cm, fig


def plot_training_history(train_loss, val_metrics, figsize=(12, 8), save_path=None, show=True):
    """
    Plot training and validation metrics history.

    Args:
        train_loss: List of training losses
        val_metrics: List of dictionaries with validation metrics
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (default: None, don't save)
        show: Whether to display the plot

    Returns:
        fig: The matplotlib figure
    """
    # Extract metrics
    epochs = range(1, len(train_loss) + 1)
    val_precision = [metrics['precision'] for metrics in val_metrics]
    val_recall = [metrics['recall'] for metrics in val_metrics]
    val_f_beta = [metrics['f_beta'] for metrics in val_metrics]

    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=figsize)

    # Plot training loss
    axs[0].plot(epochs, train_loss, 'b-', label='Training Loss')
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)

    # Plot validation metrics
    axs[1].plot(epochs, val_precision, 'r-', label='Precision')
    axs[1].plot(epochs, val_recall, 'g-', label='Recall')
    axs[1].plot(epochs, val_f_beta, 'y-', label='F-beta')
    axs[1].set_title('Validation Metrics')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Score')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    # Show plot if requested
    if show:
        plt.show()

    return fig