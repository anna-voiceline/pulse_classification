import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os


def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues,
                          figsize=(10, 8), save_path=None, show=True):
    """
    Generate and plot a confusion matrix from true and predicted labels.
    Enhanced to support multi-class classification.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        classes: List of class names (default: auto-detected)
        normalize: Whether to show normalized matrix alongside counts
        title: Plot title
        cmap: Color map for the plot
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (default: None, don't save)
        show: Whether to display the plot (plt.show())

    Returns:
        cm: The confusion matrix
        fig: The matplotlib figure
    """
    # Auto-detect classes if not provided
    if classes is None:
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        if len(unique_labels) <= 2 and set(unique_labels).issubset({0, 1}):
            classes = ['Negative', 'Positive']
        else:
            classes = [str(i) for i in range(len(unique_labels))]

    # Number of classes
    n_classes = len(classes)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Handle special case for binary classification with missing predictions
    if n_classes == 2 and cm.shape != (2, 2):
        if cm.shape == (1, 1):
            if np.all(y_pred == 1):  # All predictions are positive
                cm = np.array([[0, 0], [0, cm[0, 0]]])
            else:  # All predictions are negative
                cm = np.array([[cm[0, 0], 0], [0, 0]])
        elif cm.shape == (1, 2):
            # Only negative samples, but both predictions
            cm = np.array([[cm[0, 0], cm[0, 1]], [0, 0]])
        elif cm.shape == (2, 1):
            # Both classes, but only negative predictions
            cm = np.array([[cm[0, 0], 0], [cm[1, 0], 0]])

    # Determine plotting approach based on class count and normalize flag
    if n_classes > 2 and normalize:
        # For multi-class with normalization, we'll create two plots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # First plot: raw counts
        im1 = ax1.imshow(cm, interpolation='nearest', cmap=cmap)
        fig.colorbar(im1, ax=ax1)
        ax1.set_title(f"{title} (Counts)")

        # Second plot: normalized matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with zero
        im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap=cmap)
        fig.colorbar(im2, ax=ax2)
        ax2.set_title(f"{title} (Normalized)")

        # Configure both axes
        for ax in [ax1, ax2]:
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes,
                   ylabel='True label',
                   xlabel='Predicted label')

            # Rotate tick labels and set alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh1 = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh1 else "black")

        thresh2 = 0.5  # Threshold for normalized matrix
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[0]):
                if not np.isnan(cm_norm[i, j]):
                    ax2.text(j, i, format(cm_norm[i, j], '.2f'),
                             ha="center", va="center",
                             color="white" if cm_norm[i, j] > thresh2 else "black")

    else:
        # Use original single-plot approach for binary classification or when normalize=False
        fig, ax = plt.subplots(figsize=figsize)

        # Normalize if requested
        display_cm = cm
        if normalize:
            display_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            display_cm = np.nan_to_num(display_cm)  # Replace NaN with zero

        # Plot heatmap
        im = ax.imshow(display_cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # Set axis labels and ticks
        ax.set(xticks=np.arange(display_cm.shape[1]),
               yticks=np.arange(display_cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = display_cm.max() / 2.
        for i in range(display_cm.shape[0]):
            for j in range(display_cm.shape[1]):
                ax.text(j, i, format(display_cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if display_cm[i, j] > thresh else "black")

        # Add precision, recall and F1 as text for binary classification
        if n_classes == 2:
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
    else:
        plt.close(fig)

    return cm, fig


def plot_training_history(train_loss, val_metrics=None, figsize=(12, 8), save_path=None, show=True):
    """
    Plot training and validation metrics history.
    Enhanced to support both binary and multi-class metrics.

    Args:
        train_loss: List of training losses or list of dictionaries containing 'train_loss'
        val_metrics: List of dictionaries with validation metrics
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (default: None, don't save)
        show: Whether to display the plot

    Returns:
        fig: The matplotlib figure
    """
    # Handle different input formats
    if isinstance(train_loss, list) and train_loss and isinstance(train_loss[0], dict):
        # New format: list of dictionaries with 'train_loss', 'val_loss', etc.
        epochs = [m.get('epoch', i + 1) for i, m in enumerate(train_loss)]
        train_losses = [m.get('train_loss', 0) for m in train_loss]

        # Check what metrics are available
        has_val_loss = 'val_loss' in train_loss[0] if train_loss else False
        has_val_acc = 'val_accuracy' in train_loss[0] if train_loss else False
        has_precision = 'precision' in train_loss[0] if train_loss else False
        has_recall = 'recall' in train_loss[0] if train_loss else False
        has_f_beta = 'f_beta' in train_loss[0] if train_loss else False

        if has_val_loss:
            val_losses = [m.get('val_loss', 0) for m in train_loss]
        if has_val_acc:
            val_accuracies = [m.get('val_accuracy', 0) for m in train_loss]
        if has_precision:
            val_precision = [m.get('precision', 0) for m in train_loss]
        if has_recall:
            val_recall = [m.get('recall', 0) for m in train_loss]
        if has_f_beta:
            val_f_beta = [m.get('f_beta', 0) for m in train_loss]
    else:
        # Original format: separate train_loss and val_metrics lists
        epochs = range(1, len(train_loss) + 1)
        train_losses = train_loss

        has_val_loss = False
        has_val_acc = False
        has_precision = val_metrics and 'precision' in val_metrics[0] if val_metrics else False
        has_recall = val_metrics and 'recall' in val_metrics[0] if val_metrics else False
        has_f_beta = val_metrics and 'f_beta' in val_metrics[0] if val_metrics else False

        if has_precision:
            val_precision = [metrics.get('precision', 0) for metrics in val_metrics]
        if has_recall:
            val_recall = [metrics.get('recall', 0) for metrics in val_metrics]
        if has_f_beta:
            val_f_beta = [metrics.get('f_beta', 0) for metrics in val_metrics]

    # Determine how many subplots we need
    if has_precision or has_recall or has_f_beta:
        n_plots = 2  # Loss + metrics
    elif has_val_acc:
        n_plots = 2  # Loss + accuracy
    else:
        n_plots = 1  # Just loss

    # Create figure with subplots
    fig, axs = plt.subplots(n_plots, 1, figsize=figsize)

    # Convert to array if only one subplot
    if n_plots == 1:
        axs = np.array([axs])

    # Plot training loss
    axs[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    if has_val_loss:
        axs[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
        axs[0].legend()
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)

    # Plot second subplot
    if n_plots > 1:
        if has_precision or has_recall or has_f_beta:
            # Traditional binary metrics
            if has_precision:
                axs[1].plot(epochs, val_precision, 'r-', label='Precision')
            if has_recall:
                axs[1].plot(epochs, val_recall, 'g-', label='Recall')
            if has_f_beta:
                axs[1].plot(epochs, val_f_beta, 'y-', label='F-beta')
            axs[1].set_title('Validation Metrics')
        elif has_val_acc:
            # Accuracy for multi-class
            axs[1].plot(epochs, val_accuracies, 'g-', label='Accuracy')
            axs[1].set_title('Validation Accuracy')

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
    else:
        plt.close(fig)

    return fig


def plot_class_distribution(labels, class_names=None, title="Class Distribution",
                            figsize=(10, 6), save_path=None, show=True):
    """
    Plot the distribution of classes in a dataset.

    Args:
        labels: Array or list of class labels
        class_names: List of class names (if None, will use unique label indices)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        show: Whether to show the plot

    Returns:
        fig: The matplotlib figure
    """
    # Convert labels to numpy array if needed
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # If labels is a multi-dimensional array or list, flatten it
    if hasattr(labels, 'shape') and len(labels.shape) > 1:
        labels = labels.flatten()

    # Count class frequencies
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Set class names if not provided
    if class_names is None:
        class_names = [str(label) for label in unique_labels]
    else:
        # Ensure we have class names for each unique label
        if len(class_names) < len(unique_labels):
            # Fill missing class names with indices
            class_names = class_names + [str(i) for i in range(len(class_names), len(unique_labels))]

        # Match class names to labels
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        display_names = [class_names[label_to_idx.get(i, i)] if i < len(unique_labels) else str(i)
                         for i in range(len(class_names))]
        class_names = display_names[:len(unique_labels)]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(counts)), counts, tick_label=class_names)

    # Color the bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(counts)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add count labels on top of bars
    for i, count in enumerate(counts):
        ax.text(i, count + (max(counts) * 0.01), str(count),
                ha='center', va='bottom')

    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig