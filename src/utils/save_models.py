import os
import json
import torch
import torch.nn.functional as F


def save_pytorch_model(model, output_dir, class_mapping, test_metrics, logger=None,
                       filename='document_type_classifier.pt'):
    """
    Save PyTorch model to file with metadata.

    Args:
        model: The PyTorch model to save
        output_dir: Directory to save the model
        class_mapping: Dictionary mapping class indices to class names
        test_metrics: Evaluation metrics on test set
        logger: Optional logger for messages
        filename: Name of the output file

    Returns:
        Path to the saved model
    """
    log_fn = logger.info if logger else print

    model_path = os.path.join(output_dir, filename)
    log_fn(f"Saving model to {model_path}...")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': model.input_dim,
            'hidden_dim': model.hidden_dim,
            'num_classes': model.num_classes
        },
        'class_mapping': class_mapping,
        'test_metrics': test_metrics
    }, model_path)

    return model_path


def export_to_onnx(model, output_dir, device, logger=None,
                   filename='document_type_classifier.onnx'):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: The PyTorch model to export
        output_dir: Directory to save the ONNX model
        device: Device to create dummy input on
        logger: Optional logger for messages
        filename: Name of the output file

    Returns:
        Path to the saved ONNX model
    """
    log_fn = logger.info if logger else print

    onnx_model_path = os.path.join(output_dir, filename)
    log_fn(f"Exporting model to ONNX format: {onnx_model_path}")

    # Create wrapper for ONNX export that includes softmax
    class ModelWithSoftmax(torch.nn.Module):
        def __init__(self, model):
            super(ModelWithSoftmax, self).__init__()
            self.model = model

        def forward(self, x):
            logits = self.model(x)
            return F.softmax(logits, dim=1)

    wrapped_model = ModelWithSoftmax(model)
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

    return onnx_model_path


def save_model_config(output_dir, class_mapping, num_classes, logger=None,
                      filename='document_type_classifier_config.json'):
    """
    Save model configuration to JSON file.

    Args:
        output_dir: Directory to save the config
        class_mapping: Dictionary mapping class indices to class names
        num_classes: Number of classes
        logger: Optional logger for messages
        filename: Name of the output file

    Returns:
        Path to the saved config file
    """
    log_fn = logger.info if logger else print

    config_path = os.path.join(output_dir, filename)

    # Create model configs
    model_configs = {
        'class_mapping': class_mapping,
        'num_classes': num_classes
    }

    with open(config_path, 'w') as f:
        json.dump(model_configs, f, indent=2)

    log_fn(f"Model configuration saved to {config_path}")
    return config_path


def save_model_artifacts(model, output_dir, embeddings_paths, test_results, device, logger=None):
    """
    Save all model artifacts (PyTorch model, ONNX model, and config).

    Args:
        model: The PyTorch model to save
        output_dir: Directory to save artifacts
        embeddings_paths: Dictionary of embedding paths (used for class names)
        test_results: Evaluation results from model.evaluate()
        device: Device for ONNX export
        logger: Optional logger

    Returns:
        Dictionary with paths to all saved artifacts
    """
    # Create class mapping
    class_mapping = {i: cls for i, cls in enumerate(embeddings_paths.keys())}

    # Extract metrics from test_results (handle both formats)
    test_metrics = test_results.get('metrics', test_results)

    # Save PyTorch model
    pt_path = save_pytorch_model(
        model=model,
        output_dir=output_dir,
        class_mapping=class_mapping,
        test_metrics=test_metrics,
        logger=logger
    )

    # Export to ONNX
    onnx_path = export_to_onnx(
        model=model,
        output_dir=output_dir,
        device=device,
        logger=logger
    )

    # Save configuration
    config_path = save_model_config(
        output_dir=output_dir,
        class_mapping=class_mapping,
        num_classes=len(embeddings_paths),
        logger=logger
    )

    if logger:
        logger.info("All model artifacts saved successfully")

    return {
        'pytorch_model': pt_path,
        'onnx_model': onnx_path,
        'config': config_path
    }