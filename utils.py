"""
utils.py — Helper functions (seeds, accuracy, plotting)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_accuracy(logits, labels):
    """Compute accuracy from model logits and true labels."""
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


def count_params(model):
    """Count trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_history(history, model_name, save_path=None):
    """Plot training/validation loss and accuracy curves for one model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    ax1.set_title(f'{model_name} — Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(epochs, history['val_acc'], label='Val Acc', marker='s')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_all_models(histories, names, save_path=None):
    """Plot validation curves for all models on one figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(histories[0]['val_loss']) + 1)
    colors = ['blue', 'orange', 'green', 'red']

    for history, name, color in zip(histories, names, colors):
        ax1.plot(epochs, history['val_loss'], label=name, marker='o', color=color)
        ax2.plot(epochs, history['val_acc'], label=name, marker='o', color=color)

    ax1.set_title('Validation Loss — All Models')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)

    ax2.set_title('Validation Accuracy — All Models')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.5, 1.0); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()