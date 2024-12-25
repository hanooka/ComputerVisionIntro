"""
This code is straight from GPT as I had no desire on fiddling with matplotlib.
"""

import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import numpy as np


def plot_roc_pr_cm(y_true, y_probs, class_labels, save_path=None):
    """
    Plots ROC AUC, Precision-Recall Curves, and Confusion Matrix for each class.

    Parameters:
    - y_true: True labels (array-like, shape: [n_samples,])
    - y_probs: Predicted probabilities for each class (array-like, shape: [n_samples, n_classes])
    - class_labels: List of class labels (array-like, shape: [n_classes])
    - save_path: Optional path to save the figure (string). If None, the figure won't be saved.
    """
    # Initialize subplots
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    ax_roc, ax_pr, ax_cm = axes

    # Initialize color map
    colors = plt.cm.get_cmap("tab10", len(class_labels))

    # Iterate through each class and plot ROC and Precision-Recall curves
    for i, class_name in enumerate(class_labels):
        # Binary labels for the current class
        y_bin = (y_true == i).astype(int)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        ax_roc.plot(fpr, tpr, label=f"Class {class_name}", color=colors(i))

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, i])
        ax_pr.plot(recall, precision, label=f"Class {class_name}", color=colors(i))

    # ROC AUC Plot formatting
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set_title("ROC AUC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="best")

    # Precision-Recall Plot formatting
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="best")

    # Confusion Matrix
    cm = confusion_matrix(y_true, np.argmax(y_probs, axis=1))  # Predicted class labels
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Save the figure if save_path is provided
    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")


def denormalize(img, mean, std):
    """Reverse the normalization applied to the image for display purposes."""
    img = img.clone()  # Clone to avoid modifying the original tensor
    for c in range(3):  # Assuming 3 channels (RGB)
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1)  # Ensure values are in the range [0, 1]

def plot_samples_per_class(data_loader, label_encoder, class_labels, num_samples=8, mean=None, std=None):
    """
    Plot a few examples from each class in the dataset.

    Parameters:
    - data_loader: DataLoader object for the dataset.
    - label_encoder: Fitted LabelEncoder to decode numerical labels.
    - class_labels: List of class labels (names).
    - num_samples: Number of examples to display per class.
    - mean, std: Mean and standard deviation used for normalization in the transform.
    """
    # Dictionary to store images for each class
    samples_per_class = defaultdict(list)

    # Collect examples from the data loader
    for imgs, labels in data_loader:
        for img, label in zip(imgs, labels):
            decoded_label = label_encoder.inverse_transform([label.item()])[0]  # Decode label
            if len(samples_per_class[decoded_label]) < num_samples:
                samples_per_class[decoded_label].append(img)
            # Stop collecting if we have enough examples for all classes
            if all(len(samples_per_class[c]) >= num_samples for c in class_labels):
                break

    # Plot the samples
    num_classes = len(class_labels)
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(num_samples * 2, num_classes * 3))
    fig.suptitle("Samples Per Class", fontsize=16)

    for class_idx, (class_label, images) in enumerate(samples_per_class.items()):
        for sample_idx, img in enumerate(images):
            ax = axes[class_idx, sample_idx] if num_classes > 1 else axes[sample_idx]
            # Denormalize if mean and std are provided
            if mean and std:
                img = denormalize(img, mean, std)
            ax.imshow(img.permute(1, 2, 0))  # Convert from Tensor (C, H, W) to (H, W, C)
            ax.axis("off")
            ax.set_title(class_label, fontsize=10, y=-0.2)  # Use the decoded label as the title

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add space for the title
    plt.show()

"""
To plot this

mean = [0.485, 0.456, 0.406]  # Example: ImageNet mean
std = [0.229, 0.224, 0.225]  # Example: ImageNet std

class_labels = le.classes_
plot_samples_per_class(train_data_loader, le, class_labels, num_samples=8, mean=mean, std=std)

"""

def plot_metrics(metrics):
    epochs = metrics["epoch"]

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["train_accuracy"], label="Train Accuracy", marker='o')
    plt.plot(epochs, metrics["val_accuracy"], label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
