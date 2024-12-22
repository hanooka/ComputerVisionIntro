"""
This code is straight from GPT as I had no desire on fiddling with matplotlib.
"""

import matplotlib.pyplot as plt
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
