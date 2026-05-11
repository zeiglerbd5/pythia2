"""
Model Diagnostics Visualization

Provides visualization tools for analyzing classification model performance,
including prediction distributions, calibration curves, confusion matrices,
and precision-recall trade-offs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve
from typing import Optional


def plot_predictions(
    model,
    train_data,
    train_labels,
    test_data,
    test_labels,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize classification model predictions vs actual labels.

    Creates a 2x2 grid of diagnostic plots:
    - Prediction distributions (train/test comparison)
    - Calibration curve (reliability of probabilities)
    - Confusion matrix (error analysis)
    - Precision-Recall at different thresholds

    Args:
        model: Trained sklearn/xgboost model with predict_proba method
        train_data: Training features (X_train)
        train_labels: Training labels (y_train)
        test_data: Test features (X_test)
        test_labels: Test labels (y_test)
        title: Optional title for the entire figure
        save_path: Optional path to save the figure
        show: Whether to display the plot (default True)
    """
    # Generate predictions
    y_train_prob = model.predict_proba(train_data)[:, 1]
    y_test_prob = model.predict_proba(test_data)[:, 1]
    y_test_pred = model.predict(test_data)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title or 'Model Diagnostics', fontsize=16, fontweight='bold')

    # Plot each diagnostic view
    _plot_prediction_distribution(axes[0, 0], train_labels, y_train_prob, test_labels, y_test_prob)
    _plot_calibration_curve(axes[0, 1], test_labels, y_test_prob)
    _plot_confusion_matrix(axes[1, 0], test_labels, y_test_pred)
    _plot_threshold_metrics(axes[1, 1], test_labels, y_test_prob)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostic plot saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def _plot_prediction_distribution(ax, y_train, y_train_prob, y_test, y_test_prob):
    """
    Plot histogram of predicted probabilities split by dataset and true class.

    Shows if model behaves differently on train vs test (overfitting indicator).
    """
    ax.set_title('Prediction Distribution: Train vs Test', fontsize=12, fontweight='bold')

    # Plot train set
    ax.hist(y_train_prob[y_train == 0], bins=30, alpha=0.5, label='Train (Negative)', color='blue', edgecolor='black')
    ax.hist(y_train_prob[y_train == 1], bins=30, alpha=0.5, label='Train (Positive)', color='red', edgecolor='black')

    # Plot test set with different style
    ax.hist(y_test_prob[y_test == 0], bins=30, alpha=0.3, label='Test (Negative)', color='cyan', edgecolor='black', linestyle='--')
    ax.hist(y_test_prob[y_test == 1], bins=30, alpha=0.3, label='Test (Positive)', color='orange', edgecolor='black', linestyle='--')

    ax.set_xlabel('Predicted Probability', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(loc='upper center', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add stats
    train_mean = np.mean(y_train_prob[y_train == 1])
    test_mean = np.mean(y_test_prob[y_test == 1])
    ax.text(0.02, 0.98, f'Train Pos Mean: {train_mean:.3f}\nTest Pos Mean: {test_mean:.3f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def _plot_calibration_curve(ax, y_true, y_pred_proba):
    """
    Plot calibration curve showing reliability of predicted probabilities.

    Diagonal line = perfect calibration.
    Above diagonal = overconfident, below = underconfident.
    """
    ax.set_title('Calibration Curve', fontsize=12, fontweight='bold')

    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10, strategy='uniform'
    )

    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model', linewidth=2, markersize=8)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)

    ax.set_xlabel('Mean Predicted Probability', fontsize=10)
    ax.set_ylabel('Fraction of Positives', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add interpretation guide
    ax.text(0.02, 0.98, 'Above diagonal: Overconfident\nBelow diagonal: Underconfident',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))


def _plot_confusion_matrix(ax, y_true, y_pred):
    """
    Plot confusion matrix heatmap for test set predictions.

    Shows breakdown of TP, FP, TN, FN at default threshold (0.5).
    """
    ax.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)

    # Add percentages
    total = np.sum(cm)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    stats_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
    ax.text(1.5, 0.5, stats_text, transform=ax.transData, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))


def _plot_threshold_metrics(ax, y_true, y_pred_proba):
    """
    Plot precision and recall curves across different probability thresholds.

    Helps choose optimal threshold based on precision/recall trade-off.
    """
    ax.set_title('Precision-Recall at Different Thresholds', fontsize=12, fontweight='bold')

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Plot curves
    ax.plot(thresholds, precision[:-1], 'b-', label='Precision', linewidth=2)
    ax.plot(thresholds, recall[:-1], 'r-', label='Recall', linewidth=2)

    # Mark common thresholds
    important_thresholds = [0.5, 0.75, 0.85, 0.9]
    for thresh in important_thresholds:
        if thresh <= max(thresholds):
            idx = np.argmin(np.abs(thresholds - thresh))
            ax.axvline(x=thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(thresh, 0.05, f'{thresh:.2f}', fontsize=8, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('Probability Threshold', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add F1 score curve
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    ax.plot(thresholds, f1_scores, 'g--', label='F1 Score', linewidth=1.5, alpha=0.7)

    # Mark best F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    ax.plot(best_threshold, f1_scores[best_f1_idx], 'go', markersize=10,
            label=f'Best F1: {best_threshold:.3f}')

    ax.legend(loc='best', fontsize=9)
