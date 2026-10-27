"""Model Evaluation and Diagnostics Module.

Computes classification metrics (Accuracy, F1, Precision, Recall, ROC-AUC),
plots Confusion Matrices, ROC Curves, and Model Comparison charts.
"""

import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    class_names: List[str]
) -> Dict[str, Any]:
    """Compute standard multiclass classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Multiclass ROC AUC (One-vs-Rest)
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        roc_auc = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = np.nan
        
    return {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision (Macro)': round(prec_macro, 4),
        'Recall (Macro)': round(rec_macro, 4),
        'F1-Score (Macro)': round(f1_macro, 4),
        'F1-Score (Weighted)': round(f1_weighted, 4),
        'ROC-AUC (OvR)': round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A"
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str
) -> None:
    """Plot dual raw and normalized Confusion Matrix heatmap."""
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Raw Counts
    sns.heatmap(
        cm_raw, annot=True, fmt='d', cmap='Blues', ax=axes[0],
        xticklabels=class_names, yticklabels=class_names, cbar=False
    )
    axes[0].set_title(f"Confusion Matrix (Counts) - {model_name}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    
    # Normalized Percentage
    sns.heatmap(
        cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
        xticklabels=class_names, yticklabels=class_names, cbar=True
    )
    axes[1].set_title(f"Normalized Confusion Matrix - {model_name}", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def plot_multiclass_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_path: str
) -> None:
    """Plot One-vs-Rest ROC curve for each weather class."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    palette = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']
    
    plt.figure(figsize=(9, 7))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i], tpr[i], lw=2, color=palette[i % len(palette)],
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})"
        )
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.50)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title(f'Multiclass ROC Curves (One-vs-Rest) - {model_name}', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve: {save_path}")


def plot_model_comparison_bar(comparison_df: pd.DataFrame, save_path: str) -> None:
    """Plot visual comparison of key metrics across all evaluated models."""
    metrics_to_plot = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    plot_df = comparison_df.set_index('Model')[metrics_to_plot]
    
    ax = plot_df.plot(kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='black', alpha=0.85, width=0.75)
    plt.title("Model Performance Comparison on Test Set", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.xlabel("Classifier Model", fontsize=11)
    plt.ylim(0.0, 1.08)
    plt.legend(loc='lower right', framealpha=0.9)
    plt.xticks(rotation=15, ha='right')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot: {save_path}")


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, model_name: str, save_path: str, top_n: int = 15) -> None:
    """Plot horizontal bar chart of top feature importances."""
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(fi_df['Feature'], fi_df['Importance'], color='#16A085', edgecolor='black', alpha=0.85)
    plt.title(f"Top {top_n} Feature Importances ({model_name})", fontsize=13, fontweight='bold')
    plt.xlabel("Gini / Mean Decrease Impurity", fontsize=11)
    plt.xlim(0, max(fi_df['Importance']) * 1.15)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.003, bar.get_y() + bar.get_height()/2, f"{width:.4f}", va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot: {save_path}")
