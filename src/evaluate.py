"""Metrics and plots. Reports false-positive rate, the metric SOC teams care about."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from .config import FIG_DIR


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall (detection rate)": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in out.items()}


def plot_confusion(y_true, y_pred, labels, title, fname, normalize="true"):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, display_labels=labels, normalize=normalize,
        cmap="Blues", ax=ax, values_format=".2f" if normalize else "d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def plot_curves(scores: dict, y_true, fname="roc_pr_curves.png"):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        p, r, _ = precision_recall_curve(y_true, s)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, s):.3f})")
        axes[1].plot(r, p, label=f"{name} (AP={average_precision_score(y_true, s):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (KDDTest+)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (KDDTest+)")
    for a in axes:
        a.legend(loc="lower right", fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def plot_class_distribution(train, test, fname="class_distribution.png"):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = ["normal", "DoS", "Probe", "R2L", "U2R"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.38
    x = np.arange(len(order))
    ax.bar(x - w / 2, [(train.category == c).mean() * 100 for c in order], w, label="KDDTrain+")
    ax.bar(x + w / 2, [(test.category == c).mean() * 100 for c in order], w, label="KDDTest+")
    ax.set_xticks(x, order)
    ax.set_ylabel("% of records")
    ax.set_title("Class distribution (note the train/test shift)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
