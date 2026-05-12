from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class MetricsResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray


def compute_metrics(y_true, y_pred, average: str = "macro") -> MetricsResult:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return MetricsResult(accuracy=accuracy, precision=precision, recall=recall, f1=f1, confusion=cm)


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
