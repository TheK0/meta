import numpy as np


def compute_binary_task_metrics(*, predictions, labels):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    return {
        "avg_precision": float((predictions[labels == 1].mean() if (labels == 1).any() else 0.0)),
    }
