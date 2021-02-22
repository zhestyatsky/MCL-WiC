import numpy as np
from sklearn.metrics import roc_curve

from src.training.config import get_config
from src.util.util import get_accuracy


def find_threshold_if_needed(labels, probas, model_description):
    model_config = get_config(model_description)
    threshold = 0.5

    if model_config["top"] == "cosine_similarity":
        fpr, tpr, thlds = roc_curve(labels, probas)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thlds[optimal_idx]

    acc = get_accuracy(labels, probas, threshold)
    print("Threshold: {}, Validation Accuracy: {}".format(threshold, acc))
    return threshold, acc
