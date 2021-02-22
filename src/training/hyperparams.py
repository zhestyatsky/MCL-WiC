import numpy as np
import torch
from pytorch_lightning.metrics import Accuracy
from sklearn.metrics import roc_curve

from src.training.config import get_config


def find_threshold_if_needed(labels, proba, model_description):
    model_config = get_config(model_description)
    thld = 0.5

    if model_config["top"] == "cosine_similarity":
        fpr, tpr, thlds = roc_curve(labels, proba)
        optimal_idx = np.argmax(tpr - fpr)
        thld = thlds[optimal_idx]

    y_pred = (proba > thld).float()
    acc = Accuracy()
    acc_value = acc(y_pred, torch.tensor(labels)).item()
    print("Threshold: {}, Validation Accuracy: {}".format(thld, acc_value))
    return thld, acc_value
