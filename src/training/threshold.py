import numpy as np
import torch
from pytorch_lightning.metrics import Accuracy
from sklearn.metrics import roc_curve


def get_threshold_with_acc(labels, proba):
    fpr, tpr, thlds = roc_curve(labels, proba)
    optimal_idx = np.argmax(tpr - fpr)
    thld = thlds[optimal_idx]
    y_pred = (proba > thld).float()
    acc = Accuracy()
    acc_value = acc(y_pred, torch.tensor(labels)).item()
    print("Threshold: {}, Accuracy: {}".format(thld, acc_value))
    return thld, acc_value
