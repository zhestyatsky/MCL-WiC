import numpy as np
import torch
from pytorch_lightning.metrics import Accuracy
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve

def get_threshold_with_acc(labels, proba):
    fpr, tpr, thlds = roc_curve(labels, proba)
    optimal_idx = np.argmax(tpr - fpr)
    thld = thlds[optimal_idx]
    y_pred = (proba > thld).float()
    acc = Accuracy()
    return thld, acc(y_pred, torch.tensor(labels)).item()