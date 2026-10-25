"""Tâche 4: classical baseline = logistic regression (scikit-learn), no neural network trained.

Three inputs for the same logistic regression:
- pixels:  the 64 x 64 images of the MLP, flattened (same input as mlp.ipynb)
- colours: a colour histogram of each painting (no network at all)
- resnet:  the 512 features of a frozen ImageNet ResNet-18 (same backbone as resnet.ipynb, not trained)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from museum_guide.data import prepare_resnet

# C = 1 / lambda in scikit-learn: small C = strong L2 penalty (weight decay of session 6)
C_GRID = (0.0001, 0.0003, 0.001, 0.01, 0.1, 1.0, 10.0)   # 0.0001 and 0.0003 added: on the pixels the best C was at the edge


# 1. Features: images (N, 3, 128, 128) in [0, 1] -> one row of numbers per image
def pixel_features(X, size=64):
    # same resize as prepare_mlp in data.py, so the logistic regression sees what the MLP sees
    X = F.interpolate(X, size=size, mode="bilinear", align_corners=False)
    return X.flatten(start_dim=1).numpy()                       # (N, 3 * 64 * 64) = (N, 12 288)


def colour_histogram(X, bins=8):
    # each channel cut into 8 levels -> 8 x 8 x 8 = 512 colour boxes,
    # we count the share of the pixels of the image that fall in each box
    n, _, h, w = X.shape
    level = (X * bins).long().clamp(max=bins - 1)               # 0..7 for R, G and B
    box = level[:, 0] * bins * bins + level[:, 1] * bins + level[:, 2]
    box = box + torch.arange(n).view(n, 1, 1) * bins ** 3       # shift so each image has its own boxes
    counts = torch.bincount(box.flatten(), minlength=n * bins ** 3)
    return (counts.view(n, bins ** 3).float() / (h * w)).numpy()  # (N, 512), each row sums to 1


def load_frozen_resnet(pretrained=True):
    # ResNet-18 trained on ImageNet without its last layer -> 512 numbers per image
    from torchvision.models import ResNet18_Weights, resnet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False                                 # frozen: we never train it
    return model.eval()                                         # eval: batch norm keeps the ImageNet statistics


def resnet_features(X, backbone, batch_size=100, device="cpu"):
    backbone = backbone.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = prepare_resnet(X[i:i + batch_size]).to(device)  # ImageNet normalisation, as in resnet.ipynb
            out.append(backbone(xb).cpu())
    return torch.cat(out).numpy()                               # (N, 512)


# 2. Logistic regression, C chosen on the validation set (never on the test set)
def make_logreg(C):
    # StandardScaler: mean 0 / std 1 per feature, computed on the train set only (fit is on train)
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def fit_logreg(X_train, y_train, X_val, y_val, c_grid=C_GRID, verbose=True):
    """Train one model per C and keep the best validation accuracy. Returns (model, C, table)."""
    table, best_acc, best_C, best_model = [], -1.0, None, None
    for C in c_grid:
        model = make_logreg(C).fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        table.append({"C": C, "train_acc": train_acc, "val_acc": val_acc})
        if verbose:
            print(f"C = {C:<6g} | train acc {train_acc:.3f} | val acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc, best_C, best_model = val_acc, C, model
    if verbose:
        print(f"kept C = {best_C:g} (best val acc {best_acc:.3f})")
    return best_model, best_C, table


def scores(model, X, y):
    pred = model.predict(X)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"), "pred": pred}


def count_logreg_parameters(model):
    # weights (n_features x 5) + biases (5)
    lr = model[-1]
    return int(lr.coef_.size + lr.intercept_.size)
