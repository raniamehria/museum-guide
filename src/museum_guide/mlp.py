"""Feed-forward classifiers for the museum-guide style subset.

Mirrors TP04 Exercise 15 (nn.Sequential MLP) and TP05 Exercise 17 (weight init).
The two factories reproduce the two baselines of notebook/mlp.ipynb.
"""
from __future__ import annotations

import torch.nn as nn

IN_FEATURES: int = 3 * 64 * 64   # 12 288 — must match museum_guide.data resize
NUM_CLASSES: int = 5


def make_mlp(
    widths: list[int],
    in_features: int = IN_FEATURES,
    num_classes: int = NUM_CLASSES,
    activation: type[nn.Module] = nn.ReLU,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Flatten -> [Linear, activation, (dropout)] * len(widths) -> Linear."""
    layers: list[nn.Module] = [nn.Flatten()]
    n_in = in_features
    for w in widths:
        layers += [nn.Linear(n_in, w), activation()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        n_in = w
    layers.append(nn.Linear(n_in, num_classes))
    return nn.Sequential(*layers)


def init_weights(model: nn.Sequential, kind: str = "he") -> nn.Sequential:
    """In-place init of every nn.Linear. Same spec as TP05 Exercise 17."""
    for m in model:
        if isinstance(m, nn.Linear):
            if kind == "zeros":
                nn.init.zeros_(m.weight)
            elif kind == "small":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif kind == "large":
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
            elif kind == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif kind == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            else:
                raise ValueError(f"unknown init: {kind!r}")
            nn.init.zeros_(m.bias)
    return model


def make_baseline_mlp() -> nn.Sequential:
    """Notebook Part A: 1 hidden sigmoid layer, default PyTorch init (TP05 Part A)."""
    return make_mlp(widths=[128], activation=nn.Sigmoid)


def make_tuned_mlp() -> nn.Sequential:
    """Notebook 'Going further': 3 hidden ReLU layers + He init (TP05 Ex 20)."""
    return init_weights(make_mlp(widths=[256, 128, 64], activation=nn.ReLU), kind="he")