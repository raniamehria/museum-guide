# src/museum_guide/train.py
import copy
import torch

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running, total = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(yb); total += len(yb)
    return running / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    tl, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            tl += loss_fn(out, yb).item() * len(yb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    return tl / total, correct / total


def fit_with_best(model, train_loader, val_loader, loss_fn, optimizer, n_epochs, device):
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(n_epochs):
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        vl, va = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(tr); history["val_loss"].append(vl); history["val_acc"].append(va)
        if vl < best_val:
            best_val, best_epoch = vl, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_epoch


def fit(model, train_loader, val_loader, loss_fn, optimizer, n_epochs, device):
    """Plain training loop: keep the last epoch (no best-epoch selection)."""
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for _ in range(n_epochs):
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        vl, va = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
    return history

class EarlyStopping:
    """Stop training when val_loss has not improved for patience epochs (TP06 Ex 10)"""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict()) # deepcopy: state_dict holds references
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def fit_regularized(model, train_loader, val_loader, loss_fn, optimizer, n_epochs, device,
                    early_stopping=None, train_eval_loader=None, scheduler=None):
    """Training loop with optional early stopping and LR schedule (TP06 Part B / Ex 10).

    Returns (history, n_epochs_trained). History keys match Task 1's mlp_history.json,
    plus 'train_acc' and 'lr' for a fair train/val gap analysis.
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    for epoch in range(1, n_epochs + 1):
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        ta_loader = train_eval_loader if train_eval_loader is not None else train_loader
        _, ta = evaluate(model, ta_loader, loss_fn, device)
        vl, va = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(tr)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if early_stopping is not None:
            early_stopping(vl, model, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch} (best epoch {early_stopping.best_epoch})")
                break
        if scheduler is not None:
            scheduler.step()

    if early_stopping is not None and early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
    return history, len(history["val_loss"])