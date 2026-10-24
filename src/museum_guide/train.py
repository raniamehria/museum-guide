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