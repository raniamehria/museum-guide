"""ResNet-18 pretrained on ImageNet, adapted to the 5 WikiArt styles (TP07 Part D)."""
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from museum_guide.data import get_dataloaders

FAST_RUN = os.environ.get("FAST_RUN", "0") == "1"   # True: fewer epochs for a quick check
EPOCHS_HEAD = 1 if FAST_RUN else 5
EPOCHS_FT = 1 if FAST_RUN else 5
N_CLASSES = 5
RESULTS_DIR = Path("results/resnet")
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# TP07 helpers
def evaluate(model, loader, loss_fn):
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total += loss_fn(out, yb).item() * len(yb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return total / n, correct / n


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# TP07 fit_transfer loop, written as a TP06 train_epoch_fn
def train_transfer_epoch(model, loader, loss_fn, optimizer):
    model.eval()                                   # batch norm keeps its ImageNet statistics
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()


# TP06 helpers
def new_history():
    return {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}


def record(history, model, optimizer):
    """Measure the model on the training and validation sets and append the results to the history."""
    tl, ta = evaluate(model, train_eval_loader, loss_fn)
    vl, va = evaluate(model, val_loader, loss_fn)
    history["train_loss"].append(tl); history["train_acc"].append(ta)
    history["val_loss"].append(vl); history["val_acc"].append(va)
    history["lr"].append(optimizer.param_groups[0]["lr"])


def fit(model, optimizer, n_epochs, train_epoch_fn=train_transfer_epoch, scheduler=None, verbose=False):
    torch.manual_seed(1)                       # same order of the mini-batches for every run
    history = new_history()
    for epoch in range(n_epochs):
        train_epoch_fn(model, train_loader, loss_fn, optimizer)
        record(history, model, optimizer)
        if scheduler is not None:
            scheduler.step()
        if verbose:
            print(f"epoch {epoch + 1:2d}  train loss {history['train_loss'][-1]:.3f}  "
                  f"val loss {history['val_loss'][-1]:.3f}  val acc {history['val_acc'][-1]:.3f}")
    return history


def plot_history(history, title=""):
    e = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(e, history["train_loss"], label="train"); axes[0].plot(e, history["val_loss"], label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].plot(e, history["train_acc"], label="train"); axes[1].plot(e, history["val_acc"], label="validation")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()
    fig.suptitle(title)
    fig.savefig(RESULTS_DIR / "curves.png", dpi=120, bbox_inches="tight")   # plt.show() in the notebook


# Project additions (not in the TPs): macro-F1 and CPU latency for the director's question
def macro_f1(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu())
            labels.append(yb)
    return f1_score(torch.cat(labels), torch.cat(preds), average="macro")


def cpu_latency_ms(model, n=50):
    model = model.cpu().eval()
    x = torch.randn(1, 3, 128, 128)                # one visitor = one image (batch 1)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t0) / n * 1000


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders("resnet")
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=250)   # TP06: train set without shuffling
    loss_fn = nn.CrossEntropyLoss()

    # Exercise 13: load the model and replace its head
    torch.manual_seed(0)
    model_tl = resnet18(weights=ResNet18_Weights.DEFAULT)
    print("original head:", model_tl.fc)
    model_tl.fc = nn.Linear(512, N_CLASSES)
    model_tl = model_tl.to(device)

    # Exercise 14: freeze the backbone
    for p in model_tl.parameters():
        p.requires_grad = False
    for p in model_tl.fc.parameters():
        p.requires_grad = True
    n_trainable = count_parameters(model_tl, trainable_only=True)
    print("Trainable:", n_trainable, "of", count_parameters(model_tl))          # 512 x 5 + 5 = 2565

    # Exercise 15: train the head
    t0 = time.time()
    optimizer = torch.optim.Adam(model_tl.fc.parameters(), lr=3e-3)
    history_head = fit(model_tl, optimizer, EPOCHS_HEAD, verbose=True)
    acc_head = evaluate(model_tl, test_loader, loss_fn)[1]
    f1_head = macro_f1(model_tl, test_loader)
    time_head = time.time() - t0
    print(f"Head only: test accuracy {acc_head:.3f}, macro-F1 {f1_head:.3f} ({time_head:.0f} s). Chance level is 0.20.")

    # Exercise 16: fine-tune the last block
    t0 = time.time()
    for p in model_tl.layer4.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam([{"params": model_tl.layer4.parameters(), "lr": 1e-4},
                                  {"params": model_tl.fc.parameters(), "lr": 1e-3}])
    history_ft = fit(model_tl, optimizer, EPOCHS_FT, verbose=True)
    acc_ft = evaluate(model_tl, test_loader, loss_fn)[1]
    f1_ft = macro_f1(model_tl, test_loader)
    time_ft = time.time() - t0
    print(f"Fine-tuned: test accuracy {acc_ft:.3f}, macro-F1 {f1_ft:.3f} (head only {acc_head:.3f}) in {time_ft:.0f} s.")

    # Save the weights (outside Git), the curves and the numbers of the slides
    torch.save(model_tl.state_dict(), "models/resnet18.pt")
    history = {k: history_head[k] + history_ft[k] for k in history_head}
    plot_history(history, f"ResNet-18: epochs 1-{EPOCHS_HEAD} head only, then layer4 + head")
    results = {
        "A_head_only": {"test_acc": acc_head, "test_macro_f1": f1_head, "train_seconds": time_head},
        "B_layer4_finetuned": {"test_acc": acc_ft, "test_macro_f1": f1_ft, "train_seconds": time_ft},
        "cpu_latency_ms_batch1": cpu_latency_ms(model_tl),
        "size_mb": os.path.getsize("models/resnet18.pt") / 1e6,
        "n_parameters": count_parameters(model_tl),
    }
    (RESULTS_DIR / "history.json").write_text(json.dumps(history, indent=2))
    (RESULTS_DIR / "resnet.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))