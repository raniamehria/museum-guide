"""Tâche 3: ResNet-18 transfer learning on the 5 WikiArt styles (the recipe of notebook/resnet.ipynb).

- Step A: ResNet-18 pretrained on ImageNet, backbone frozen, new head 512 -> 5 (2 565 trained parameters).
- Step B: step A, then layer4 unfrozen with a smaller learning rate (1e-4 for layer4, 1e-3 for the head).
- Step C: step B trained longer with regularisation: random flip + crop, dropout 0.5 before the head,
  AdamW weight decay 1e-2, 3 + 15 epochs, cosine learning rate.
Every run keeps the epoch with the best validation accuracy (early stopping); BatchNorm keeps its
ImageNet statistics during training (model.eval()), only the dropout of the head is switched on.

notebook/resnet.ipynb keeps its own copy of these functions so that it reads top to bottom; this module
is the same recipe for scripts and other notebooks (for example qa_hybrid.ipynb loads the saved weights).

Run from the repo root (images in data/wikiart_style5/):
    uv run python -m museum_guide.resnet --step B            # trains steps A + B, saves models/resnet18.pt
    uv run python -m museum_guide.resnet --step C --seed 1   # step C with another random seed
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

CLASSES = ["Baroque", "Cubism", "Impressionism", "Pop_Art", "Ukiyo_e"]
N_CLASSES = len(CLASSES)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 1. Model
def build_resnet(dropout=0.0, pretrained=True):
    """ResNet-18 with a new head for the 5 styles; dropout > 0 puts a Dropout layer before the head."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    head = nn.Linear(512, N_CLASSES)
    model.fc = nn.Sequential(nn.Dropout(dropout), head) if dropout > 0 else head
    return model


def freeze_backbone(model):
    """Only the head is trained."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_layer4(model):
    for p in model.layer4.parameters():
        p.requires_grad = True


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def load_resnet(path, device=DEFAULT_DEVICE):
    """Weights saved by the notebook or by this module (step B: one linear head; step C: dropout + linear)."""
    state = torch.load(path, map_location="cpu")
    model = build_resnet(dropout=0.5 if "fc.1.weight" in state else 0.0, pretrained=False)
    model.load_state_dict(state)
    return model.to(device).eval()


# 2. Training
def evaluate(model, loader, loss_fn, device=DEFAULT_DEVICE):
    """(mean loss, accuracy) of the model on one loader."""
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


def augment(xb, shift=12):
    """Random left-right flip (per image) + random crop (same shift for the batch). No colour change:
    colour is part of the style."""
    flip = torch.rand(len(xb), device=xb.device) < 0.5
    xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(-1), xb)
    size = xb.shape[-1]
    xb = F.pad(xb, (shift, shift, shift, shift), mode="reflect")
    i, j = torch.randint(0, 2 * shift + 1, (2,)).tolist()
    return xb[..., i:i + size, j:j + size]


def train_one_epoch(model, loader, loss_fn, optimizer, device=DEFAULT_DEVICE, regularised=False):
    model.eval()                       # BatchNorm keeps its ImageNet statistics
    if regularised:
        model.fc.train()               # ...but the dropout of the head must be on
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if regularised:
            xb = augment(xb)
        optimizer.zero_grad()
        loss_fn(model(xb), yb).backward()
        optimizer.step()


def fit(model, optimizer, n_epochs, loaders, loss_fn, device=DEFAULT_DEVICE, regularised=False,
        scheduler=None, seed=1, verbose=False):
    """Train n_epochs, record train/validation loss and accuracy, keep the best validation epoch."""
    train_loader, val_loader, train_eval_loader = loaders
    torch.manual_seed(seed)            # same order of the mini-batches for the same seed
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(n_epochs):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, regularised)
        tl, ta = evaluate(model, train_eval_loader, loss_fn, device)
        vl, va = evaluate(model, val_loader, loss_fn, device)
        for key, value in zip(history, [tl, ta, vl, va]):
            history[key].append(value)
        if scheduler is not None:
            scheduler.step()
        if va > best_acc:
            best_acc, best_epoch = va, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch + 1:2d}  train loss {tl:.3f}  val loss {vl:.3f}  val acc {va:.3f}")
    model.load_state_dict(best_state)
    if verbose:
        print(f"kept epoch {best_epoch} (best val acc {best_acc:.3f})")
    return history


# 3. The three steps
def train_step_b(loaders, seed=0, epochs_head=5, epochs_ft=5, pretrained=True, device=DEFAULT_DEVICE, verbose=False):
    """Step A (head only) then step B (layer4 + head). Seed 0 reproduces resnet.ipynb.
    Returns (model, history of step A, history of step B)."""
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(seed)
    model = build_resnet(pretrained=pretrained)
    freeze_backbone(model)
    model = model.to(device)
    history_a = fit(model, torch.optim.Adam(model.fc.parameters(), lr=3e-3), epochs_head, loaders, loss_fn,
                    device, seed=seed + 1, verbose=verbose)
    unfreeze_layer4(model)
    optimizer = torch.optim.Adam([{"params": model.layer4.parameters(), "lr": 1e-4},
                                  {"params": model.fc.parameters(), "lr": 1e-3}])
    history_b = fit(model, optimizer, epochs_ft, loaders, loss_fn, device, seed=seed + 1, verbose=verbose)
    return model, history_a, history_b


def train_step_c(loaders, seed=0, epochs_head=3, epochs_ft=15, dropout=0.5, weight_decay=1e-2,
                 pretrained=True, device=DEFAULT_DEVICE, verbose=False):
    """Step C: augmentation + dropout + AdamW weight decay + cosine learning rate. Returns (model, history)."""
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(seed)
    model = build_resnet(dropout=dropout, pretrained=pretrained)
    freeze_backbone(model)
    model = model.to(device)
    fit(model, torch.optim.AdamW(model.fc.parameters(), lr=3e-3, weight_decay=weight_decay), epochs_head,
        loaders, loss_fn, device, regularised=True, seed=seed + 1, verbose=verbose)
    unfreeze_layer4(model)
    optimizer = torch.optim.AdamW([{"params": model.layer4.parameters(), "lr": 1e-4},
                                   {"params": model.fc.parameters(), "lr": 1e-3}], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_ft)
    history = fit(model, optimizer, epochs_ft, loaders, loss_fn, device, regularised=True, scheduler=scheduler,
                  seed=seed + 1, verbose=verbose)
    return model, history


# 4. Cost on a kiosk
def cpu_latency_ms(model, n=50, shape=(1, 3, 128, 128), threads=1):
    """Milliseconds to classify ONE image on the CPU (threads=1: one core, like a modest kiosk CPU)."""
    model = model.cpu().eval()
    x = torch.randn(*shape)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
    torch.set_num_threads(old_threads)
    return (time.perf_counter() - t0) / n * 1000


def main():
    from museum_guide.data import get_dataloaders

    p = argparse.ArgumentParser(description="Train ResNet-18 step A+B or step C and save the weights.")
    p.add_argument("--step", choices=["B", "C"], default="B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/resnet18.pt")
    p.add_argument("--epochs-head", type=int, default=None, help="default: 5 (step B) or 3 (step C)")
    p.add_argument("--epochs-ft", type=int, default=None, help="default: 5 (step B) or 15 (step C)")
    p.add_argument("--no-pretrained", action="store_true", help="random weights (only for a quick test)")
    args = p.parse_args()

    device = DEFAULT_DEVICE
    train_loader, val_loader, test_loader = get_dataloaders("resnet", batch_size=64)
    loaders = (train_loader, val_loader, DataLoader(train_loader.dataset, batch_size=250))
    t0 = time.time()
    if args.step == "B":
        model, _, history = train_step_b(loaders, args.seed, args.epochs_head or 5, args.epochs_ft or 5,
                                         not args.no_pretrained, device, verbose=True)
    else:
        model, history = train_step_c(loaders, args.seed, args.epochs_head or 3, args.epochs_ft or 15,
                                      pretrained=not args.no_pretrained, device=device, verbose=True)
    _, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)                 # weights stay outside Git (*.pt is ignored)
    print(f"step {args.step}, seed {args.seed}: best val acc {max(history['val_acc']):.3f} | "
          f"test acc {test_acc:.3f} | {time.time() - t0:.0f} s | saved {args.out}")


if __name__ == "__main__":
    main()
