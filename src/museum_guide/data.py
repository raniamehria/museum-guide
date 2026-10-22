"""Data pipeline of the museum guide: WikiArt subset -> train / validation / test DataLoaders."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

DATA_DIR = Path("data/wikiart_style5")
SPLIT_FILE = Path("splits/split.csv")
STATS_FILE = Path("splits/stats.json")
N_TRAIN, N_VAL = 1000, 250            # the test set gets the remaining 250 images
UNKNOWN_ARTIST = "Unknown Artist"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# 1. Load the images (ImageFolder, as in make_wikiart_subset.py; one tensor as in TP07 Part D)
def load_images(data_dir=DATA_DIR):
    ds = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    X = torch.stack([x for x, _ in ds])          # (1500, 3, 128, 128), in [0, 1]
    y = torch.tensor(ds.targets)                 # = the label_idx column of metadata.csv
    meta = pd.read_csv(data_dir / "metadata.csv")
    paths = [Path(p).relative_to(data_dir).as_posix() for p, _ in ds.samples]
    assert paths == list(meta["path"]), "ImageFolder and metadata.csv must list the images in the same order"
    assert ds.targets == meta["label_idx"].tolist(), "ImageFolder indices must equal label_idx"
    return X, y, meta


# 2. Train / validation / test split (TP04 Ex. 12), grouped by artist (session 8 checklist)
def make_split(meta, seed=0):
    groups = [a if a != UNKNOWN_ARTIST else f"unknown_{i}" for i, a in enumerate(meta["artist"])]
    names = sorted(set(groups))
    size = Counter(groups)                       # number of images of each group
    perm = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed))
    split_of, n_train, n_val = {}, 0, 0
    for g in perm.tolist():
        name = names[g]
        if n_train + size[name] <= N_TRAIN:
            split_of[name] = "train"
            n_train += size[name]
        elif n_val + size[name] <= N_VAL:
            split_of[name] = "val"
            n_val += size[name]
        else:
            split_of[name] = "test"
    return pd.DataFrame({"path": meta["path"], "split": [split_of[g] for g in groups]})


# 4. Checks (TP04 Ex. 12 asserts)
def check_split(split, meta):
    counts = split["split"].value_counts()
    assert counts["train"] == N_TRAIN and counts["val"] == N_VAL, f"sizes should be 1000 / 250 / 250, found {dict(counts)}"
    assert split["path"].is_unique, "an image appears twice"
    named = meta["artist"] != UNKNOWN_ARTIST
    sides = split[named].groupby(meta.loc[named, "artist"])["split"].nunique()
    assert (sides == 1).all(), f"artists on two sides: {list(sides[sides > 1].index)}"
    print(pd.crosstab(meta["label"], split["split"]))
    print("Split validated!")


# 5. MLP input: 64 px (TP07 prepare), standardised with the training mean and std (TP05 Ex. 2)
def prepare_mlp(X, train_mask):
    X = F.interpolate(X, size=64, mode="bilinear", align_corners=False)   # (1500, 3, 64, 64)
    mean = X[train_mask].mean().item()
    std = X[train_mask].std().item()
    return (X - mean) / std, mean, std


# 6. ResNet input: ImageNet normalisation (TP07 prepare)
def prepare_resnet(X):
    return (X - IMAGENET_MEAN) / IMAGENET_STD


# 3 + 7. Saved split and DataLoaders (TP05 Ex. 3)
def get_dataloaders(model_type="mlp", batch_size=64):
    X, y, meta = load_images()
    if SPLIT_FILE.exists():
        split = pd.read_csv(SPLIT_FILE)              # everybody uses the same split
    else:
        split = make_split(meta)
        SPLIT_FILE.parent.mkdir(exist_ok=True)
        split.to_csv(SPLIT_FILE, index=False)
    assert (split["path"] == meta["path"]).all(), "split.csv does not match metadata.csv"
    check_split(split, meta)
    train = torch.tensor((split["split"] == "train").values)
    val = torch.tensor((split["split"] == "val").values)
    test = torch.tensor((split["split"] == "test").values)

    if model_type == "mlp":
        X, mean, std = prepare_mlp(X, train)
        STATS_FILE.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    elif model_type == "resnet":
        X = prepare_resnet(X)
    else:
        raise ValueError("model_type must be 'mlp' or 'resnet'")

    torch.manual_seed(0)
    train_loader = DataLoader(TensorDataset(X[train], y[train]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[val], y[val]), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X[test], y[test]), batch_size=batch_size)
    return train_loader, val_loader, test_loader


# 8. Sanity check (TP05 Ex. 3): python -m src.data
if __name__ == "__main__":
    for model_type in ["mlp", "resnet"]:
        train_loader, val_loader, test_loader = get_dataloaders(model_type)
        xb, yb = next(iter(train_loader))
        print(model_type, "one batch:", tuple(xb.shape), tuple(yb.shape), "| batches per epoch:", len(train_loader))