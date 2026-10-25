"""Tâche 4: cost of each model on a kiosk (CPU time per image, size, parameters).

All the models are measured the same way: one 128 x 128 image in [0, 1] (what the camera gives
after resizing), the whole pipeline (preparing the image + the model), on 1 CPU core.
Same idea as cpu_latency_ms in resnet.ipynb, but with the preprocessing included.
"""
import os
import pickle
import time

import numpy as np
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def torch_size_mb(model, path):
    # size of the saved weights = what the kiosk has to store
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    return os.path.getsize(path) / 1e6


def sklearn_size_mb(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return os.path.getsize(path) / 1e6


def cpu_latency_ms(predict_one, n_runs=50, n_warmup=5, threads=1):
    """Median time (ms) of predict_one(image) for ONE image, on `threads` cores."""
    image = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(0))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)                 # 1 core = a modest kiosk CPU
    try:
        with torch.no_grad():
            for _ in range(n_warmup):              # the first calls are slower, not timed
                predict_one(image)
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                predict_one(image)
                times.append(time.perf_counter() - t0)
    finally:
        torch.set_num_threads(old_threads)
    return float(np.median(times) * 1000)          # median: one slow run does not change the result
