# museum-guide
Final project – Neural Networks & Deep Learning Foundations (MSA-DATB09-03, Albert School).

A (fictional) fine-arts museum wants a guide on visitors' phones and on gallery kiosks: the visitor points the camera at a painting and the guide recognises its **style** and tells the story of the movement. The kiosks run on modest CPUs, so the director also asks: **is a neural network worth it, or would a simpler classical model do the job?**

## Data
A WikiArt subset built with the course script `scripts/make_wikiart_subset.py`: **5 styles** (Baroque, Cubism, Impressionism, Pop Art, Ukiyo-e) x **300 paintings**, 128 x 128 px, at most 50 paintings per artist (the exact command is in `splits/subset_info.json`).

- The images are **not in Git** (WikiArt licence). Build them with the script, or unzip the group's copy into `data/wikiart_style5/`.
- One shared split for everybody, `splits/split.csv`: 1 000 train / 250 validation / 250 test, **grouped by artist** (a painter is never in two sets), so every model is tested on the same 250 paintings.

## Repository
| Path | What | Who |
|---|---|---|
| `src/museum_guide/data.py` | loading, split by artist, DataLoaders (MLP: 64 px standardised, ResNet: ImageNet normalisation) | Chanelle |
| `src/museum_guide/mlp.py`, `train.py` | MLP in PyTorch primitives, training loop with best-epoch checkpoint | Chanelle, Luli |
| `notebook/mlp.ipynb` | Tâche 1: MLP, learning curves | Chanelle, Luli |
| `notebook/mlp_diagnosis.ipynb` | Tâche 2: diagnosis, dropout / weight decay / early stopping, learning rate | Luli |
| `notebook/resnet.ipynb` | Tâche 3: ResNet-18 transfer learning (frozen backbone, then `layer4`), guardrail and similar-artwork search | Wei-Ling |
| `src/museum_guide/baseline.py`, `notebook/baseline.ipynb` | Tâche 4: classical baseline, logistic regression on pixels, colour histograms and frozen ResNet features | Rania |
| `src/museum_guide/benchmark.py`, `notebook/comparison.ipynb` | Tâche 4: CPU time / size of every model, final table and recommendation | Rania |
| `reports/` | measured results of Tâche 4 (`baseline_metrics.json`, `comparison.csv`, `accuracy_vs_cpu.png`) | Rania |

## Run it
Locally (Python 3.11, [uv](https://docs.astral.sh/uv/)): `uv sync`, put the images in `data/wikiart_style5/`, then run the notebooks in this order: `mlp.ipynb`, `mlp_diagnosis.ipynb`, `resnet.ipynb`, `baseline.ipynb`, `comparison.ipynb` (it needs the models saved by `baseline.ipynb`).

On Colab: `resnet.ipynb` and `baseline.ipynb` start with a "Colab only" cell that clones the repo and loads the images; the last cell of `baseline.ipynb` also runs `comparison.ipynb` in the same session.

## Results (same 250 test paintings)
| Model | Test accuracy | CPU time / image (1 core) | Size | Trained parameters |
|---|---|---|---|---|
| chance | 0.200 | – | – | – |
| MLP, 1 sigmoid layer | 0.284 | 0.8 ms | 6.3 MB | 1.57 M |
| MLP, 3 ReLU layers (Task 1) | 0.356 | 1.0 ms | 12.8 MB | 3.19 M |
| MLP, 3 ReLU layers + weight decay (Task 2) | 0.392 | 1.0 ms | 12.8 MB | 3.19 M |
| Logistic regression, pixels | 0.356 | 0.8 ms | 0.8 MB | 61 k |
| Logistic regression, colour histogram | 0.428 | 0.9 ms | 0.03 MB | 2.6 k |
| Logistic regression, frozen ResNet-18 features | 0.728 | 25 ms | 45 MB | 2.6 k |
| ResNet-18, head only | 0.732 | 24 ms | 45 MB | 2.6 k |
| ResNet-18, layer4 + head | **0.776** | 24 ms | 45 MB | 8.4 M |

With 250 test paintings, one standard deviation of the accuracy is about 3 points.
Best MLP (Task 2): AdamW, lr 1e-3, weight decay 5e-1, 20 epochs, seed 0, saved as `checkpoints/mlp_regularized.pt` (not in Git); it is the run with the best test accuracy of `mlp_diagnosis.ipynb`, so 0.392 is slightly optimistic. CPU times are measured on a Colab CPU, one image at a time, preparation of the image included: the ratios between models matter more than the absolute values. Details and figures in `notebook/comparison.ipynb`.

## Recommendation for the director
A neural network is worth it, **but a pretrained one (ResNet-18 with transfer learning), not the MLP**:
- the regularised MLP reaches 0.392: slightly above a logistic regression on the same pixels (0.356, about one standard deviation, so not a clear gain), and still below a colour histogram with no network at all (0.428) and ResNet-18 (0.732 to 0.776);
- ResNet-18 gives +30 points (0.73 to 0.78) for about 25 ms per painting on one CPU core and 45 MB, which a modest kiosk can handle without a GPU;
- we recommend the fine-tuned ResNet-18 with the confidence guardrail of `resnet.ipynb` (answers about 85% of the paintings, right 84% of the time when it answers, on validation); the head-only version is an almost equivalent fallback that retrains in seconds.

Limits: 250 test paintings, 5 well-separated styles, no test yet on real photos taken in the galleries.

## Team
Chanelle, Luli, Wei-Ling and Rania.
