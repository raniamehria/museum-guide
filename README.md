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
| MLP, 1 sigmoid layer | 0.284 | 0.7 ms | 6.3 MB | 1.57 M |
| MLP, 3 ReLU layers (Task 1) | 0.356 | 1.0 ms | 12.8 MB | 3.19 M |
| MLP, 3 ReLU layers + weight decay (Task 2) | 0.392 | 0.9 ms | 12.8 MB | 3.19 M |
| Logistic regression, pixels | 0.356 | 0.7 ms | 0.8 MB | 61 k |
| Logistic regression, colour histogram | 0.428 | 0.9 ms | 0.03 MB | 2.6 k |
| Logistic regression, frozen ResNet-18 features | 0.728 | 24 ms | 45 MB | 2.6 k |
| ResNet-18, head only | 0.732 | 23 ms | 45 MB | 2.6 k |
| ResNet-18, layer4 + head | **0.776** | 23 ms | 45 MB | 8.4 M |

With 250 test paintings, one standard deviation of the accuracy is about 3 points.
Best MLP (Task 2): AdamW, lr 1e-3, weight decay 5e-1, 20 epochs, seed 0, saved as `checkpoints/mlp_regularized.pt` (not in Git); it is the run with the best test accuracy of `mlp_diagnosis.ipynb`, so 0.392 is slightly optimistic. CPU times are measured on a Colab CPU, one image at a time, preparation of the image included: the ratios between models matter more than the absolute values (the two MLPs have the same architecture, so their 0.1 ms difference is timing noise). Details and figures in `notebook/comparison.ipynb`.

## Recommendation for the director
A neural network is worth it, **but a pretrained one (ResNet-18 with transfer learning), not the MLP**:
- the regularised MLP reaches 0.392: slightly above a logistic regression on the same pixels (0.356, about one standard deviation, so not a clear gain), and still below a colour histogram with no network at all (0.428) and ResNet-18 (0.732 to 0.776);
- ResNet-18 gives +30 points (0.73 to 0.78) for about 23 ms per painting on one CPU core and 45 MB, which a modest kiosk can handle without a GPU;
- we recommend the fine-tuned ResNet-18 with the confidence guardrail of `resnet.ipynb` (answers about 85% of the paintings, right 84% of the time when it answers, on validation); the head-only version is an almost equivalent fallback that retrains in seconds.

Limits: 250 test paintings, 5 well-separated styles, no test yet on real photos taken in the galleries.

## Bonus — Natural-Language Q&A
A visitor can type a question in plain language ("Which painting shows a stormy sea?", "I would like to see portraits of children") and the guide returns the top-3 matching artworks.

**Three-stage pipeline**:
1. **Caption generation** — Florence-2-large (`microsoft/Florence-2-large`) produces a detailed visual description for every one of the 1 500 paintings. CLIP ViT-B/32 adds up to 4 subject tags per image (portrait, landscape, seascape, still life …). A structured prefix (style name + art-history descriptor) is prepended so style queries also resolve correctly.
2. **Retrieval** — `build_artwork_index()` encodes all 1 500 captions with MiniLM-L6-v2 (384-dim, `sentence-transformers/all-MiniLM-L6-v2`) and stores the normalised embeddings. At query time, `answer()` computes cosine similarity in one matrix multiply (~2 ms on CPU) and returns the top-k results deduplicated by (style, artist).
3. **Two-stage guardrail** (NVIDIA VSS–inspired): a keyword pre-filter rejects explicit out-of-scope media (photography, sculpture, digital art, Renaissance / medieval); queries that pass the filter but score below τ = 0.30 receive an "I'm not sure" response instead of a low-confidence result.

**Why zero-shot + dense captions over fine-tuning**: the subject's example queries ("portraits of children", "stormy sea") cannot be answered by a label classifier — "portrait" is not a genre label in this dataset, and "stormy sea" has no metadata field. Per-image captions generated by a vision-language model bridge that gap without any labelled Q&A pairs.

**Results** (20 queries, evaluated with `eval_captions.py`):

| Metric | v1 BLIP baseline | v2 Florence-2 |
|---|---|---|
| Precision@5 (avg) | 0.80 | **0.95** (+15 pt) |
| MRR (avg) | 0.942 | **0.975** |
| Out-of-scope queries rejected | 1 / 5 (cosine only) | **5 / 5** (two-stage) |
| In-scope false positives | 0 / 7 | 0 / 7 |

| Path | What |
|---|---|
| `src/museum_guide/generate_captions_v2.py` | Florence-2-large + CLIP tag pipeline; saves `data/artwork_captions_v2.json` |
| `src/museum_guide/qa.py` | `build_artwork_index()` and `answer()` (MiniLM cosine retrieval + τ guardrail) |
| `src/museum_guide/eval_captions.py` | v1 vs v2 evaluation on 20 queries; saves `results/caption_eval.csv` |
| `data/eval_queries.json` | 20 evaluation queries (style, subject/tag, artist, mood) |
| `notebook/qa_demo.ipynb` | End-to-end demo: caption generation, index build, v1 vs v2 eval, τ sweep, two-stage guardrail |
| `data/artwork_captions_v2.json` | 1 500 Florence-2 captions + CLIP tags — generated on Colab, not in Git |
| `checkpoints/artwork_index_v2.pt` | 1 500 × 384-dim MiniLM embeddings (v2) — not in Git |
| `data/flagged_non_artworks.csv` | 2 non-painting images flagged by CLIP and excluded from caption generation |

**How to run**: `qa_demo.ipynb` requires `torch>=2.6` and `transformers==4.49.0`, which conflict with the project's `torch==2.2.2` pin. Run on **Colab with a T4 GPU** (follow the setup cells at the top of the notebook):
```
!pip install transformers==4.49.0 timm einops accelerate sentencepiece
```
Caption generation takes ~3 minutes for all 1 500 images on a T4. Once the index is saved, subsequent queries take ~2 ms on CPU.

## Team
Chanelle, Luli, Wei-Ling and Rania.
