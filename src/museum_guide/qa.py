"""Bonus — Natural-language Q&A for the museum guide.

Requires: transformers>=4.40, torch>=2.6  (run on Colab, same as resnet.ipynb).
Install in Colab: !pip install transformers sentence-transformers

Step 1 — build_artwork_index(): run once offline, embeds all 1 500 artwork
          captions with MiniLM and saves them to checkpoints/artwork_index.pt.
Step 2 — answer(): encodes a visitor question, ranks artworks by cosine
          similarity, applies a confidence guardrail (tau), returns top-k.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN     = 120
CAPTIONS_PATH = Path("data/artwork_captions.json")
METADATA_PATH = Path("data/wikiart_style5/metadata.csv")
INDEX_PATH    = Path("checkpoints/artwork_index.pt")
TAU = 0.30


def _mean_pool(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool the last hidden states, weighted by the attention mask."""
    token_embeddings = model_output.last_hidden_state          # (B, T, H)
    mask = attention_mask.unsqueeze(-1).float()                # (B, T, 1)
    summed = (token_embeddings * mask).sum(dim=1)              # (B, H)
    return F.normalize(summed / mask.sum(dim=1).clamp(min=1e-9), dim=-1)


def _encode(texts: list[str] | str, tokenizer, model) -> torch.Tensor:
    """Encode one string or a list of strings → L2-normalised (N, 384) tensor."""
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(
        texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**enc)
    return _mean_pool(out, enc["attention_mask"])


def build_artwork_index(
    captions_path: Path = CAPTIONS_PATH,
    metadata_path: Path = METADATA_PATH,
    index_path: Path = INDEX_PATH,
) -> list[dict]:
    """Embed all 1 500 artwork captions and save the index.

    Each entry: {path, style, artist, caption, embedding (384-dim, normalised)}.
    Run once offline on Colab; the demo notebook loads the saved index.
    """
    caption_lookup = {
        (c["style"], c["artist"]): c["caption"]
        for c in json.loads(Path(captions_path).read_text(encoding="utf-8"))
    }

    meta = pd.read_csv(metadata_path)
    meta["caption"] = meta.apply(
        lambda r: caption_lookup[(r["style"], r["artist"])], axis=1
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).eval()

    captions   = meta["caption"].tolist()
    batch_size = 64
    all_embs   = []
    for i in range(0, len(captions), batch_size):
        batch_embs = _encode(captions[i : i + batch_size], tokenizer, model)
        all_embs.append(batch_embs)
        print(f"  encoded {min(i + batch_size, len(captions))}/{len(captions)}")
    embeddings = torch.cat(all_embs, dim=0)                    # (1500, 384)

    index = [
        {
            "path":      row["path"],
            "style":     row["style"],
            "artist":    row["artist"],
            "caption":   row["caption"],
            "embedding": embeddings[i],
        }
        for i, (_, row) in enumerate(meta.iterrows())
    ]

    Path(index_path).parent.mkdir(exist_ok=True)
    torch.save(index, index_path)
    print(f"Saved {len(index)}-artwork index → {index_path}")
    return index


def answer(
    question: str,
    index: list[dict],
    tokenizer,
    model,
    tau: float = TAU,
    top_k: int = 3,
) -> dict:
    """Retrieve the top_k artworks most similar to the visitor's question.

    Applies a guardrail: if the best cosine score < tau the guide says
    'not sure' instead of returning a potentially irrelevant result.
    tokenizer and model are passed in so the caller loads them once.
    """
    q_emb  = _encode(question, tokenizer, model).squeeze(0)    # (384,)
    stack  = torch.stack([art["embedding"] for art in index])  # (N, 384)
    scores = (stack @ q_emb).tolist()                          # cosine = dot (normalised)

    ranked    = sorted(zip(scores, index), key=lambda x: -x[0])
    top_score = ranked[0][0]

    if top_score < tau:
        return {
            "answered":  False,
            "top_score": round(top_score, 3),
            "message": (
                "I'm not sure which painting you mean. "
                "Our collection includes Baroque, Cubism, Impressionism, "
                "Pop Art, and Ukiyo-e."
            ),
        }

    return {
        "answered":  True,
        "top_score": round(top_score, 3),
        "results": [
            {
                "rank":   rank + 1,
                "score":  round(score, 3),
                "path":   art["path"],
                "style":  art["style"],
                "artist": art["artist"],
            }
            for rank, (score, art) in enumerate(ranked[:top_k])
        ],
    }


if __name__ == "__main__":
    build_artwork_index()
