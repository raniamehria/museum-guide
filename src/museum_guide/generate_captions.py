"""Generate per-image captions with BLIP for all 1 500 artworks.

Run once on Colab (T4 GPU, ~10 min) before building the artwork index.
Saves data/artwork_captions.json — one caption per image path.

Why BLIP: our previous approach gave every image the same caption as all other
images by the same artist. BLIP reads the actual pixels and produces an
image-specific description, so "stormy sea" can now match the Great Wave
directly instead of all Hokusai paintings equally.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

BLIP_MODEL    = "Salesforce/blip-image-captioning-base"
METADATA_PATH = Path("data/wikiart_style5/metadata.csv")
IMG_DIR       = Path("data/wikiart_style5")
CAPTIONS_PATH = Path("data/artwork_captions.json")


def generate_captions(
    metadata_path: Path = METADATA_PATH,
    img_dir: Path = IMG_DIR,
    captions_path: Path = CAPTIONS_PATH,
    batch_size: int = 16,
) -> list[dict]:
    """Caption all 1 500 artworks with BLIP and save to JSON.

    Each entry: {"path": "Baroque/s54_r0037.jpg", "caption": "..."}
    Caption = BLIP visual description prefixed with style/artist context so
    MiniLM can still match style/artist queries as well as visual queries.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model     = (
        BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
        .to(device)
        .eval()
    )

    meta    = pd.read_csv(metadata_path)
    results = []

    for i in range(0, len(meta), batch_size):
        batch  = meta.iloc[i : i + batch_size]
        images = [
            Image.open(img_dir / row["path"]).convert("RGB")
            for _, row in batch.iterrows()
        ]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60)

        for j, (_, row) in enumerate(batch.iterrows()):
            blip_cap = processor.decode(out[j], skip_special_tokens=True)
            artist   = (
                row["artist"].replace("-", " ").title()
                if row["artist"] != "Unknown Artist"
                else "unknown artist"
            )
            # Prefix keeps style/artist queries working; BLIP suffix adds
            # image-specific visual content (waves, colours, figures, etc.)
            full_caption = (
                f"A {row['style']} painting by {artist}. {blip_cap}."
            )
            results.append({"path": row["path"], "caption": full_caption})

        print(f"  captioned {min(i + batch_size, len(meta))}/{len(meta)}")

    Path(captions_path).parent.mkdir(exist_ok=True)
    Path(captions_path).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(results)} captions → {captions_path}")
    return results


if __name__ == "__main__":
    generate_captions()
