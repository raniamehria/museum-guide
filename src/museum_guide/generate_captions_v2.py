"""Improved per-image caption generation for the museum-guide Q&A bonus (v2).

Design choices (each defensible in Q&A):
  Structured prefix   — style → artist → genre sentence anchors each part of
                        the caption to a different query type in MiniLM's
                        embedding space.  v1 gave every image the same
                        artist-level boilerplate, so rankings inside a style
                        were near-random.
  Fixed style descriptor — 8-12 art-history keywords per style (brushwork,
                        palette, composition).  Short enough not to dominate;
                        provides the visual vocabulary visitors use that BLIP
                        alone omits ("impressionist colours", "chiaroscuro").
  CLIP subject tags   — zero-shot, no manual labelling.  Adds subject nouns
                        (portrait, seascape, child) that are decisive for the
                        subject queries from the brief.
  Non-painting flag   — CLIP binary (painting vs photo/poster).  Saved to CSV
                        so the team reviews; nothing is deleted automatically.
  Beam search + post-processing — no_repeat_ngram_size=3 and
                        repetition_penalty=1.3 reduce looping in BLIP/Florence;
                        a regex pass removes any remaining repeated 4-grams and
                        sentences that contain wrong-medium words.
  Checkpoint every 100 images — Colab session timeout does not lose work.

Captioner candidates (choose with --model):
  blip-large  : Salesforce/blip-image-captioning-large, conditional prompt
                "a painting of".  ~400 MB, batch 16, ~8 min on T4.
  florence2   : microsoft/Florence-2-large, task <MORE_DETAILED_CAPTION>.
                ~830 MB, batch 4, ~20 min on T4.  Recommended default.

Colab install (T4 runtime, run before this script):
  !pip install -q transformers timm einops pillow pandas

Usage:
  python -m museum_guide.generate_captions_v2 --analyse        # v1 stats
  python -m museum_guide.generate_captions_v2 --compare        # side-by-side on 20 images
  python -m museum_guide.generate_captions_v2                  # full run (florence2)
  python -m museum_guide.generate_captions_v2 --model blip-large
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA_PATH = Path("data/wikiart_style5/metadata.csv")
IMG_DIR       = Path("data/wikiart_style5")
CAPTIONS_PATH = Path("data/artwork_captions.json")       # v1 (BLIP-base, artist-level)
CAPTIONS_V2   = Path("data/artwork_captions_v2.json")   # v2 output
CHECKPOINT    = Path("data/artwork_captions_v2_checkpoint.json")
FLAGGED_CSV   = Path("data/flagged_non_artworks.csv")

# ---------------------------------------------------------------------------
# Style vocabulary  (human-readable, grammatically correct)
# Design: visitors query by movement name or adjective; both forms must appear
# so MiniLM can match either ("Impressionist art" and "Impressionism").
# ---------------------------------------------------------------------------
STYLE_SENTENCES = {
    "Baroque":       "A Baroque painting",
    "Cubism":        "A Cubist painting (Cubism)",
    "Impressionism": "An Impressionist painting (Impressionism)",
    "Pop_Art":       "A Pop Art work",
    "Ukiyo_e":       "A Ukiyo-e Japanese woodblock print or painting",
}

# 8-12 words of art-history vocabulary, fixed per style.
# Short enough not to swamp the image-specific part of the caption.
STYLE_DESCRIPTORS = {
    "Baroque":       "dramatic chiaroscuro lighting, rich warm colours, emotional intensity",
    "Cubism":        "geometric facets, multiple viewpoints, muted earthy palette, flat planes",
    "Impressionism": "loose visible brushstrokes, natural light, vivid outdoor colours, everyday scenes",
    "Pop_Art":       "bold flat colours, hard edges, mass-culture imagery, graphic design influence",
    "Ukiyo_e":       "flat colour areas, bold black outlines, elegant line, decorative patterning",
}

GENRE_READABLE = {
    "still_life":         "still life",
    "genre_painting":     "genre scene",
    "landscape":          "landscape",
    "portrait":           "portrait",
    "religious_painting": "religious scene",
    "cityscape":          "cityscape",
    "abstract":           "abstract",
    "figurative":         "figurative",
}

# ---------------------------------------------------------------------------
# CLIP zero-shot subject tags
# Design: each key is the display label added to the caption; each value is
# the text prompt fed to CLIP.  Prompts are phrased as captions ("a …") to
# match CLIP's training distribution.
# ---------------------------------------------------------------------------
CLIP_TAG_PROMPTS: dict[str, str] = {
    "portrait":        "a portrait of a person looking at the viewer",
    "child":           "a child or baby in the scene",
    "woman":           "a woman or women in the scene",
    "man":             "a man or men in the scene",
    "group of people": "a group of people or crowd",
    "landscape":       "an outdoor landscape with sky and terrain",
    "seascape":        "a sea, ocean, or stormy crashing waves",
    "river":           "a river, canal or waterway",
    "city street":     "a city street or urban scene with buildings",
    "interior":        "an indoor room or interior scene",
    "still life":      "still life objects — fruit, bottles or flowers on a table",
    "flowers":         "flowers or a floral arrangement",
    "animals":         "animals — birds, horses or other creatures",
    "boats":           "boats, ships or sailing vessels on water",
    "religious scene": "a religious or biblical scene with saints or angels",
    "night":           "a nighttime or dark evening scene",
    "snow":            "snow or a winter landscape",
}
CLIP_TAG_THRESHOLD = 0.22   # absolute cosine similarity (calibrated on ViT-B/32)
CLIP_MAX_TAGS      = 4

# CLIP non-painting detection
NON_PAINTING_PROMPTS = [
    "a photograph, poster, book cover, screenshot or digital image",
    "a painting, drawing, woodblock print or artwork on canvas or paper",
]
NON_PAINTING_MARGIN = 0.03  # flag if "photo" prompt beats "painting" prompt by this margin

# Beam-search parameters applied to both captioners for reproducibility
GENERATE_KWARGS: dict = {
    "max_new_tokens":      80,
    "num_beams":           3,
    "no_repeat_ngram_size": 3,
    "repetition_penalty":  1.3,
}

# Words that indicate BLIP / Florence mistook the medium
WRONG_MEDIA: set[str] = {
    "photo", "photograph", "photographed", "photography",
    "album cover", "album", "cover", "poster", "screenshot",
    "digital art", "3d render", "render", "animation",
    "book cover", "printed on",
}


# ---------------------------------------------------------------------------
# A.  Diagnostic: analyse v1 captions
# ---------------------------------------------------------------------------
def analyse_v1(captions_path: Path = CAPTIONS_PATH) -> None:
    """Print statistics that confirm the five diagnosed v1 problems."""
    raw  = json.loads(Path(captions_path).read_text(encoding="utf-8"))
    texts = [c["caption"] for c in raw]
    n     = len(texts)

    unique = len(set(texts))
    ua     = sum(1 for t in texts if "unknown artist" in t.lower())
    media  = {w: sum(1 for t in texts if w in t.lower()) for w in WRONG_MEDIA}
    media  = {k: v for k, v in media.items() if v}

    # Repeated-bigram detection
    def has_repeat(t: str) -> bool:
        words = t.split()
        for i in range(2, len(words) - 1):
            if words[i:i+2] == words[i-2:i]:
                return True
        return False
    repeated = sum(1 for t in texts if has_repeat(t))

    avg_words = sum(len(t.split()) for t in texts) / n

    print("=" * 60)
    print(f"V1 caption analysis  ({n} captions)")
    print("=" * 60)
    print(f"  Unique captions          : {unique:4d} / {n}  ({100*unique/n:.0f}%)")
    print(f"  'unknown artist'         : {ua:4d} / {n}  ({100*ua/n:.0f}%)")
    print(f"  Avg caption length       : {avg_words:.0f} words")
    print(f"  Captions with repetition : {repeated:4d}")
    if media:
        print(f"  Wrong-medium words       : {media}")
    for style in ["A Impressionism", "A Ukiyo_e", "A Pop_Art"]:
        n_bad = sum(1 for t in texts if style in t)
        if n_bad:
            print(f"  Ungrammatical prefix '{style}': {n_bad}")
    print()


# ---------------------------------------------------------------------------
# B-helper.  Post-processing
# ---------------------------------------------------------------------------
def _remove_repeated_4grams(text: str) -> str:
    """Delete any 4-gram that immediately repeats itself."""
    words = text.split()
    out, i = [], 0
    while i < len(words):
        if i + 4 <= len(words) and i >= 4 and words[i:i+4] == words[i-4:i]:
            i += 4
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _postprocess(text: str) -> str:
    """Remove wrong-medium sentences and repeated phrases from a caption."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    clean = [
        s for s in sentences
        if not any(w in s.lower() for w in WRONG_MEDIA)
    ]
    text = " ".join(clean) if clean else text
    text = _remove_repeated_4grams(text)
    return text.strip()


# ---------------------------------------------------------------------------
# B-helper.  Caption assembly
# Design: each sentence answers a different query type.  Total target < 60
# words so MiniLM's 256-token window is not wasted on boilerplate.
# ---------------------------------------------------------------------------
def _build_caption(row: pd.Series, visual: str, tags: list[str]) -> str:
    """Assemble a structured caption from metadata + visual + CLIP tags."""
    parts: list[str] = []

    # 1. Style sentence (grammatically correct, both adjective and noun form)
    parts.append(STYLE_SENTENCES.get(row["style"], f"A {row['style']} painting"))

    # 2. Artist — only when known; never write "unknown artist"
    if row["artist"] != "Unknown Artist":
        readable = row["artist"].replace("-", " ").title()
        parts[-1] += f" by {readable}"

    parts[-1] += "."

    # 3. Genre — only when informative
    genre = row.get("genre", "Unknown Genre")
    if genre and genre.lower() not in ("unknown genre", "unknown", ""):
        parts.append(f"Genre: {GENRE_READABLE.get(genre, genre.replace('_', ' '))}.")

    # 4. Image-specific visual description (BLIP or Florence-2 output)
    visual = _postprocess(visual)
    if visual:
        parts.append(visual if visual.endswith(".") else visual + ".")

    # 5. CLIP subject tags (max 4, threshold-filtered)
    if tags:
        parts.append(f"Subject: {', '.join(tags)}.")

    # 6. Fixed per-style descriptor (art-history vocabulary)
    descriptor = STYLE_DESCRIPTORS.get(row["style"], "")
    if descriptor:
        parts.append(descriptor + ".")

    caption = " ".join(parts)
    # Hard-cap at 60 words by truncating at the last full stop within limit
    words = caption.split()
    if len(words) > 60:
        truncated = " ".join(words[:60])
        cut = truncated.rfind(".")
        caption = truncated[:cut + 1] if cut != -1 else truncated
    return caption


# ---------------------------------------------------------------------------
# C.  CLIP loading and tag/flag computation
# ---------------------------------------------------------------------------
def _load_clip(device: str):
    """Load CLIP ViT-B/32 and pre-compute text embeddings for tags and flags."""
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    tag_labels  = list(CLIP_TAG_PROMPTS.keys())
    tag_prompts = list(CLIP_TAG_PROMPTS.values())

    with torch.no_grad():
        tag_inputs  = proc(text=tag_prompts + NON_PAINTING_PROMPTS,
                           return_tensors="pt", padding=True).to(device)
        text_feats  = model.get_text_features(**tag_inputs)
        text_feats  = F.normalize(text_feats, dim=-1)

    tag_embeds = text_feats[:len(tag_prompts)]      # (n_tags, 512)
    np_embeds  = text_feats[len(tag_prompts):]      # (2, 512)  non-painting prompts

    return proc, model, tag_labels, tag_embeds, np_embeds


def _clip_batch(
    images: list[Image.Image],
    clip_proc,
    clip_model,
    tag_labels: list[str],
    tag_embeds: torch.Tensor,
    np_embeds: torch.Tensor,
    device: str,
) -> tuple[list[list[str]], list[bool]]:
    """Return (tags, flagged) for a batch of PIL images."""
    with torch.no_grad():
        img_inputs = clip_proc(images=images, return_tensors="pt").to(device)
        img_feats  = F.normalize(clip_model.get_image_features(**img_inputs), dim=-1)

    # Subject tags: keep labels above threshold, at most CLIP_MAX_TAGS
    scores = (img_feats @ tag_embeds.T).cpu()                  # (B, n_tags)
    tags_batch = []
    for row_scores in scores:
        above = [(tag_labels[i], float(row_scores[i]))
                 for i in range(len(tag_labels))
                 if float(row_scores[i]) >= CLIP_TAG_THRESHOLD]
        above.sort(key=lambda x: -x[1])
        tags_batch.append([t for t, _ in above[:CLIP_MAX_TAGS]])

    # Non-painting flag: photo prompt score > painting prompt score + margin
    np_scores = (img_feats @ np_embeds.T).cpu()               # (B, 2)
    flagged = [
        float(np_scores[i, 0]) > float(np_scores[i, 1]) + NON_PAINTING_MARGIN
        for i in range(len(images))
    ]
    return tags_batch, flagged


# ---------------------------------------------------------------------------
# D.  Captioner loading (BLIP-large or Florence-2-large)
# ---------------------------------------------------------------------------
def _load_captioner(model_name: str, device: str):
    """Load captioner and return a caption_fn(images) -> list[str]."""

    if model_name == "blip-large":
        # Conditional prompt "a painting of" biases BLIP toward art vocabulary
        proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device).eval()

        def caption_fn(images: list[Image.Image]) -> list[str]:
            prompt = ["a painting of"] * len(images)
            inputs = proc(images=images, text=prompt,
                          return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, **GENERATE_KWARGS)
            return [proc.decode(o, skip_special_tokens=True) for o in out]

        batch_size = 16

    elif model_name == "florence2":
        # Florence-2-large <MORE_DETAILED_CAPTION> produces richer art descriptions
        # trust_remote_code=True is required by Florence-2's custom modeling code
        proc  = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        ).to(device).eval()

        def caption_fn(images: list[Image.Image]) -> list[str]:
            task   = "<MORE_DETAILED_CAPTION>"
            texts  = [task] * len(images)
            inputs = proc(text=texts, images=images,
                          return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    **GENERATE_KWARGS,
                )
            raw = proc.batch_decode(out, skip_special_tokens=False)
            captions = []
            for r, img in zip(raw, images):
                parsed = proc.post_process_generation(
                    r, task=task, image_size=(img.width, img.height)
                )
                captions.append(parsed.get(task, ""))
            return captions

        batch_size = 4

    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose blip-large or florence2.")

    return caption_fn, batch_size


# ---------------------------------------------------------------------------
# E.  Comparison: 20 sample images, two captioners side by side
# ---------------------------------------------------------------------------
def compare_captioners(
    n_samples: int = 20,
    metadata_path: Path = METADATA_PATH,
    img_dir: Path = IMG_DIR,
) -> None:
    """Run BLIP-large and Florence-2-large on n_samples images and print side by side."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta   = pd.read_csv(metadata_path).sample(n_samples, random_state=42)

    results: dict[str, list[str]] = {"blip-large": [], "florence2": []}
    for name in ("blip-large", "florence2"):
        print(f"\nLoading {name} …")
        fn, _ = _load_captioner(name, device)
        for _, row in meta.iterrows():
            img = Image.open(img_dir / row["path"]).convert("RGB")
            results[name].append(_postprocess(fn([img])[0]))
        # free VRAM before loading the next model
        del fn
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("CAPTIONER COMPARISON  (20 random images)")
    print("=" * 70)
    for i, (_, row) in enumerate(meta.iterrows()):
        print(f"\n[{i+1:02d}] {row['style']} / {row['artist']}")
        print(f"     {row['path']}")
        for name in ("blip-large", "florence2"):
            wrapped = textwrap.fill(results[name][i], width=60,
                                    initial_indent="     ", subsequent_indent="          ")
            print(f"  {name:12s}: {wrapped.strip()}")
    print("\n→ Re-run with --model blip-large or --model florence2 to generate all 1 500 captions.")


# ---------------------------------------------------------------------------
# F.  Main generation function
# ---------------------------------------------------------------------------
def generate_captions_v2(
    model_name: str  = "florence2",
    metadata_path: Path = METADATA_PATH,
    img_dir: Path    = IMG_DIR,
    captions_path: Path = CAPTIONS_V2,
    checkpoint_path: Path = CHECKPOINT,
    flagged_path: Path = FLAGGED_CSV,
    batch_size: int  = None,       # None → use model default
) -> list[dict]:
    """Generate v2 captions for all 1 500 artworks.

    Checkpoints every 100 images so Colab session timeouts do not lose work.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Model: {model_name}")

    # Resume from checkpoint if it exists
    if checkpoint_path.exists():
        results  = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        done     = {r["path"] for r in results}
        print(f"Resuming from checkpoint: {len(done)} done.")
    else:
        results, done = [], set()

    meta = pd.read_csv(metadata_path)
    meta = meta[~meta["path"].isin(done)].reset_index(drop=True)
    if meta.empty:
        print("All images already captioned.")
    else:
        print(f"Images remaining: {len(meta)}")

        # Load CLIP first (small, stays in memory throughout)
        print("Loading CLIP ViT-B/32 …")
        clip_proc, clip_model, tag_labels, tag_embeds, np_embeds = _load_clip(device)

        # Load captioner
        print(f"Loading captioner ({model_name}) …")
        caption_fn, default_bs = _load_captioner(model_name, device)
        bs = batch_size if batch_size is not None else default_bs

        flagged_rows: list[dict] = []

        for i in range(0, len(meta), bs):
            batch    = meta.iloc[i : i + bs]
            images   = [Image.open(img_dir / r["path"]).convert("RGB")
                        for _, r in batch.iterrows()]

            visuals  = caption_fn(images)
            tags_b, flagged_b = _clip_batch(
                images, clip_proc, clip_model, tag_labels, tag_embeds, np_embeds, device
            )

            for j, (_, row) in enumerate(batch.iterrows()):
                caption = _build_caption(row, visuals[j], tags_b[j])
                entry   = {
                    "path":    row["path"],
                    "caption": caption,
                    "tags":    tags_b[j],
                    "flagged": flagged_b[j],
                }
                results.append(entry)
                if flagged_b[j]:
                    flagged_rows.append({"path": row["path"], "style": row["style"],
                                         "artist": row["artist"]})

            done_total = len(done) + len(results) - len(done)
            print(f"  {min(i + bs, len(meta)) + len(done)}/{len(meta) + len(done)} done")

            # Checkpoint every 100 images
            if len(results) % 100 < bs:
                checkpoint_path.write_text(
                    json.dumps(results, ensure_ascii=False), encoding="utf-8"
                )

    # Save final output
    Path(captions_path).parent.mkdir(exist_ok=True)
    Path(captions_path).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(results)} captions → {captions_path}")

    # Save flagged non-artworks
    if flagged_rows:
        pd.DataFrame(flagged_rows).to_csv(flagged_path, index=False)
        print(f"Flagged {len(flagged_rows)} possible non-paintings → {flagged_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate v2 captions with BLIP/Florence-2 + CLIP tags."
    )
    parser.add_argument("--analyse", action="store_true",
                        help="Print v1 diagnostic stats and exit.")
    parser.add_argument("--compare", action="store_true",
                        help="Compare captioners on 20 images and exit.")
    parser.add_argument("--model", default="florence2",
                        choices=["blip-large", "florence2"],
                        help="Captioner to use for full generation.")
    args = parser.parse_args()

    if args.analyse:
        analyse_v1()
        return

    if args.compare:
        compare_captioners()
        return

    analyse_v1()                          # always print v1 stats before starting
    generate_captions_v2(model_name=args.model)


if __name__ == "__main__":
    main()
