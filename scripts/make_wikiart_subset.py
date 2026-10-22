#!/usr/bin/env python
"""Build a small WikiArt subset for the museum-guide project (MSA-DATB09-03).

Streams the ``huggan/wikiart`` dataset from the Hugging Face Hub (81 444 paintings labelled with
27 styles, 11 genres and 129 artist labels including the catch-all "Unknown Artist"; 72 parquet shards,
33.7 GB in total) and writes, for K classes of your choice x N images per class:

    OUT/
      <class_a>/s03_r0457.jpg ...   an ImageFolder tree (torchvision.datasets.ImageFolder reads it as is)
      <class_b>/...
      metadata.csv                  one row per image: path, label, label_idx, id, style, genre, artist, ...
      subset_info.json              command, dataset revision, counts, shards used (reproducibility)

USAGE
-----
    pip install "datasets>=3.0" pillow

    # 1. List the classes of a label with their frequencies (the first run builds a small label
    #    index, 2-4 minutes, cached for the next runs).
    python make_wikiart_subset.py --list-classes --label style

    # 2. Check the plan (how many shards, how many GB will be streamed) without downloading images.
    python make_wikiart_subset.py --label style \
        --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
        --per-class 300 --size 128 --out data/wikiart_style5 --dry-run

    # 3. Build it: the same command without --dry-run. Re-running it resumes (existing files are kept).
    python make_wikiart_subset.py --label style \
        --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
        --per-class 300 --size 128 --out data/wikiart_style5

    # Genres instead of styles:
    python make_wikiart_subset.py --label genre \
        --classes portrait landscape cityscape still_life religious_painting \
        --per-class 300 --out data/wikiart_genre5

    Then, in PyTorch:  torchvision.datasets.ImageFolder("data/wikiart_style5", transform=...)
    (class indices = sorted folder names = the ``label_idx`` column of metadata.csv).

HOW IT WORKS (and why)
----------------------
1. Label pass. Each shard is a single ~470 MB parquet row group: fetching ANY image of a shard
   downloads the whole shard. So we first stream only the three label columns of every shard
   (``datasets.load_dataset(..., streaming=True, columns=[...])``, a few KB per shard) and cache
   them in a small JSON index.
2. Plan. With a fixed seed, pick N rows per class, preferring the shards that contain the most wanted
   rows (``--strategy fewest-shards``, the default) or visiting shards in a random order
   (``--strategy spread``, more diverse, more download). The dataset is ordered in blocks of artists:
   without care, 280 of 300 "Impressionism" images would come from Renoir and the classifier would
   learn the painter, not the style. ``--max-per-artist`` (default 50, 0 = off) caps how many images
   one named painter contributes ("Unknown Artist", 51 % of the rows, is a catch-all and is not capped).
3. Image pass. Stream only the planned shards with ``datasets.load_dataset(..., split="train",
   streaming=True)``, decode only the selected images, resize (shorter side -> --size, then a centre
   crop to size x size unless --no-crop) and save them as JPEG.

Typical cost (5-8 classes x 300 images): 4-10 shards, i.e. 2-5 GB streamed (a few minutes on Colab,
10-40 min on a home connection), peak RAM about 1.5 GB, output about 5 MB per 1 000 images at 128 px.
Measured plans with the defaults: 5 genres (portrait, landscape, cityscape, still_life,
religious_painting) -> 5 shards (~2.3 GB); 6 frequent styles -> 4 shards (~1.9 GB); 5 contrasted styles
(Impressionism, Baroque, Cubism, Ukiyo_e, Pop_Art) -> 9 shards (~4.2 GB). Always --dry-run first.

DATA AND LICENCE
----------------
WikiArt images remain the property of their authors / rights holders (the dataset card says
"license: unknown"). Use the subset for this coursework only: do NOT commit images to Git and do not
publish them. Commit this script and the exact command instead (they are in subset_info.json).
Pinned dataset revision by default, so that everybody gets the same data.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = "huggan/wikiart"
REVISION = "d559852d2b232e0fcf195e775866964f0564f2b5"  # dataset commit of 2023-03-22 (pinned)
LABEL_COLUMNS = ("style", "genre", "artist")
UNKNOWN_ARTIST = "Unknown Artist"
GB_PER_SHARD = 33.74 / 72  # average shard size


def log(msg: str = "") -> None:
    print(msg, flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream a K-class x N-image subset of huggan/wikiart into an ImageFolder tree + metadata.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--label", choices=["style", "genre"], default="style", help="which label defines the classes")
    p.add_argument("--classes", nargs="+", metavar="NAME",
                   help="class names (case-insensitive; see --list-classes), e.g. Impressionism Cubism Ukiyo_e")
    p.add_argument("--per-class", type=int, default=300, help="images per class (N)")
    p.add_argument("--size", type=int, default=128, help="output size in pixels (64-224 is sensible)")
    p.add_argument("--no-crop", action="store_true", help="keep the aspect ratio (shorter side = --size, no crop)")
    p.add_argument("--out", type=Path, default=Path("data/wikiart_subset"), help="output directory")
    p.add_argument("--seed", type=int, default=0, help="random seed of the plan (which rows are picked)")
    p.add_argument("--strategy", choices=["fewest-shards", "spread"], default="fewest-shards",
                   help="fewest-shards: least download; spread: shards in random order (more diverse, more download)")
    p.add_argument("--max-per-artist", type=int, default=50,
                   help="cap on images per named artist over the whole subset (0 = no cap); "
                        "without it, e.g. 280 of 300 'Impressionism' images come from Renoir")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality")
    p.add_argument("--revision", default=REVISION, help="dataset revision (commit hash or 'main')")
    p.add_argument("--index-cache", type=Path, default=None,
                   help="label-index cache file (default: ~/.cache/wikiart_label_index_<revision>.json)")
    p.add_argument("--retries", type=int, default=3, help="attempts per shard on network errors")
    p.add_argument("--list-classes", action="store_true", help="print the classes of --label with counts, then exit")
    p.add_argument("--dry-run", action="store_true", help="print the plan (shards, GB) without downloading images")
    a = p.parse_args(argv)
    if not a.list_classes and not a.classes:
        p.error("--classes is required (use --list-classes to see the names)")
    if a.per_class < 1 or a.size < 8:
        p.error("--per-class must be >= 1 and --size >= 8")
    return a


# --------------------------------------------------------------------------------------------------
# Streaming helpers
# --------------------------------------------------------------------------------------------------
def open_stream(revision: str, columns=None):
    """The whole dataset as a streaming IterableDataset (optionally only some columns)."""
    from datasets import load_dataset
    kwargs = {"columns": list(columns)} if columns else {}
    return load_dataset(REPO, split="train", streaming=True, revision=revision, **kwargs)


def with_retries(fn, what: str, retries: int):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:  # network hiccups: retry with a growing pause
            if attempt == retries:
                raise
            wait = 10 * attempt
            log(f"  ! {what}: {type(e).__name__}: {str(e)[:160]} -> retry {attempt}/{retries - 1} in {wait} s")
            time.sleep(wait)


def default_index_path(revision: str) -> Path:
    return Path.home() / ".cache" / f"wikiart_label_index_{revision[:12]}.json"


def load_label_index(revision: str, cache: Path, retries: int) -> dict:
    """{'names': {label: [class names]}, 'shards': [{'style': [...], 'genre': [...], 'artist': [...]}, ...]}"""
    if cache.exists():
        index = json.loads(cache.read_text(encoding="utf-8"))
        if index.get("revision") == revision:
            log(f"Label index: {cache} ({sum(len(s['style']) for s in index['shards'])} rows, cached)")
            return index
    log("Label pass: streaming the label columns of every shard (no images, a few KB each)...")
    t0 = time.time()
    ds = with_retries(lambda: open_stream(revision, LABEL_COLUMNS), "load_dataset", retries)
    names = {k: list(ds.features[k].names) for k in LABEL_COLUMNS}
    shards = []
    for i in range(ds.n_shards):
        def read_shard(i=i):
            cols = {k: [] for k in LABEL_COLUMNS}
            for ex in ds.shard(num_shards=ds.n_shards, index=i):
                for k in LABEL_COLUMNS:
                    cols[k].append(ex[k])
            return cols
        shards.append(with_retries(read_shard, f"labels of shard {i}", retries))
        if (i + 1) % 12 == 0 or i + 1 == ds.n_shards:
            log(f"  {i + 1}/{ds.n_shards} shards indexed ({time.time() - t0:.0f} s)")
    index = {"repo": REPO, "revision": revision, "names": names, "shards": shards}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index), encoding="utf-8")
    log(f"Label index saved to {cache}")
    return index


# --------------------------------------------------------------------------------------------------
# Classes and plan
# --------------------------------------------------------------------------------------------------
def _norm(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def resolve_classes(requested, names):
    lookup = {_norm(n): i for i, n in enumerate(names)}
    ids = []
    for r in requested:
        if _norm(r) not in lookup:
            sys.exit(f"Unknown class {r!r}. Valid names: {', '.join(names)}")
        if lookup[_norm(r)] not in ids:
            ids.append(lookup[_norm(r)])
    return ids


def list_classes(index: dict, label: str) -> None:
    names, shards = index["names"], index["shards"]
    unknown = names["artist"].index(UNKNOWN_ARTIST) if UNKNOWN_ARTIST in names["artist"] else -1
    count, n_shards, artists, unk = Counter(), Counter(), {}, Counter()
    for s in shards:
        seen = set()
        for c, a in zip(s[label], s["artist"]):
            count[c] += 1
            seen.add(c)
            artists.setdefault(c, set()).add(a)
            unk[c] += a == unknown
        for c in seen:
            n_shards[c] += 1
    log(f"\n{label} classes of {REPO} ({sum(count.values())} images):")
    log(f"  {'name':28s} {'images':>7s} {'shards':>7s} {'named artists':>14s} {'unknown artist':>15s}")
    for c, n in count.most_common():
        named = len(artists[c] - {unknown})
        log(f"  {names[label][c]:28s} {n:7d} {n_shards[c]:7d} {named:14d} {unk[c] / n:14.0%}")


def make_plan(index: dict, label: str, class_ids, per_class: int, max_per_artist: int, strategy: str, seed: int):
    """Choose the rows to download: returns ({shard: [rows]}, {class_id: still missing})."""
    names, shards = index["names"], index["shards"]
    unknown = names["artist"].index(UNKNOWN_ARTIST) if UNKNOWN_ARTIST in names["artist"] else -1
    rng = random.Random(seed)
    wanted = set(class_ids)
    candidates = {}
    for s, cols in enumerate(shards):
        rows = [r for r, c in enumerate(cols[label]) if c in wanted]
        rng.shuffle(rows)  # which rows of a shard are taken is random but reproducible
        candidates[s] = rows
    order = list(range(len(shards)))
    rng.shuffle(order)
    need = {c: per_class for c in class_ids}
    per_artist = Counter()

    def rows_from(s):
        """Rows that shard s would add, given what has already been planned (no side effect)."""
        need2, art2, take = dict(need), Counter(), []
        for r in candidates[s]:
            c, a = shards[s][label][r], shards[s]["artist"][r]
            if need2[c] == 0:
                continue
            if max_per_artist and a != unknown and per_artist[a] + art2[a] >= max_per_artist:
                continue
            take.append(r)
            need2[c] -= 1
            art2[a] += 1
        return take

    plan, remaining = {}, list(order)
    while any(need.values()) and remaining:
        if strategy == "fewest-shards":
            best = max(remaining, key=lambda s: len(rows_from(s)))  # ties: first in the seeded order
        else:
            best = next((s for s in remaining if rows_from(s)), remaining[0])
        take = rows_from(best)
        remaining.remove(best)
        if not take:
            break
        plan[best] = sorted(take)
        for r in take:
            need[shards[best][label][r]] -= 1
            per_artist[shards[best]["artist"][r]] += 1
    return plan, need


def print_plan(plan, need, index, label, class_ids, per_class):
    names, shards = index["names"], index["shards"]
    got = Counter(shards[s][label][r] for s, rows in plan.items() for r in rows)
    artists = Counter(shards[s]["artist"][r] for s, rows in plan.items() for r in rows)
    log(f"\nPlan: {len(plan)} shard(s) to stream (~{len(plan) * GB_PER_SHARD:.1f} GB): {sorted(plan)}")
    for c in class_ids:
        flag = "" if got[c] == per_class else f"   <- only {got[c]} available with these settings"
        log(f"  {names[label][c]:28s} {got[c]:5d} images{flag}")
    top = [(names["artist"][a], n) for a, n in artists.most_common(6)]
    log(f"  most frequent artists in the subset: {top}")
    if any(need.values()):
        log("  Warning: some classes are incomplete (try another seed, --max-per-artist 0 or fewer images).")


# --------------------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------------------
def save_image(data: bytes, dst: Path, size: int, crop: bool, quality: int):
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None  # some WikiArt scans are very large (trusted source)
    with Image.open(io.BytesIO(data)) as im:
        orig = im.size
        im.draft("RGB", (size, size))  # fast JPEG decoding at reduced scale (never below the requested size)
        im = im.convert("RGB")
        if crop:
            im = ImageOps.fit(im, (size, size), method=Image.Resampling.BICUBIC)
        else:
            w, h = im.size
            scale = size / min(w, h)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BICUBIC)
        tmp = dst.with_name(dst.name + ".part")
        im.save(tmp, "JPEG", quality=quality)
        tmp.replace(dst)
    return orig


def row_id(shard: int, row: int) -> str:
    return f"s{shard:02d}_r{row:04d}"


def download(plan, index, label, class_names, args, sizes):
    """Stream the planned shards and save the selected images; returns {row_id: (w, h)}."""
    from datasets import Image as HFImage
    shards = index["shards"]
    todo = {}
    for s, rows in plan.items():
        missing = [r for r in rows
                   if not (args.out / class_names[shards[s][label][r]] / f"{row_id(s, r)}.jpg").exists()]
        if missing:
            todo[s] = missing
    done_already = sum(len(r) for r in plan.values()) - sum(len(r) for r in todo.values())
    if done_already:
        log(f"Resuming: {done_already} images already on disk.")
    if not todo:
        return sizes
    log(f"Image pass: {len(todo)} shard(s) to stream. Each shard arrives in one block (~{GB_PER_SHARD:.2f} GB).")
    ds = with_retries(lambda: open_stream(args.revision), "load_dataset", args.retries)
    ds = ds.cast_column("image", HFImage(decode=False))  # raw bytes: decode only what we keep
    n_shards = ds.n_shards
    for k, (s, rows) in enumerate(sorted(todo.items()), 1):
        t0 = time.time()
        wanted = set(rows)

        def stream_shard(s=s, wanted=wanted):
            left = set(wanted)
            for r, ex in enumerate(ds.shard(num_shards=n_shards, index=s)):
                if r not in left:
                    continue
                c = shards[s][label][r]
                if ex[label] != c:  # safety net: the index must describe this exact revision
                    raise RuntimeError(f"label mismatch at shard {s}, row {r}: delete the index cache and retry")
                dst = args.out / class_names[c] / f"{row_id(s, r)}.jpg"
                sizes[row_id(s, r)] = save_image(ex["image"]["bytes"], dst, args.size, not args.no_crop, args.quality)
                left.discard(r)
                if not left:
                    break
            if left:
                raise RuntimeError(f"shard {s}: {len(left)} planned rows were not found")

        with_retries(stream_shard, f"shard {s}", args.retries)
        log(f"  shard {s:2d} ({k}/{len(todo)}): {len(rows):4d} images saved in {time.time() - t0:5.0f} s")
    return sizes


def write_metadata(plan, index, label, class_ids, class_names, args, sizes) -> Counter:
    names, shards = index["names"], index["shards"]
    folder_order = sorted(class_names[c] for c in class_ids)  # = ImageFolder's class_to_idx
    label_idx = {n: i for i, n in enumerate(folder_order)}
    rows_out, counts = [], Counter()
    for s in sorted(plan):
        for r in plan[s]:
            c = shards[s][label][r]
            rel = f"{class_names[c]}/{row_id(s, r)}.jpg"
            if not (args.out / rel).exists():
                continue
            w, h = sizes.get(row_id(s, r), ("", ""))
            counts[class_names[c]] += 1
            rows_out.append({
                "path": rel, "label": class_names[c], "label_idx": label_idx[class_names[c]], "id": row_id(s, r),
                "style": names["style"][shards[s]["style"][r]], "genre": names["genre"][shards[s]["genre"][r]],
                "artist": names["artist"][shards[s]["artist"][r]], "shard": s, "row": r,
                "orig_width": w, "orig_height": h})
    rows_out.sort(key=lambda d: (d["label_idx"], d["id"]))
    with open(args.out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0]) if rows_out else ["path"])
        wr.writeheader()
        wr.writerows(rows_out)
    info = {
        "repo": REPO, "revision": args.revision, "label": label, "classes": folder_order,
        "per_class": args.per_class, "size": args.size, "crop": not args.no_crop, "seed": args.seed,
        "strategy": args.strategy, "max_per_artist": args.max_per_artist, "shards": sorted(plan),
        "counts": {n: counts[n] for n in folder_order}, "total": sum(counts.values()),
        "command": "python " + " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / "subset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return counts


def previous_sizes(out: Path) -> dict:
    """Original sizes recorded by a previous (interrupted) run, so that resuming keeps them."""
    meta = out / "metadata.csv"
    if not meta.exists():
        return {}
    with open(meta, newline="", encoding="utf-8") as f:
        return {d["id"]: (d.get("orig_width", ""), d.get("orig_height", "")) for d in csv.DictReader(f)}


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # artist names with accents on Windows consoles
    args = parse_args(argv)
    cache = args.index_cache or default_index_path(args.revision)
    index = load_label_index(args.revision, cache, args.retries)
    if args.list_classes:
        list_classes(index, args.label)
        return 0
    names = index["names"][args.label]
    class_ids = resolve_classes(args.classes, names)
    class_names = {c: names[c] for c in class_ids}
    plan, need = make_plan(index, args.label, class_ids, args.per_class, args.max_per_artist, args.strategy, args.seed)
    print_plan(plan, need, index, args.label, class_ids, args.per_class)
    if args.dry_run:
        return 0
    for c in class_ids:
        (args.out / class_names[c]).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sizes = download(plan, index, args.label, class_names, args, previous_sizes(args.out))
    counts = write_metadata(plan, index, args.label, class_ids, class_names, args, sizes)
    log(f"\nDone in {time.time() - t0:.0f} s: {sum(counts.values())} images in {args.out.resolve()}")
    log("  " + ", ".join(f"{n}: {k}" for n, k in sorted(counts.items())))
    log(f"  metadata: {args.out / 'metadata.csv'} | reproducibility: {args.out / 'subset_info.json'}")
    complete = all(counts[class_names[c]] >= args.per_class for c in class_ids)
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
