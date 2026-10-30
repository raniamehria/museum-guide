"""Evaluation: compare v1 vs v2 caption quality via retrieval metrics.

Metrics (all on CPU, no GPU needed):
  Precision@5       — fraction of top-5 results that match the relevance rule.
  Style hit rate@5  — for style queries, fraction of top-5 with the target style.
  MRR               — mean reciprocal rank of the first relevant result.
  Score gap 1→10    — score(rank-1) − score(rank-10).  A large gap means the
                      model can distinguish relevant from irrelevant; a near-
                      zero gap means the template dominates and rankings are
                      arbitrary.

Relevance rules (from data/eval_queries.json):
  {"type": "style",  "value": "Impressionism"}  → artwork.style == value
  {"type": "tag",    "value": "seascape"}        → value in artwork["tags"]
                                                   (tags come from captions_v2,
                                                   used as ground truth for both
                                                   v1 and v2 to keep eval fair)
  {"type": "artist", "value": "katsushika-hokusai"} → artwork.artist == value

Usage:
  python -m museum_guide.eval_captions          # compare v1 vs v2
  python -m museum_guide.eval_captions --rebuild  # force index rebuild first
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from museum_guide.qa import (
    CAPTIONS_PATH,
    METADATA_PATH,
    MODEL_NAME,
    _encode,
    build_artwork_index,
)
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CAPTIONS_V2    = Path("data/artwork_captions_v2.json")
INDEX_V1       = Path("checkpoints/artwork_index.pt")
INDEX_V2       = Path("checkpoints/artwork_index_v2.pt")
EVAL_QUERIES   = Path("data/eval_queries.json")
EVAL_RESULTS   = Path("results/caption_eval.csv")


# ---------------------------------------------------------------------------
# Relevance lookup
# ---------------------------------------------------------------------------
def _build_tag_lookup(captions_v2_path: Path) -> dict[str, list[str]]:
    """Build {path: [tag, …]} from captions_v2.json.

    Used as ground-truth relevance for tag-based queries for BOTH v1 and v2,
    so the comparison is fair (same relevance definition, different embeddings).
    """
    if not captions_v2_path.exists():
        print(f"Warning: {captions_v2_path} not found — tag-based queries skipped.")
        return {}
    raw = json.loads(captions_v2_path.read_text(encoding="utf-8"))
    return {c["path"]: c.get("tags", []) for c in raw}


def _is_relevant(artwork: dict, rule: dict, tag_lookup: dict[str, list[str]]) -> bool:
    rtype = rule["type"]
    val   = rule["value"]
    if rtype == "style":
        return artwork["style"] == val
    if rtype == "tag":
        return val in tag_lookup.get(artwork["path"], [])
    if rtype == "artist":
        return artwork["artist"] == val
    return False


# ---------------------------------------------------------------------------
# Retrieval (CPU)
# ---------------------------------------------------------------------------
def _retrieve(query: str, index: list[dict], tokenizer, model, top_n: int = 20) -> list[dict]:
    """Return top_n artworks ranked by cosine similarity (no deduplication)."""
    q_emb  = _encode(query, tokenizer, model).squeeze(0)
    stack  = torch.stack([art["embedding"] for art in index])
    scores = (stack @ q_emb).tolist()
    ranked = sorted(zip(scores, index), key=lambda x: -x[0])
    return [{"score": s, **art} for s, art in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(
    results: list[dict],
    rule: dict,
    tag_lookup: dict[str, list[str]],
    k: int = 5,
) -> dict:
    """Compute precision@k, style_hit@k, MRR, and score gap (rank1 − rank10)."""
    relevant = [_is_relevant(r, rule, tag_lookup) for r in results]

    prec_at_k = sum(relevant[:k]) / k

    style_hit = (
        sum(1 for r in results[:k] if r["style"] == rule["value"]) / k
        if rule["type"] == "style"
        else float("nan")
    )

    mrr = 0.0
    for rank, rel in enumerate(relevant, start=1):
        if rel:
            mrr = 1.0 / rank
            break

    score_gap = (
        results[0]["score"] - results[9]["score"]
        if len(results) >= 10
        else float("nan")
    )

    return {
        "precision@5":    round(prec_at_k, 3),
        "style_hit@5":    round(style_hit, 3) if not (style_hit != style_hit) else "—",
        "MRR":            round(mrr, 3),
        "score_gap_1_10": round(score_gap, 3),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(rebuild: bool = False) -> None:
    if not EVAL_QUERIES.exists():
        raise FileNotFoundError(f"{EVAL_QUERIES} not found.")

    queries    = json.loads(EVAL_QUERIES.read_text(encoding="utf-8"))
    tag_lookup = _build_tag_lookup(CAPTIONS_V2)

    # Build or load indexes
    for label, caps_path, idx_path in [
        ("v1", CAPTIONS_PATH, INDEX_V1),
        ("v2", CAPTIONS_V2,   INDEX_V2),
    ]:
        if rebuild or not idx_path.exists():
            if not caps_path.exists():
                print(f"Skipping {label}: {caps_path} not found.")
                continue
            print(f"Building {label} index …")
            build_artwork_index(captions_path=caps_path, index_path=idx_path)

    print("Loading MiniLM tokenizer/model on CPU …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    minilm    = AutoModel.from_pretrained(MODEL_NAME).eval()

    rows: list[dict] = []

    for entry in queries:
        qid   = entry["id"]
        query = entry["query"]
        rule  = entry["relevance"]

        row: dict = {"id": qid, "query": query, "relevance_type": rule["type"],
                     "relevance_value": rule["value"]}

        for label, idx_path in [("v1", INDEX_V1), ("v2", INDEX_V2)]:
            if not idx_path.exists():
                for m in ("precision@5", "style_hit@5", "MRR", "score_gap_1_10"):
                    row[f"{label}_{m}"] = "—"
                continue

            index   = torch.load(idx_path, weights_only=False)
            results = _retrieve(query, index, tokenizer, minilm)
            m       = _metrics(results, rule, tag_lookup)
            for k, v in m.items():
                row[f"{label}_{k}"] = v

        rows.append(row)
        print(f"  [{qid:02d}] {query[:50]:<50}  v1_P@5={row.get('v1_precision@5','—')}  "
              f"v2_P@5={row.get('v2_precision@5','—')}")

    df = pd.DataFrame(rows)

    # Summary row
    numeric_cols = [c for c in df.columns if any(m in c for m in
                    ("precision", "MRR", "score_gap"))]
    summary: dict = {"id": "AVG", "query": "— average —",
                     "relevance_type": "", "relevance_value": ""}
    for c in numeric_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        summary[c] = round(float(vals.mean()), 3) if len(vals) else "—"
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    # Print table
    print("\n" + "=" * 80)
    print("CAPTION EVALUATION  v1 vs v2")
    print("=" * 80)
    cols = ["id", "query",
            "v1_precision@5", "v2_precision@5",
            "v1_MRR",         "v2_MRR",
            "v1_score_gap_1_10", "v2_score_gap_1_10"]
    print(df[cols].to_string(index=False))

    # Save
    Path(EVAL_RESULTS).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EVAL_RESULTS, index=False)
    print(f"\nSaved → {EVAL_RESULTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v1 vs v2 captions.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild both indexes before evaluation.")
    args = parser.parse_args()
    evaluate(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
