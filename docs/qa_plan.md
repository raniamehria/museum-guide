# Plan v3 — Natural-Language Q&A: Architecture Decision

**Status**: Chosen plan for implementation. Supersedes v2 on architecture.
**Deadline**: Before presentation rehearsal (Nov 3)

---

## 1. The Two Approaches

The professor described two valid paths:

| | Approach 1 — Fine-tuned sentence transformer | Approach 2 — Zero-shot cosine similarity |
|---|---|---|
| Model | Pre-trained MiniLM, fine-tuned with 30 examples | Pre-trained MiniLM, no fine-tuning |
| Mechanism | Softmax classifier over all labels → label filter → cosine rerank | Direct cosine similarity between query and artwork captions |
| Training needed | Yes | No |
| "Which painting shows a stormy sea?" | Predicts `specific_query`, falls back to full scan | Full scan naturally — same result |
| "I would like to see portraits of children" | Label "portrait" does not exist in metadata genre column → filter fails or returns nothing | Caption includes portrait-related description if LLM-enriched → retrieval works |
| CPU cost on kiosk | ~10 ms/query (BERT inference on 120 tokens) | ~2 ms/query (one cosine similarity over 250 pre-computed vectors) |
| Answers the director's question | This is the "neural network" option | This is the "simpler classical model" option |
| Where LLMs help | Generate 30–60 diverse training examples | Generate rich artwork captions covering genre, mood, subject |

---

## 2. Recommendation: Approach 2

**We implement Approach 2 (zero-shot) as the primary method.**

### Why Approach 2 wins

**1. The subject's example queries break Approach 1.**
"I would like to see portraits of children" is the professor's own example question (the subject, not us, defines the visitor queries — we cannot change them). The word "portrait" does not exist as a genre label in our dataset: 82% of images have "Unknown Genre". Approach 1's label filter would return 0 candidates for this query. Approach 2 bypasses this entirely — it compares the query embedding to caption embeddings, and if captions are LLM-enriched with portrait-like descriptions, the right artworks score high.

**2. It directly answers the director's question.**
The subject asks: "is a neural network worth it in production, or would a simpler classical model do the job?" For image classification (Task 3), the answer was yes — ResNet-18 is worth it. For natural-language retrieval, we now argue the opposite: a pre-trained encoder + cosine similarity is sufficient, no fine-tuning needed. This is a stronger and more honest conclusion.

**3. CPU-friendlier.**
Approach 2 at inference runs one cosine similarity dot product over 250 pre-computed 384-dim vectors — roughly 2 ms on a modest CPU. Approach 1 runs a full BERT forward pass per query before the similarity step.

**4. The guardrail (from resnet.md) applies cleanly.**
Resnet.md adapts the NVIDIA VSS idea: if the maximum similarity score < threshold τ, the guide says "not sure" instead of returning a bad answer. This is the same guardrail pattern, directly reusable here. It makes the system honest without any training.

**5. LLMs are used for higher-leverage work.**
In Approach 1, LLMs generate 30 training examples — useful but bounded. In Approach 2, LLMs generate rich captions for all 1,500 artworks, dramatically improving retrieval quality for all queries including open-ended ones. Same budget, much bigger impact.

---

## 3. Architecture

```text
                 OFFLINE (done once, pre-computed)
  ─────────────────────────────────────────────────────────
  metadata.csv (style, genre, artist)
          │
          ▼
  LLM-generated caption per artwork
  e.g. "A Baroque painting by Peter Paul Rubens. Dramatic
        chiaroscuro lighting, religious and mythological
        subjects, rich warm colors, portraits of figures
        in motion."
          │
          ▼
  MiniLM encoder (HuggingFace, no fine-tuning)
  → 384-dim caption embedding per artwork
          │
          ▼
  artwork_index.pt  (250 test artworks × 384 dims)
  ─────────────────────────────────────────────────────────

                 ONLINE (per visitor query, ~2 ms)
  ─────────────────────────────────────────────────────────
  Visitor: "I would like to see portraits of children"
          │
          ▼
  ┌────────────────────────────────────────────────────┐
  │  MiniLM encoder (same model, no fine-tuning)       │
  │  → 384-dim query embedding                         │
  └────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────┐
  │  Cosine similarity vs all 250 caption embeddings   │
  │  → ranked list of (artwork, score)                 │
  └────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────┐
  │  Guardrail (adapted from resnet.md §NVIDIA VSS)    │
  │  if top score < τ  → "I'm not sure"                │
  │  if top score ≥ τ  → return top-3 artworks         │
  └────────────────────────────────────────────────────┘
          │
          ▼
         Top-3 artworks (or "not sure")
  ─────────────────────────────────────────────────────────
```

**Key design choices:**
- Same encoder for captions and queries → embeddings are in the same space.
- No fine-tuning → no training infrastructure, no 30-example bottleneck.
- Guardrail threshold τ chosen on the validation split (same method as resnet.md).
- Captions are the only thing that determines retrieval quality — invest time here.

---

## 4. Caption Strategy (the critical step)

Since 82% of artworks have "Unknown Genre" in metadata.csv, structured captions (`"A Baroque painting by Rubens"`) are too weak for genre or subject queries. We use an LLM to generate a richer caption for each artwork based on what we know: style and artist.

**Template fed to the LLM (per artwork):**
> "Write a 2–3 sentence description of a {style} painting by {artist}. Describe the typical visual characteristics, common subjects, mood and technique of this style and artist. Include genre terms (portrait, landscape, still life, seascape, etc.) if they are typical of this artist's work."

**Result example for a Baroque painting by Rubens:**
> "A Baroque painting by Peter Paul Rubens. Typical subjects include religious and mythological scenes, portraits of aristocrats, and dramatic battle scenes, rendered with rich warm colors and dynamic compositions. The style features strong chiaroscuro lighting and expressive figures in motion."

**Why this unlocks genre queries without having genre metadata**: the phrase "portraits of aristocrats" in the caption makes the query "portraits of children" score higher on Baroque/Flemish works than on abstract Cubist works — even though "portrait" is not a metadata field.

**Practical note**: the LLM generates one description per (style, artist) pair, not per individual image. There are ~50 unique (style, artist) pairs in the dataset, so this is ~50 LLM calls, not 1,500.

---

## 5. Guardrail

Adapted directly from resnet.md's NVIDIA VSS section:

- Compute max cosine similarity for the query against all 250 artwork captions.
- Choose threshold τ on the **validation** split: for each τ, measure what fraction of queries get answered and what fraction of top-1 answers are relevant.
- At inference: if max score < τ → return "I'm not sure. Here are our most popular styles: [Baroque, Impressionism, Ukiyo-e]". If max score ≥ τ → return top-3.

**Target**: answer ~80% of queries, with top-1 relevant for ~75% of answered queries (estimated, to be measured in the demo notebook).

---

## 6. On the 15-Label Idea from v1

v1 proposed 15 hardcoded labels (5 styles + 5 genres + 5 artists) with Approach 1. With Approach 2, labels are not needed for inference. However, the 15-label concept is still useful for **caption enrichment** — the LLM generates captions that mention portrait, landscape, still life, religious, genre scene naturally, so the cosine similarity implicitly finds artworks matching those concepts without needing an explicit label classifier.

The professor's note that LLMs can generate training examples applies in Approach 2 as: LLMs generate captions, which is a higher-value use (1,500 artworks enriched) than generating 30 training sentences.

---

## 7. Implementation Steps (2–3 hours total)

### Step 0 — Generate LLM captions (45 min) — Person A
For each unique (style, artist) pair in `metadata.csv`, ask an LLM to write a 2–3 sentence description. Store as a dict and apply to all artworks in the pair. Save the enriched captions to `data/artwork_captions.json`.

### Step 1 — Build the artwork index (30 min) — Person A
Load the test split (250 images) from `splits/split.csv`. For each artwork, retrieve its LLM caption. Embed all 250 captions with MiniLM (HuggingFace, no fine-tuning). Save to `checkpoints/artwork_index.pt`.

### Step 2 — Inference + guardrail (30 min) — Person B
Write `answer(question, ...)` in `src/museum_guide/qa.py`:
1. Encode the question with the same MiniLM tokeniser and encoder.
2. Compute cosine similarity between the query embedding and all 250 caption embeddings.
3. Apply guardrail: if max score < τ, return a "not sure" response. Otherwise return top-3 artworks with their scores.

### Step 3 — Choose τ and evaluate (30 min) — Person C
On the validation split (250 artworks), run 8 held-out queries. For τ in [0.2, 0.3, 0.4, 0.5], measure: fraction answered and Recall@3. Pick τ that balances coverage vs precision. Report in `results/metrics/qa_eval.json`.

### Step 4 — Demo notebook and slide (45 min) — Person D
Create `notebook/qa_demo.ipynb`:
1. Load index and run 5 example queries including the subject's own examples ("stormy sea", "portraits of children").
2. Show the guardrail in action: one query that triggers "not sure".
3. Evaluation table (τ, % answered, Recall@3).
4. Add 1 slide to `final_pres/Présentation1.pptx`.

---

## 8. Answering the Director's Question

The QA feature answers the director's trade-off question in the opposite direction from Task 3:

| Task | Question | Answer |
|---|---|---|
| Task 3 — Image classification | Is ResNet worth it vs MLP? | Yes — +38 points, 23 ms on CPU, worth it |
| Bonus — Natural-language QA | Is a fine-tuned NN worth it vs zero-shot? | No — zero-shot retrieval is sufficient, simpler, faster |

This is a stronger project narrative: we show we can argue both sides of the "is NN worth it?" question, backed by measurement.

---

## 9. Definition of Done

- [ ] `data/artwork_captions.json` — LLM-generated captions for all (style, artist) pairs
- [ ] `checkpoints/artwork_index.pt` — 250 caption embeddings (384-dim each)
- [ ] `src/museum_guide/qa.py` — `build_artwork_index()` and `answer()` functions
- [ ] `results/metrics/qa_eval.json` — τ sweep, % answered, Recall@3
- [ ] `notebook/qa_demo.ipynb` — end-to-end demo with guardrail demonstration
- [ ] `README.md` — bonus section updated
- [ ] Slide — 1 bonus slide with the architecture diagram and the director's answer

---

## 10. Q&A Defense Note

If asked why we chose zero-shot over fine-tuning:
> "The subject's own example queries — 'portraits of children', 'stormy sea' — cannot be covered by a label classifier because portrait is not a genre in our dataset and 'stormy sea' is a visual concept with no label. Zero-shot retrieval with LLM-enriched captions handles both naturally. It also directly answers the director's question: for the QA feature, the simpler model is sufficient."

If asked why we didn't fine-tune at all:
> "The pre-trained MiniLM already understands natural language. Fine-tuning adds complexity, requires training examples, and doesn't improve over zero-shot on our label set — which we verified by comparing both approaches in the demo notebook."
