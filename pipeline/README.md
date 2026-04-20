# Rebuilding PluRule from Scratch

This directory contains the end-to-end pipeline that turns raw Pushshift
archives into the published PluRule benchmark. It matches the five-phase
construction described in the paper (§5). Run this only if you want to
**reproduce or modify** the dataset — to consume the released benchmark, use
[`../hydrate/README.md`](../hydrate/README.md) instead.

> Reconstruction takes **1–2 days** on a reasonable machine and requires
> multiple GPUs (8B embedding model + 30B LLM judge), Reddit API keys,
> and ~1–2 TB of disk. Hydration takes a few hours and needs no GPU.

## Paper phases ↔ pipeline stages

| Paper phase | Stages | What happens |
|---|---|---|
| **1. Data Collection** | 0, 1, 2 | Download Pushshift archives, extract moderator comments, pull current rules from Reddit API for SFW subreddits |
| **2. Rule Matching** | 3 | Embed comments + rules, apply global percentile thresholds to assign rule labels |
| **3. Instance Construction** | 4, 5, 6, 7 | Build comment trees, pair violating threads with compliant alternatives, collect submission objects and media |
| **4. Verification** | 8 | LLM judge (Qwen3-30B-A3B-Instruct) filters matches that don't actually cite the rule; build train/val/test |
| **5. Splitting & Clustering** | 9a, 9b, 9c, 9d, 10 | Embed + UMAP + HDBSCAN subreddits and rules, label clusters, attach labels to the dataset |

## Prerequisites

- Python **3.10+**
- **GPUs** (vLLM):
  - Stage 3 and Stage 9a use **Qwen3-Embedding-8B** (≥ 16 GB VRAM; multi-GPU helps)
  - Stage 8 and Stage 9c use **Qwen/Qwen3-30B-A3B-*** (≥ 80 GB VRAM, or split across GPUs)
- `aria2c` on `PATH` (Stage 0)
- **Reddit API credentials** in `credentials/reddit_api_keys.json` — see
  `credentials/reddit_api_keys.json.template`. Multiple keys are used
  round-robin to stay under rate limits; 4+ recommended.
- **Disk:** ~1–2 TB for the full Pushshift archive, 200–500 GB for
  intermediate stages, ~10–20 GB for final datasets.

### Install

```bash
conda env create -f ../environment-pipeline.yml
conda activate plurule-pipeline
```

The yml pulls `aria2` from conda-forge (needed by Stage 0) plus the full GPU
stack (torch, transformers, vllm), clustering deps (umap-learn, hdbscan),
and `praw` for Reddit API calls in Stage 2.

### Configure

Edit base directories in `../config.py` before running:

```python
BASE_DATA      = "/your/working/dir"        # outputs land under ./data and ./output
PUSHSHIFT_DATA = "/your/pushshift/mirror"   # where Stage 0 downloads land
```

Other knobs (thresholds, min-comment counts, embedding model) also live in
`config.py` but the defaults reproduce the paper numbers.

## Running

There is **no top-level orchestrator** (`run_pipeline.py` was removed because
its stage-numbering assumptions didn't survive the split sub-stages). Run
stages in order, from the repo root:

```bash
python pipeline/0_download_data.py
python pipeline/1_collect_mod_comments.py
python pipeline/2_get_top_sfw_subreddits.py
python pipeline/3_match_rules.py
python pipeline/4_collect_submission_comments.py
python pipeline/5_build_trees_and_threads.py
python pipeline/6_collect_submissions.py
python pipeline/7_collect_media.py
python pipeline/8_create_dehydrated_dataset.py
python pipeline/9a_embed_clusters.py
python pipeline/9b_cluster_embeddings.py
python pipeline/9c_label_clusters.py
# Optional manual refinement:
python pipeline/9d_reapply_cluster_labels.py
python pipeline/10_assign_cluster_labels.py
```

Each stage is independently re-runnable; most skip work that's already been
produced (see **Re-running** below).

---

## Stages

### 0. `0_download_data.py` — download Pushshift archive

Fetches all ~80K per-subreddit `{sub}_{comments,submissions}.zst` files from
the
[Academic Torrents Pushshift/Arctic Shift archive](https://academictorrents.com/details/3e3f64dee22dc304cdd2546254ca1f8e8ae542b4)
via `aria2c` and normalizes them into a first-letter bucket layout
(`<PUSHSHIFT_DATA>/<letter>/<sub>_*.zst`) that later stages rely on.

Shares the core torrent/aria2c logic with `hydrate/0_download.py` via
`utils/pushshift_download.py`. Use the hydrate variant if you only want the
dataset's subset.

**Key flags:** `--output-dir`, `--torrent-file`, `--from-dir` (skip torrent
when you already have a local mirror).

### 1. `1_collect_mod_comments.py` — extract moderator comments

Streams every `{sub}_comments.zst` in parallel, keeps lines where
`distinguished=="moderator"` and `parent_id.startswith("t1_")`, drops
bot-like usernames (`bot`, `automod`, …). Writes per-subreddit
`output/top_subreddits/{sub}_mod_comments.jsonl.zst` plus
`data/stage1_subreddit_mod_comment_rankings.json` (subreddits ranked by
mod-comment count).

**Caches the GPFS directory walk** to
`data/stage1_arctic_shift_discovery_cache.json` (delete to force
re-discovery; the walk can take ~20 min on slow filesystems).

### 2. `2_get_top_sfw_subreddits.py` — Reddit API metadata

For every subreddit ranking ≥ `MIN_MATCHED_COMMENTS` mod comments, hits the
Reddit API (PRAW) to:

- Check NSFW (`over18`) — drop NSFW.
- Pull subreddit metadata (title, description, language, subscriber count).
- Pull current community rules — drop subreddits with < `MIN_RULES_FOR_MATCHING`.

Output: `data/stage2_sfw_subreddits_min_{N}_comments.json`.

**Rate limiting:** uses round-robin across every key in
`credentials/reddit_api_keys.json`. Thread-pool = # of keys. Retries on 429
with exponential backoff.

### 3. `3_match_rules.py` — match comments to rules

Two phases:

**Phase 1 — similarity matrices (GPU-heavy).**
Subreddits are assigned to buckets via greedy load balancing (one bucket per
visible CUDA device). Each bucket is processed by a subprocess
(`utils/match_rules_bucket.py`) that loads **Qwen3-Embedding-8B** once via
vLLM and computes one similarity matrix per subreddit → saved as
`{sub}_similarity_matrix.pt`.

**Phase 2 — global thresholds (CPU).**
All matrices are loaded into a single distribution; the
`GOLD_PERCENTILE` (default 99.2) and `AMBIGUOUS_PERCENTILE` (default 98)
thresholds come from this pooled distribution. Comments whose best score
exceeds `gold` are assigned that rule; comments with > 1 rule above
`ambiguous` are dropped to avoid noisy labels. A distribution plot of
cosine-similarity scores is written alongside.

**Key flag:** `--phase2-only` to re-run just the matching step if only
thresholds changed.

**Outputs:**
`output/matched_comments/{sub}_match.jsonl.zst` + `{sub}_stats.json`,
`data/stage3_matching_summary.json`, `data/stage3_subreddit_submission_ids.json`,
`output/matched_comments/cosine_similarity_distribution_all_percentiles.png`.

Pre-cleanup: stale `*_match.jsonl.zst` files from previous runs are deleted
before Phase 2, to prevent downstream stages from reading outdated data.

### 4. `4_collect_submission_comments.py` — gather discussion context

Memory-optimized two-pass stream per subreddit:

1. **Pass 1:** filter the Pushshift `{sub}_comments.zst` down to only comments
   belonging to target `submission_id`s (from Stage 3), written to a temp
   file. Counts expected per-submission line totals.
2. **Pass 2:** deduplicate (prefer non-`[removed]`/`[deleted]` duplicates),
   write `output/organized_comments/{sub}/submission_{id}.pkl` as soon as all
   expected lines for that submission are seen.

Peak memory per worker: **under 1 GB** (only one submission's comments
resident at a time).

### 5. `5_build_trees_and_threads.py` — violating + compliant pairs

Builds hierarchical comment trees from the pickles (parent/child/depth),
saves them to `output/comment_trees/{sub}_comment_trees.pkl`, then pairs each
moderator-flagged (violating) thread with a **compliant** thread from the
same submission.

**Six filtering rules** applied to threads (see paper §5.3):

1. No `[removed]` / `[deleted]` content anywhere in the thread
2. No media in any comment of the thread
3. No moderator-authored comments in the thread (avoid mod ↔ user back-and-forth)
4. *Violating only:* leaf comment not edited after posting
5. *Compliant only:* leaf has no moderator replies as direct children
6. *Instance level:* mod comment posted before 2023-03-01 (Pushshift cutoff)

Compliant candidates are drawn from depth *n* or *n-1* of the same submission
and ranked by **(common ancestors ↑, length ↑, score ↓)**.

Output: `output/discussion_threads/{sub}_discussion_threads.pkl`,
`data/stage5_trees_and_threads_summary.json`. Trees and threads for subs no
longer in the Stage 3 manifest are cleaned up before this stage runs.

### 6. `6_collect_submissions.py` — submission objects

Streams Pushshift `{sub}_submissions.zst` for each qualifying subreddit,
keeps only the `submission_id`s referenced by Stage 5's thread pairs,
validates structure, writes `output/submissions/{sub}_submissions.zst`.

### 7. `7_collect_media.py` — images

Priority hierarchy: `media_metadata` → `url` → `oembed` → `preview`. Validates
Content-Type, caps at 50 MB, skips NSFW/crosspost/video submissions. Writes
`output/media/{sub}/{submission_id}_{media_id}.{ext}` plus
`data/stage7_media_collection_stats.json` and
`data/stage7_successful_submission_ids.json`.

Extraction + download logic lives in `utils/media.py` and is shared with
`hydrate/2_download_media.py`.

### 8. `8_create_dehydrated_dataset.py` — LLM verification + splits

**Loads** the output of Stages 5–7, filters to Pushshift-era instances (mod
comment before 2023-03-01), removes instances whose submissions have
`[removed]` / `[deleted]` content or moderator-authored submissions.

**LLM judge** (Qwen3-30B-A3B-Instruct via vLLM): for each instance presents
the mod comment + matched rule and classifies as:

- `(a)` stating a violation of the rule  — **kept**
- `(b)` discussing the rule — dropped
- `(c)` unrelated to the rule — dropped

Pass rate ≈ 82% (→ 13,371 instances in the paper).

**Adaptive per-subreddit splits:**

| Instances / sub | Test | Val | Train |
|---|---|---|---|
| 1 | 1 | 0 | 0 |
| 2 | 1 | 0 | 1 |
| 3–9 | 1 | 1 | n−2 |
| ≥ 10 | 10% | 10% | 80% |

Every subreddit appears in test; big subreddits don't dominate evaluation.

**Outputs:**
`data/{train,val,test}_hydrated.json.zst`,
`data/{train,val,test}_dehydrated.json.zst`,
`data/test_hydrated.json` (uncompressed for manual inspection),
`data/stage8_llm_verification_results.json`,
`data/stage8_thread_distribution_analysis.json`,
`data/stage8_final_datasets_stats.json`.

### 9. Clustering (four sub-stages)

#### 9a. `9a_embed_clusters.py` — embed subreddits and rules

Uses **Qwen3-Embedding-8B** on all three splits combined (unique subs + unique
rules, dedup by text).

- Subreddit text: `title + public_description`
- Rule text: `rule_comprehensive` (violation reason + short name + description)

Writes TSVs of embeddings + metadata to `output/embeddings/`.

#### 9b. `9b_cluster_embeddings.py` — UMAP + HDBSCAN

- **Grid search** over UMAP `(n_neighbors, n_components, min_dist)` × HDBSCAN
  `(min_cluster_size, min_samples, metric)` in parallel (`multiprocessing`,
  one UMAP fit per param set). Scored by DBCV, filtered to 5–30 clusters.
- **Apply best** params, write reduced 2D-ish embeddings + cluster_id columns
  back into the metadata TSVs.

```bash
python pipeline/9b_cluster_embeddings.py                # grid search + apply
python pipeline/9b_cluster_embeddings.py --grid-search  # grid only
python pipeline/9b_cluster_embeddings.py --apply-best   # apply only
```

Entity-specific UMAP `random_state` is pinned for reproducibility
(`UMAP_RANDOM_STATE = {'subreddit': 140, 'rule': 218}`).

#### 9c. `9c_label_clusters.py` — LLM names the clusters

Uses **Qwen3-30B-A3B-Thinking** to propose a 1–2 word label per cluster via
**majority vote** over 10 samples (temperature 0.6, top-p 0.92). Writes
`{entity}_cluster_labels.json`, `{entity}_cluster_analysis.txt` (with
thinking traces + vote counts), and updates the metadata TSV with a
`cluster_label` column.

#### 9d. `9d_reapply_cluster_labels.py` — optional manual refinement

After reviewing `{entity}_cluster_analysis.txt`, you can create
`output/clustering/{entity}_label_overrides.txt`:

```
# CLUSTER_ID: NEW_LABEL
5: Image Context
10: Text Context
3: Spam/Self-Promotion
```

Running this script applies the overrides to the JSON + metadata, merges
clusters whose NEW_LABEL collides (lowest cluster_id wins), renumbers
contiguously. The paper uses this pass to tidy cluster names.

### 10. `10_assign_cluster_labels.py` — join labels onto datasets

Reads the cluster labels from Stage 9b/c/d, joins them onto every thread
pair (rule cluster) and subreddit (subreddit cluster) in each split, and
writes out both hydrated and dehydrated *clustered* versions plus a LaTeX
table for the paper.

**Outputs:**
`data/{train,val,test}_hydrated_clustered.json.zst`,
`data/{train,val,test}_dehydrated_clustered.json.zst`,
`data/test_hydrated_clustered.json` (uncompressed),
`data/stage10_cluster_assignment_stats.json`,
`data/stage10_dataset_stats_table.tex`.

The **dehydrated-clustered** files are what you publish / share; hydrators
consume them via [`../hydrate/`](../hydrate/README.md).

---

## Data flow

```
Pushshift torrent
      │
      ▼
[Stage 0]   <PUSHSHIFT_DATA>/<letter>/<sub>_{comments,submissions}.zst
      │
      ▼
[Stage 1]   output/top_subreddits/{sub}_mod_comments.jsonl.zst
            data/stage1_subreddit_mod_comment_rankings.json
      │         │
      │         ▼
      │     [Stage 2]   data/stage2_sfw_subreddits_min_{N}_comments.json
      │         │
      └─────────┤
                ▼
       [Stage 3]   output/matched_comments/{sub}_match.jsonl.zst
                    data/stage3_matching_summary.json
                    data/stage3_subreddit_submission_ids.json
                    │
                    ▼
        [Stage 4]   output/organized_comments/{sub}/submission_{id}.pkl
                    │
                    ▼
        [Stage 5]   output/comment_trees/{sub}_comment_trees.pkl
                    output/discussion_threads/{sub}_discussion_threads.pkl
                    │
                    ▼
        [Stage 6]   output/submissions/{sub}_submissions.zst
                    │
                    ▼
        [Stage 7]   output/media/{sub}/*.{jpg,png,…}
                    │
                    ▼
        [Stage 8]   data/{train,val,test}_hydrated.json.zst
                    data/{train,val,test}_dehydrated.json.zst
                    │
                    ▼
        [Stage 9a]  output/embeddings/all_{subreddit,rule}_{embeddings,metadata}.tsv
                    │
                    ▼
        [Stage 9b]  output/clustering/{subreddit,rule}_grid_search_results.json
                    + cluster_id written back into metadata TSVs
                    │
                    ▼
        [Stage 9c]  output/clustering/{entity}_cluster_labels.json
                    + cluster_label written back into metadata TSVs
                    │
                    ▼ (optional manual pass)
        [Stage 9d]  ... overrides applied in place ...
                    │
                    ▼
        [Stage 10]  data/{split}_hydrated_clustered.json.zst      ← final hydrated
                    data/{split}_dehydrated_clustered.json.zst    ← publish this
                    data/stage10_dataset_stats_table.tex
```

## Re-running

All stages are independently re-runnable but **not fully idempotent**. Safe
to re-run individually after any config change:

- **Stages 0, 1, 4, 6, 7** skip files that already exist on disk.
- **Stage 2** re-queries Reddit API every time (no cache — rules evolve).
- **Stage 3 pre-cleanup** removes stale match files before running Phase 2.
- **Stage 5 pre-cleanup** removes stale tree/thread files for subs no longer
  in the Stage 3 manifest.
- **Stages 8, 9a–d, 10** overwrite their outputs.

If you change a threshold in `config.py` (e.g. `GOLD_PERCENTILE`), rerun from
Stage 3 onwards. If you only change a Stage 9b grid boundary, rerun only 9b
onwards.

## Caveats

- **CUDA device defaults in Stages 8, 9a, 9c**: each script sets a default
  `CUDA_VISIBLE_DEVICES` via `os.environ.setdefault(...)`, so your shell's
  `CUDA_VISIBLE_DEVICES=<ids> python …` takes precedence. Edit the default
  (4, 3, 0 respectively) only if you always want a different fallback.
- **Stage 2 requires Reddit API access** — no cached snapshot is shipped with
  the pipeline because rules evolve. See `../credentials/` for the credential
  template.
- **Stage 9d is manual** — the overrides file is hand-edited based on
  Stage 9c's output. The paper's clusters reflect these refinements.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `praw.exceptions.ResponseException: 429` in Stage 2 | add more keys to `credentials/reddit_api_keys.json`, reduce concurrency |
| Stage 3 OOMs a GPU | shrink `max_model_len` in `utils/embedding_matcher.py` or reduce per-bucket subreddit count |
| Stage 8 vLLM OOM | lower `max_model_len=8192`; use a smaller judge model; or set `tensor_parallel_size>1` across GPUs |
| Stage 9b grid search slow | reduce `PROCESSES` or parameter grid in `9b_cluster_embeddings.py` |
| Stage 4 reports "Arctic Shift file not found" for some subs | re-check Stage 0 coverage for those subreddits; their comment/submission `.zst` may be missing from the torrent snapshot |
| Stage 7 can't download many URLs | normal — historical URLs decay; the pass rate is used as a filter, not a hard error |
