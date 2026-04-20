# Hydrating PluRule

This directory contains the three scripts a user runs to **reconstitute the full
PluRule benchmark** from the released dehydrated dataset.

If you instead want to rebuild PluRule from scratch starting from the raw
Pushshift archives, see [`../pipeline/README.md`](../pipeline/README.md).

## Why hydration?

The released dataset ships only IDs, metadata, rules, cluster labels, and answer
options — every comment body, submission, and media file is replaced with a
`[NEEDS_HYDRATION]` placeholder. This keeps the distribution small and avoids
redistributing Reddit content that originates from the Pushshift archives. To
run the benchmark you first populate those placeholders from a local Pushshift
mirror (which you download from Academic Torrents) and then download the
submission media.

## Prerequisites

- Python **3.10+**
- `aria2c` on `PATH` (for torrent download)
- A BitTorrent port open in your firewall
- **Disk**: plan for several hundred GB to ~1–2 TB for the Pushshift subset.
  Large subreddits (`r/askreddit`, `r/worldnews`, …) contribute most of the
  volume; small subreddits are tens of MB each.
- **Bandwidth**: torrent throughput depends on seeders; budget several hours.

### Install

The quickest path uses the bundled conda env (pulls `aria2` from conda-forge so
you don't need root):

```bash
conda env create -f ../environment-hydrate.yml
conda activate plurule-hydrate
```

If you already have a Python environment, install the minimal hydrate deps
from the yml's `pip:` section (`zstandard`, `orjson`, `tqdm`, `requests`,
`torf`) and make sure `aria2c` is on PATH:

```
Debian/Ubuntu:  sudo apt install aria2
macOS:          brew install aria2
Fedora/CentOS:  sudo dnf install aria2
No root:        conda install -c conda-forge aria2
```

### Get the dehydrated dataset

Place the three dehydrated split files under `./data/`:

```
data/
├── train_dehydrated_clustered.json.zst
├── val_dehydrated_clustered.json.zst
└── test_dehydrated_clustered.json.zst
```

<!-- TODO: replace with the HuggingFace / Zenodo URL once the dataset is published -->

## Quick start

Three steps, from repo root:

```bash
# 1. Download the Pushshift subset referenced by the dataset (~3,978 files)
python hydrate/0_download.py

# 2. Fill every [NEEDS_HYDRATION] placeholder using the downloaded archives
python hydrate/1_hydrate_dataset.py

# 3. (Optional) Download submission images
python hydrate/2_download_media.py
```

After step 1 the Pushshift subset lives under the path configured in
`config.PUSHSHIFT_DATA`. After step 2 you have
`data/{train,val,test}_hydrated_clustered.json.zst`. After step 3 those same
files have their `media_files` arrays populated with local paths.

---

## 0. `0_download.py` — fetch the Pushshift subset

Reads the dehydrated splits, computes the set of per-subreddit comment and
submission files referenced (~3,978 files across ~1,989 subreddits), fetches
only those from the Arctic Shift / Pushshift
[academictorrent](https://academictorrents.com/details/3e3f64dee22dc304cdd2546254ca1f8e8ae542b4)
via `aria2c`, and reorganizes them into a first-letter bucket layout:

```
<output-dir>/
├── a/
│   ├── askreddit_comments.zst
│   ├── askreddit_submissions.zst
│   └── …
├── b/
│   └── …
└── hydrate_manifest.json
```

### Common invocations

```bash
# Default (reads ./data, writes to config.PUSHSHIFT_DATA)
python hydrate/0_download.py

# Preview torrent match without downloading
python hydrate/0_download.py --dry-run

# Custom output directory
python hydrate/0_download.py --output-dir /mnt/big/pushshift

# Skip the torrent; build manifest from an existing local mirror
python hydrate/0_download.py --from-dir /gpfs/.../Arcticshift/Subreddits/subreddits
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--dataset-dir` | `./data` | where the three `*_dehydrated_clustered.json.zst` files live |
| `--output-dir` | `config.PUSHSHIFT_DATA` | destination for Pushshift files |
| `--torrent-file` | *(fetched)* | use a pre-downloaded `.torrent` instead of the Academic Torrents URL |
| `--dry-run` | off | preview match report without downloading |
| `--from-dir` | *(off)* | skip torrent; use an existing local mirror |

### What it writes

- Downloaded files under `<output-dir>/<letter>/`
- `<output-dir>/hydrate_manifest.json` — `basename_to_path` map consumed by step 1
- `<output-dir>/pushshift.torrent` — cached `.torrent` so re-runs don't re-fetch

### Resuming

`aria2c` keeps `.aria2` control files next to each download. Re-running the
script picks up where it left off. Files already in the letter-bucket layout
are detected and not re-downloaded.

### Subreddits missing from the torrent

Expect a small tail (<2%) of subreddits in the dataset that aren't in this
particular torrent snapshot (renamed, banned, or post-cutoff subs). The script
reports them and writes their names to `hydrate_manifest.json`; step 1 marks
those subreddits with `hydration_status: source_unavailable`.

---

## 1. `1_hydrate_dataset.py` — fill the placeholders

Streams each Pushshift file exactly **once** across all three splits (most
subreddits appear in multiple splits), extracts only the referenced comment
and submission IDs, and fills the placeholders in each split's JSON.

### Run

```bash
# Default (reads ./data + config.PUSHSHIFT_DATA, writes ./data)
python hydrate/1_hydrate_dataset.py

# Only one split
python hydrate/1_hydrate_dataset.py --splits test

# Tune parallelism (default from config.PROCESSES)
python hydrate/1_hydrate_dataset.py --num-workers 16
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--dataset-dir` | `./data` | input dehydrated files |
| `--pushshift-dir` | `config.PUSHSHIFT_DATA` | where `hydrate_manifest.json` lives |
| `--output-dir` | `./data` | output hydrated files |
| `--splits` | all | subset of {train, val, test} |
| `--num-workers` | `config.PROCESSES` | parallel subreddit workers |

### How it fills things

| Placeholder in dehydrated JSON | Filled by step 1 from |
|---|---|
| `submissions[sid].submission_object` | `{sub}_submissions.zst` |
| `thread_pairs[i].mod_comment` | `{sub}_comments.zst` (id = `mod_comment_id`) |
| `thread_pairs[i].violating_thread` | root→leaf walk of `violating_thread_ids` |
| `thread_pairs[i].compliant_thread` | root→leaf walk of `compliant_thread_ids` |
| `submissions[sid].media_files` | **not filled** — see step 2 |

Missing IDs (the Pushshift archive doesn't contain them) become
`{"hydration_status": "missing", "id": ...}` instead of aborting the script.
Subreddits whose Pushshift files aren't in the manifest have a
`hydration_status: "source_unavailable"` flag set on their `sub_data`; their
thread pairs are left with placeholders in place. Partial hydration is fine —
the script always exits 0 and records everything in the summary.

### Output

- `./data/{train,val,test}_hydrated_clustered.json.zst` — same schema as
  `pipeline/10_assign_cluster_labels.py`'s hydrated output
- `./data/hydrate_summary.json` — per-split counts + list of
  source-unavailable subreddits

---

## 2. `2_download_media.py` — submission images (optional)

For each hydrated submission, follows the priority hierarchy
(`media_metadata` → `url` → `oembed` → `preview`), validates Content-Type,
caps files at 50 MB, and writes actual local paths into each submission's
`media_files` array in the hydrated JSON.

This step reuses the same extraction + download logic as
`pipeline/7_collect_media.py` via `utils/media.py`.

### Run

```bash
# Default
python hydrate/2_download_media.py

# Only test split, more parallelism
python hydrate/2_download_media.py --splits test --num-workers 32

# Skip submissions whose media is already on disk
python hydrate/2_download_media.py --skip-existing
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--dataset-dir` | `./data` | hydrated files from step 1 |
| `--media-dir` | `./data/media` | where images land (per-subreddit subdirs) |
| `--splits` | all | subset of {train, val, test} |
| `--num-workers` | 16 | HTTP threads (I/O-bound; threads, not processes) |
| `--skip-existing` | off | keep existing `media_files` paths that still exist on disk |

### What to expect

- Media is **best-effort**. Many historical Reddit URLs are dead or rate-limit.
  A 60–80% success rate is typical. The benchmark works fine without 100%
  media coverage; models that don't consume images are unaffected.
- Videos, crossposts, and NSFW submissions are skipped at the top (same rule
  as the pipeline).
- Files are named `{submission_id}_{media_id}.{ext}` or
  `{submission_id}_{index}_{safe_media_id}.{ext}` for gallery items.

### Output

- `./data/media/<subreddit>/<submission_id>_*.{jpg,png,gif,webp,bmp}`
- Each `submission.media_files` array in the hydrated JSON now holds real paths
- `./data/hydrate_media_summary.json` — per-split status / error counts

---

## Output format

After all three steps, each `{split}_hydrated_clustered.json.zst` matches the
schema produced by `pipeline/10_assign_cluster_labels.py`:

```jsonc
{
  "metadata": { /* split-level, hydration dates, version */ },
  "subreddits": [
    {
      "subreddit": "excel",
      "title": "...",
      "description": "...",
      "language": "en",
      "rules": [ /* full rule objects with cluster ids */ ],
      "subreddit_cluster_id":    2,
      "subreddit_cluster_label": "tech communities",
      "submissions": {
        "<submission_id>": {
          "submission_object": { /* full submission JSON */ },
          "num_media": 1,
          "media_files": ["data/media/excel/<id>_direct.png"]
        }
      },
      "thread_pairs": [
        {
          "mod_comment_id": "...",
          "mod_comment":    { /* full comment */ },
          "violating_thread":  [ /* root→leaf comments, each with level */ ],
          "compliant_thread":  [ /* same */ ],
          "violating_answer_options": [ /* shuffled MCQ */ ],
          "violating_correct_answer": "(c)",
          "compliant_answer_options": [ /* shuffled MCQ */ ],
          "compliant_correct_answer": "(b)",
          "metadata": {
            "rule": "No low-effort posts",
            "rule_cluster_id":    5,
            "rule_cluster_label": "spam / self-promotion",
            /* plus similarity score, depths, scores, ancestor IDs, … */
          }
        }
      ]
    }
  ]
}
```

## Re-running

All three scripts are safe to re-run:

- Step 0: `aria2c` resumes from `.aria2` control files. Files already in the
  letter-bucket layout are detected and skipped.
- Step 1: overwrites `*_hydrated_clustered.json.zst` each run.
- Step 2: with `--skip-existing`, submissions whose media already exists on
  disk are not re-downloaded.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `aria2c not found` | install it (see Prerequisites) |
| Step 0 very slow | few seeders for some files; try again later, or use `--from-dir` with a local mirror |
| Step 1 OOMs on a big subreddit | use `--num-workers 1`; the streaming hydrator caps per-subreddit memory at only the needed IDs, not the full file — if you still OOM, file an issue |
| Step 1 reports many missing IDs for one sub | that subreddit's Pushshift file is truncated or corrupt; re-download just that pair via `aria2c --torrent-file=... --select-file=<idx>` |
| Step 2 dies with 429s | lower `--num-workers`, the retry logic backs off but heavy parallelism against single hosts (e.g. Imgur) can trip limits |
| `hydration_status: source_unavailable` on several subs | those subs aren't in the Pushshift torrent snapshot — expected for a small tail of renamed/banned subreddits |
