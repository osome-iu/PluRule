#!/usr/bin/env python3
"""
PluRule Hydrate Step 2: Download media for hydrated submissions.

Iterates the hydrated JSON.zst files produced by hydrate/1_hydrate_dataset.py,
downloads each submission's media via the shared `utils.media` helper, and
writes the actual local file paths back into each submission's `media_files`
(replacing the `[NEEDS_HYDRATION]` placeholders).

Priority hierarchy and filtering rules match pipeline/7_collect_media.py
(same shared helper).  Media downloads are best-effort — dead URLs are
expected and don't abort the script; failures are counted and reported.

Usage:
    python hydrate/2_download_media.py
    python hydrate/2_download_media.py --splits test
    python hydrate/2_download_media.py --num-workers 8
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from tqdm import tqdm
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e.name}.\n"
        f"Install hydrate requirements: pip install -r requirements-hydrate.txt"
    )

from utils.files import read_compressed_json, write_compressed_json
from utils.media import categorize_error, create_session, download_submission_media

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Per-submission worker (thread-pool; I/O-bound HTTP)
# ---------------------------------------------------------------------------

def _download_one(args: Tuple[Dict, str]) -> Dict[str, Any]:
    """Download media for one submission; per-thread requests.Session."""
    submission_obj, media_dir = args
    session = _thread_local_session()
    return download_submission_media(submission_obj, media_dir, session)


# ThreadPoolExecutor workers don't persist locals; use module-level cache keyed
# by thread id so each worker reuses one Session across its tasks.
import threading
_SESSIONS = threading.local()


def _thread_local_session():
    s = getattr(_SESSIONS, "session", None)
    if s is None:
        s = create_session()
        _SESSIONS.session = s
    return s


# ---------------------------------------------------------------------------
# Split driver
# ---------------------------------------------------------------------------

def hydrate_split_media(
    split: str,
    dataset_dir: Path,
    media_root: Path,
    num_workers: int,
    skip_existing: bool,
) -> Dict[str, Any]:
    """Update media_files in a hydrated split in place; return stats."""
    in_path = dataset_dir / f"{split}_hydrated_clustered.json.zst"
    if not in_path.exists():
        print(f"⚠️  Skipping {split}: {in_path} not found")
        return {}

    print(f"\n📂 Loading {in_path}")
    data = read_compressed_json(str(in_path))

    # Collect tasks: (subreddit_idx, submission_id, submission_obj, media_dir)
    tasks: List[Tuple[int, str, Dict, str]] = []
    skipped_cached = 0

    for sub_idx, sub_data in enumerate(data.get("subreddits", [])):
        if sub_data.get("hydration_status") == "source_unavailable":
            continue
        subreddit = (sub_data.get("subreddit") or "").lower().strip()
        if not subreddit:
            continue
        media_dir = str(media_root / subreddit)

        for sub_id, entry in sub_data.get("submissions", {}).items():
            sub_obj = entry.get("submission_object")
            if not isinstance(sub_obj, dict):
                continue
            if sub_obj.get("hydration_status") == "missing":
                continue

            existing = entry.get("media_files") or []
            already_on_disk = [p for p in existing if isinstance(p, str) and os.path.exists(p)]
            expected = entry.get("num_media", 0)
            if skip_existing and len(already_on_disk) >= expected > 0:
                entry["media_files"] = already_on_disk
                skipped_cached += 1
                continue

            tasks.append((sub_idx, sub_id, sub_obj, media_dir))

    if skipped_cached:
        print(f"   ✓ {skipped_cached} submissions already have media on disk (skip-existing)")
    print(f"   {len(tasks)} submissions to process")

    if not tasks:
        out_path = dataset_dir / f"{split}_hydrated_clustered.json.zst"
        size_mb = write_compressed_json(data, str(out_path))
        return {
            "split": split, "output": str(out_path), "size_mb": round(size_mb, 1),
            "submissions_processed": 0, "files_downloaded": 0,
            "status_breakdown": {}, "error_breakdown": {},
        }

    # Parallel downloads. HTTP is I/O-bound → threads, not processes.
    status_counts: Dict[str, int] = defaultdict(int)
    error_counts: Dict[str, int] = defaultdict(int)
    total_files = 0
    submissions: Dict[str, int] = {"processed": 0}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as pool, \
         tqdm(total=len(tasks), desc=f"{split} media", unit="sub") as pbar:
        futures = {
            pool.submit(_download_one, (sub_obj, media_dir)): (sub_idx, sub_id)
            for sub_idx, sub_id, sub_obj, media_dir in tasks
        }
        for future in as_completed(futures):
            sub_idx, sub_id = futures[future]
            try:
                result = future.result()
            except Exception as e:
                error_counts[categorize_error(str(e))] += 1
                pbar.update(1)
                continue

            status_counts[result["status"]] += 1
            total_files += result["files_downloaded"]
            submissions["processed"] += 1

            for err in result.get("errors", []):
                error_counts[categorize_error(err)] += 1

            # Update media_files paths in place
            entry = data["subreddits"][sub_idx]["submissions"][sub_id]
            entry["media_files"] = result["file_paths"]

            pbar.update(1)

    elapsed = time.time() - t0
    print(f"   ✅ {submissions['processed']} submissions, {total_files} files in {elapsed:.1f}s")

    # Write updated hydrated JSON
    data.setdefault("metadata", {})
    data["metadata"]["media_hydrated_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_path = dataset_dir / f"{split}_hydrated_clustered.json.zst"
    size_mb = write_compressed_json(data, str(out_path))
    print(f"💾 {out_path} ({size_mb:.1f} MB)")

    return {
        "split": split,
        "output": str(out_path),
        "size_mb": round(size_mb, 1),
        "submissions_processed": submissions["processed"],
        "files_downloaded": total_files,
        "status_breakdown": dict(status_counts),
        "error_breakdown": dict(sorted(error_counts.items(), key=lambda x: -x[1])[:20]),
        "processing_time_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download media for hydrated PluRule submissions."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("./data"),
        help="Directory containing {split}_hydrated_clustered.json.zst",
    )
    parser.add_argument(
        "--media-dir", type=Path, default=Path("./data/media"),
        help="Root directory for downloaded media (per-subreddit subdirs auto-created)",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
    )
    parser.add_argument(
        "--num-workers", type=int, default=16,
        help="Parallel HTTP workers (threads, I/O-bound). Default 16.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip submissions whose expected media files already exist on disk.",
    )
    args = parser.parse_args()

    args.media_dir.mkdir(parents=True, exist_ok=True)

    all_stats: Dict[str, Dict] = {}
    totals = {"submissions_processed": 0, "files_downloaded": 0}
    start = time.time()

    for split in args.splits:
        stats = hydrate_split_media(
            split, args.dataset_dir, args.media_dir,
            args.num_workers, args.skip_existing,
        )
        if not stats:
            continue
        all_stats[split] = stats
        totals["submissions_processed"] += stats.get("submissions_processed", 0)
        totals["files_downloaded"] += stats.get("files_downloaded", 0)

    elapsed = time.time() - start

    summary = {
        "media_hydration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(args.dataset_dir),
        "media_dir": str(args.media_dir),
        "splits": list(all_stats.keys()),
        "totals": totals,
        "total_time_seconds": round(elapsed, 1),
        "per_split": all_stats,
    }

    summary_path = args.dataset_dir / "hydrate_media_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n📊 Summary")
    print(f"   processed:       {totals['submissions_processed']:,} submissions")
    print(f"   files downloaded:{totals['files_downloaded']:,}")
    print(f"   time:            {elapsed:.1f}s")
    print(f"   summary:         {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
