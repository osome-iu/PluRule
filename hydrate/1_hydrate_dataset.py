#!/usr/bin/env python3
"""
PluRule Hydrate Step 1: Fill [NEEDS_HYDRATION] placeholders.

Reads the dehydrated PluRule dataset + the Pushshift subset produced by
hydrate/0_download.py, writes hydrated JSON.zst files whose structure matches
what pipeline/10_assign_cluster_labels.py's hydrated output produces -- except
that `media_files` paths remain placeholders until hydrate/2_download_media.py
runs.

Subreddits appear across multiple splits (every sub appears in test, many
in val/train), so we hydrate subreddit-centric rather than split-centric:
each Pushshift zst file is streamed exactly once, and the extracted
submissions/comments are distributed to whichever split(s) need them.

Per unique subreddit we:
  1. Union the comment + submission IDs referenced across all splits.
  2. Stream `{sub}_comments.zst` once, keep only matching IDs.
  3. Stream `{sub}_submissions.zst` once, keep only matching IDs.
  4. Fill placeholders in every split's copy of the sub_data.
  5. Missing IDs become {"hydration_status": "missing", "id": ...}.
  6. Subreddits whose Pushshift files are absent from the manifest get
     `hydration_status: "source_unavailable"` on every split's sub_data.

Usage:
    python hydrate/1_hydrate_dataset.py
    python hydrate/1_hydrate_dataset.py --splits test
    python hydrate/1_hydrate_dataset.py --num-workers 8
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Add repo root to path (for utils/)
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import zstandard
    from tqdm import tqdm
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e.name}.\n"
        f"Install hydrate requirements: pip install -r requirements-hydrate.txt"
    )

from config import PROCESSES, PUSHSHIFT_DATA
from utils.files import json_loads, read_compressed_json, write_compressed_json
from utils.logging import setup_stage_logger

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------

_READ_CHUNK = 1 << 24  # 16 MB — matches utils.files.read_and_decode


def iter_zst_jsonl(path: str):
    """Yield parsed JSON objects from a .zst-compressed JSONL file.

    Bytes-based line split (safe because '\\n' = 0x0A is never inside a
    multibyte UTF-8 sequence). Uses a mutable `bytearray` buffer with a
    cursor to avoid repeated slice allocation.
    """
    with open(path, "rb") as f:
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(f) as reader:
            buf = bytearray()
            while True:
                chunk = reader.read(_READ_CHUNK)
                if not chunk:
                    break
                buf.extend(chunk)
                start = 0
                n = len(buf)
                while start < n:
                    nl = buf.find(b"\n", start)
                    if nl < 0:
                        break
                    if nl > start:
                        try:
                            yield json_loads(bytes(buf[start:nl]))
                        except Exception:
                            pass
                    start = nl + 1
                if start:
                    del buf[:start]
            if buf:
                try:
                    yield json_loads(bytes(buf))
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Per-subreddit helpers
# ---------------------------------------------------------------------------

def _collect_needed_ids(sub_data: Dict) -> Tuple[Set[str], Set[str]]:
    """Union of comment IDs and submission IDs referenced by one sub_data entry."""
    comment_ids: Set[str] = set()
    submission_ids: Set[str] = set(sub_data.get("submissions", {}).keys())

    for pair in sub_data.get("thread_pairs", []):
        mid = pair.get("mod_comment_id")
        if mid:
            comment_ids.add(mid)
        for tid in pair.get("violating_thread_ids") or []:
            if tid:
                comment_ids.add(tid)
        for tid in pair.get("compliant_thread_ids") or []:
            if tid:
                comment_ids.add(tid)
    return comment_ids, submission_ids


def _extract_by_id(path: str, id_field: str, needed: Set[str]) -> Dict[str, Dict]:
    """Stream a .zst JSONL file; return {id: obj} for ids in `needed`. Early-exits once full."""
    out: Dict[str, Dict] = {}
    if not needed:
        return out
    target = len(needed)
    for obj in iter_zst_jsonl(path):
        oid = obj.get(id_field)
        if oid in needed and oid not in out:
            out[oid] = obj
            if len(out) == target:
                break
    return out


def _fill_sub_data(
    sub_data: Dict,
    comments_by_id: Dict[str, Dict],
    submissions_by_id: Dict[str, Dict],
    split_counts: Dict[str, Any],
) -> None:
    """Fill [NEEDS_HYDRATION] placeholders in one sub_data in place."""
    # Submissions
    for sid, entry in sub_data.get("submissions", {}).items():
        obj = submissions_by_id.get(sid)
        if obj is not None:
            entry["submission_object"] = obj
            split_counts["hydrated_submissions"] += 1
        else:
            entry["submission_object"] = {"hydration_status": "missing", "id": sid}
            split_counts["missing_submissions"] += 1

    # Thread pairs
    for pair in sub_data.get("thread_pairs", []):
        # Mod comment
        mid = pair.get("mod_comment_id")
        mod_obj = comments_by_id.get(mid) if mid else None
        if mod_obj is not None:
            pair["mod_comment"] = mod_obj
            split_counts["hydrated_comments"] += 1
        else:
            pair["mod_comment"] = {"hydration_status": "missing", "id": mid}
            split_counts["missing_comments"] += 1

        # Threads (root -> leaf, matching Stage 5 output)
        for mode in ("violating", "compliant"):
            ids = pair.get(f"{mode}_thread_ids") or []
            thread: List[Dict] = []
            for level, cid in enumerate(ids):
                obj = comments_by_id.get(cid)
                if obj is not None:
                    c = dict(obj)
                    c["level"] = level
                    thread.append(c)
                    split_counts["hydrated_comments"] += 1
                else:
                    thread.append({"hydration_status": "missing", "id": cid, "level": level})
                    split_counts["missing_comments"] += 1
            pair[f"{mode}_thread"] = thread


def hydrate_subreddit_unified(args: Tuple[str, List[Dict], List[Tuple[str, int]], str, str, Set[str], Set[str]]
                              ) -> Tuple[str, List[Dict], List[Tuple[str, int]], Dict[str, Any]]:
    """
    Stream Pushshift files once for a subreddit, fill every split's sub_data copy.

    Returns (name, hydrated_sub_dicts, placements, stats).
    `placements[i]` is (split, pos) in splits_data[split]["subreddits"] that sub_dicts[i] came from.
    """
    name, sub_dicts, placements, c_path, s_path, c_ids, s_ids = args

    comments_by_id = _extract_by_id(c_path, "id", c_ids) if c_ids else {}
    submissions_by_id = _extract_by_id(s_path, "id", s_ids) if s_ids else {}

    stats: Dict[str, Any] = {
        "subreddit": name,
        "splits": [s for s, _ in placements],
        "per_split": {},
    }

    for (split, _pos), sub_data in zip(placements, sub_dicts):
        per = {
            "hydrated_submissions": 0, "missing_submissions": 0,
            "hydrated_comments": 0, "missing_comments": 0,
        }
        _fill_sub_data(sub_data, comments_by_id, submissions_by_id, per)
        stats["per_split"][split] = per

    return name, sub_dicts, placements, stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_manifest(pushshift_dir: Path, logger) -> Dict[str, str]:
    """Load basename -> absolute path map written by hydrate/0_download.py."""
    manifest_path = pushshift_dir / "hydrate_manifest.json"
    if not manifest_path.exists():
        logger.error(
            f"Manifest not found: {manifest_path}. Run hydrate/0_download.py first."
        )
        sys.exit(1)
    with open(manifest_path) as f:
        m = json.load(f)
    b2p = m.get("basename_to_path", {})
    if not b2p:
        logger.error(
            f"{manifest_path} has no `basename_to_path`. Re-run hydrate/0_download.py."
        )
        sys.exit(1)
    return b2p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hydrate the PluRule dehydrated dataset using downloaded Pushshift files."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("./data"))
    parser.add_argument("--pushshift-dir", type=Path, default=Path(PUSHSHIFT_DATA),
                        help=f"Directory containing hydrate_manifest.json (default: {PUSHSHIFT_DATA})")
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
        help="Splits to hydrate (default: all)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=PROCESSES,
        help=f"Parallel workers (default: {PROCESSES}, from config.PROCESSES). "
             f"Recommend 8-16 for I/O-bound work on network filesystems.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_stage_logger("hydrate1_hydrate_dataset")
    logger.info("=" * 60)
    logger.info("🚀 Hydrate Step 1: Fill [NEEDS_HYDRATION] placeholders")
    logger.info("=" * 60)

    # 1. Load manifest
    logger.info(f"📋 Loading manifest from {args.pushshift_dir}...")
    basename_to_path = load_manifest(args.pushshift_dir, logger)
    logger.info(f"   {len(basename_to_path)} Pushshift files available")

    # 2. Load ALL selected splits upfront
    splits_data: Dict[str, Dict] = {}
    for split in args.splits:
        path = args.dataset_dir / f"{split}_dehydrated_clustered.json.zst"
        if not path.exists():
            logger.warning(f"⚠️  Skipping {split}: {path} not found")
            continue
        logger.info(f"📂 Loading {path}")
        splits_data[split] = read_compressed_json(str(path))

    if not splits_data:
        logger.error("No split files loaded")
        sys.exit(1)

    # 3. Build cross-split index: subreddit -> [(split, pos, sub_data_ref), ...]
    by_subreddit: Dict[str, List[Tuple[str, int, Dict]]] = defaultdict(list)
    for split, data in splits_data.items():
        for pos, sub_data in enumerate(data.get("subreddits", [])):
            name = (sub_data.get("subreddit") or "").lower()
            if name:
                by_subreddit[name].append((split, pos, sub_data))

    n_unique = len(by_subreddit)
    n_entries = sum(len(v) for v in by_subreddit.values())
    logger.info(f"   {n_unique} unique subreddits across {n_entries} split entries "
                f"(saves {n_entries - n_unique} redundant stream passes)")

    # 4. Build tasks (one per unique subreddit)
    tasks = []
    source_unavailable: List[str] = []
    task_sizes: List[int] = []  # parallels `tasks`; comments-file bytes

    for name, entries in by_subreddit.items():
        c_path = basename_to_path.get(f"{name}_comments.zst")
        s_path = basename_to_path.get(f"{name}_submissions.zst")
        if not c_path or not s_path:
            for _split, _pos, sub_data in entries:
                sub_data["hydration_status"] = "source_unavailable"
            source_unavailable.append(name)
            continue

        all_c_ids: Set[str] = set()
        all_s_ids: Set[str] = set()
        for _split, _pos, sub_data in entries:
            c, s = _collect_needed_ids(sub_data)
            all_c_ids |= c
            all_s_ids |= s

        sub_dicts = [sd for _, _, sd in entries]
        placements = [(split, pos) for split, pos, _ in entries]
        tasks.append((name, sub_dicts, placements, c_path, s_path, all_c_ids, all_s_ids))
        task_sizes.append(0)  # filled in next step

    if source_unavailable:
        logger.warning(f"⚠️  {len(source_unavailable)} subreddits have no Pushshift source "
                       f"(marked `hydration_status: source_unavailable`)")

    # 4b. Stat comments files in parallel and sort tasks largest-first (LPT heuristic).
    # Streaming the whole comments file is the dominant cost; dispatching biggest
    # tasks first keeps workers saturated to the end rather than finishing one
    # huge subreddit alone while everyone else idles.
    import os as _os
    from concurrent.futures import ThreadPoolExecutor as _TPE

    logger.info(f"📏 Stat-ing {len(tasks)} comments files to sort largest-first...")
    t_stat = time.time()

    def _stat(path: str) -> int:
        try:
            return _os.path.getsize(path)
        except OSError:
            return 0

    with _TPE(max_workers=32) as _p:
        task_sizes = list(_p.map(lambda t: _stat(t[3]), tasks))
    logger.info(f"   done in {time.time() - t_stat:.1f}s")

    order = sorted(range(len(tasks)), key=lambda i: task_sizes[i], reverse=True)
    tasks = [tasks[i] for i in order]
    task_sizes = [task_sizes[i] for i in order]
    if tasks:
        top5 = ", ".join(f"r/{tasks[i][0]} ({task_sizes[i] / (1 << 30):.1f} GB)"
                         for i in range(min(5, len(tasks))))
        logger.info(f"   largest first: {top5}")

    # 5. Parallel hydration
    logger.info(f"🚀 Hydrating {len(tasks)} unique subreddits ({args.num_workers} workers)...")
    t0 = time.time()

    # Aggregate per-split stats across all subreddits
    totals_by_split: Dict[str, Dict[str, int]] = {
        split: {"hydrated_submissions": 0, "missing_submissions": 0,
                "hydrated_comments": 0, "missing_comments": 0}
        for split in splits_data
    }

    with Pool(args.num_workers) as pool, tqdm(total=len(tasks), desc="subreddits") as pbar:
        for name, hydrated_dicts, placements, stats in pool.imap_unordered(
                hydrate_subreddit_unified, tasks):
            # Replace sub_data in each split (pickling lost the by-reference link)
            for (split, pos), hydrated in zip(placements, hydrated_dicts):
                splits_data[split]["subreddits"][pos] = hydrated
            # Accumulate per-split totals
            for split, per in stats["per_split"].items():
                for k, v in per.items():
                    totals_by_split[split][k] += v
            pbar.update(1)

    elapsed = time.time() - t0
    logger.info(f"✅ Hydrated in {elapsed:.1f}s")

    # 6. Write each split
    summary: Dict[str, Any] = {
        "hydration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(args.dataset_dir),
        "pushshift_dir": str(args.pushshift_dir),
        "output_dir": str(args.output_dir),
        "unique_subreddits": n_unique,
        "split_entries": n_entries,
        "source_unavailable_count": len(source_unavailable),
        "source_unavailable": source_unavailable,
        "hydration_elapsed_s": round(elapsed, 1),
        "splits": {},
    }

    for split, data in splits_data.items():
        data.setdefault("metadata", {})
        data["metadata"]["hydrated_from"] = "dehydrated_clustered"
        data["metadata"]["hydration_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["metadata"].pop("instructions", None)

        out_path = args.output_dir / f"{split}_hydrated_clustered.json.zst"
        size_mb = write_compressed_json(data, str(out_path))

        totals = totals_by_split[split]
        logger.info(f"💾 {split}: {out_path} ({size_mb:.1f} MB) — "
                    f"subs:{totals['hydrated_submissions']:,}/{totals['missing_submissions']:,} miss, "
                    f"cmts:{totals['hydrated_comments']:,}/{totals['missing_comments']:,} miss")

        summary["splits"][split] = {
            "output": str(out_path),
            "size_mb": round(size_mb, 1),
            "totals": totals,
        }

    summary_path = args.output_dir / "hydrate_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"📊 Summary: {summary_path}")

    # Partial hydration is acceptable; exit 0 regardless.
    return 0


if __name__ == "__main__":
    sys.exit(main())
