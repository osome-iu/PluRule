#!/usr/bin/env python3
"""
PluRule Hydrate Step 0: Download the Pushshift subset needed to hydrate the benchmark.

Reads the dehydrated PluRule dataset, determines which per-subreddit comment
and submission files are referenced (~3,978 files for ~1,989 subreddits), and
fetches just those from the academictorrents Pushshift/Arctic Shift archive via
aria2c. Output layout is first-letter buckets (`<output-dir>/<letter>/<Sub>_...zst`)
to match the existing pipeline layout.

Modes:
  (default)      Download via torrent
  --from-dir X   Skip download; build manifest from an existing local mirror at X
  --dry-run      Print torrent match report, no download

Usage:
    python hydrate/0_download.py
    python hydrate/0_download.py --dry-run
    python hydrate/0_download.py --from-dir /gpfs/.../Arcticshift/Subreddits/subreddits
    python hydrate/0_download.py --output-dir /mnt/big/pushshift
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PUSHSHIFT_DATA
from utils.files import read_compressed_json
from utils.pushshift_download import (
    PUSHSHIFT_TORRENT_URL,
    check_aria2c,
    ensure_torrent,
    match_basenames,
    parse_torrent,
    reorganize_to_letter_buckets,
    run_aria2c,
    scan_local_files,
)

SPLITS = ("train", "val", "test")


def collect_subreddits(dataset_dir: Path) -> Set[str]:
    """Union of subreddit names referenced across all three dehydrated splits."""
    subs: Set[str] = set()
    for split in SPLITS:
        path = dataset_dir / f"{split}_dehydrated_clustered.json.zst"
        if not path.exists():
            sys.exit(f"Missing dataset file: {path}")
        data = read_compressed_json(str(path))
        for entry in data.get("subreddits", []):
            name = (entry.get("subreddit") or "").lower().strip()
            if name:
                subs.add(name)
    return subs


def required_basenames(subreddits: Set[str]) -> Set[str]:
    """Two files per subreddit (comments + submissions)."""
    files: Set[str] = set()
    for sub in subreddits:
        files.add(f"{sub}_comments.zst")
        files.add(f"{sub}_submissions.zst")
    return files


def _write_manifest(output_dir: Path, manifest: Dict) -> Path:
    path = output_dir / "hydrate_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def _print_summary(required: int, present: int, missing: int, manifest_path: Path) -> None:
    print("\n📊 Summary")
    print(f"   required:  {required}")
    print(f"   present:   {present}")
    print(f"   missing:   {missing}")
    print(f"   manifest:  {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Pushshift subset referenced by the dehydrated PluRule dataset."
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("./data"),
        help="Directory containing {train,val,test}_dehydrated_clustered.json.zst",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(PUSHSHIFT_DATA),
        help=f"Destination (default from config.PUSHSHIFT_DATA: {PUSHSHIFT_DATA})",
    )
    parser.add_argument(
        "--torrent-file", type=Path, default=None,
        help="Pre-downloaded .torrent file (skip fetch from academictorrents.com)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview torrent matches without downloading",
    )
    parser.add_argument(
        "--from-dir", type=Path, default=None,
        help="Skip torrent; build manifest from an existing local mirror.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect needed basenames from dehydrated dataset
    print(f"📋 Collecting subreddits from dehydrated datasets in {args.dataset_dir}...")
    subreddits = collect_subreddits(args.dataset_dir)
    needed = required_basenames(subreddits)
    print(f"   {len(subreddits)} unique subreddits → {len(needed)} files required")

    # 2a. --from-dir fast path: build manifest from existing mirror
    if args.from_dir is not None:
        if not args.from_dir.exists():
            sys.exit(f"--from-dir not found: {args.from_dir}")
        print(f"📁 Scanning existing mirror at {args.from_dir}...")
        present, missing, basename_to_path = scan_local_files(args.from_dir, needed)

        manifest = {
            "dataset_dir": str(args.dataset_dir),
            "output_dir": str(args.output_dir),
            "source": "from-dir",
            "source_dir": str(args.from_dir),
            "subreddits_count": len(subreddits),
            "files_required_count": len(needed),
            "files_present_count": len(present),
            "files_missing_in_source": sorted(missing),
            "basename_to_path": basename_to_path,
        }
        manifest_path = _write_manifest(args.output_dir, manifest)
        _print_summary(len(needed), len(present), len(missing), manifest_path)
        if missing and len(missing) <= 20:
            print(f"   missing list: {sorted(missing)}")
        elif missing:
            print(f"   missing sample: {sorted(missing)[:20]} ... (+{len(missing) - 20} more)")
        return 0

    # 2b. Torrent path
    torrent_path = args.torrent_file or (args.output_dir / "pushshift.torrent")
    if args.torrent_file is None:
        ensure_torrent(torrent_path, PUSHSHIFT_TORRENT_URL)
    elif not torrent_path.exists():
        sys.exit(f"Torrent file not found: {torrent_path}")

    print("🔎 Matching required files against torrent contents...")
    all_files = parse_torrent(torrent_path)
    matched, missing_in_torrent = match_basenames(all_files, needed)
    print(f"   matched: {len(matched)} / not in torrent: {len(missing_in_torrent)}")
    if missing_in_torrent and len(missing_in_torrent) <= 20:
        print(f"   missing: {sorted(missing_in_torrent)}")
    elif missing_in_torrent:
        print(f"   missing sample: {sorted(missing_in_torrent)[:20]} ... "
              f"(+{len(missing_in_torrent) - 20} more)")

    manifest: Dict = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "source": "torrent",
        "torrent_file": str(torrent_path),
        "torrent_url": PUSHSHIFT_TORRENT_URL if args.torrent_file is None else None,
        "subreddits_count": len(subreddits),
        "files_required_count": len(needed),
        "files_matched_count": len(matched),
        "files_missing_in_torrent": sorted(missing_in_torrent),
    }

    if args.dry_run:
        manifest["torrent_relative_paths"] = {b: p for b, (_, p) in matched.items()}
        path = args.output_dir / "hydrate_manifest_dry_run.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n✅ Dry run complete: {path}")
        for _, (idx, rel) in list(matched.items())[:3]:
            print(f"     [{idx:>5}] {rel}")
        return 0

    if not matched:
        sys.exit("No files matched in torrent; nothing to download.")

    # 3. Check what's already in letter-bucket layout (re-run skip)
    print(f"🔎 Checking existing files in {args.output_dir}...")
    present, missing_to_dl, _ = scan_local_files(args.output_dir, set(matched.keys()))
    print(f"   already present: {len(present)} / still need: {len(missing_to_dl)}")

    if missing_to_dl:
        check_aria2c()
        print(f"\n🚀 aria2c: downloading to {args.output_dir}")
        rc = run_aria2c(torrent_path, [i for i, _ in matched.values()], args.output_dir)
        if rc != 0:
            print(f"⚠️  aria2c exited with code {rc} (partial set may still have downloaded)")

        # 4. Reorganize freshly-downloaded files into letter buckets
        print("🗂️  Reorganizing into first-letter bucket layout...")
        _, _, downloaded_map = scan_local_files(args.output_dir, set(matched.keys()))
        reorganize_to_letter_buckets(args.output_dir, downloaded_map)
    else:
        print("✓ All needed files already present; skipping aria2c.")

    # 5. Re-verify and write manifest (authoritative after any moves)
    present_final, missing_final, verified_map = scan_local_files(args.output_dir, needed)
    manifest["files_downloaded_count"] = len(present_final)
    manifest["files_missing_after_download"] = sorted(missing_final)
    manifest["basename_to_path"] = verified_map
    manifest_path = _write_manifest(args.output_dir, manifest)
    _print_summary(len(needed), len(present_final),
                   len(missing_final) + len(missing_in_torrent), manifest_path)

    return 0 if not missing_final else 1


if __name__ == "__main__":
    sys.exit(main())
