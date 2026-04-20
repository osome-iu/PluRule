#!/usr/bin/env python3
"""
Stage 0: Download the full Pushshift / Arctic Shift subreddit torrent.

Fetches all ~80K per-subreddit `{sub}_comments.zst` + `{sub}_submissions.zst`
files from the academictorrents Pushshift archive into `config.PUSHSHIFT_DATA`,
normalised to first-letter bucket layout
(`<PUSHSHIFT_DATA>/<letter>/<Subreddit>_comments.zst`) so Stages 1, 4, and 6 can
locate files via `<PUSHSHIFT_DATA>/<first_letter>/<sub>_<kind>.zst`.

For hydrating the released PluRule benchmark (not end-to-end reconstruction),
use `hydrate/0_download.py` instead — it downloads only the ~3,978 files
actually referenced by the benchmark.

Requirements:
    pip install -r requirements.txt
    aria2c on PATH (conda install -c conda-forge aria2)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PUSHSHIFT_DATA, create_directories
from utils.logging import (
    get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue,
)
from utils.pushshift_download import (
    PUSHSHIFT_TORRENT_URL,
    check_aria2c,
    ensure_torrent,
    parse_torrent,
    reorganize_to_letter_buckets,
    run_aria2c,
    scan_local_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 0: Download full Pushshift/Arctic Shift subreddit torrent."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(PUSHSHIFT_DATA),
        help=f"Destination (default from config.PUSHSHIFT_DATA: {PUSHSHIFT_DATA})",
    )
    parser.add_argument(
        "--torrent-file", type=Path, default=None,
        help="Pre-downloaded .torrent file (skip fetch from academictorrents.com)",
    )
    args = parser.parse_args()

    logger = get_stage_logger(0, "download_data")
    log_stage_start(logger, 0, "Download Pushshift/Arctic Shift subreddit torrent")
    start_time = time.time()

    try:
        create_directories()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Obtain .torrent
        torrent_path = args.torrent_file or (args.output_dir / "pushshift.torrent")
        if args.torrent_file is None:
            ensure_torrent(torrent_path, PUSHSHIFT_TORRENT_URL)
        elif not torrent_path.exists():
            sys.exit(f"Torrent file not found: {torrent_path}")

        # 2. Parse torrent to know the full expected set
        logger.info("Parsing torrent metadata...")
        all_files = parse_torrent(torrent_path)
        expected = {basename_lower for _, basename_lower, _ in all_files}
        logger.info(f"   torrent declares {len(expected):,} files")

        # 3. Check what's already present (from previous runs or existing mirror)
        logger.info(f"Scanning existing files in {args.output_dir}...")
        _, missing_to_dl, current_map = scan_local_files(args.output_dir, expected)
        logger.info(f"   already present: {len(current_map):,}  /  still need: {len(missing_to_dl):,}")

        if missing_to_dl:
            check_aria2c()
            logger.info(f"Starting aria2c → {args.output_dir}")
            rc = run_aria2c(torrent_path, indices=None, output_dir=args.output_dir)
            if rc != 0:
                logger.warning(f"aria2c exited with code {rc} (may have partial set)")

            # 4. Normalise layout into letter buckets
            logger.info("Reorganizing into first-letter bucket layout...")
            _, _, downloaded_map = scan_local_files(args.output_dir, expected)
            final_map = reorganize_to_letter_buckets(args.output_dir, downloaded_map)
        else:
            logger.info("All files already present; skipping aria2c.")
            final_map = current_map

        # 5. Final verify
        _, missing_final, verified_map = scan_local_files(args.output_dir, expected)
        elapsed = time.time() - start_time

        log_data = {
            "output_dir": str(args.output_dir),
            "torrent_url": PUSHSHIFT_TORRENT_URL if args.torrent_file is None else None,
            "torrent_file": str(torrent_path),
            "total_files_in_torrent": len(expected),
            "files_present_count": len(verified_map),
            "files_missing_count": len(missing_final),
            "elapsed_seconds": round(elapsed, 1),
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        log_path = os.path.join(PATHS["logs"], "stage0_download_log.json")
        os.makedirs(PATHS["logs"], exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Stage 0 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Files: {len(verified_map):,} / {len(expected):,} "
                    f"({len(missing_final):,} missing)")
        logger.info(f"Log: {log_path}")

        log_stage_end(logger, 0, success=not missing_final, elapsed_time=elapsed)
        return 0 if not missing_final else 1

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 0 execution")
        log_stage_end(logger, 0, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    sys.exit(main())
