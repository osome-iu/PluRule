#!/usr/bin/env python3
"""
Stage 1: Collect Moderator Comments from Arctic Shift

Extracts distinguished moderator comments from Arctic Shift subreddit files,
filtering out bots and AutoModerator. Directly writes organized output by subreddit.

This replaces the old Stage 1 (collect from RC files) + Stage 3 (consolidate by subreddit).

Input:  Arctic Shift subreddit comment files
Output: top_subreddits/{subreddit}_mod_comments.jsonl.zst + stage1_subreddit_mod_comment_rankings.json
"""

import sys
import os
import time
from collections import defaultdict
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, ARCTIC_SHIFT_DATA, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, process_zst_file_multi)
from utils.reddit import is_bot_or_automoderator, is_moderator_reply_to_comment, normalize_subreddit_name, clean_rule_text


def get_all_arctic_shift_subreddits(logger):
    """Get list of all subreddit comment files from Arctic Shift.

    Returns a list of (subreddit_name, filepath, file_size) tuples sorted
    by file_size descending so the largest subreddits start first (better
    load balancing across workers). Uses os.scandir() so directory metadata
    comes back in the same RPC as readdir() on supporting filesystems.
    """
    subreddit_files = []

    with os.scandir(ARCTIC_SHIFT_DATA) as shard_iter:
        for shard_entry in shard_iter:
            if not shard_entry.is_dir(follow_symlinks=False):
                continue
            with os.scandir(shard_entry.path) as file_iter:
                for file_entry in file_iter:
                    if not file_entry.name.endswith('_comments.zst'):
                        continue
                    subreddit_name = file_entry.name.replace('_comments.zst', '')
                    try:
                        size = file_entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        size = 0
                    subreddit_files.append((
                        normalize_subreddit_name(subreddit_name),
                        file_entry.path,
                        size,
                    ))

    subreddit_files.sort(key=lambda t: t[2], reverse=True)
    logger.info(f"Found {len(subreddit_files)} subreddits in Arctic Shift")
    return subreddit_files


def process_subreddit(args: tuple) -> Dict[str, Any]:
    """
    Process a single subreddit from Arctic Shift: filter for mod comments and write to output.
    Uses process_zst_file_multi for efficient streaming.
    """
    subreddit, arctic_file, _ = args  # Third element is file_size (for sorting only)
    worker_logger = get_stage_logger(1, "collect_mod_comments", worker_identifier=f"subreddits/{subreddit}")

    worker_logger.info(f"🔄 Processing r/{subreddit}")
    start_time = time.time()

    output_file = os.path.join(PATHS['top_subreddits'], f"{subreddit}_mod_comments.jsonl.zst")

    # Skip if output already exists
    if os.path.exists(output_file):
        worker_logger.info(f"⏭️  Skipping r/{subreddit} - output already exists")
        return {'subreddit': subreddit, 'skipped': True, 'mod_comment_count': 0}

    try:
        def mod_comment_filter(line: str, state: Dict) -> Dict[str, Any]:
            """Filter for moderator comments, excluding bots."""
            # Quick pre-filter before JSON parsing
            if not ('"distinguished":"moderator"' in line and '"parent_id":"t1_' in line):
                return {'matched': False}

            try:
                comment = json_loads(line)

                # Check if it's a moderator comment
                if not is_moderator_reply_to_comment(comment):
                    return {'matched': False}

                # Filter out bots and AutoModerator
                author = comment.get('author', '')
                if is_bot_or_automoderator(author):
                    return {'matched': False}

                # Add cleaned body text
                body = comment.get('body', '')
                comment['body_clean'] = clean_rule_text(body)

                # Valid mod comment - write to output
                return {
                    'matched': True,
                    'output_files': [output_file],
                    'data': comment
                }

            except Exception:
                return {'matched': False}

        # Use process_zst_file_multi for efficient streaming
        stats = process_zst_file_multi(arctic_file, mod_comment_filter, {},
                                      progress_interval=10_000_000, logger=worker_logger)

        elapsed = time.time() - start_time
        mod_count = stats['lines_matched']

        worker_logger.info(f"✅ r/{subreddit}: {mod_count:,} mod comments from {stats['lines_processed']:,} lines in {elapsed:.1f}s")

        return {
            'subreddit': subreddit,
            'mod_comment_count': mod_count,
            'lines_processed': stats['lines_processed'],
            'processing_time': elapsed,
            'success': True
        }

    except Exception as e:
        worker_logger.error(f"❌ Error processing r/{subreddit}: {e}")
        return {'subreddit': subreddit, 'error': str(e), 'mod_comment_count': 0, 'success': False}


def generate_rankings(results: list) -> dict:
    """Generate subreddit rankings from processing results."""
    # Filter successful results and sort by mod comment count
    successful = [r for r in results if r.get('success') and not r.get('skipped')]
    sorted_results = sorted(successful, key=lambda x: x['mod_comment_count'], reverse=True)

    total_mod_comments = sum(r['mod_comment_count'] for r in successful)

    rankings_data = {
        "summary": {
            "total_subreddits": len(sorted_results),
            "total_mod_comments": total_mod_comments,
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Arctic Shift subreddit files"
        },
        "rankings": [
            {
                "rank": rank,
                "subreddit": r['subreddit'],
                "mod_comment_count": r['mod_comment_count'],
                "lines_processed": r.get('lines_processed', 0),
                "processing_time": round(r.get('processing_time', 0), 2)
            }
            for rank, r in enumerate(sorted_results, 1)
        ]
    }

    return rankings_data


def main():
    """Main execution function."""
    logger = get_stage_logger(1, "collect_mod_comments")
    log_stage_start(logger, 1, "Collect Moderator Comments from Arctic Shift")
    start_time = time.time()

    try:
        create_directories()

        if not os.path.exists(ARCTIC_SHIFT_DATA):
            logger.error(f"❌ Arctic Shift directory not found: {ARCTIC_SHIFT_DATA}")
            log_stage_end(logger, 1, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Get all subreddit files from Arctic Shift (with disk cache to avoid
        # re-walking GPFS metadata on every rerun -- the walk takes ~20 minutes).
        cache_file = os.path.join(PATHS['data'], 'stage1_arctic_shift_discovery_cache.json')
        if os.path.exists(cache_file):
            logger.info(f"📁 Loading Arctic Shift discovery cache: {cache_file}")
            cache_data = read_json_file(cache_file)
            subreddit_files = [tuple(t) for t in cache_data.get('subreddit_files', [])]
            logger.info(f"   Loaded {len(subreddit_files)} subreddits from cache")
            logger.info(f"   (Delete {os.path.basename(cache_file)} to force rediscovery)")
        else:
            logger.info("📁 Discovering subreddits from Arctic Shift...")
            subreddit_files = get_all_arctic_shift_subreddits(logger)
            logger.info(f"💾 Saving discovery cache: {cache_file}")
            write_json_file(
                {'arctic_shift_root': ARCTIC_SHIFT_DATA,
                 'subreddit_files': subreddit_files},
                cache_file, pretty=False
            )

        if not subreddit_files:
            logger.error("❌ No subreddit files found in Arctic Shift")
            log_stage_end(logger, 1, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Process subreddits in parallel
        logger.info(f"🗂️  Processing {len(subreddit_files)} subreddits with {PROCESSES} processes")
        processing_start = time.time()

        results = process_files_parallel(subreddit_files, process_subreddit, PROCESSES, logger)

        processing_elapsed = time.time() - processing_start

        # Count results
        successful = [r for r in results if r.get('success')]
        skipped = [r for r in results if r.get('skipped')]
        failed = [r for r in results if not r.get('success') and not r.get('skipped')]

        logger.info(f"✅ Processing complete in {processing_elapsed:.1f}s")
        logger.info(f"   📊 {len(successful)} successful, {len(skipped)} skipped, {len(failed)} failed")

        if failed:
            logger.warning(f"⚠️  Failed subreddits ({len(failed)}):")
            for r in failed[:10]:
                logger.warning(f"     r/{r['subreddit']}: {r.get('error', 'unknown')}")
            if len(failed) > 10:
                logger.warning(f"     ... and {len(failed) - 10} more")

        # Generate and save rankings
        logger.info("📈 Generating subreddit rankings...")
        rankings_data = generate_rankings(results)
        rankings_file = os.path.join(PATHS['data'], 'stage1_subreddit_mod_comment_rankings.json')
        write_json_file(rankings_data, rankings_file, pretty=True)

        # Summary
        elapsed = time.time() - start_time
        total_comments = rankings_data['summary']['total_mod_comments']
        total_subreddits = rankings_data['summary']['total_subreddits']

        logger.info(f"🎉 Stage 1 Complete!")
        logger.info(f"   ⏱️  Total time: {elapsed:.1f}s")
        logger.info(f"   💬 Total mod comments: {total_comments:,}")
        logger.info(f"   🗂️  Unique subreddits: {total_subreddits:,}")
        logger.info(f"   📈 Rankings: {rankings_file}")

        # Show top 10 subreddits
        logger.info(f"🏆 Top 10 subreddits by mod comment count:")
        for entry in rankings_data['rankings'][:10]:
            logger.info(f"   {entry['rank']:2d}. r/{entry['subreddit']}: {entry['mod_comment_count']:,}")

        log_stage_end(logger, 1, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 1 execution")
        log_stage_end(logger, 1, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
