#!/usr/bin/env python3
"""
Stage 7: Collect Media for Submissions (end-to-end reconstruction).

Streams per-subreddit `{subreddit}_submissions.zst` from Stage 6 output and
downloads each submission's media via the shared helper in `utils/media.py`.
For benchmark hydration (not reconstruction) see `hydrate/2_download_media.py`.

Input:
- output/submissions/{subreddit}_submissions.zst  (from Stage 6)
- data/stage6_submission_collection_stats.json    (subreddit list)

Output:
- output/media/{subreddit}/{submission_id}_{media_id}.{ext}
- data/stage7_media_collection_stats.json
- data/stage7_successful_submission_ids.json
"""

import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES
from utils.files import (
    json_loads, process_files_parallel, read_json_file, read_zst_lines, write_json_file,
)
from utils.logging import get_stage_logger, log_stage_end, log_stage_start
from utils.media import categorize_error, create_session, download_submission_media


# ---------------------------------------------------------------------------
# Subreddit processor
# ---------------------------------------------------------------------------

def process_subreddit(args: Tuple) -> Dict[str, Any]:
    """Download media for every submission in one subreddit."""
    subreddit, = args
    logger = get_stage_logger(7, "collect_media", worker_identifier=f"subreddits/{subreddit}")
    logger.info(f"🔄 Processing r/{subreddit}")

    start_time = time.time()
    submissions_file = os.path.join(PATHS['submissions'], f"{subreddit}_submissions.zst")
    media_dir = os.path.join(PATHS['media'], subreddit)

    if not os.path.exists(submissions_file):
        logger.warning("⚠️  Submissions file not found")
        return {'subreddit': subreddit, 'error': 'submissions_file_not_found'}

    try:
        session = create_session()
        status_counts: Dict[str, int] = defaultdict(int)
        error_counts: Dict[str, int] = defaultdict(int)
        successful_ids = []
        total_files = 0
        submission_count = 0

        for line in read_zst_lines(submissions_file):
            if not line.strip():
                continue

            submission = json_loads(line)
            result = download_submission_media(submission, media_dir, session)

            status_counts[result['status']] += 1
            total_files += result['files_downloaded']
            submission_count += 1

            if result['status'] in ('complete', 'no_media'):
                successful_ids.append(result['submission_id'])

            for err in result.get('errors', []):
                error_counts[categorize_error(err)] += 1

            if submission_count % 100 == 0:
                logger.info(f"  Progress: {submission_count} submissions, {total_files} files")

        elapsed = time.time() - start_time
        logger.info(f"✅ {submission_count} submissions, {total_files} files in {elapsed:.1f}s")

        return {
            'subreddit': subreddit,
            'submission_count': submission_count,
            'total_files': total_files,
            'status_counts': dict(status_counts),
            'error_counts': dict(error_counts),
            'successful_ids': successful_ids,
            'processing_time': elapsed,
        }

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'subreddit': subreddit, 'error': str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger = get_stage_logger(7, "collect_media")
    log_stage_start(logger, 7, "Collect Media for Submissions")
    start_time = time.time()

    try:
        logger.info("📋 Loading subreddits from Stage 6...")
        summary_file = os.path.join(PATHS['data'], 'stage6_submission_collection_stats.json')
        if not os.path.exists(summary_file):
            logger.error(f"❌ Stage 6 summary not found: {summary_file}")
            log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
            return 1

        summary = read_json_file(summary_file)
        qualified_stats = summary.get('subreddit_stats', [])
        if not qualified_stats:
            logger.error("❌ No subreddits found from Stage 6!")
            log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
            return 1

        subreddits = [s['subreddit'] for s in qualified_stats]
        logger.info(f"Processing {len(subreddits)} subreddits with {PROCESSES} workers")

        args_list = [(s,) for s in subreddits]
        results = process_files_parallel(args_list, process_subreddit, PROCESSES, logger)

        valid = [r for r in results if 'error' not in r]
        errors = [r for r in results if 'error' in r]

        total_submissions = sum(r.get('submission_count', 0) for r in valid)
        total_files = sum(r.get('total_files', 0) for r in valid)

        global_status: Dict[str, int] = defaultdict(int)
        global_errors: Dict[str, int] = defaultdict(int)
        for r in valid:
            for k, v in r.get('status_counts', {}).items():
                global_status[k] += v
            for k, v in r.get('error_counts', {}).items():
                global_errors[k] += v

        successful_ids_by_subreddit = {
            r['subreddit']: r.get('successful_ids', []) for r in valid
        }
        total_successful_ids = sum(len(ids) for ids in successful_ids_by_subreddit.values())

        elapsed = time.time() - start_time

        out_summary = {
            'summary': {
                'total_subreddits': len(subreddits),
                'successful_subreddits': len(valid),
                'failed_subreddits': len(errors),
                'total_submissions': total_submissions,
                'total_files_downloaded': total_files,
                'total_successful_submissions': total_successful_ids,
                'processing_time_seconds': round(elapsed, 1),
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'status_breakdown': dict(global_status),
            'error_breakdown': dict(sorted(global_errors.items(), key=lambda x: -x[1])[:20]),
            'per_subreddit': [
                {
                    'subreddit': r['subreddit'],
                    'submissions': r.get('submission_count', 0),
                    'files_downloaded': r.get('total_files', 0),
                    'successful_submissions': len(r.get('successful_ids', [])),
                    'processing_time': round(r.get('processing_time', 0), 2),
                }
                for r in sorted(valid, key=lambda x: x.get('total_files', 0), reverse=True)
            ],
            'failed_subreddits': [
                {'subreddit': r['subreddit'], 'error': r.get('error', 'unknown')}
                for r in errors
            ],
        }

        stats_file = os.path.join(PATHS['data'], 'stage7_media_collection_stats.json')
        write_json_file(out_summary, stats_file, pretty=True)

        ids_output = {
            'metadata': {
                'total_subreddits': len(successful_ids_by_subreddit),
                'total_successful_submissions': total_successful_ids,
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'criteria': 'Submissions with status: complete or no_media',
            },
            'subreddit_submission_ids': successful_ids_by_subreddit,
        }
        ids_file = os.path.join(PATHS['data'], 'stage7_successful_submission_ids.json')
        write_json_file(ids_output, ids_file, pretty=True)

        logger.info("\n🎉 Stage 7 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Subreddits: {len(valid)}/{len(subreddits)}")
        logger.info(f"Submissions: {total_submissions:,}")
        logger.info(f"Files: {total_files:,}")
        logger.info(f"Successful submissions: {total_successful_ids:,}")

        logger.info("\n📊 Status Breakdown:")
        for status, count in sorted(global_status.items(), key=lambda x: -x[1]):
            logger.info(f"  {status}: {count:,}")

        if global_errors:
            logger.info("\n⚠️  Top Errors:")
            for err, count in list(sorted(global_errors.items(), key=lambda x: -x[1]))[:5]:
                logger.info(f"  {err}: {count:,}")

        logger.info(f"\nResults: {stats_file}")
        logger.info(f"Successful IDs: {ids_file}")

        if errors:
            logger.warning(f"\n❌ Failed subreddits: {len(errors)}")
            for r in errors[:10]:
                logger.warning(f"  {r['subreddit']}: {r.get('error', 'unknown')}")

        log_stage_end(logger, 7, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        logger.error(f"❌ Stage 7 failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    sys.exit(main())
