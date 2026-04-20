#!/usr/bin/env python3
"""
Stage 6: Collect Submissions from Discussion Threads

Collects submission data for submissions referenced in discussion thread pairs.
Uses Pushshift subreddit-specific submission files for efficient lookup.

Input:
- discussion_threads/{subreddit}_discussion_threads.pkl (from Stage 5)
- Pushshift: {first_letter}/{subreddit}_submissions.zst

Output:
- submissions/{subreddit}_submissions.zst
- stage6_submission_collection_stats.json
"""

import sys
import os
import time
import pickle
from typing import Dict, Set, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, PUSHSHIFT_DATA, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import write_json_file, process_files_parallel, json_loads, process_zst_file_multi, read_json_file
from utils.reddit import validate_submission_structure


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_submission_ids_from_threads(subreddit: str, logger) -> Set[str]:
    """Extract unique submission IDs from a subreddit's discussion thread pairs."""
    threads_file = os.path.join(PATHS['discussion_threads'], f"{subreddit}_discussion_threads.pkl")

    if not os.path.exists(threads_file):
        logger.warning(f"⚠️  No discussion threads file found for {subreddit}")
        return set()

    try:
        with open(threads_file, 'rb') as f:
            threads_data = pickle.load(f)

        submission_ids = set()
        for pair in threads_data.get('thread_pairs', []):
            submission_id = pair.get('metadata', {}).get('submission_id')
            if submission_id:
                submission_ids.add(submission_id)

        return submission_ids

    except Exception as e:
        logger.error(f"❌ Error reading threads for {subreddit}: {e}")
        return set()


def get_pushshift_submission_file(subreddit: str) -> str:
    """Get path to Pushshift submission file for a subreddit (case-insensitive)."""
    first_char = subreddit[0].lower() if subreddit else 'unknown'

    # Handle numeric first characters
    if not first_char.isalpha() and first_char.isdigit():
        first_char = str(first_char)

    dir_path = os.path.join(PUSHSHIFT_DATA, first_char)
    if not os.path.exists(dir_path):
        return None

    # Case-insensitive filename lookup
    target_filename_lower = f"{subreddit.lower()}_submissions.zst"
    try:
        for filename in os.listdir(dir_path):
            if filename.lower() == target_filename_lower:
                return os.path.join(dir_path, filename)
    except Exception:
        pass

    return None


# ============================================================================
# SUBREDDIT PROCESSING
# ============================================================================

def process_subreddit_submissions(args: tuple) -> Dict[str, Any]:
    """Collect submissions for a single subreddit from Pushshift."""
    subreddit, target_submission_ids, output_dir = args
    worker_logger = get_stage_logger(6, "collect_submissions", worker_identifier=f"subreddits/{subreddit}")

    worker_logger.info(f"🔄 Processing r/{subreddit} ({len(target_submission_ids)} target submissions)")
    start_time = time.time()

    # Locate Pushshift file
    pushshift_file = get_pushshift_submission_file(subreddit)
    if not pushshift_file:
        worker_logger.warning(f"⚠️  No Pushshift submission file found for {subreddit}")
        return {
            'subreddit': subreddit,
            'submissions_collected': 0,
            'lines_processed': 0,
            'processing_time': 0,
            'success': False,
            'error': 'Pushshift file not found'
        }

    output_file = os.path.join(output_dir, f"{subreddit}_submissions.zst")

    try:
        def submission_filter(line: str, _state: Dict) -> Dict[str, Any]:
            """Filter for target submission IDs and validate structure."""
            try:
                submission = json_loads(line)
                submission_id = submission.get('id')

                # Check if this is a target submission with valid structure
                if submission_id in target_submission_ids and validate_submission_structure(submission):
                    return {
                        'matched': True,
                        'output_files': [output_file],
                        'data': submission
                    }

                return {'matched': False}

            except Exception:
                return {'matched': False}

        # Stream Pushshift file and extract matching submissions
        stats = process_zst_file_multi(
            pushshift_file,
            submission_filter,
            {},
            progress_interval=10_000_000,
            logger=worker_logger
        )

        elapsed = time.time() - start_time
        submissions_collected = stats['lines_matched']

        worker_logger.info(f"✅ r/{subreddit}: {submissions_collected:,} submissions from "
                          f"{stats['lines_processed']:,} lines in {elapsed:.1f}s")

        return {
            'subreddit': subreddit,
            'submissions_collected': submissions_collected,
            'lines_processed': stats['lines_processed'],
            'processing_time': elapsed,
            'success': True
        }

    except Exception as e:
        worker_logger.error(f"❌ Error processing r/{subreddit}: {e}")
        return {
            'subreddit': subreddit,
            'submissions_collected': 0,
            'lines_processed': 0,
            'processing_time': 0,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger = get_stage_logger(6, "collect_submissions")
    log_stage_start(logger, 6, "Collect Submissions from Pushshift")
    start_time = time.time()

    try:
        create_directories()

        # Validate Pushshift directory exists
        if not os.path.exists(PUSHSHIFT_DATA):
            logger.error(f"❌ Pushshift directory not found: {PUSHSHIFT_DATA}")
            log_stage_end(logger, 6, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Load subreddits from Stage 6
        logger.info("📋 Loading subreddits from Stage 5...")
        summary_file = os.path.join(PATHS['data'], 'stage5_trees_and_threads_summary.json')

        if not os.path.exists(summary_file):
            logger.error(f"❌ Stage 5 summary not found: {summary_file}")
            log_stage_end(logger, 6, success=False, elapsed_time=time.time() - start_time)
            return 1

        summary = read_json_file(summary_file)
        qualified_subreddits = summary.get('subreddit_stats', [])

        if not qualified_subreddits:
            logger.error("❌ No subreddits found from Stage 5!")
            log_stage_end(logger, 6, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"Loaded {len(qualified_subreddits)} subreddits from Stage 5")

        # Extract submission IDs from discussion thread pairs
        logger.info("🔍 Extracting submission IDs from discussion threads...")
        subreddit_submission_ids = {}

        for subreddit_stat in qualified_subreddits:
            subreddit = subreddit_stat.get('subreddit')
            if not subreddit:
                continue

            submission_ids = extract_submission_ids_from_threads(subreddit, logger)
            if submission_ids:
                subreddit_submission_ids[subreddit] = submission_ids

        total_submission_ids = sum(len(ids) for ids in subreddit_submission_ids.values())
        logger.info(f"📊 Found {len(subreddit_submission_ids)} subreddits with submission IDs")
        logger.info(f"📊 Total unique submission IDs to collect: {total_submission_ids:,}")

        if not subreddit_submission_ids:
            logger.error("❌ No submission IDs found!")
            log_stage_end(logger, 6, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Process subreddits in parallel
        logger.info(f"🗂️  Processing {len(subreddit_submission_ids)} subreddits with {PROCESSES} processes")
        processing_start = time.time()

        subreddit_args = [
            (subreddit, submission_ids, PATHS['submissions'])
            for subreddit, submission_ids in subreddit_submission_ids.items()
        ]
        results = process_files_parallel(subreddit_args, process_subreddit_submissions, PROCESSES, logger)

        processing_elapsed = time.time() - processing_start

        # Aggregate results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        total_submissions = sum(r['submissions_collected'] for r in successful_results)
        total_lines = sum(r['lines_processed'] for r in successful_results)

        # Create summary statistics
        summary = {
            'summary': {
                'total_subreddits': len(results),
                'successful_subreddits': len(successful_results),
                'failed_subreddits': len(failed_results),
                'total_submissions_collected': total_submissions,
                'total_lines_processed': total_lines,
                'processing_time_seconds': round(processing_elapsed, 1),
                'total_time_seconds': round(time.time() - start_time, 1),
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'Pushshift subreddit submission files'
            },
            'subreddit_stats': [
                {
                    'subreddit': r['subreddit'],
                    'submissions_collected': r['submissions_collected'],
                    'lines_processed': r['lines_processed'],
                    'processing_time': round(r['processing_time'], 2)
                }
                for r in sorted(successful_results, key=lambda x: x['submissions_collected'], reverse=True)
            ],
            'failed_subreddits': [
                {'subreddit': r['subreddit'], 'error': r.get('error', 'unknown')}
                for r in failed_results
            ]
        }

        # Save summary
        stats_file = os.path.join(PATHS['data'], 'stage6_submission_collection_stats.json')
        write_json_file(summary, stats_file, pretty=True)

        # Log results
        overall_elapsed = time.time() - start_time
        logger.info(f"🎉 Stage 6 Complete!")
        logger.info(f"Time: {overall_elapsed:.1f}s")
        logger.info(f"📊 Subreddits processed: {len(successful_results)}/{len(subreddit_submission_ids)}")
        logger.info(f"📄 Submissions collected: {total_submissions:,}")
        logger.info(f"📈 Lines processed: {total_lines:,}")
        logger.info(f"Summary: {stats_file}")

        if failed_results:
            logger.warning(f"⚠️  Failed subreddits ({len(failed_results)}):")
            for r in failed_results[:10]:
                logger.warning(f"  r/{r['subreddit']}: {r.get('error', 'unknown')}")
            if len(failed_results) > 10:
                logger.warning(f"  ... and {len(failed_results) - 10} more")

        log_stage_end(logger, 6, success=True, elapsed_time=overall_elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 6 execution")
        log_stage_end(logger, 6, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
