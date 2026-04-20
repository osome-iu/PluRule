#!/usr/bin/env python3
"""
Stage 4: Collect and Organize Submission Comments (MEMORY-OPTIMIZED)

Pass 1: Filter Pushshift to temp file (all matching comments), count per submission
Pass 2: Stream temp file, deduplicate, write submissions when complete

Peak memory: <1 GB (only one submission in memory at a time)
Output: organized_comments/{subreddit}/submission_{id}.pkl
"""

import sys
import os
import time
import pickle
import tempfile
import shutil
from collections import defaultdict
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, PUSHSHIFT_DATA, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, process_zst_file_multi)
from utils.reddit import extract_submission_id, normalize_subreddit_name


def load_submission_ids(logger):
    """Load submission IDs from Stage 3."""
    data = read_json_file(os.path.join(PATHS['data'], 'stage3_subreddit_submission_ids.json'))
    subreddit_to_ids = {}
    for subreddit, submission_ids in data['subreddit_submission_ids'].items():
        subreddit_to_ids[normalize_subreddit_name(subreddit)] = set(submission_ids)
    logger.info(f"📋 Loaded {sum(len(ids) for ids in subreddit_to_ids.values()):,} submission IDs from {len(subreddit_to_ids)} subreddits")
    return subreddit_to_ids


def get_pushshift_comment_file(subreddit: str) -> str:
    """Get Pushshift comment file path."""
    first_char = subreddit[0].lower() if subreddit else 'unknown'
    if not first_char.isalpha() and first_char.isdigit():
        first_char = str(first_char)

    dir_path = os.path.join(PUSHSHIFT_DATA, first_char)
    if not os.path.exists(dir_path):
        return None

    target_filename_lower = f"{subreddit.lower()}_comments.zst"
    try:
        for filename in os.listdir(dir_path):
            if filename.lower() == target_filename_lower:
                return os.path.join(dir_path, filename)
    except Exception:
        pass
    return None


def check_is_removed(comment: dict) -> bool:
    """Check if comment is [removed] or [deleted]."""
    body = comment.get('body', '')
    author = comment.get('author', '')
    return body in ['[removed]', '[deleted]'] or author in ['[deleted]', '[removed]']


def process_subreddit_comments(args: tuple) -> Dict[str, Any]:
    """
    Two-pass processing:
    Pass 1: Filter to temp file, count expected comments per submission
    Pass 2: Deduplicate and write submissions
    """
    subreddit, target_submission_ids, output_dir = args
    worker_logger = get_stage_logger(4, "collect_submission_comments", worker_identifier=f"subreddits/{subreddit}")

    worker_logger.info(f"🔄 Processing {subreddit} ({len(target_submission_ids)} target submissions)")
    start_time = time.time()

    pushshift_file = get_pushshift_comment_file(subreddit)
    if not pushshift_file:
        worker_logger.warning(f"⚠️  No Pushshift file found for {subreddit}")
        return {'subreddit': subreddit, 'comments_collected': 0, 'submissions_with_comments': 0,
                'lines_processed': 0, 'removed_deleted_count': 0, 'preserved_from_removal_count': 0,
                'overwritten_with_better_count': 0, 'processing_time': 0, 'success': False,
                'error': 'Pushshift file not found'}

    temp_dir = None
    try:
        # ==================== PASS 1: Filter to Temp File ====================
        worker_logger.info(f"   📊 Pass 1: Filtering comments...")
        pass1_start = time.time()

        temp_dir = tempfile.mkdtemp(prefix=f"stage4_{subreddit}_")
        temp_file = os.path.join(temp_dir, "filtered_comments.zst")

        # Track expected line count per submission (including duplicates)
        expected_lines = defaultdict(int)

        def comment_filter(line: str, state: Dict) -> Dict[str, Any]:
            try:
                comment = json_loads(line)
                submission_id = extract_submission_id(comment.get('link_id', ''))
                if submission_id in target_submission_ids:
                    state['expected_lines'][submission_id] += 1
                    return {'matched': True, 'output_files': [temp_file], 'data': comment}
            except Exception:
                pass
            return {'matched': False}

        state = {'expected_lines': expected_lines}
        filter_stats = process_zst_file_multi(pushshift_file, comment_filter, state, progress_interval=10_000_000, logger=worker_logger)
        expected_lines = state['expected_lines']

        pass1_elapsed = time.time() - pass1_start
        worker_logger.info(f"   ✅ Pass 1: {filter_stats['lines_matched']:,} comments from {filter_stats['lines_processed']:,} lines in {pass1_elapsed:.1f}s")

        # ==================== PASS 2: Deduplicate and Write ====================
        worker_logger.info(f"   💾 Pass 2: Deduplicating and writing submissions...")
        pass2_start = time.time()

        # Create output directory
        subreddit_output_dir = os.path.join(output_dir, subreddit)
        os.makedirs(subreddit_output_dir, exist_ok=True)

        # Process temp file with deduplication
        submission_comments = defaultdict(dict)
        lines_processed_per_submission = defaultdict(int)
        submissions_written = 0
        removed_deleted_count = 0
        preserved_from_removal_count = 0
        overwritten_with_better_count = 0

        for line_data in read_zst_lines(temp_file):
            try:
                comment = json_loads(line_data)
                submission_id = extract_submission_id(comment.get('link_id', ''))
                comment_id = comment.get('id')
                is_removed = check_is_removed(comment)

                # Track lines processed for this submission
                lines_processed_per_submission[submission_id] += 1

                if comment_id not in submission_comments[submission_id]:
                    # First encounter
                    submission_comments[submission_id][comment_id] = comment
                    if is_removed:
                        removed_deleted_count += 1
                else:
                    # Duplicate - apply deduplication logic
                    existing = submission_comments[submission_id][comment_id]
                    existing_is_removed = check_is_removed(existing)

                    if not is_removed and existing_is_removed:
                        # New is clean, old was removed - OVERWRITE
                        submission_comments[submission_id][comment_id] = comment
                        overwritten_with_better_count += 1
                    elif is_removed and not existing_is_removed:
                        # New is removed, old was clean - PRESERVE
                        preserved_from_removal_count += 1
                    # else: keep first occurrence

                # Check if submission is complete (processed all lines for this submission)
                if lines_processed_per_submission[submission_id] == expected_lines[submission_id]:
                    # Write to file
                    output_file = os.path.join(subreddit_output_dir, f"submission_{submission_id}.pkl")
                    with open(output_file, 'wb') as f:
                        pickle.dump(submission_comments[submission_id], f, protocol=pickle.HIGHEST_PROTOCOL)

                    submissions_written += 1
                    if submissions_written % 100 == 0:
                        worker_logger.info(f"      Written {submissions_written}/{len(expected_lines)} submissions")

                    # Free memory
                    del submission_comments[submission_id]
                    del lines_processed_per_submission[submission_id]

            except Exception as e:
                worker_logger.debug(f"Failed to process line in Pass 2: {e}")
                pass

        pass2_elapsed = time.time() - pass2_start
        worker_logger.info(f"   ✅ Pass 2: {submissions_written} submissions in {pass2_elapsed:.1f}s")

        # Cleanup
        shutil.rmtree(temp_dir)
        temp_dir = None

        elapsed = time.time() - start_time
        avg = filter_stats['lines_matched'] / submissions_written if submissions_written > 0 else 0

        worker_logger.info(f"✅ {subreddit}: {filter_stats['lines_matched']:,} comments across {submissions_written} submissions (avg {avg:.1f}) in {elapsed:.1f}s")
        if removed_deleted_count > 0:
            worker_logger.info(f"   📊 {removed_deleted_count:,} [removed]/[deleted]")
        if preserved_from_removal_count > 0:
            worker_logger.info(f"   🛡️  {preserved_from_removal_count:,} preserved")
        if overwritten_with_better_count > 0:
            worker_logger.info(f"   ✨ {overwritten_with_better_count:,} recovered")

        return {
            'subreddit': subreddit,
            'comments_collected': filter_stats['lines_matched'],
            'submissions_with_comments': submissions_written,
            'lines_processed': filter_stats['lines_processed'],
            'removed_deleted_count': removed_deleted_count,
            'preserved_from_removal_count': preserved_from_removal_count,
            'overwritten_with_better_count': overwritten_with_better_count,
            'processing_time': elapsed,
            'success': True
        }

    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        worker_logger.error(f"❌ Error: {e}")
        import traceback
        worker_logger.error(traceback.format_exc())
        return {'subreddit': subreddit, 'comments_collected': 0, 'submissions_with_comments': 0,
                'lines_processed': 0, 'removed_deleted_count': 0, 'preserved_from_removal_count': 0,
                'overwritten_with_better_count': 0, 'processing_time': 0, 'success': False, 'error': str(e)}


def main():
    """Main orchestration."""
    logger = get_stage_logger(4, "collect_submission_comments")
    log_stage_start(logger, 4, "Collect and Organize Submission Comments")
    overall_start = time.time()

    try:
        create_directories()

        if not os.path.exists(PUSHSHIFT_DATA):
            logger.error(f"❌ Pushshift not found: {PUSHSHIFT_DATA}")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - overall_start)
            return 1

        logger.info("📋 Loading submission IDs...")
        subreddit_to_ids = load_submission_ids(logger)

        if not subreddit_to_ids:
            logger.error("❌ No submission IDs found")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - overall_start)
            return 1

        logger.info(f"🗂️  Processing {len(subreddit_to_ids)} subreddits with {PROCESSES} processes")
        logger.info(f"⚡ Memory-optimized: <1GB per worker")

        # Prepare arguments for parallel processing
        subreddit_args = [(subreddit, target_ids, PATHS['organized_comments'])
                         for subreddit, target_ids in subreddit_to_ids.items()]
        results = process_files_parallel(subreddit_args, process_subreddit_comments, PROCESSES, logger)

        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        total_comments = sum(r['comments_collected'] for r in successful_results)
        total_submissions = sum(r['submissions_with_comments'] for r in successful_results)
        total_lines = sum(r['lines_processed'] for r in successful_results)

        logger.info(f"✅ Complete: {len(successful_results)}/{len(subreddit_to_ids)} subreddits")
        logger.info(f"   📊 {total_lines:,} lines → {total_comments:,} comments → {total_submissions:,} submissions")

        # Write stats
        stats_data = {
            'summary': {
                'total_subreddits': len(results),
                'successful_subreddits': len(successful_results),
                'failed_subreddits': len(failed_results),
                'total_comments_collected': total_comments,
                'total_submissions_with_comments': total_submissions,
                'total_lines_processed': total_lines,
                'total_removed_deleted': sum(r.get('removed_deleted_count', 0) for r in successful_results),
                'total_preserved_from_removal': sum(r.get('preserved_from_removal_count', 0) for r in successful_results),
                'total_overwritten_with_better': sum(r.get('overwritten_with_better_count', 0) for r in successful_results),
                'avg_comments_per_submission': round(total_comments / total_submissions, 2) if total_submissions > 0 else 0,
                'processing_time': round(time.time() - overall_start, 1),
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'Pushshift',
                'version': 'memory_optimized_two_pass'
            },
            'subreddit_stats': [
                {
                    'subreddit': r['subreddit'],
                    'comments_collected': r['comments_collected'],
                    'submissions_with_comments': r['submissions_with_comments'],
                    'lines_processed': r['lines_processed'],
                    'removed_deleted_count': r.get('removed_deleted_count', 0),
                    'preserved_from_removal_count': r.get('preserved_from_removal_count', 0),
                    'overwritten_with_better_count': r.get('overwritten_with_better_count', 0),
                    'processing_time': round(r['processing_time'], 2)
                }
                for r in sorted(successful_results, key=lambda x: x['comments_collected'], reverse=True)
            ],
            'failed_subreddits': [{'subreddit': r['subreddit'], 'error': r.get('error', 'unknown')} for r in failed_results]
        }

        stats_file = os.path.join(PATHS['data'], 'stage4_submission_comment_collection_stats.json')
        write_json_file(stats_data, stats_file, pretty=True)

        if failed_results:
            logger.warning(f"⚠️  Failed: {len(failed_results)} subreddits")

        log_stage_end(logger, 4, success=True, elapsed_time=time.time() - overall_start)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 4")
        log_stage_end(logger, 4, success=False, elapsed_time=time.time() - overall_start)
        return 1


if __name__ == "__main__":
    exit(main())
