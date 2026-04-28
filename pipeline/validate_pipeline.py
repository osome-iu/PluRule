#!/usr/bin/env python3
"""
Pipeline Validation Script

Compares summary/manifest counts against actual file counts on disk to detect
data drift caused by partial re-runs of pipeline stages. Reports mismatches
that could lead to stale data being consumed by downstream stages.

Usage:
    python pipeline/validate_pipeline.py
    python pipeline/validate_pipeline.py --stage 3    # Validate specific stage
    python pipeline/validate_pipeline.py --verbose     # Show per-file details
"""

import sys
import os
import glob
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS
from utils.files import read_json_file


def validate_stage3(verbose=False):
    """Validate Stage 3: Match Rules consistency."""
    issues = []
    info = []

    output_dir = PATHS['matched_comments']

    # Count files on disk
    matrix_files = glob.glob(os.path.join(output_dir, '*_similarity_matrix.pt'))
    match_files = glob.glob(os.path.join(output_dir, '*_match.jsonl.zst'))
    stats_files = glob.glob(os.path.join(output_dir, '*_stats.json'))

    info.append(f"Files on disk: {len(matrix_files)} matrices, {len(match_files)} match files, {len(stats_files)} stats files")

    # Load summary
    summary_file = os.path.join(PATHS['data'], 'stage3_matching_summary.json')
    if not os.path.exists(summary_file):
        issues.append("CRITICAL: stage3_matching_summary.json not found")
        return issues, info

    summary = read_json_file(summary_file)
    summary_subreddits = {s['subreddit'] for s in summary.get('subreddit_stats', [])}
    summary_with_matches = {s['subreddit'] for s in summary.get('subreddit_stats', [])
                            if s.get('matched_comments', 0) > 0}

    info.append(f"Summary: {len(summary_subreddits)} subreddits total, {len(summary_with_matches)} with matches")

    # Check matrix count vs summary
    if len(matrix_files) != len(summary_subreddits):
        issues.append(f"Matrix count mismatch: {len(matrix_files)} on disk vs {len(summary_subreddits)} in summary")

    # Check match file subreddits vs summary subreddits with matches
    match_subreddits = {os.path.basename(f).replace('_match.jsonl.zst', '') for f in match_files}
    extra_match_files = match_subreddits - summary_with_matches
    missing_match_files = summary_with_matches - match_subreddits

    if extra_match_files:
        issues.append(f"CRITICAL: {len(extra_match_files)} match files on disk NOT in summary (stale from previous runs)")
        if verbose:
            for sub in sorted(extra_match_files)[:20]:
                issues.append(f"  - {sub}_match.jsonl.zst (stale)")

    if missing_match_files:
        issues.append(f"WARNING: {len(missing_match_files)} subreddits in summary missing match files on disk")
        if verbose:
            for sub in sorted(missing_match_files)[:20]:
                issues.append(f"  - {sub} (missing)")

    # Check date consistency of match files
    if match_files:
        mtimes = defaultdict(list)
        for f in match_files:
            mtime = os.path.getmtime(f)
            date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
            mtimes[date_str].append(os.path.basename(f))

        if len(mtimes) > 1:
            issues.append(f"WARNING: Match files span {len(mtimes)} different dates: {sorted(mtimes.keys())}")
            for date, files in sorted(mtimes.items()):
                info.append(f"  {date}: {len(files)} match files")

    # Check submission IDs manifest
    sub_ids_file = os.path.join(PATHS['data'], 'stage3_subreddit_submission_ids.json')
    if os.path.exists(sub_ids_file):
        sub_ids_data = read_json_file(sub_ids_file)
        manifest_subreddits = set(sub_ids_data.get('subreddit_submission_ids', {}).keys())
        if manifest_subreddits != summary_with_matches:
            diff = manifest_subreddits.symmetric_difference(summary_with_matches)
            issues.append(f"WARNING: Submission IDs manifest differs from summary by {len(diff)} subreddits")

    # Verify total_matched in summary vs sum of subreddit stats
    reported_total = summary.get('total_matched', 0)
    computed_total = sum(s.get('matched_comments', 0) for s in summary.get('subreddit_stats', []))
    if reported_total != computed_total:
        issues.append(f"WARNING: Summary total_matched ({reported_total}) != sum of subreddit stats ({computed_total})")

    return issues, info


def validate_stage5(verbose=False):
    """Validate Stage 5: Build Trees and Threads consistency."""
    issues = []
    info = []

    trees_dir = PATHS['comment_trees']
    threads_dir = PATHS['discussion_threads']

    # Count files on disk
    tree_files = glob.glob(os.path.join(trees_dir, '*_comment_trees.pkl')) if os.path.exists(trees_dir) else []
    thread_files = glob.glob(os.path.join(threads_dir, '*_discussion_threads.pkl')) if os.path.exists(threads_dir) else []

    tree_subreddits = {os.path.basename(f).replace('_comment_trees.pkl', '') for f in tree_files}
    thread_subreddits = {os.path.basename(f).replace('_discussion_threads.pkl', '') for f in thread_files}

    info.append(f"Files on disk: {len(tree_files)} tree files, {len(thread_files)} thread files")

    # Tree and thread files should match
    if tree_subreddits != thread_subreddits:
        only_trees = tree_subreddits - thread_subreddits
        only_threads = thread_subreddits - tree_subreddits
        if only_trees:
            issues.append(f"WARNING: {len(only_trees)} subreddits have trees but no threads")
        if only_threads:
            issues.append(f"WARNING: {len(only_threads)} subreddits have threads but no trees")

    # Load summary
    summary_file = os.path.join(PATHS['data'], 'stage5_trees_and_threads_summary.json')
    if not os.path.exists(summary_file):
        issues.append("CRITICAL: stage5_trees_and_threads_summary.json not found")
        return issues, info

    summary = read_json_file(summary_file)
    summary_stats = summary.get('subreddit_stats', [])
    summary_subreddits = {s['subreddit'] for s in summary_stats if s.get('status') == 'completed'}

    info.append(f"Summary: {len(summary_subreddits)} completed subreddits")

    # Check for stale files not in summary
    extra_trees = tree_subreddits - summary_subreddits
    extra_threads = thread_subreddits - summary_subreddits

    if extra_trees:
        issues.append(f"CRITICAL: {len(extra_trees)} tree files on disk NOT in summary (stale from previous runs)")
        if verbose:
            for sub in sorted(extra_trees)[:20]:
                issues.append(f"  - {sub}_comment_trees.pkl (stale)")

    if extra_threads:
        issues.append(f"CRITICAL: {len(extra_threads)} thread files on disk NOT in summary (stale from previous runs)")

    # Check date consistency
    for label, files in [("tree", tree_files), ("thread", thread_files)]:
        if files:
            mtimes = defaultdict(int)
            for f in files:
                mtime = os.path.getmtime(f)
                date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
                mtimes[date_str] += 1

            if len(mtimes) > 1:
                issues.append(f"WARNING: {label.capitalize()} files span {len(mtimes)} different dates: {sorted(mtimes.keys())}")
                for date, count in sorted(mtimes.items()):
                    info.append(f"  {date}: {count} {label} files")

    # Cross-validate with Stage 3
    stage3_summary_file = os.path.join(PATHS['data'], 'stage3_matching_summary.json')
    if os.path.exists(stage3_summary_file):
        stage3_summary = read_json_file(stage3_summary_file)
        stage3_with_matches = {s['subreddit'] for s in stage3_summary.get('subreddit_stats', [])
                                if s.get('matched_comments', 0) > 0}
        extra_vs_stage3 = tree_subreddits - stage3_with_matches
        if extra_vs_stage3:
            issues.append(f"CRITICAL: {len(extra_vs_stage3)} tree files for subreddits NOT in Stage 3 manifest")

    return issues, info


def validate_stage4(verbose=False):
    """Validate Stage 4: Organized Comments consistency."""
    issues = []
    info = []

    organized_dir = PATHS['organized_comments']
    if not os.path.exists(organized_dir):
        issues.append("WARNING: organized_comments directory not found")
        return issues, info

    # Count subreddit dirs on disk
    subreddit_dirs = [d for d in os.listdir(organized_dir)
                      if os.path.isdir(os.path.join(organized_dir, d))]

    # Count submission_*.pkl files inside each subreddit dir
    pkl_per_sub = {}
    total_pkls = 0
    empty_dirs = []
    for sub in subreddit_dirs:
        sub_path = os.path.join(organized_dir, sub)
        try:
            pkls = [f for f in os.listdir(sub_path)
                    if f.startswith('submission_') and f.endswith('.pkl')]
        except OSError:
            pkls = []
        pkl_per_sub[sub] = len(pkls)
        total_pkls += len(pkls)
        if not pkls:
            empty_dirs.append(sub)

    info.append(f"Files on disk: {len(subreddit_dirs)} subreddit directories, {total_pkls} submission_*.pkl files")

    if empty_dirs:
        info.append(f"{len(empty_dirs)} organized_comments dirs are empty cleanup candidates")
        if verbose:
            for sub in sorted(empty_dirs)[:20]:
                info.append(f"  empty: {sub}/")

    # Cross-validate with Stage 3 submission IDs manifest
    sub_ids_file = os.path.join(PATHS['data'], 'stage3_subreddit_submission_ids.json')
    if os.path.exists(sub_ids_file):
        sub_ids_data = read_json_file(sub_ids_file)
        manifest_map = sub_ids_data.get('subreddit_submission_ids', {})
        manifest_subreddits = set(manifest_map.keys())
        extra_dirs = set(subreddit_dirs) - manifest_subreddits
        missing_dirs = manifest_subreddits - set(subreddit_dirs)

        if extra_dirs:
            extra_empty = [sub for sub in extra_dirs if pkl_per_sub.get(sub, 0) == 0]
            extra_nonempty = [sub for sub in extra_dirs if pkl_per_sub.get(sub, 0) > 0]
            if extra_empty:
                info.append(f"{len(extra_empty)} empty organized_comments dirs are not in Stage 3 manifest")
            if extra_nonempty:
                issues.append(
                    f"WARNING: {len(extra_nonempty)} non-empty organized_comments dirs NOT in Stage 3 submission IDs manifest"
                )
                if verbose:
                    for sub in sorted(extra_nonempty)[:20]:
                        issues.append(f"  - {sub}/ ({pkl_per_sub.get(sub, 0)} .pkl files)")
        if missing_dirs:
            issues.append(f"WARNING: {len(missing_dirs)} subreddits in manifest missing organized_comments dirs")

        # Compare per-subreddit pkl counts to expected submission IDs
        undercollected = []
        for sub in sorted(set(subreddit_dirs) & manifest_subreddits):
            expected = len(manifest_map.get(sub, []))
            actual = pkl_per_sub.get(sub, 0)
            if expected and actual < expected:
                undercollected.append((sub, actual, expected))

        if undercollected:
            issues.append(f"WARNING: {len(undercollected)} subreddits have fewer .pkl files than expected from Stage 3 manifest")
            if verbose:
                for sub, actual, expected in undercollected[:20]:
                    issues.append(f"  - {sub}: {actual}/{expected} .pkl files")

    # Load Stage 4 summary
    stage4_summary_file = os.path.join(PATHS['data'], 'stage4_submission_comment_collection_stats.json')
    if os.path.exists(stage4_summary_file):
        stage4_summary = read_json_file(stage4_summary_file)
        summary_subreddits = set()
        for stats in stage4_summary.get('subreddit_stats', []):
            sub = stats.get('subreddit')
            if sub:
                summary_subreddits.add(sub)
        info.append(f"Stage 4 summary: {len(summary_subreddits)} subreddits")

        extra_vs_summary = set(subreddit_dirs) - summary_subreddits
        if extra_vs_summary:
            extra_empty = [sub for sub in extra_vs_summary if pkl_per_sub.get(sub, 0) == 0]
            extra_nonempty = [sub for sub in extra_vs_summary if pkl_per_sub.get(sub, 0) > 0]
            if extra_empty:
                info.append(f"{len(extra_empty)} empty organized_comments dirs are not in Stage 4 summary")
            if extra_nonempty:
                issues.append(f"WARNING: {len(extra_nonempty)} non-empty organized_comments dirs NOT in Stage 4 summary")

    return issues, info


def validate_stage6(verbose=False):
    """Validate Stage 6: Submissions consistency."""
    issues = []
    info = []

    submissions_dir = PATHS['submissions']
    if not os.path.exists(submissions_dir):
        issues.append("WARNING: submissions directory not found")
        return issues, info

    submission_files = glob.glob(os.path.join(submissions_dir, '*_submissions.zst'))
    submission_subreddits = {os.path.basename(f).replace('_submissions.zst', '') for f in submission_files}

    info.append(f"Files on disk: {len(submission_files)} submission files")

    # Load Stage 6 summary
    summary_file = os.path.join(PATHS['data'], 'stage6_submission_collection_stats.json')
    if os.path.exists(summary_file):
        summary = read_json_file(summary_file)
        summary_subreddits = set()
        for stats in summary.get('subreddit_stats', []):
            sub = stats.get('subreddit')
            if sub:
                summary_subreddits.add(sub)

        extra = submission_subreddits - summary_subreddits
        if extra:
            issues.append(f"WARNING: {len(extra)} submission files on disk NOT in Stage 6 summary (potentially stale)")

    # Cross-validate with Stage 5
    stage5_summary_file = os.path.join(PATHS['data'], 'stage5_trees_and_threads_summary.json')
    if os.path.exists(stage5_summary_file):
        stage5_summary = read_json_file(stage5_summary_file)
        stage5_subreddits = {s['subreddit'] for s in stage5_summary.get('subreddit_stats', [])
                             if s.get('status') == 'completed'}
        extra_vs_stage5 = submission_subreddits - stage5_subreddits
        if extra_vs_stage5:
            issues.append(f"WARNING: {len(extra_vs_stage5)} submission files for subreddits NOT in Stage 5 summary")

    return issues, info


def _check_data_files(expected, label):
    """Check existence of a list of (filename, severity) pairs in PATHS['data']."""
    issues = []
    found = 0
    for fname, sev in expected:
        fpath = os.path.join(PATHS['data'], fname)
        if os.path.exists(fpath):
            found += 1
        else:
            issues.append(f"{sev}: {label} output missing: {fname}")
    return issues, found


def _check_stage7_input_readiness():
    """Cross-check that Stage 7's successful IDs have backing tree/thread/submission files."""
    issues = []
    info = []

    stage7_file = os.path.join(PATHS['data'], 'stage7_successful_submission_ids.json')
    if not os.path.exists(stage7_file):
        info.append("Stage 7 successful IDs not found (stage 7 may not have run yet)")
        return issues, info

    stage7_data = read_json_file(stage7_file)
    if isinstance(stage7_data, dict) and 'subreddit_submission_ids' in stage7_data:
        stage7_subreddits = set(stage7_data['subreddit_submission_ids'].keys())
    elif isinstance(stage7_data, dict):
        stage7_subreddits = set(stage7_data.keys())
    else:
        stage7_subreddits = set()

    missing_trees, missing_threads, missing_submissions = [], [], []
    for subreddit in stage7_subreddits:
        if not os.path.exists(os.path.join(PATHS['comment_trees'], f"{subreddit}_comment_trees.pkl")):
            missing_trees.append(subreddit)
        if not os.path.exists(os.path.join(PATHS['discussion_threads'], f"{subreddit}_discussion_threads.pkl")):
            missing_threads.append(subreddit)
        if not os.path.exists(os.path.join(PATHS['submissions'], f"{subreddit}_submissions.zst")):
            missing_submissions.append(subreddit)

    if missing_trees:
        issues.append(f"WARNING: {len(missing_trees)} Stage 7 subreddits missing tree files")
    if missing_threads:
        issues.append(f"WARNING: {len(missing_threads)} Stage 7 subreddits missing thread files")
    if missing_submissions:
        issues.append(f"WARNING: {len(missing_submissions)} Stage 7 subreddits missing submission files")

    info.append(f"Stage 7: {len(stage7_subreddits)} subreddits with successful submissions")
    return issues, info


def validate_stage8(verbose=False):
    """Validate Stage 8: Final dataset outputs."""
    issues = []
    info = []

    expected = [
        ('train_hydrated.json.zst', 'CRITICAL'),
        ('val_hydrated.json.zst', 'CRITICAL'),
        ('test_hydrated.json.zst', 'CRITICAL'),
        ('train_dehydrated.json.zst', 'CRITICAL'),
        ('val_dehydrated.json.zst', 'CRITICAL'),
        ('test_dehydrated.json.zst', 'CRITICAL'),
        ('test_hydrated.json', 'WARNING'),
        ('stage8_final_datasets_stats.json', 'CRITICAL'),
        ('stage8_llm_verification_results.json', 'WARNING'),
        ('stage8_thread_distribution_analysis.json', 'WARNING'),
    ]
    file_issues, found = _check_data_files(expected, 'Stage 8')
    info.append(f"Stage 8 outputs found: {found}/{len(expected)} expected files")
    issues.extend(file_issues)

    # Also check Stage 7 → Stage 8 input readiness (informative; useful when re-running)
    readiness_issues, readiness_info = _check_stage7_input_readiness()
    info.extend(readiness_info)
    issues.extend(readiness_issues)

    return issues, info


def validate_stage10(verbose=False):
    """Validate Stage 10: Cluster-labeled dataset outputs."""
    issues = []
    info = []

    expected = [
        ('train_hydrated_clustered.json.zst', 'CRITICAL'),
        ('val_hydrated_clustered.json.zst', 'CRITICAL'),
        ('test_hydrated_clustered.json.zst', 'CRITICAL'),
        ('train_dehydrated_clustered.json.zst', 'WARNING'),
        ('val_dehydrated_clustered.json.zst', 'WARNING'),
        ('test_dehydrated_clustered.json.zst', 'WARNING'),
        ('test_hydrated_clustered.json', 'WARNING'),
        ('stage10_cluster_assignment_stats.json', 'CRITICAL'),
        ('stage10_dataset_stats_table.tex', 'WARNING'),
    ]
    file_issues, found = _check_data_files(expected, 'Stage 10')
    info.append(f"Stage 10 outputs found: {found}/{len(expected)} expected files")
    issues.extend(file_issues)

    # Cross-check that Stage 8 hydrated splits exist as inputs
    for split in ('train', 'val', 'test'):
        src = os.path.join(PATHS['data'], f'{split}_hydrated.json.zst')
        if not os.path.exists(src):
            issues.append(f"WARNING: Stage 10 input missing: {split}_hydrated.json.zst")

    return issues, info


def validate_stage1(verbose=False):
    """Validate Stage 1: Mod comment collection consistency."""
    issues = []
    info = []

    output_dir = PATHS['top_subreddits']
    if not os.path.exists(output_dir):
        issues.append("WARNING: top_subreddits directory not found")
        return issues, info

    mod_files = glob.glob(os.path.join(output_dir, '*_mod_comments.jsonl.zst'))
    disk_subreddits = {os.path.basename(f).replace('_mod_comments.jsonl.zst', '') for f in mod_files}

    info.append(f"Files on disk: {len(mod_files)} mod_comments files")

    rankings_file = os.path.join(PATHS['data'], 'stage1_subreddit_mod_comment_rankings.json')
    if not os.path.exists(rankings_file):
        issues.append("CRITICAL: stage1_subreddit_mod_comment_rankings.json not found")
        return issues, info

    rankings = read_json_file(rankings_file)
    ranking_entries = rankings.get('rankings', [])
    ranked_subreddits = {r['subreddit'] for r in ranking_entries}
    ranked_with_comments = {
        r['subreddit'] for r in ranking_entries
        if r.get('mod_comment_count', 0) > 0
    }
    zero_count_ranked = ranked_subreddits - ranked_with_comments
    info.append(
        f"Rankings: {len(ranked_subreddits)} subreddits "
        f"({len(ranked_with_comments)} with >0 mod comments, {len(zero_count_ranked)} with 0)"
    )

    extra_files = disk_subreddits - ranked_subreddits
    zero_count_files = disk_subreddits & zero_count_ranked
    missing_files = ranked_with_comments - disk_subreddits

    if extra_files:
        issues.append(f"CRITICAL: {len(extra_files)} mod_comments files on disk NOT in rankings (stale from previous runs)")
        if verbose:
            for sub in sorted(extra_files)[:20]:
                issues.append(f"  - {sub}_mod_comments.jsonl.zst (stale)")

    if zero_count_files:
        issues.append(f"WARNING: {len(zero_count_files)} mod_comments files exist for zero-count ranking entries")
        if verbose:
            for sub in sorted(zero_count_files)[:20]:
                issues.append(f"  - {sub}_mod_comments.jsonl.zst (zero-count entry)")

    if missing_files:
        issues.append(f"WARNING: {len(missing_files)} ranked subreddits with >0 mod comments missing files on disk")
        if verbose:
            for sub in sorted(missing_files)[:20]:
                issues.append(f"  - {sub} (missing)")

    # Verify summary total_subreddits vs rankings length
    summary_total = rankings.get('summary', {}).get('total_subreddits')
    if summary_total is not None and summary_total != len(ranked_subreddits):
        issues.append(f"WARNING: Rankings summary total_subreddits ({summary_total}) != length of rankings list ({len(ranked_subreddits)})")

    # Date consistency
    if mod_files:
        mtimes = defaultdict(int)
        for f in mod_files:
            mtime = os.path.getmtime(f)
            date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
            mtimes[date_str] += 1
        if len(mtimes) > 1:
            issues.append(f"WARNING: Mod comment files span {len(mtimes)} different dates: {sorted(mtimes.keys())}")
            for date, count in sorted(mtimes.items()):
                info.append(f"  {date}: {count} files")

    return issues, info


def validate_stage7(verbose=False):
    """Validate Stage 7: Media collection consistency."""
    issues = []
    info = []

    media_dir = PATHS['media']
    if not os.path.exists(media_dir):
        issues.append("WARNING: media directory not found")
        return issues, info

    # Subreddit dirs on disk. Empty dirs are common cleanup candidates and do
    # not imply data drift.
    media_subreddits = []
    media_per_sub = {}
    total_media = 0
    for sub in os.listdir(media_dir):
        sub_path = os.path.join(media_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        try:
            files = [
                f for f in os.listdir(sub_path)
                if not f.startswith('.') and os.path.isfile(os.path.join(sub_path, f))
            ]
        except OSError:
            files = []
        media_subreddits.append(sub)
        media_per_sub[sub] = len(files)
        total_media += len(files)

    info.append(f"Files on disk: {len(media_subreddits)} media subreddit dirs, {total_media} media files")

    # Cross-check against successful submission IDs manifest. This manifest
    # includes both `complete` and `no_media` submissions, so it is not a list of
    # submissions expected to have media files.
    ids_file = os.path.join(PATHS['data'], 'stage7_successful_submission_ids.json')
    if not os.path.exists(ids_file):
        issues.append("CRITICAL: stage7_successful_submission_ids.json not found")
        return issues, info

    ids_data = read_json_file(ids_file)
    if isinstance(ids_data, dict) and 'subreddit_submission_ids' in ids_data:
        manifest_map = ids_data['subreddit_submission_ids']
    elif isinstance(ids_data, dict):
        manifest_map = ids_data
    else:
        manifest_map = {}

    manifest_subreddits = set(manifest_map.keys())
    info.append(f"Manifest: {len(manifest_subreddits)} subreddits with successful submissions")

    if not manifest_subreddits:
        issues.append("WARNING: stage7_successful_submission_ids.json contains no submission IDs (empty manifest)")

    extra_dirs = set(media_subreddits) - manifest_subreddits

    if extra_dirs:
        extra_empty = [sub for sub in extra_dirs if media_per_sub.get(sub, 0) == 0]
        extra_nonempty = [sub for sub in extra_dirs if media_per_sub.get(sub, 0) > 0]
        if extra_empty:
            info.append(f"{len(extra_empty)} empty media dirs are not in successful IDs manifest")
        if extra_nonempty:
            issues.append(f"WARNING: {len(extra_nonempty)} non-empty media dirs NOT in successful IDs manifest")
            if verbose:
                for sub in sorted(extra_nonempty)[:20]:
                    issues.append(f"  - media/{sub}/ ({media_per_sub.get(sub, 0)} files)")

    # Verify stats file exists
    stats_file = os.path.join(PATHS['data'], 'stage7_media_collection_stats.json')
    if not os.path.exists(stats_file):
        issues.append("WARNING: stage7_media_collection_stats.json not found")
    else:
        stats = read_json_file(stats_file)
        summary = stats.get('summary', {})
        status_breakdown = stats.get('status_breakdown', {})

        complete_count = status_breakdown.get('complete', 0)
        no_media_count = status_breakdown.get('no_media', 0)
        expected_successful = complete_count + no_media_count
        reported_successful = summary.get('total_successful_submissions')
        if reported_successful is not None and reported_successful != expected_successful:
            issues.append(
                f"WARNING: Stage 7 total_successful_submissions ({reported_successful}) "
                f"!= complete + no_media ({expected_successful})"
            )

        reported_files = summary.get('total_files_downloaded')
        if reported_files is not None and reported_files != total_media:
            issues.append(
                f"WARNING: Stage 7 summary reports {reported_files} downloaded files, "
                f"but {total_media} media files are on disk"
            )

        info.append(
            "Status breakdown: "
            + ", ".join(f"{k}={v}" for k, v in sorted(status_breakdown.items()))
        )

    return issues, info


def validate_stage9a(verbose=False):
    """Validate Stage 9a: Embeddings + metadata TSV consistency."""
    issues = []
    info = []

    embeddings_dir = PATHS['embeddings']
    if not os.path.exists(embeddings_dir):
        issues.append("WARNING: embeddings directory not found")
        return issues, info

    pairs = [
        ('all_subreddit_embeddings.tsv', 'all_subreddit_metadata.tsv'),
        ('all_rule_embeddings.tsv', 'all_rule_metadata.tsv'),
    ]

    # Metadata TSVs are written by pandas and may contain quoted multi-line fields
    # (e.g. rule descriptions with embedded newlines), so naive line counts are
    # unreliable. Use pandas for the metadata; embedding TSV is one row per line.
    try:
        import pandas as pd
    except ImportError:
        pd = None

    for emb_name, meta_name in pairs:
        emb_path = os.path.join(embeddings_dir, emb_name)
        meta_path = os.path.join(embeddings_dir, meta_name)

        emb_exists = os.path.exists(emb_path)
        meta_exists = os.path.exists(meta_path)

        if not emb_exists:
            issues.append(f"CRITICAL: Stage 9a output missing: {emb_name}")
        if not meta_exists:
            issues.append(f"CRITICAL: Stage 9a output missing: {meta_name}")

        if not (emb_exists and meta_exists):
            continue

        try:
            with open(emb_path, 'r') as f:
                emb_rows = sum(1 for _ in f)
        except OSError as e:
            issues.append(f"WARNING: Could not read {emb_name}: {e}")
            continue

        if pd is not None:
            try:
                meta_rows = len(pd.read_csv(meta_path, sep='\t'))
            except Exception as e:
                issues.append(f"WARNING: Could not parse {meta_name} with pandas: {e}")
                continue
        else:
            issues.append(f"WARNING: pandas unavailable; skipping row-count check for {meta_name}")
            continue

        info.append(f"{emb_name}: {emb_rows} rows, {meta_name}: {meta_rows} rows")
        if emb_rows != meta_rows:
            issues.append(f"CRITICAL: Row count mismatch between {emb_name} ({emb_rows}) and {meta_name} ({meta_rows}) — possible partial-write")

    return issues, info


VALIDATORS = {
    1: ("Stage 1: Mod Comments", validate_stage1),
    3: ("Stage 3: Match Rules", validate_stage3),
    4: ("Stage 4: Organized Comments", validate_stage4),
    5: ("Stage 5: Trees & Threads", validate_stage5),
    6: ("Stage 6: Submissions", validate_stage6),
    7: ("Stage 7: Media", validate_stage7),
    8: ("Stage 8: Dataset", validate_stage8),
    9: ("Stage 9a: Embeddings", validate_stage9a),
    10: ("Stage 10: Clustered Dataset", validate_stage10),
}


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline data consistency across stages")
    parser.add_argument("--stage", type=int, help="Validate a specific stage only")
    parser.add_argument("--verbose", action="store_true", help="Show per-file details")
    args = parser.parse_args()

    print("=" * 70)
    print("Pipeline Data Consistency Validation")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_issues = 0
    stages_to_validate = {args.stage: VALIDATORS[args.stage]} if args.stage else VALIDATORS

    for stage_num, (label, validator) in sorted(stages_to_validate.items()):
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        issues, info_lines = validator(verbose=args.verbose)

        for line in info_lines:
            print(f"  ℹ️  {line}")

        if issues:
            for issue in issues:
                if issue.startswith("CRITICAL"):
                    print(f"  🔴 {issue}")
                elif issue.startswith("WARNING"):
                    print(f"  🟡 {issue}")
                else:
                    print(f"     {issue}")
            total_issues += len([i for i in issues if i.startswith(("CRITICAL", "WARNING"))])
        else:
            print(f"  ✅ No issues found")

    print(f"\n{'=' * 70}")
    if total_issues > 0:
        print(f"  ⚠️  Found {total_issues} issue(s) across validated stages")
        print(f"  Run with --verbose for more details")
    else:
        print(f"  ✅ All validated stages are consistent")
    print(f"{'=' * 70}")

    return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    exit(main())
