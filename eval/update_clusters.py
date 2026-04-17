#!/usr/bin/env python3
"""
Cluster Assignment Update Script

Updates cluster IDs and labels in all evaluation result files when clustering is re-run.

Usage:
    python update_clusters.py --source data/test_hydrated_clustered.json
    python update_clusters.py --source data/test_hydrated_clustered.json --dry-run

The script:
1. Reads new cluster assignments from the source dataset
2. Finds all reasoning_*.json files in output/eval/
3. Updates cluster IDs/labels based on stable keys (subreddit, rule text)
4. Regenerates performance_*.json files with recalculated metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from datetime import datetime

# Add project root and eval directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import config

# Default output directory
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "eval"


def load_cluster_assignments(source_path: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Load cluster assignments from source dataset.

    Args:
        source_path: Path to clustered dataset JSON

    Returns:
        Tuple of (subreddit_clusters, rule_clusters)
        - subreddit_clusters: {subreddit_name: {cluster_id, cluster_label}}
        - rule_clusters: {rule_text: {cluster_id, cluster_label}}
    """
    print(f"Loading cluster assignments from: {source_path}")

    # Handle compressed files
    if source_path.suffix == '.zst':
        import zstandard as zstd
        with open(source_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                data = json.loads(reader.read())
    else:
        with open(source_path, 'r') as f:
            data = json.load(f)

    subreddit_clusters = {}
    rule_clusters = {}

    for subreddit_data in data['subreddits']:
        subreddit = subreddit_data['subreddit']

        # Subreddit cluster assignment
        subreddit_clusters[subreddit] = {
            'cluster_id': subreddit_data.get('subreddit_cluster_id', -1),
            'cluster_label': subreddit_data.get('subreddit_cluster_label', 'Other')
        }

        # Rule cluster assignments (from thread pairs metadata)
        for pair in subreddit_data['thread_pairs']:
            rule_text = pair['metadata']['rule']
            if rule_text not in rule_clusters:
                rule_clusters[rule_text] = {
                    'cluster_id': pair['metadata'].get('rule_cluster_id', -1),
                    'cluster_label': pair['metadata'].get('rule_cluster_label', 'Other')
                }

    print(f"  Loaded {len(subreddit_clusters)} subreddit cluster assignments")
    print(f"  Loaded {len(rule_clusters)} rule cluster assignments")

    return subreddit_clusters, rule_clusters


def find_reasoning_files(output_dir: Path, latest_only: bool = False) -> List[Path]:
    """Find all reasoning JSON files in output directory.

    Args:
        output_dir: Root directory to search
        latest_only: If True, only return the most recent file per model/split/context/phrase
    """
    reasoning_files = list(output_dir.rglob("reasoning_*.json"))
    print(f"Found {len(reasoning_files)} reasoning files")

    if not latest_only:
        return reasoning_files

    # Group by model/split/context/phrase directory
    groups = defaultdict(list)

    for file_path in reasoning_files:
        # Group key is the parent directory path (model/split/context/phrase)
        group_key = file_path.parent
        groups[group_key].append(file_path)

    # Select most recent file from each group (by filename timestamp)
    latest_files = []
    for group_key, files in groups.items():
        # Sort by filename (timestamps are in filename) and take the last one
        latest_file = sorted(files, key=lambda p: p.name)[-1]
        latest_files.append(latest_file)

    print(f"  Filtered to {len(latest_files)} latest files (one per model/context)")
    return latest_files


def update_reasoning_file(
    reasoning_path: Path,
    subreddit_clusters: Dict[str, Dict],
    rule_clusters: Dict[str, Dict],
    dry_run: bool = False
) -> Tuple[int, int, List[str]]:
    """
    Update cluster assignments in a reasoning file.

    Args:
        reasoning_path: Path to reasoning JSON file
        subreddit_clusters: Subreddit cluster assignments
        rule_clusters: Rule cluster assignments
        dry_run: If True, don't write changes

    Returns:
        Tuple of (updated_count, total_count, warnings)
    """
    with open(reasoning_path, 'r') as f:
        results = json.load(f)

    updated_count = 0
    warnings = []

    for result in results:
        subreddit = result['subreddit']
        rule_text = result['metadata']['rule']

        changed = False

        # Update subreddit cluster
        if subreddit in subreddit_clusters:
            new_sub = subreddit_clusters[subreddit]
            old_id = result['metadata'].get('subreddit_cluster_id')
            old_label = result['metadata'].get('subreddit_cluster_label')

            if old_id != new_sub['cluster_id'] or old_label != new_sub['cluster_label']:
                result['metadata']['subreddit_cluster_id'] = new_sub['cluster_id']
                result['metadata']['subreddit_cluster_label'] = new_sub['cluster_label']
                changed = True
        else:
            warnings.append(f"Subreddit '{subreddit}' not found in new cluster assignments")

        # Update rule cluster
        if rule_text in rule_clusters:
            new_rule = rule_clusters[rule_text]
            old_id = result['metadata'].get('rule_cluster_id')
            old_label = result['metadata'].get('rule_cluster_label')

            if old_id != new_rule['cluster_id'] or old_label != new_rule['cluster_label']:
                result['metadata']['rule_cluster_id'] = new_rule['cluster_id']
                result['metadata']['rule_cluster_label'] = new_rule['cluster_label']
                changed = True
        else:
            warnings.append(f"Rule '{rule_text[:50]}...' not found in new cluster assignments")

        if changed:
            updated_count += 1

    # Write updated file
    if not dry_run and updated_count > 0:
        with open(reasoning_path, 'w') as f:
            json.dump(results, f, indent=2)

    return updated_count, len(results), list(set(warnings))  # Dedupe warnings


def recalculate_performance(reasoning_path: Path) -> Dict[str, Any]:
    """
    Recalculate performance metrics from reasoning file.

    Args:
        reasoning_path: Path to reasoning JSON file

    Returns:
        Performance metrics dictionary
    """
    with open(reasoning_path, 'r') as f:
        results = json.load(f)

    total_pairs = len(results)

    # Overall accuracy
    violating_correct = sum(r['violating']['score'] for r in results)
    compliant_correct = sum(r['compliant']['score'] for r in results)
    total_correct = violating_correct + compliant_correct
    total_threads = total_pairs * 2

    overall_accuracy = total_correct / total_threads if total_threads > 0 else 0
    violating_accuracy = violating_correct / total_pairs if total_pairs > 0 else 0
    compliant_accuracy = compliant_correct / total_pairs if total_pairs > 0 else 0

    # Per-cluster accuracy helper
    def calculate_cluster_stats(results: List[Dict], cluster_key: str) -> Dict[str, Dict]:
        cluster_stats = defaultdict(lambda: {
            'violating_correct': 0,
            'compliant_correct': 0,
            'total_correct': 0,
            'count': 0
        })

        for result in results:
            cluster_label = result['metadata'].get(cluster_key)
            if cluster_label is None:
                continue
            cluster_stats[cluster_label]['violating_correct'] += result['violating']['score']
            cluster_stats[cluster_label]['compliant_correct'] += result['compliant']['score']
            cluster_stats[cluster_label]['total_correct'] += result['violating']['score'] + result['compliant']['score']
            cluster_stats[cluster_label]['count'] += 1

        final_stats = {}
        for cluster, stats in cluster_stats.items():
            count = stats['count']
            total_threads = count * 2
            final_stats[cluster] = {
                'overall_accuracy': stats['total_correct'] / total_threads if total_threads > 0 else 0,
                'violating_accuracy': stats['violating_correct'] / count if count > 0 else 0,
                'compliant_accuracy': stats['compliant_correct'] / count if count > 0 else 0,
                'count': count,
                'total_threads': total_threads
            }

        return final_stats

    # Calculate per-cluster stats
    rule_cluster_stats = calculate_cluster_stats(results, 'rule_cluster_label')
    subreddit_cluster_stats = calculate_cluster_stats(results, 'subreddit_cluster_label')

    # Calculate per-language stats
    language_stats = calculate_cluster_stats(results, 'subreddit_language')

    return {
        'overall': {
            'total_pairs': total_pairs,
            'total_threads': total_threads,
            'overall_accuracy': overall_accuracy,
            'violating_accuracy': violating_accuracy,
            'compliant_accuracy': compliant_accuracy,
            'violating_correct': violating_correct,
            'compliant_correct': compliant_correct,
            'total_correct': total_correct
        },
        'per_rule_cluster': rule_cluster_stats,
        'per_subreddit_cluster': subreddit_cluster_stats,
        'per_language': language_stats
    }


def update_performance_file(
    reasoning_path: Path,
    dry_run: bool = False
) -> Path:
    """
    Regenerate performance file from updated reasoning file.

    Args:
        reasoning_path: Path to reasoning JSON file
        dry_run: If True, don't write changes

    Returns:
        Path to performance file
    """
    # Find corresponding performance file (same directory, same timestamp)
    reasoning_dir = reasoning_path.parent
    reasoning_name = reasoning_path.name
    timestamp = reasoning_name.replace('reasoning_', '').replace('.json', '')

    performance_path = reasoning_dir / f"performance_{timestamp}.json"

    # Load existing performance file to get config info
    if performance_path.exists():
        with open(performance_path, 'r') as f:
            existing_perf = json.load(f)

        model = existing_perf.get('model', 'unknown')
        split = existing_perf.get('split', 'unknown')
        context = existing_perf.get('context', 'unknown')
        phrase = existing_perf.get('phrase', 'unknown')
        mode = existing_perf.get('mode', 'unknown')
    else:
        # Parse from path if performance file doesn't exist
        parts = reasoning_path.parts
        model = 'unknown'
        split = 'unknown'
        context = 'unknown'
        phrase = 'unknown'
        mode = 'unknown'

    # Recalculate metrics
    metrics = recalculate_performance(reasoning_path)

    performance_data = {
        'model': model,
        'split': split,
        'context': context,
        'phrase': phrase,
        'mode': mode,
        'metrics': metrics,
        'cluster_updated_at': datetime.now().isoformat()
    }

    if not dry_run:
        with open(performance_path, 'w') as f:
            json.dump(performance_data, f, indent=2)

    return performance_path


def main():
    parser = argparse.ArgumentParser(
        description="Update cluster assignments in evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Path to source dataset with new cluster assignments'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Output directory containing evaluation results'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be updated without making changes'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    parser.add_argument(
        '--latest-only', '-l',
        action='store_true',
        help='Only update the most recent reasoning file per model/context combination'
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)

    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}")
        sys.exit(1)

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    print("=" * 60)
    print("CLUSTER ASSIGNMENT UPDATE")
    print("=" * 60)
    print(f"Source: {source_path}")
    print(f"Output dir: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Latest only: {args.latest_only}")
    print("=" * 60)

    # Load new cluster assignments
    subreddit_clusters, rule_clusters = load_cluster_assignments(source_path)

    # Find all reasoning files
    reasoning_files = find_reasoning_files(output_dir, latest_only=args.latest_only)

    if not reasoning_files:
        print("No reasoning files found. Nothing to update.")
        return

    # Update each file
    total_updated = 0
    total_results = 0
    all_warnings = []

    print("\nUpdating reasoning files...")
    for reasoning_path in reasoning_files:
        updated, total, warnings = update_reasoning_file(
            reasoning_path,
            subreddit_clusters,
            rule_clusters,
            dry_run=args.dry_run
        )

        total_updated += updated
        total_results += total
        all_warnings.extend(warnings)

        if args.verbose or updated > 0:
            status = "[DRY RUN] " if args.dry_run else ""
            print(f"  {status}{reasoning_path.relative_to(output_dir)}: {updated}/{total} results updated")

    # Regenerate performance files
    print("\nRegenerating performance files...")
    for reasoning_path in reasoning_files:
        perf_path = update_performance_file(reasoning_path, dry_run=args.dry_run)
        if args.verbose:
            status = "[DRY RUN] " if args.dry_run else ""
            print(f"  {status}{perf_path.relative_to(output_dir)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Reasoning files processed: {len(reasoning_files)}")
    print(f"Results updated: {total_updated}/{total_results}")
    print(f"Performance files regenerated: {len(reasoning_files)}")

    if all_warnings:
        unique_warnings = list(set(all_warnings))
        print(f"\nWarnings ({len(unique_warnings)} unique):")
        for warning in unique_warnings[:10]:  # Show first 10
            print(f"  ⚠️  {warning}")
        if len(unique_warnings) > 10:
            print(f"  ... and {len(unique_warnings) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
