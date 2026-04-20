#!/usr/bin/env python3
"""
Stage 10: Assign Cluster Labels to Datasets

Reads cluster labels from analysis/label_clusters.py output and assigns them
to thread pairs and subreddits in the train/val/test datasets.

Input:
- output/embeddings/all_rule_metadata.tsv (with cluster_id and cluster_label columns)
- output/embeddings/all_subreddit_metadata.tsv (with cluster_id and cluster_label columns)
- data/{split}_hydrated.json.zst (train/val/test datasets from Stage 8)

Output:
- data/{split}_hydrated_clustered.json.zst (updated datasets with cluster labels)
- data/{split}_dehydrated_clustered.json.zst (dehydrated versions)
- data/test_hydrated_clustered.json (uncompressed test set)
- data/stage10_cluster_assignment_stats.json
- data/stage10_dataset_stats_table.tex (LaTeX table for paper)

The script adds cluster labels to:
1. Thread pairs (rule clusters):
   - rule_cluster_id: The cluster ID for the matched rule (-1 for Other)
   - rule_cluster_label: The semantic label for the cluster (e.g., "spoiler tags", "civility rules")
   - rule_cluster_probability: Cluster assignment probability

2. Subreddits (subreddit clusters):
   - subreddit_cluster_id: The cluster ID for the subreddit (-1 for Other)
   - subreddit_cluster_label: The semantic label for the cluster
   - subreddit_cluster_probability: Cluster assignment probability
"""

import sys
import os
import time
import json
import pandas as pd
from typing import Dict, Tuple
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import write_json_file, read_compressed_json, write_compressed_json


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_language(lang_code: str) -> str:
    """Normalize language code by taking root (e.g., en-au → en, pt_BR → pt)."""
    return lang_code.replace('_', '-').split('-')[0]


# ============================================================================
# Data Loading
# ============================================================================

def load_cluster_metadata(filename: str, required_cols: list, logger) -> pd.DataFrame:
    """Load and validate cluster metadata from a TSV file.

    Args:
        filename: Name of the metadata file (e.g., 'all_rule_metadata.tsv')
        required_cols: List of required column names
        logger: Logger instance

    Returns:
        DataFrame with cluster metadata, or empty DataFrame on error
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_file = os.path.join(base_dir, 'output', 'embeddings', filename)

    if not os.path.exists(metadata_file):
        logger.error(f"❌ Metadata not found: {metadata_file}")
        logger.error("Please run analysis/label_clusters.py first")
        return pd.DataFrame()

    logger.info(f"📋 Loading cluster mappings from {metadata_file}...")
    df = pd.read_csv(metadata_file, sep='\t')

    # Check for required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ Missing required columns in metadata: {missing}")
        return pd.DataFrame()

    return df


def load_rule_cluster_mapping(logger) -> Dict[Tuple[str, str], Dict]:
    """Load cluster labels from all_rule_metadata.tsv.

    Returns:
        Dict mapping (subreddit, short_name) -> {cluster_id, cluster_label, cluster_probability}
    """
    required_cols = ['subreddit', 'short_name', 'cluster_id', 'cluster_label']
    df = load_cluster_metadata('all_rule_metadata.tsv', required_cols, logger)

    if df.empty:
        return {}

    # Build mapping using short_name (not short_name_clean) to match dataset rules
    # Note: Stage 9 uses short_name_clean as the 'rule' field in metadata
    # Note: The subreddit field may contain comma-separated subreddits for shared rules
    mapping = {}
    for _, row in df.iterrows():
        subreddits_str = str(row['subreddit'])
        short_name = str(row['short_name'])

        # Split comma-separated subreddits and create entry for each
        subreddits = [s.strip().lower() for s in subreddits_str.split(',')]

        cluster_id = int(row['cluster_id'])
        cluster_label = str(row['cluster_label'])

        # Replace "Noise" with "Other" for cluster_id -1
        if cluster_id == -1:
            cluster_label = 'Other'

        cluster_info = {
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'cluster_probability': float(row.get('cluster_probability', 0.0))
        }

        for subreddit in subreddits:
            key = (subreddit, short_name)
            mapping[key] = cluster_info

    logger.info(f"  ✅ Loaded cluster mappings for {len(mapping)} rules")
    return mapping


def load_subreddit_cluster_mapping(logger) -> Dict[str, Dict]:
    """Load cluster labels from all_subreddit_metadata.tsv.

    Returns:
        Dict mapping subreddit -> {cluster_id, cluster_label, cluster_probability}
    """
    required_cols = ['subreddit', 'cluster_id', 'cluster_label']
    df = load_cluster_metadata('all_subreddit_metadata.tsv', required_cols, logger)

    if df.empty:
        return {}

    # Build mapping
    mapping = {}
    for _, row in df.iterrows():
        subreddit = str(row['subreddit']).strip().lower()

        cluster_id = int(row['cluster_id'])
        cluster_label = str(row['cluster_label'])

        # Replace "Noise" with "Other" for cluster_id -1
        if cluster_id == -1:
            cluster_label = 'Other'

        cluster_info = {
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'cluster_probability': float(row.get('cluster_probability', 0.0))
        }

        mapping[subreddit] = cluster_info

    logger.info(f"  ✅ Loaded cluster mappings for {len(mapping)} subreddits")
    return mapping




# ============================================================================
# Cluster Assignment
# ============================================================================

def assign_clusters_to_dataset(dataset: Dict, rule_mapping: Dict[Tuple[str, str], Dict],
                               subreddit_mapping: Dict[str, Dict], logger) -> Tuple[Dict, Dict]:
    """Assign cluster labels to all thread pairs and subreddits in a dataset.

    Args:
        dataset: The hydrated dataset from Stage 9
        rule_mapping: Dict mapping (subreddit, short_name) -> cluster info
        subreddit_mapping: Dict mapping subreddit -> cluster info
        logger: Logger instance

    Returns:
        (updated_dataset, statistics)
    """
    stats = {
        'total_subreddits': 0,
        'total_thread_pairs': 0,
        'total_comments': 0,  # Total comments across all threads
        'total_images': 0,  # Total images across all submissions
        'pairs_with_rule_clusters': 0,
        'subreddits_with_clusters': 0,
        'unique_rules': set(),  # Track unique (subreddit, rule) pairs
        'unique_rule_clusters': set(),  # Track unique rule cluster IDs
        'unique_subreddit_clusters': set(),  # Track unique subreddit cluster IDs
        'language_counts': Counter(),  # Track thread pairs per language
        'rule_cluster_distribution': Counter(),
        'subreddit_cluster_distribution': Counter(),
        'subreddit_cluster_pair_distribution': Counter(),  # Track pairs by subreddit cluster
        'rule_cluster_subreddits': defaultdict(set),
        'rule_cluster_rules': defaultdict(set),
        'subreddit_cluster_subreddits': defaultdict(set),
        'subreddit_cluster_rules': defaultdict(set),
    }

    for sub_data in dataset['subreddits']:
        subreddit = sub_data['subreddit'].lower()
        stats['total_subreddits'] += 1

        # Get normalized language for this subreddit
        language = sub_data.get('language', 'unknown')
        normalized_lang = normalize_language(language)

        # Count images from submissions
        for sub_id, submission in sub_data.get('submissions', {}).items():
            stats['total_images'] += submission.get('num_media', 0)

        # Assign subreddit cluster
        subreddit_cluster_info = subreddit_mapping[subreddit]
        sub_data['subreddit_cluster_id'] = subreddit_cluster_info['cluster_id']
        sub_data['subreddit_cluster_label'] = subreddit_cluster_info['cluster_label']
        sub_data['subreddit_cluster_probability'] = subreddit_cluster_info['cluster_probability']
        stats['subreddits_with_clusters'] += 1
        stats['subreddit_cluster_distribution'][subreddit_cluster_info['cluster_label']] += 1
        stats['unique_subreddit_clusters'].add(subreddit_cluster_info['cluster_id'])

        # Assign rule clusters to each thread pair
        for pair in sub_data['thread_pairs']:
            stats['total_thread_pairs'] += 1

            # Count comments in both threads
            num_comments = len(pair['violating_thread']) + len(pair['compliant_thread'])
            stats['total_comments'] += num_comments

            # Track language counts (per thread pair)
            stats['language_counts'][normalized_lang] += 1

            # Get the matched rule from metadata
            matched_rule = pair['metadata']['rule']

            # Track unique rules
            stats['unique_rules'].add((subreddit, matched_rule))

            # Look up cluster info
            key = (subreddit, matched_rule)
            cluster_info = rule_mapping[key]

            # Assign cluster info
            pair['metadata']['rule_cluster_id'] = cluster_info['cluster_id']
            pair['metadata']['rule_cluster_label'] = cluster_info['cluster_label']
            pair['metadata']['rule_cluster_probability'] = cluster_info['cluster_probability']
            stats['pairs_with_rule_clusters'] += 1
            stats['rule_cluster_distribution'][cluster_info['cluster_label']] += 1
            stats['unique_rule_clusters'].add(cluster_info['cluster_id'])
            stats['rule_cluster_subreddits'][cluster_info['cluster_label']].add(subreddit)
            stats['rule_cluster_rules'][cluster_info['cluster_label']].add((subreddit, matched_rule))

            # Track thread pairs by subreddit cluster label
            stats['subreddit_cluster_pair_distribution'][subreddit_cluster_info['cluster_label']] += 1
            stats['subreddit_cluster_subreddits'][subreddit_cluster_info['cluster_label']].add(subreddit)
            stats['subreddit_cluster_rules'][subreddit_cluster_info['cluster_label']].add((subreddit, matched_rule))

    return dataset, stats


def dehydrate_dataset(hydrated: Dict) -> Dict:
    """Create dehydrated version (IDs only)."""
    dehydrated = {'metadata': hydrated['metadata'].copy(), 'subreddits': []}

    for sub_data in hydrated['subreddits']:
        dehydrated_subs = {}
        for sub_id, sub in sub_data['submissions'].items():
            dehydrated_subs[sub_id] = {
                'id': sub_id,
                'submission_object': '[NEEDS_HYDRATION]',
                'num_media': sub.get('num_media', 0),
                'media_files': ['[NEEDS_HYDRATION]'] * sub.get('num_media', 0)
            }

        dehydrated_pairs = []
        for pair in sub_data['thread_pairs']:
            dehydrated_pairs.append({
                'mod_comment_id': pair['mod_comment_id'],
                'mod_comment': '[NEEDS_HYDRATION]',
                'violating_thread': ['[NEEDS_HYDRATION]'] * len(pair['violating_thread']),
                'compliant_thread': ['[NEEDS_HYDRATION]'] * len(pair['compliant_thread']),
                # Root-to-leaf comment IDs. Enables single-pass hydration of thread
                # bodies without having to walk parent_id pointers from Arctic Shift.
                'violating_thread_ids': [c['id'] for c in pair['violating_thread']],
                'compliant_thread_ids': [c['id'] for c in pair['compliant_thread']],
                'violating_answer_options': pair['violating_answer_options'],
                'violating_correct_answer': pair['violating_correct_answer'],
                'compliant_answer_options': pair['compliant_answer_options'],
                'compliant_correct_answer': pair['compliant_correct_answer'],
                'metadata': pair['metadata']  # Keep full metadata (includes cluster labels)
            })

        dehydrated['subreddits'].append({
            'subreddit': sub_data['subreddit'],
            'title': sub_data.get('title', ''),
            'description': sub_data.get('description', ''),
            'language': sub_data['language'],
            'data_version': sub_data['data_version'],
            'last_updated': sub_data['last_updated'],
            'total_thread_pairs': sub_data['total_thread_pairs'],
            'jsd_from_uniform': sub_data['jsd_from_uniform'],
            'rules': sub_data['rules'],
            'submissions': dehydrated_subs,
            'thread_pairs': dehydrated_pairs,
            'rank': sub_data.get('rank'),
            'subreddit_cluster_id': sub_data.get('subreddit_cluster_id', -1),
            'subreddit_cluster_label': sub_data.get('subreddit_cluster_label', 'Other'),
            'subreddit_cluster_probability': sub_data.get('subreddit_cluster_probability', 0.0)
        })

    dehydrated['metadata']['instructions'] = 'Use hydration script. All text fields contain [NEEDS_HYDRATION].'
    return dehydrated


# ============================================================================
# LaTeX Table Generation
# ============================================================================

def generate_latex_table(all_stats: Dict, overall_totals: Dict) -> str:
    """Generate LaTeX table for dataset statistics.

    Args:
        all_stats: Per-split statistics
        overall_totals: Overall totals across all splits

    Returns:
        LaTeX table string
    """
    # Format numbers with commas
    def fmt(n):
        return f"{n:,}"

    # Determine qualifying languages (≥10 instances across all splits)
    all_language_counts = overall_totals.get('language_counts', {})
    qualifying_languages = {l for l, c in all_language_counts.items() if c >= 10}

    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lrrrrrr}",
        r"  \toprule",
        r"  \textbf{Split} & \textbf{Thread Pairs} & \textbf{Comments} & \textbf{Images} & \textbf{Subreddits / Clusters} & \textbf{Rules / Clusters} & \textbf{Languages} \\",
        r"  \midrule",
    ]

    # Add rows for each split in order
    for split in ['train', 'val', 'test']:
        if split not in all_stats:
            continue
        s = all_stats[split]
        split_name = split.capitalize()
        thread_pairs = fmt(s['total_thread_pairs'])
        comments = fmt(s['total_comments'])
        images = fmt(s['total_images'])
        subs_clusters = f"{fmt(s['total_subreddits'])} / {s['unique_subreddit_clusters']}"
        rules_clusters = f"{fmt(s['unique_rules'])} / {s['unique_rule_clusters']}"
        # Count qualifying languages present in this split (at least 1 instance)
        split_lang_counts = s.get('language_counts', {})
        languages_in_split = sum(1 for l in qualifying_languages if split_lang_counts.get(l, 0) > 0)
        lines.append(f"  {split_name} & {thread_pairs} & {comments} & {images} & {subs_clusters} & {rules_clusters} & {languages_in_split} \\\\")

    # Add totals row
    lines.append(r"  \midrule")
    total_pairs = fmt(overall_totals['total_thread_pairs'])
    total_comments = fmt(overall_totals['total_comments'])
    total_images = fmt(overall_totals['total_images'])
    total_subs = fmt(overall_totals['total_subreddits'])
    total_sub_clusters = overall_totals['total_subreddit_clusters']
    total_rules = fmt(overall_totals['total_unique_rules'])
    total_rule_clusters = overall_totals['total_rule_clusters']
    total_langs = overall_totals['total_languages']

    lines.append(
        f"  \\textbf{{Total}} & \\textbf{{{total_pairs}}} & \\textbf{{{total_comments}}} & "
        f"\\textbf{{{total_images}}} & \\textbf{{{total_subs} / {total_sub_clusters}}} & "
        f"\\textbf{{{total_rules} / {total_rule_clusters}}} & \\textbf{{{total_langs}}} \\\\"
    )

    lines.extend([
        r"  \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Dataset statistics. Each thread pair contains one rule-violating and one compliant thread from the same submission.}",
        r"  \label{tab:dataset-stats}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    logger = get_stage_logger(10, "assign_cluster_labels")
    log_stage_start(logger, 10, "Assign Cluster Labels to Datasets")
    start_time = time.time()

    try:
        create_directories()

        # Load cluster mappings
        rule_mapping = load_rule_cluster_mapping(logger)
        if not rule_mapping:
            logger.error("❌ Failed to load rule cluster mappings")
            log_stage_end(logger, 10, success=False, elapsed_time=time.time() - start_time)
            return 1

        subreddit_mapping = load_subreddit_cluster_mapping(logger)
        if not subreddit_mapping:
            logger.error("❌ Failed to load subreddit cluster mappings")
            log_stage_end(logger, 10, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Process each split
        logger.info("\n" + "="*80)
        logger.info("PROCESSING DATASETS")
        logger.info("="*80)

        splits = ['test', 'val', 'train']
        all_stats = {}
        output_files = {}
        global_rule_cluster_subreddits = defaultdict(set)
        global_rule_cluster_rules = defaultdict(set)
        global_subreddit_cluster_subreddits = defaultdict(set)
        global_subreddit_cluster_rules = defaultdict(set)

        for split in splits:
            logger.info(f"\n{'='*80}")
            logger.info(f"{split.upper()} SPLIT")
            logger.info(f"{'='*80}")

            # Check if input file exists
            input_file = os.path.join(PATHS['data'], f'{split}_hydrated.json.zst')
            if not os.path.exists(input_file):
                logger.warning(f"  ⚠️  {split} dataset not found, skipping...")
                continue

            # Load dataset
            dataset = read_compressed_json(input_file, logger)

            # Assign clusters
            logger.info(f"  Assigning cluster labels to {split} dataset...")
            updated_dataset, stats = assign_clusters_to_dataset(dataset, rule_mapping, subreddit_mapping, logger)

            # Union per-split cluster size sets into globals
            for label, subs in stats['rule_cluster_subreddits'].items():
                global_rule_cluster_subreddits[label].update(subs)
            for label, rules in stats['rule_cluster_rules'].items():
                global_rule_cluster_rules[label].update(rules)
            for label, subs in stats['subreddit_cluster_subreddits'].items():
                global_subreddit_cluster_subreddits[label].update(subs)
            for label, rules in stats['subreddit_cluster_rules'].items():
                global_subreddit_cluster_rules[label].update(rules)

            # Update metadata version
            updated_dataset['metadata']['version'] = '1.1'
            updated_dataset['metadata']['cluster_labels_added'] = time.strftime('%Y-%m-%d')

            # Log statistics
            logger.info(f"  📊 Statistics:")
            logger.info(f"    Total subreddits: {stats['total_subreddits']}")
            logger.info(f"    Total thread pairs: {stats['total_thread_pairs']}")
            logger.info(f"    Total comments: {stats['total_comments']}")
            logger.info(f"    Total images: {stats['total_images']}")
            logger.info(f"    Unique rules: {len(stats['unique_rules'])}")
            logger.info(f"    ")
            logger.info(f"    Rule Clusters:")
            logger.info(f"      Pairs with rule clusters: {stats['pairs_with_rule_clusters']}")
            logger.info(f"      Unique rule clusters: {len(stats['unique_rule_clusters'])}")
            logger.info(f"    ")
            logger.info(f"    Subreddit Clusters:")
            logger.info(f"      Subreddits with clusters: {stats['subreddits_with_clusters']}")
            logger.info(f"      Unique subreddit clusters: {len(stats['unique_subreddit_clusters'])}")

            # Save hydrated version
            hydrated_output = os.path.join(PATHS['data'], f'{split}_hydrated_clustered.json.zst')
            hydrated_size = write_compressed_json(updated_dataset, hydrated_output, logger=logger)

            # Create and save dehydrated version
            dehydrated = dehydrate_dataset(updated_dataset)
            dehydrated_output = os.path.join(PATHS['data'], f'{split}_dehydrated_clustered.json.zst')
            dehydrated_size = write_compressed_json(dehydrated, dehydrated_output, logger=logger)

            # Save uncompressed test set
            if split == 'test':
                uncompressed_file = os.path.join(PATHS['data'], 'test_hydrated_clustered.json')
                with open(uncompressed_file, 'w') as f:
                    json.dump(updated_dataset, f, indent=2)
                uncompressed_size = os.path.getsize(uncompressed_file) / (1024 * 1024)
                logger.info(f"  ✅ {uncompressed_file} ({uncompressed_size:.1f} MB)")
                output_files[split] = {
                    'hydrated': {'path': hydrated_output, 'size_mb': hydrated_size},
                    'dehydrated': {'path': dehydrated_output, 'size_mb': dehydrated_size},
                    'uncompressed': {'path': uncompressed_file, 'size_mb': uncompressed_size}
                }
            else:
                output_files[split] = {
                    'hydrated': {'path': hydrated_output, 'size_mb': hydrated_size},
                    'dehydrated': {'path': dehydrated_output, 'size_mb': dehydrated_size}
                }

            # Save split statistics
            all_stats[split] = {
                'total_subreddits': stats['total_subreddits'],
                'total_thread_pairs': stats['total_thread_pairs'],
                'total_comments': stats['total_comments'],
                'total_images': stats['total_images'],
                'unique_rules': len(stats['unique_rules']),  # Convert set to count
                'unique_rule_clusters': len(stats['unique_rule_clusters']),  # Convert set to count
                'unique_subreddit_clusters': len(stats['unique_subreddit_clusters']),  # Convert set to count
                'language_counts': dict(stats['language_counts']),  # Full language counts; unique_languages/languages added after global threshold
                'pairs_with_rule_clusters': stats['pairs_with_rule_clusters'],
                'subreddits_with_clusters': stats['subreddits_with_clusters'],
                'top_10_rule_clusters': dict(stats['rule_cluster_distribution'].most_common(10)),
                'rule_clusters': dict(stats['rule_cluster_distribution']),  # Save all clusters for plotting
                'top_10_subreddit_clusters': dict(stats['subreddit_cluster_distribution'].most_common(10)),
                'subreddit_clusters': dict(stats['subreddit_cluster_pair_distribution']),  # Save for plotting (by thread pairs)
                'cluster_size_stats': {
                    'rule': {
                        label: {
                            'n_subreddits': len(stats['rule_cluster_subreddits'][label]),
                            'n_rules': len(stats['rule_cluster_rules'][label]),
                            'n_thread_pairs': stats['rule_cluster_distribution'][label],
                        }
                        for label in stats['rule_cluster_distribution']
                    },
                    'subreddit': {
                        label: {
                            'n_subreddits': len(stats['subreddit_cluster_subreddits'][label]),
                            'n_rules': len(stats['subreddit_cluster_rules'][label]),
                            'n_thread_pairs': stats['subreddit_cluster_pair_distribution'][label],
                        }
                        for label in stats['subreddit_cluster_pair_distribution']
                    },
                },
            }

        # Save overall statistics
        logger.info("\n" + "="*80)
        logger.info("SAVING STATISTICS")
        logger.info("="*80)

        # Compute overall totals across all splits
        overall_totals = {
            'total_thread_pairs': sum(s.get('total_thread_pairs', 0) for s in all_stats.values()),
            'total_comments': sum(s.get('total_comments', 0) for s in all_stats.values()),
            'total_images': sum(s.get('total_images', 0) for s in all_stats.values()),
            'total_subreddits': len(subreddit_mapping),  # Unique subreddits across all splits
            'total_unique_rules': len(rule_mapping),  # Unique rules across all splits
            'total_rule_clusters': len(set(
                cluster_id for s in all_stats.values()
                for cluster_id in range(s.get('unique_rule_clusters', 0))
            )) if all_stats else 0,
            'total_subreddit_clusters': len(set(
                cluster_id for s in all_stats.values()
                for cluster_id in range(s.get('unique_subreddit_clusters', 0))
            )) if all_stats else 0,
        }
        # Get actual cluster counts from the distributions (more accurate)
        all_rule_clusters = set()
        all_subreddit_clusters = set()
        all_language_counts = Counter()
        all_rule_pair_counts = Counter()
        all_subreddit_pair_counts = Counter()
        for s in all_stats.values():
            all_rule_clusters.update(s.get('rule_clusters', {}).keys())
            all_subreddit_clusters.update(s.get('subreddit_clusters', {}).keys())
            # Aggregate language counts across all splits
            for lang, count in s.get('language_counts', {}).items():
                all_language_counts[lang] += count
            # Aggregate pair counts across all splits
            for label, count in s.get('rule_clusters', {}).items():
                all_rule_pair_counts[label] += count
            for label, count in s.get('subreddit_clusters', {}).items():
                all_subreddit_pair_counts[label] += count
        overall_totals['total_rule_clusters'] = len(all_rule_clusters)
        overall_totals['total_subreddit_clusters'] = len(all_subreddit_clusters)
        # Filter languages to those with ≥10 instances across all splits
        languages_with_threshold = sorted([l for l, c in all_language_counts.items() if c >= 10])
        overall_totals['total_languages'] = len(languages_with_threshold)
        overall_totals['languages'] = languages_with_threshold
        overall_totals['language_counts'] = dict(all_language_counts)  # Full counts for reference

        # Update per-split language stats using global qualifying languages
        qualifying_languages_set = set(languages_with_threshold)
        for split, s in all_stats.items():
            split_lang_counts = s.get('language_counts', {})
            langs_in_split = sorted(l for l in qualifying_languages_set if split_lang_counts.get(l, 0) > 0)
            s['unique_languages'] = len(langs_in_split)
            s['languages'] = langs_in_split
            logger.info(f"  {split}: Languages (≥10 globally, ≥1 in split): {len(langs_in_split)} ({', '.join(langs_in_split)})")
            logger.info(f"  {split}: Total unique languages: {len(split_lang_counts)}")
        overall_totals['cluster_size_stats'] = {
            'rule': {
                label: {
                    'n_subreddits': len(global_rule_cluster_subreddits[label]),
                    'n_rules': len(global_rule_cluster_rules[label]),
                    'n_thread_pairs': all_rule_pair_counts[label],
                }
                for label in all_rule_pair_counts
            },
            'subreddit': {
                label: {
                    'n_subreddits': len(global_subreddit_cluster_subreddits[label]),
                    'n_rules': len(global_subreddit_cluster_rules[label]),
                    'n_thread_pairs': all_subreddit_pair_counts[label],
                }
                for label in all_subreddit_pair_counts
            },
        }

        summary_stats = {
            'metadata': {
                'stage': 10,
                'stage_name': 'Assign Cluster Labels',
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time_seconds': time.time() - start_time
            },
            'overall_totals': overall_totals,
            'cluster_assignment_statistics': all_stats,
            'output_files': output_files,
            'total_unique_rules_mapped': len(rule_mapping),
            'total_unique_subreddits_mapped': len(subreddit_mapping)
        }

        stats_file = os.path.join(PATHS['data'], 'stage10_cluster_assignment_stats.json')
        write_json_file(summary_stats, stats_file, pretty=True)
        logger.info(f"  ✅ Saved statistics to: {stats_file}")

        # Generate and save LaTeX table
        latex_table = generate_latex_table(all_stats, overall_totals)
        latex_file = os.path.join(PATHS['data'], 'stage10_dataset_stats_table.tex')
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        logger.info(f"  ✅ Saved LaTeX table to: {latex_file}")

        elapsed = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info(f"🎉 Stage 10 Complete! ({elapsed:.1f}s)")
        logger.info("="*80)
        log_stage_end(logger, 10, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 10 execution")
        log_stage_end(logger, 10, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
