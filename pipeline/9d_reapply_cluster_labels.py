#!/usr/bin/env python3
"""
Reapply Cluster Labels from Override Files

Applies manual label overrides from simple override files to JSON and metadata.

OVERRIDE FILE FORMAT (e.g., rule_label_overrides.txt):
    # Comments start with #
    # Format: CLUSTER_ID: NEW_LABEL
    5: Image Context
    10: Text Context
    3: Spam/Self-Promotion

CLUSTER MERGING: If multiple clusters are given the same label in the override
file, they will be merged into a single cluster (keeping lowest cluster_id).

Workflow:
1. Run label_clusters.py to generate auto-labels
2. Review output/clustering/{entity}_cluster_analysis.txt (shows LABEL: xxx, ID: N)
3. Create/edit output/clustering/{entity}_label_overrides.txt with corrections
4. Run this script to apply overrides to JSON and metadata TSV

Usage:
    python reapply_cluster_labels.py                    # Reapply both subreddits and rules
    python reapply_cluster_labels.py --entity subreddit # Reapply only subreddits
    python reapply_cluster_labels.py --entity rule      # Reapply only rules

Input (override files - you create these):
- output/clustering/subreddit_label_overrides.txt
- output/clustering/rule_label_overrides.txt

Output (updated):
- output/clustering/subreddit_cluster_labels.json (updated with new labels)
- output/clustering/rule_cluster_labels.json (updated with new labels)
- output/embeddings/all_subreddit_metadata.tsv (cluster_label column + cluster_id merged)
- output/embeddings/all_rule_metadata.tsv (cluster_label column + cluster_id merged)
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import time
from typing import Dict, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PATHS
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue


def parse_label_overrides(override_file: Path, logger) -> Dict[int, str]:
    """Parse label overrides from simple override file.

    Format of override file:
        # Comments start with #
        # Format: CLUSTER_ID: NEW_LABEL
        5: Image Context
        10: Text Context
        3: Spam/Self-Promotion

    Args:
        override_file: Path to override file (e.g., rule_label_overrides.txt)
        logger: Logger instance

    Returns:
        Dict mapping cluster_id -> new_label (only for overridden clusters)
    """
    if not override_file.exists():
        logger.info(f"No override file found at {override_file}")
        logger.info(f"  To create overrides, create this file with format:")
        logger.info(f"  CLUSTER_ID: NEW_LABEL")
        logger.info(f"  Example: 5: Image Context")
        return {}

    logger.info(f"Parsing label overrides from {override_file}...")

    overrides = {}

    with open(override_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse "CLUSTER_ID: NEW_LABEL" format
            if ':' in line:
                try:
                    cluster_id_str, new_label = line.split(':', 1)
                    cluster_id = int(cluster_id_str.strip())
                    new_label = new_label.strip()

                    if new_label:
                        overrides[cluster_id] = new_label
                        logger.info(f"  Cluster {cluster_id}: {new_label}")
                    else:
                        logger.warning(f"  Line {line_num}: Empty label for cluster {cluster_id}, skipping")
                except ValueError:
                    logger.warning(f"  Line {line_num}: Invalid format '{line}', skipping")
            else:
                logger.warning(f"  Line {line_num}: Missing ':' separator in '{line}', skipping")

    if not overrides:
        logger.info(f"  No valid overrides found in {override_file}")
    else:
        logger.info(f"  Loaded {len(overrides)} label overrides")

    return overrides


def detect_cluster_merges(overrides: Dict[int, str], all_labels: Dict[int, str], logger) -> Tuple[Dict[int, int], Dict[str, list]]:
    """Detect clusters that should be merged based on identical NEW LABELs.

    ONLY merges clusters when the user explicitly assigns the same NEW LABEL
    to multiple clusters. Does NOT auto-merge existing duplicate labels.

    Args:
        overrides: Dict mapping cluster_id -> new_label (only overridden clusters)
        all_labels: Dict mapping cluster_id -> label (all clusters, including non-overridden)
        logger: Logger instance

    Returns:
        (merge_mapping, label_groups) where:
            - merge_mapping: Dict mapping old_cluster_id -> new_cluster_id (lowest in group)
            - label_groups: Dict mapping label -> list of cluster_ids with that label
    """
    # Only consider clusters that have explicit NEW LABEL overrides
    # Group overridden clusters by their new label
    override_label_to_clusters = defaultdict(list)
    for cluster_id, new_label in overrides.items():
        override_label_to_clusters[new_label].append(cluster_id)

    # Build merge mapping only for clusters with duplicate NEW LABELs
    merge_mapping = {}
    label_groups = {}

    for label, cluster_ids in override_label_to_clusters.items():
        if len(cluster_ids) > 1:
            # Multiple clusters explicitly given the same NEW LABEL - merge them
            sorted_ids = sorted(cluster_ids)
            target_id = sorted_ids[0]
            label_groups[label] = sorted_ids

            for cid in sorted_ids:
                if cid != target_id:
                    merge_mapping[cid] = target_id

            logger.info(f"  Merging clusters {sorted_ids} → {target_id} (NEW LABEL: '{label}')")

    if not merge_mapping:
        logger.info("  No merges detected (no duplicate NEW LABELs)")

    return merge_mapping, label_groups


def reapply_entity_labels(entity_type: str, embeddings_dir: Path, clustering_dir: Path, logger) -> None:
    """Reapply cluster labels from analysis text file to JSON and metadata.

    Args:
        entity_type: 'subreddit' or 'rule'
        embeddings_dir: Path to embeddings directory
        clustering_dir: Path to clustering directory
        logger: Logger instance
    """
    logger.info("\n" + "="*80)
    logger.info(f"{entity_type.upper()} - Reapplying Labels")
    logger.info("="*80)

    # Load existing JSON labels first
    labels_file = clustering_dir / f'{entity_type}_cluster_labels.json'
    if not labels_file.exists():
        logger.error(f"❌ Error: {labels_file} not found")
        logger.error("Run label_clusters.py first to generate initial labels")
        return

    logger.info(f"Loading existing labels from {labels_file}...")
    with open(labels_file) as f:
        cluster_data = json.load(f)

    # Build all_labels dict (existing labels for all clusters)
    all_labels = {int(cid): data['label'] for cid, data in cluster_data.items()}

    # Parse manual overrides from override file (source of truth for manual corrections)
    override_file = clustering_dir / f'{entity_type}_label_overrides.txt'
    overrides = parse_label_overrides(override_file, logger)

    # Detect cluster merges (only when user explicitly assigns same NEW LABEL to multiple clusters)
    logger.info("\nDetecting cluster merges from NEW LABELs...")
    merge_mapping, label_groups = detect_cluster_merges(overrides, all_labels, logger)

    # Log if no overrides to apply (but still continue to write cluster_label column)
    if not overrides and not merge_mapping:
        logger.info(f"No overrides to apply for {entity_type}, writing cluster_label column from existing labels")

    if not merge_mapping:
        logger.info("  No merges detected (all labels are unique)")

    # Apply overrides to JSON
    logger.info("Applying overrides to JSON...")
    for cluster_id, new_label in overrides.items():
        cluster_key = str(cluster_id)
        if cluster_key in cluster_data:
            old_label = cluster_data[cluster_key]['label']
            cluster_data[cluster_key]['label'] = new_label
            logger.info(f"  Cluster {cluster_id}: '{old_label}' → '{new_label}'")
        else:
            logger.warning(f"  ⚠️  Cluster {cluster_id} not found in JSON, skipping")

    # Save updated JSON
    with open(labels_file, 'w') as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Updated {labels_file}")

    # Check for duplicate labels (warn but don't merge - only explicit overrides cause merges)
    final_labels = {int(cid): data['label'] for cid, data in cluster_data.items()}
    label_to_clusters = defaultdict(list)
    for cid, label in final_labels.items():
        label_to_clusters[label].append(cid)

    duplicates = {label: cids for label, cids in label_to_clusters.items() if len(cids) > 1}
    if duplicates:
        logger.warning("\n⚠️  Duplicate labels detected (NOT merging - only explicit overrides merge):")
        for label, cids in duplicates.items():
            # Check if these are from explicit override (would have been merged) or from auto-labeling
            override_cids = [c for c in cids if c in overrides]
            if len(override_cids) < 2:
                # At least one is from auto-labeling, warn user
                logger.warning(f"  '{label}': clusters {sorted(cids)}")
                logger.warning(f"    To merge, add ALL these IDs to override file with same label")

    # Update metadata TSV
    # Use 'all_rule' for rules, 'all_subreddit' for subreddits (both from train/val/test)
    prefix = 'all_rule' if entity_type == 'rule' else 'all_subreddit'
    metadata_file = embeddings_dir / f'{prefix}_metadata.tsv'
    if not metadata_file.exists():
        logger.error(f"❌ Error: {metadata_file} not found")
        return

    logger.info(f"\nUpdating metadata file: {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep='\t')

    if 'cluster_id' not in metadata.columns:
        logger.error(f"❌ Error: cluster_id column not found in {metadata_file}")
        logger.error("Run cluster_test_1k.py --apply-best first")
        return

    # Apply cluster merges (remap cluster_ids)
    if merge_mapping:
        logger.info("Applying cluster merges to metadata...")
        original_counts = metadata['cluster_id'].value_counts().to_dict()

        metadata['cluster_id'] = metadata['cluster_id'].map(
            lambda x: merge_mapping.get(x, x)
        )

        # Log merge statistics
        for label, cluster_ids in label_groups.items():
            target_id = min(cluster_ids)
            total_items = sum(original_counts.get(cid, 0) for cid in cluster_ids)
            logger.info(f"  Merged '{label}': {len(cluster_ids)} clusters → cluster {target_id} ({total_items} items)")

    # Renumber clusters to be contiguous (1, 2, 3, ...) excluding noise (-1)
    logger.info("\nRenumbering clusters to be contiguous...")
    unique_metadata_ids = sorted([x for x in metadata['cluster_id'].unique() if x != -1])
    renumber_map = {old_id: new_id for new_id, old_id in enumerate(unique_metadata_ids, start=1)}

    # Apply renumbering to metadata
    metadata['cluster_id'] = metadata['cluster_id'].map(lambda x: renumber_map.get(x, x) if x != -1 else -1)
    new_metadata_ids = sorted([x for x in metadata['cluster_id'].unique() if x != -1])
    logger.info(f"  Renumbered {len(unique_metadata_ids)} clusters to 1-{len(new_metadata_ids)}")

    # Map labels from JSON to metadata by sorted position
    # JSON keys stay at original HDBSCAN IDs (matching override file)
    # Metadata IDs are renumbered to 1-N
    json_sorted_keys = sorted(cluster_data.keys(), key=int)
    cluster_labels = {}
    if len(json_sorted_keys) == len(new_metadata_ids):
        for json_key, metadata_id in zip(json_sorted_keys, new_metadata_ids):
            cluster_labels[metadata_id] = cluster_data[json_key]['label']
        logger.info(f"  Mapped {len(json_sorted_keys)} JSON labels (keys {json_sorted_keys[0]}-{json_sorted_keys[-1]}) → metadata IDs {new_metadata_ids[0]}-{new_metadata_ids[-1]}")
    else:
        logger.warning(f"  ⚠️  JSON has {len(json_sorted_keys)} clusters but metadata has {len(new_metadata_ids)}; falling back to direct key mapping")
        cluster_labels = {int(cid): data['label'] for cid, data in cluster_data.items()}

    # Apply labels
    metadata['cluster_label'] = metadata['cluster_id'].map(
        lambda x: cluster_labels.get(x, 'Noise' if x == -1 else f'Cluster {x}')
    )

    # Save updated metadata
    metadata.to_csv(metadata_file, sep='\t', index=False)
    logger.info(f"✅ Updated cluster_label column in {metadata_file}")


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Reapply cluster labels from analysis text files to JSON and metadata'
    )
    parser.add_argument(
        '--entity',
        choices=['subreddit', 'rule'],
        help='Reapply labels for only one entity type'
    )
    args = parser.parse_args()

    logger = get_stage_logger("9d", "reapply_cluster_labels")
    log_stage_start(logger, "9d", "Reapply/Override Cluster Labels")

    start_time = time.time()

    # Create directories using PATHS
    embeddings_dir = Path(PATHS['embeddings'])
    clustering_dir = Path(PATHS['clustering'])

    logger.info("\nManual Override Workflow:")
    logger.info("  1. Review output/clustering/{entity}_cluster_analysis.txt")
    logger.info("  2. Create/edit output/clustering/{entity}_label_overrides.txt")
    logger.info("  3. Format: CLUSTER_ID: NEW_LABEL (e.g., '5: Image Context')")
    logger.info("  4. Run this script to apply changes")
    logger.info("\nCluster Merging:")
    logger.info("  - Give multiple clusters the same label to merge them")
    logger.info("  - Merged clusters keep the lowest cluster_id\n")

    try:
        # Determine entity types to process
        entity_types = [args.entity] if args.entity else ['subreddit', 'rule']

        # Reapply labels for each entity type
        for entity_type in entity_types:
            reapply_entity_labels(entity_type, embeddings_dir, clustering_dir, logger)

        logger.info("\nNext steps:")
        logger.info("  - Review updated JSON files in output/clustering/")
        logger.info("  - Review updated metadata files in output/embeddings/")
        logger.info("  - Regenerate plots with: python analysis/plot_clusters.py")

        elapsed = time.time() - start_time
        logger.info(f"🎉 Stage 9d Complete!")
        log_stage_end(logger, "9d", success=True, elapsed_time=elapsed)

        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 9d execution")
        log_stage_end(logger, "9d", success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
