#!/usr/bin/env python3
"""
Cluster Test Communities and All Rules

Two modes:
1. Grid search: Find optimal UMAP + HDBSCAN parameters
2. Apply best: Apply best parameters and save clustered metadata

Usage:
    python cluster_test_1k.py --grid-search    # Run grid search only
    python cluster_test_1k.py --apply-best     # Apply best params and save clusters
    python cluster_test_1k.py                  # Run both (grid search then apply)

Output:
- Grid search mode:
  - output/clustering/subreddit_grid_search_results.json
  - output/clustering/rule_grid_search_results.json

- Apply best mode:
  - output/embeddings/all_subreddit_metadata.tsv (updated with cluster columns)
  - output/embeddings/all_rule_metadata.tsv (updated with cluster columns)
  - output/embeddings/all_subreddit_embeddings_reduced.tsv
  - output/embeddings/all_rule_embeddings_reduced.tsv
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple
import multiprocessing
from pathlib import Path
import time
import argparse

# Fix OpenMP fork issue - must be set before importing numpy-based libraries
multiprocessing.set_start_method('spawn', force=True)

# Suppress benign warnings
warnings.filterwarnings('ignore', message='overflow encountered in power', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='The TBB threading layer')
warnings.filterwarnings('ignore', message='n_jobs value .* overridden')
warnings.filterwarnings('ignore', message="'force_all_finite' was renamed", category=FutureWarning)

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from config import PATHS, PROCESSES
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
import umap
import hdbscan
from hdbscan import validity_index

# Default static parameters - single source of truth for reproducibility
DEFAULT_UMAP_STATIC_PARAMS = {'metric': 'cosine', 'init': 'pca', 'n_epochs': 1000, 'verbose': False}
UMAP_RANDOM_STATE = {'subreddit': 1, 'rule': 0}
DEFAULT_HDBSCAN_STATIC_PARAMS = {'cluster_selection_method': 'eom'}


def load_embeddings(embeddings_file: str, metadata_file: str, logger) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata from TSV files."""
    logger.info(f"Loading embeddings from {embeddings_file}...")
    embeddings = np.loadtxt(embeddings_file, delimiter='\t')

    logger.info(f"Loading metadata from {metadata_file}...")
    metadata = pd.read_csv(metadata_file, sep='\t')

    logger.info(f"  Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    logger.info(f"  Loaded {len(metadata)} metadata rows")

    assert len(embeddings) == len(metadata), "Embeddings and metadata must have same length"

    return embeddings, metadata


def run_cluster(embeddings: np.ndarray, umap_params: Dict, hdbscan_params: Dict,
                umap_static_params: Dict = None, hdbscan_static_params: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Run UMAP dimensionality reduction followed by HDBSCAN clustering.

    This is the core reusable function used by both grid search and final application.

    Args:
        embeddings: Original high-dimensional embeddings
        umap_params: Dict with 'n_neighbors', 'n_components', 'min_dist'
        hdbscan_params: Dict with 'min_cluster_size', 'min_samples', 'metric'
        umap_static_params: Optional static UMAP params (defaults: cosine metric, random_state=0)
        hdbscan_static_params: Optional static HDBSCAN params (defaults: eom selection)

    Returns:
        Tuple of (reduced_embeddings, labels, probabilities, clusterer)
    """
    # Use module defaults if not provided
    if umap_static_params is None:
        umap_static_params = DEFAULT_UMAP_STATIC_PARAMS
    if hdbscan_static_params is None:
        hdbscan_static_params = DEFAULT_HDBSCAN_STATIC_PARAMS

    # UMAP reduction
    reducer = umap.UMAP(
        n_neighbors=umap_params['n_neighbors'],
        n_components=umap_params['n_components'],
        min_dist=umap_params['min_dist'],
        **umap_static_params
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_params['min_cluster_size'],
        min_samples=hdbscan_params['min_samples'],
        metric=hdbscan_params['metric'],
        **hdbscan_static_params
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    probabilities = clusterer.probabilities_

    return reduced_embeddings, labels, probabilities, clusterer


def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray, clusterer) -> Dict:
    """Evaluate clustering quality with multiple metrics.

    Note: Evaluates DBCV on UMAP-reduced embeddings to avoid curse of dimensionality
    issues. The reduced space is where clustering was performed and where density-based
    metrics are most meaningful.
    """
    # Count clusters and noise
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = (labels == -1).sum()
    noise_ratio = n_noise / len(labels)

    # DBCV score on reduced embeddings (only if we have clusters)
    if n_clusters > 0:
        try:
            dbcv = validity_index(embeddings.astype(np.float64), labels, metric='euclidean')
        except Exception as e:
            # Can't use logger in worker process, print to stderr which is captured
            print(f"Warning: Failed to compute DBCV: {e}", file=sys.stderr)
            dbcv = -999  # Flag as failed
    else:
        dbcv = -999

    # Relative validity
    relative_validity = clusterer.relative_validity_ if hasattr(clusterer, 'relative_validity_') else None

    # Cluster sizes
    cluster_sizes = []
    for label in sorted([l for l in unique_labels if l != -1]):
        cluster_sizes.append(int((labels == label).sum()))

    # Average probability (confidence)
    avg_probability = clusterer.probabilities_.mean() if hasattr(clusterer, 'probabilities_') else None

    return {
        'dbcv': float(dbcv) if dbcv != -999 else None,
        'relative_validity': float(relative_validity) if relative_validity is not None else None,
        'n_clusters': int(n_clusters),
        'n_noise': int(n_noise),
        'noise_ratio': float(noise_ratio),
        'avg_probability': float(avg_probability) if avg_probability is not None else None,
        'cluster_sizes': cluster_sizes,
        'min_cluster_size': int(min(cluster_sizes)) if cluster_sizes else 0,
        'max_cluster_size': int(max(cluster_sizes)) if cluster_sizes else 0,
        'avg_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0
    }


def process_umap_params(umap_params_tuple, embeddings, hdbscan_params_list, iteration_start,
                       umap_static_params, hdbscan_static_params):
    """Process a single UMAP parameter combination with all HDBSCAN combinations.

    Used by grid search for parallel processing.
    """
    n_neighbors, n_components, min_dist = umap_params_tuple

    # Run UMAP once
    umap_start = time.time()
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, **umap_static_params)
    reduced_embeddings = reducer.fit_transform(embeddings)
    umap_time = time.time() - umap_start

    results = []
    for idx, hdbscan_params_tuple in enumerate(hdbscan_params_list):
        iteration = iteration_start + idx

        # Unpack HDBSCAN parameters
        min_cluster_size, min_samples, metric = hdbscan_params_tuple

        # Run HDBSCAN
        hdbscan_start = time.time()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, **hdbscan_static_params)
        labels = clusterer.fit_predict(reduced_embeddings)
        hdbscan_time = time.time() - hdbscan_start

        # Evaluate (pass reduced embeddings for DBCV)
        metrics = evaluate_clustering(reduced_embeddings, labels, clusterer)

        result = {
            'iteration': iteration,
            'umap': {'n_neighbors': n_neighbors, 'n_components': n_components, 'min_dist': min_dist, 'time_seconds': umap_time},
            'hdbscan': {'min_cluster_size': min_cluster_size, 'min_samples': min_samples, 'metric': metric, 'time_seconds': hdbscan_time},
            'metrics': metrics
        }
        results.append(result)

    return results


def run_grid_search(embeddings: np.ndarray, param_grid: Dict[str, List], data_name: str, logger,
                    umap_static_params: Dict = None) -> List[Dict]:
    """Run grid search over UMAP and HDBSCAN parameters in parallel."""
    if umap_static_params is None:
        umap_static_params = DEFAULT_UMAP_STATIC_PARAMS

    # Generate all parameter combinations
    umap_params_list = list(product(param_grid['n_neighbors'], param_grid['n_components'], param_grid['min_dist']))
    hdbscan_params_list = list(product(param_grid['min_cluster_size'], param_grid['min_samples'], param_grid['metric']))

    total_combinations = len(umap_params_list) * len(hdbscan_params_list)
    logger.info(f"Testing {total_combinations} parameter combinations for {data_name}")
    logger.info(f"  UMAP combinations: {len(umap_params_list)}")
    logger.info(f"  HDBSCAN combinations: {len(hdbscan_params_list)}")
    logger.info(f"  Using {PROCESSES} parallel processes")
    logger.info(f"  UMAP static params: {umap_static_params}")
    logger.info(f"  HDBSCAN static params: {DEFAULT_HDBSCAN_STATIC_PARAMS}")

    # Prepare tasks for each UMAP param set
    tasks = []
    for i, umap_params in enumerate(umap_params_list):
        iteration_start = i * len(hdbscan_params_list) + 1
        tasks.append((umap_params, embeddings, hdbscan_params_list, iteration_start, umap_static_params, DEFAULT_HDBSCAN_STATIC_PARAMS))

    # Run in parallel
    logger.info(f"Starting parallel grid search...")
    start_time = time.time()

    with multiprocessing.Pool(processes=PROCESSES) as pool:
        results_nested = pool.starmap(process_umap_params, tasks)

    # Flatten results
    results = [item for sublist in results_nested for item in sublist]

    elapsed = time.time() - start_time
    logger.info(f"Grid search completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    logger.info(f"Processed {len(results)} combinations")

    return results


def analyze_results(results: List[Dict], logger, min_clusters: int = 5, max_clusters: int = 30) -> Dict:
    """Analyze grid search results and find best parameters.

    Args:
        results: List of grid search results
        logger: Logger instance
        min_clusters: Minimum number of clusters required (default: 5)
        max_clusters: Maximum number of clusters allowed (default: 30)
    """
    # Filter out failed runs and enforce cluster count constraints
    valid_results = [r for r in results
                     if r['metrics']['dbcv'] is not None
                     and min_clusters <= r['metrics']['n_clusters'] <= max_clusters]

    logger.info(f"Filtered to {len(valid_results)} results with {min_clusters}-{max_clusters} clusters")

    if not valid_results:
        logger.error("No valid results found!")
        return None

    # Sort by DBCV (primary) and noise_ratio (secondary)
    sorted_results = sorted(valid_results, key=lambda x: (x['metrics']['dbcv'], -x['metrics']['noise_ratio']), reverse=True)

    # Get top 10
    top_10 = sorted_results[:10]

    logger.info(f"\n{'='*80}")
    logger.info("TOP 10 PARAMETER COMBINATIONS:")
    logger.info(f"{'='*80}")

    for i, result in enumerate(top_10, 1):
        umap_params = result['umap']
        hdbscan_params = result['hdbscan']
        metrics = result['metrics']

        logger.info(f"\n#{i} - DBCV: {metrics['dbcv']:.4f}")
        logger.info(f"  UMAP: n_neighbors={umap_params['n_neighbors']}, n_components={umap_params['n_components']}, min_dist={umap_params['min_dist']}")
        logger.info(f"  HDBSCAN: min_cluster_size={hdbscan_params['min_cluster_size']}, min_samples={hdbscan_params['min_samples']}, metric={hdbscan_params['metric']}")
        logger.info(f"  Metrics: {metrics['n_clusters']} clusters, {metrics['noise_ratio']:.1%} noise, avg_size={metrics['avg_cluster_size']:.1f}")

    best = sorted_results[0]

    return {
        'best_params': {
            'umap': {k: v for k, v in best['umap'].items() if k != 'time_seconds'},
            'hdbscan': {k: v for k, v in best['hdbscan'].items() if k != 'time_seconds'}
        },
        'best_metrics': best['metrics'],
        'top_10': top_10,
        'all_results': sorted_results
    }


def apply_best_clustering(embeddings: np.ndarray, metadata: pd.DataFrame, best_params: Dict,
                          entity_type: str, embeddings_dir: Path, logger,
                          umap_static_params: Dict = None) -> None:
    """Apply best clustering parameters and save results with cluster assignments."""
    logger.info(f"\nApplying best parameters for {entity_type}:")
    logger.info(f"  UMAP: {best_params['umap']}")
    logger.info(f"  HDBSCAN: {best_params['hdbscan']}")

    reduced_embeddings, labels, probabilities, _ = run_cluster(
        embeddings, best_params['umap'], best_params['hdbscan'],
        umap_static_params=umap_static_params
    )

    # Print cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"  Result: {n_clusters} clusters, {n_noise}/{len(labels)} noise ({n_noise/len(labels)*100:.1f}%)")

    # Add cluster assignments to metadata
    metadata['cluster_id'] = labels
    metadata['cluster_probability'] = probabilities

    # Drop stale cluster_label column if it exists (will be regenerated by label_clusters.py)
    if 'cluster_label' in metadata.columns:
        metadata = metadata.drop(columns=['cluster_label'])
        logger.info("  Dropped stale cluster_label column (will be regenerated by label_clusters.py)")

    # Save updated metadata
    # Use 'all_rule' for rules, 'all_subreddit' for subreddits (both from train/val/test)
    prefix = 'all_rule' if entity_type == 'rule' else 'all_subreddit'
    metadata_file = embeddings_dir / f'{prefix}_metadata.tsv'
    metadata.to_csv(metadata_file, sep='\t', index=False)
    logger.info(f"  ✅ Saved clustered metadata to: {metadata_file}")

    # Save reduced embeddings
    reduced_file = embeddings_dir / f'{prefix}_embeddings_reduced.tsv'
    np.savetxt(reduced_file, reduced_embeddings, delimiter='\t')
    logger.info(f"  ✅ Saved reduced embeddings to: {reduced_file}")


def process_entity_type(entity_type: str, param_grid: Dict, embeddings_dir: Path, output_dir: Path,
                        run_grid: bool, run_apply: bool, logger) -> None:
    """Process a single entity type (subreddit or rule) through grid search and/or application."""
    logger.info("\n" + "="*80)
    logger.info(f"{entity_type.upper()} EMBEDDINGS")
    logger.info("="*80)

    # Build entity-specific UMAP static params with fixed random_state
    umap_static_params = {**DEFAULT_UMAP_STATIC_PARAMS, 'random_state': UMAP_RANDOM_STATE[entity_type]}
    logger.info(f"  UMAP random_state: {UMAP_RANDOM_STATE[entity_type]}")

    # Load data
    prefix = 'all_rule' if entity_type == 'rule' else 'all_subreddit'
    embeddings, metadata = load_embeddings(
        str(embeddings_dir / f'{prefix}_embeddings.tsv'),
        str(embeddings_dir / f'{prefix}_metadata.tsv'),
        logger
    )

    # Grid search
    if run_grid:
        logger.info(f"\nRunning grid search for {entity_type}s...")
        results = run_grid_search(embeddings, param_grid, f'{entity_type}s', logger, umap_static_params)
        analysis = analyze_results(results, logger)

        output_file = output_dir / f'{entity_type}_grid_search_results.json'
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"\n✅ {entity_type.capitalize()} grid search results saved to: {output_file}")

    # Apply best params
    if run_apply:
        if run_grid:
            best_params = analysis['best_params']
        else:
            output_file = output_dir / f'{entity_type}_grid_search_results.json'
            with open(output_file) as f:
                best_params = json.load(f)['best_params']

        apply_best_clustering(embeddings, metadata, best_params, entity_type, embeddings_dir, logger, umap_static_params)


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Cluster test_1k embeddings')
    parser.add_argument('--grid-search', action='store_true', help='Run grid search only')
    parser.add_argument('--apply-best', action='store_true', help='Apply best params only')
    args = parser.parse_args()

    # Determine mode
    if args.grid_search and args.apply_best:
        print("Error: Cannot specify both --grid-search and --apply-best")
        return 1

    run_grid = args.grid_search or not args.apply_best  # Default is to run both
    run_apply = args.apply_best or not args.grid_search

    logger = get_stage_logger("9b", "cluster_embeddings")
    log_stage_start(logger, "9b", "Cluster Embeddings with UMAP + HDBSCAN")

    # Create directories using PATHS
    output_dir = Path(PATHS['clustering'])
    embeddings_dir = Path(PATHS['embeddings'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Mode: {'Grid search only' if args.grid_search else ('Apply best only' if args.apply_best else 'Full pipeline')}")

    start_time = time.time()

    try:
        # Define parameter grids
        # Note: UMAP uses cosine on original 4096D, HDBSCAN uses euclidean on reduced space
        param_grids = {
            'subreddit': {
                'n_neighbors': [5, 10, 15, 20, 30],
                'n_components': [5, 10, 15, 20, 25, 30],
                'min_dist': [0.0],
                'min_cluster_size': [15, 20, 25, 30],
                'min_samples': [5, 10, 15, 20],
                'metric': ['euclidean']
            },
            'rule': {
                'n_neighbors': [5, 10, 15, 20, 30],
                'n_components': [5, 10, 15, 20, 25, 30],
                'min_dist': [0.0],
                'min_cluster_size': [15, 20, 25, 30],
                'min_samples': [5, 10, 15, 20],
                'metric': ['euclidean']
            },
        }

        # Process both entity types
        for entity_type in ['subreddit', 'rule']:
            process_entity_type(entity_type, param_grids[entity_type], embeddings_dir, output_dir, run_grid, run_apply, logger)

        # Summary
        elapsed = time.time() - start_time
        logger.info(f"🎉 Stage 9b Complete!")
        log_stage_end(logger, "9b", success=True, elapsed_time=elapsed)

        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 9b execution")
        log_stage_end(logger, "9b", success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
