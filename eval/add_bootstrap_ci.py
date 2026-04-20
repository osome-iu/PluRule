#!/usr/bin/env python3
"""
Add Bootstrap Confidence Intervals to All Performance Results

Generates bootstrap indices ONCE from the test set, then applies them to ALL
evaluated models to ensure comparable CIs across models.

Uses CuPy/CUDA GPU acceleration for fast computation (100k iterations in seconds).

Usage:
    python add_bootstrap_ci.py                          # Process all results
    python add_bootstrap_ci.py --n-bootstrap 100000     # More bootstrap samples
    python add_bootstrap_ci.py --dry-run                # Preview without saving
    python add_bootstrap_ci.py --regenerate             # Force regenerate indices
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime

# Set CUDA device before importing cupy. Respects shell override.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

import cupy as cp

# Suppress expected warnings for empty cluster slices
warnings.filterwarnings('ignore', message='Mean of empty slice')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.config import DATASET_FILES, OUTPUT_DIR


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_language(lang_code: str) -> str:
    """Normalize language code by taking root (e.g., en-au → en, pt_BR → pt)."""
    return lang_code.replace('_', '-').split('-')[0]


# =============================================================================
# CONFIGURATION
# =============================================================================

EVAL_OUTPUT_DIR = OUTPUT_DIR
BOOTSTRAP_INDICES_FILE = EVAL_OUTPUT_DIR / "bootstrap_indices.npz"


# =============================================================================
# TEST SET LOADING
# =============================================================================

def load_test_set() -> Dict[str, Any]:
    """
    Load test set and return structured data.

    Returns:
        Dict with n_samples, cluster arrays, cluster names, and mod_comment_ids
    """
    test_path = DATASET_FILES['test']
    print(f"📂 Loading test set from: {test_path}")

    if test_path.suffix == '.zst':
        import zstandard as zstd
        with open(test_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                data = json.loads(reader.read())
    else:
        with open(test_path, 'r') as f:
            data = json.load(f)

    # Extract cluster labels and language for each pair
    rule_clusters = []
    subreddit_clusters = []
    languages = []
    mod_comment_ids = []

    for subreddit_data in data['subreddits']:
        subreddit_cluster_label = subreddit_data.get('subreddit_cluster_label', 'Other')
        language = subreddit_data.get('language', 'unknown')
        normalized_lang = normalize_language(language)

        for pair in subreddit_data['thread_pairs']:
            rule_cluster_label = pair['metadata'].get('rule_cluster_label', 'Other')
            rule_clusters.append(rule_cluster_label)
            subreddit_clusters.append(subreddit_cluster_label)
            languages.append(normalized_lang)
            mod_comment_ids.append(pair['mod_comment_id'])

    n_samples = len(mod_comment_ids)

    # Convert to numeric indices for fast array operations
    unique_rules = sorted(set(rule_clusters))
    unique_subreddits = sorted(set(subreddit_clusters))
    unique_languages = sorted(set(languages))

    rule_to_idx = {r: i for i, r in enumerate(unique_rules)}
    subreddit_to_idx = {s: i for i, s in enumerate(unique_subreddits)}
    language_to_idx = {l: i for i, l in enumerate(unique_languages)}

    rule_cluster_ids = np.array([rule_to_idx[r] for r in rule_clusters], dtype=np.int32)
    subreddit_cluster_ids = np.array([subreddit_to_idx[s] for s in subreddit_clusters], dtype=np.int32)
    language_ids = np.array([language_to_idx[l] for l in languages], dtype=np.int32)

    print(f"   ✅ {n_samples} pairs, {len(unique_rules)} rule clusters, {len(unique_subreddits)} subreddit clusters, {len(unique_languages)} languages")

    return {
        'n_samples': n_samples,
        'rule_cluster_ids': rule_cluster_ids,
        'subreddit_cluster_ids': subreddit_cluster_ids,
        'language_ids': language_ids,
        'rule_cluster_names': unique_rules,
        'subreddit_cluster_names': unique_subreddits,
        'language_names': unique_languages,
        'mod_comment_ids': mod_comment_ids
    }


# =============================================================================
# DISCOVERY FUNCTIONS
# =============================================================================

def find_all_result_dirs(eval_dir: Path) -> List[Path]:
    """Find all directories containing evaluation results."""
    result_dirs = []

    for perf_file in eval_dir.glob("*/test/*/*/performance_*.json"):
        result_dir = perf_file.parent
        if result_dir not in result_dirs:
            if list(result_dir.glob("reasoning_*.json")):
                result_dirs.append(result_dir)

    return sorted(result_dirs)


def get_latest_files(result_dir: Path) -> Tuple[Path, Path]:
    """Get the most recent reasoning and performance files from a directory."""
    reasoning_files = sorted(result_dir.glob("reasoning_*.json"))
    performance_files = sorted(result_dir.glob("performance_*.json"))

    if not reasoning_files or not performance_files:
        raise FileNotFoundError(f"Missing files in {result_dir}")

    return reasoning_files[-1], performance_files[-1]


def parse_result_dir(result_dir: Path) -> Dict[str, str]:
    """Parse result directory path into config components."""
    parts = result_dir.parts
    return {
        'model': parts[-4],
        'split': parts[-3],
        'context': parts[-2],
        'phrase': parts[-1]
    }


# =============================================================================
# SCORE LOADING
# =============================================================================

def load_scores_from_reasoning(reasoning_path: Path,
                                mod_comment_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load scores from reasoning JSON, aligned to test set order.

    Args:
        reasoning_path: Path to reasoning JSON
        mod_comment_ids: List of mod_comment_ids in test set order

    Returns:
        Tuple of (violating_scores, compliant_scores) as numpy arrays
    """
    with open(reasoning_path) as f:
        results = json.load(f)

    # Build lookup by mod_comment_id
    scores_by_id = {
        r['mod_comment_id']: (r['violating']['score'], r['compliant']['score'])
        for r in results
    }

    # Align to test set order
    violating_scores = np.zeros(len(mod_comment_ids), dtype=np.float64)
    compliant_scores = np.zeros(len(mod_comment_ids), dtype=np.float64)

    for i, mid in enumerate(mod_comment_ids):
        if mid in scores_by_id:
            violating_scores[i], compliant_scores[i] = scores_by_id[mid]
        else:
            violating_scores[i] = np.nan
            compliant_scores[i] = np.nan

    return violating_scores, compliant_scores


# =============================================================================
# BOOTSTRAP FUNCTIONS (VECTORIZED)
# =============================================================================

def generate_bootstrap_indices(n_samples: int, n_bootstrap: int = 100000,
                                seed: int = 42) -> np.ndarray:
    """
    Generate bootstrap resample indices.

    Args:
        n_samples: Number of samples in dataset
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_bootstrap, n_samples) with resample indices
    """
    rng = np.random.default_rng(seed)
    return rng.choice(n_samples, size=(n_bootstrap, n_samples), replace=True).astype(np.int32)


def save_bootstrap_indices(indices: np.ndarray, path: Path,
                           test_data: Dict[str, Any]):
    """Save bootstrap indices and test set metadata to NPZ for efficiency."""
    np.savez_compressed(
        path,
        indices=indices,
        n_samples=test_data['n_samples'],
        rule_cluster_ids=test_data['rule_cluster_ids'],
        subreddit_cluster_ids=test_data['subreddit_cluster_ids'],
        language_ids=test_data['language_ids'],
        rule_cluster_names=np.array(test_data['rule_cluster_names'], dtype=object),
        subreddit_cluster_names=np.array(test_data['subreddit_cluster_names'], dtype=object),
        language_names=np.array(test_data['language_names'], dtype=object),
        mod_comment_ids=np.array(test_data['mod_comment_ids'], dtype=object),
        seed=42,
        created=datetime.now().isoformat()
    )
    print(f"💾 Saved bootstrap indices to: {path}")


def load_bootstrap_indices(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load bootstrap indices and test set metadata from NPZ."""
    data = np.load(path, allow_pickle=True)

    test_data = {
        'n_samples': int(data['n_samples']),
        'rule_cluster_ids': data['rule_cluster_ids'],
        'subreddit_cluster_ids': data['subreddit_cluster_ids'],
        'language_ids': data['language_ids'],
        'rule_cluster_names': list(data['rule_cluster_names']),
        'subreddit_cluster_names': list(data['subreddit_cluster_names']),
        'language_names': list(data['language_names']),
        'mod_comment_ids': list(data['mod_comment_ids'])
    }

    return data['indices'], test_data


def compute_cis_gpu(violating_scores: np.ndarray,
                    compliant_scores: np.ndarray,
                    indices_gpu: cp.ndarray,
                    rule_cluster_ids_gpu: cp.ndarray,
                    subreddit_cluster_ids_gpu: cp.ndarray,
                    language_ids_gpu: cp.ndarray,
                    rule_cluster_names: List[str],
                    subreddit_cluster_names: List[str],
                    language_names: List[str],
                    ci: float = 0.95) -> Dict[str, Any]:
    """
    Compute all CIs using GPU-accelerated CuPy operations.

    Args:
        violating_scores: Violating scores array (n_samples,) - CPU numpy
        compliant_scores: Compliant scores array (n_samples,) - CPU numpy
        indices_gpu: Bootstrap indices (n_bootstrap, n_samples) - GPU cupy
        rule_cluster_ids_gpu: Rule cluster index per sample (n_samples,) - GPU cupy
        subreddit_cluster_ids_gpu: Subreddit cluster index per sample (n_samples,) - GPU cupy
        language_ids_gpu: Language index per sample (n_samples,) - GPU cupy
        rule_cluster_names: List of rule cluster names
        subreddit_cluster_names: List of subreddit cluster names
        language_names: List of language names
        ci: Confidence level

    Returns:
        Dict with 'overall', 'per_rule_cluster', 'per_subreddit_cluster', 'per_language' CIs
    """
    alpha = 1 - ci

    # Move scores to GPU
    violating_scores_gpu = cp.array(violating_scores, dtype=cp.float32)
    compliant_scores_gpu = cp.array(compliant_scores, dtype=cp.float32)

    # Vectorized resampling on GPU: (n_bootstrap, n_samples)
    boot_violating = violating_scores_gpu[indices_gpu]
    boot_compliant = compliant_scores_gpu[indices_gpu]

    # Overall accuracy: mean across samples for each bootstrap
    overall_violating_acc = cp.nanmean(boot_violating, axis=1)
    overall_compliant_acc = cp.nanmean(boot_compliant, axis=1)
    overall_acc = (overall_violating_acc + overall_compliant_acc) / 2

    def get_ci_gpu(values_gpu: cp.ndarray) -> List[float]:
        """Compute CI from GPU array."""
        values_cpu = cp.asnumpy(values_gpu)
        valid = values_cpu[~np.isnan(values_cpu)]
        if len(valid) == 0:
            return [0.0, 0.0]
        return [
            float(np.percentile(valid, 100 * alpha / 2)),
            float(np.percentile(valid, 100 * (1 - alpha / 2)))
        ]

    output = {
        'overall': {
            'overall_accuracy_ci': get_ci_gpu(overall_acc),
            'violating_accuracy_ci': get_ci_gpu(overall_violating_acc),
            'compliant_accuracy_ci': get_ci_gpu(overall_compliant_acc)
        },
        'per_rule_cluster': {},
        'per_subreddit_cluster': {},
        'per_language': {}
    }

    # Per-cluster CIs (on GPU)
    # Rule clusters
    for cluster_idx, cluster_name in enumerate(rule_cluster_names):
        mask = (rule_cluster_ids_gpu == cluster_idx)
        if int(cp.sum(mask)) == 0:
            continue

        boot_mask = mask[indices_gpu]
        cluster_violating = cp.where(boot_mask, boot_violating, cp.nan)
        cluster_compliant = cp.where(boot_mask, boot_compliant, cp.nan)

        cluster_violating_acc = cp.nanmean(cluster_violating, axis=1)
        cluster_compliant_acc = cp.nanmean(cluster_compliant, axis=1)
        cluster_acc = (cluster_violating_acc + cluster_compliant_acc) / 2

        output['per_rule_cluster'][cluster_name] = {
            'overall_accuracy_ci': get_ci_gpu(cluster_acc),
            'violating_accuracy_ci': get_ci_gpu(cluster_violating_acc),
            'compliant_accuracy_ci': get_ci_gpu(cluster_compliant_acc)
        }

    # Subreddit clusters
    for cluster_idx, cluster_name in enumerate(subreddit_cluster_names):
        mask = (subreddit_cluster_ids_gpu == cluster_idx)
        if int(cp.sum(mask)) == 0:
            continue

        boot_mask = mask[indices_gpu]
        cluster_violating = cp.where(boot_mask, boot_violating, cp.nan)
        cluster_compliant = cp.where(boot_mask, boot_compliant, cp.nan)

        cluster_violating_acc = cp.nanmean(cluster_violating, axis=1)
        cluster_compliant_acc = cp.nanmean(cluster_compliant, axis=1)
        cluster_acc = (cluster_violating_acc + cluster_compliant_acc) / 2

        output['per_subreddit_cluster'][cluster_name] = {
            'overall_accuracy_ci': get_ci_gpu(cluster_acc),
            'violating_accuracy_ci': get_ci_gpu(cluster_violating_acc),
            'compliant_accuracy_ci': get_ci_gpu(cluster_compliant_acc)
        }

    # Languages
    for lang_idx, lang_name in enumerate(language_names):
        mask = (language_ids_gpu == lang_idx)
        if int(cp.sum(mask)) == 0:
            continue

        boot_mask = mask[indices_gpu]
        lang_violating = cp.where(boot_mask, boot_violating, cp.nan)
        lang_compliant = cp.where(boot_mask, boot_compliant, cp.nan)

        lang_violating_acc = cp.nanmean(lang_violating, axis=1)
        lang_compliant_acc = cp.nanmean(lang_compliant, axis=1)
        lang_acc = (lang_violating_acc + lang_compliant_acc) / 2

        output['per_language'][lang_name] = {
            'overall_accuracy_ci': get_ci_gpu(lang_acc),
            'violating_accuracy_ci': get_ci_gpu(lang_violating_acc),
            'compliant_accuracy_ci': get_ci_gpu(lang_compliant_acc)
        }

    # Free GPU memory
    del boot_violating, boot_compliant, violating_scores_gpu, compliant_scores_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return output


def merge_cis_into_performance(performance: Dict, cis: Dict, n_bootstrap: int) -> Dict:
    """Merge CI results into existing performance structure."""

    # Overall
    for key, ci in cis['overall'].items():
        performance['metrics']['overall'][key] = ci

    # Per-rule-cluster
    for cluster, cluster_cis in cis['per_rule_cluster'].items():
        if cluster in performance['metrics']['per_rule_cluster']:
            for key, ci in cluster_cis.items():
                performance['metrics']['per_rule_cluster'][cluster][key] = ci

    # Per-subreddit-cluster
    for cluster, cluster_cis in cis['per_subreddit_cluster'].items():
        if cluster in performance['metrics']['per_subreddit_cluster']:
            for key, ci in cluster_cis.items():
                performance['metrics']['per_subreddit_cluster'][cluster][key] = ci

    # Per-language
    for lang, lang_cis in cis['per_language'].items():
        if lang in performance['metrics']['per_language']:
            for key, ci in lang_cis.items():
                performance['metrics']['per_language'][lang][key] = ci

    # Metadata
    performance['metrics']['bootstrap_ci'] = {
        'confidence_level': 0.95,
        'n_bootstrap': n_bootstrap,
        'method': 'global_resample_gpu_cupy'
    }

    return performance


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Add bootstrap CIs to all evaluation results"
    )
    parser.add_argument('--n-bootstrap', type=int, default=100000,
                        help='Number of bootstrap samples (default: 100000)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without saving changes')
    parser.add_argument('--regenerate', action='store_true',
                        help='Force regeneration of bootstrap indices')
    parser.add_argument('--eval-dir', type=Path, default=EVAL_OUTPUT_DIR,
                        help='Evaluation output directory')

    args = parser.parse_args()

    print("=" * 70)
    print("BOOTSTRAP CI COMPUTATION (VECTORIZED)")
    print("=" * 70)
    print(f"Bootstrap iterations: {args.n_bootstrap:,}")

    # Check for existing indices
    indices_path = args.eval_dir / "bootstrap_indices.npz"

    if indices_path.exists() and not args.regenerate:
        print(f"\n📂 Loading existing bootstrap data from: {indices_path}")
        indices, test_data = load_bootstrap_indices(indices_path)

        if indices.shape[0] != args.n_bootstrap:
            print(f"   ⚠️  Bootstrap count mismatch ({indices.shape[0]:,} vs {args.n_bootstrap:,})")
            print(f"   Regenerating indices...")
            test_data = load_test_set()
            indices = generate_bootstrap_indices(test_data['n_samples'], args.n_bootstrap)
            if not args.dry_run:
                save_bootstrap_indices(indices, indices_path, test_data)
        else:
            print(f"   ✅ Loaded {indices.shape[0]:,} bootstrap samples for {test_data['n_samples']} pairs")
    else:
        # Load test set and generate indices
        print(f"\n🎲 Generating {args.n_bootstrap:,} bootstrap samples...")
        test_data = load_test_set()
        indices = generate_bootstrap_indices(test_data['n_samples'], args.n_bootstrap)
        if not args.dry_run:
            save_bootstrap_indices(indices, indices_path, test_data)

    # Find all result directories
    result_dirs = find_all_result_dirs(args.eval_dir)
    print(f"\n📁 Found {len(result_dirs)} result directories")

    if not result_dirs:
        print("❌ No results found!")
        return

    # Move data to GPU once
    print(f"\n🚀 Moving data to GPU...")
    indices_gpu = cp.array(indices, dtype=cp.int32)
    rule_cluster_ids_gpu = cp.array(test_data['rule_cluster_ids'], dtype=cp.int32)
    subreddit_cluster_ids_gpu = cp.array(test_data['subreddit_cluster_ids'], dtype=cp.int32)
    language_ids_gpu = cp.array(test_data['language_ids'], dtype=cp.int32)
    print(f"   ✅ GPU memory allocated: {indices_gpu.nbytes / 1e9:.2f} GB")

    # Process directories sequentially (GPU is fast enough)
    print(f"\n{'=' * 70}")
    print(f"PROCESSING RESULTS (GPU-accelerated, CUDA 0)")
    print("=" * 70)

    processed = 0
    skipped = 0
    errors = 0

    for result_dir in result_dirs:
        config = parse_result_dir(result_dir)
        config_str = f"{config['model']}/{config['context']}/{config['phrase']}"

        try:
            reasoning_path, performance_path = get_latest_files(result_dir)

            # Check if _ci file already exists
            ci_performance_path = performance_path.parent / f"{performance_path.stem}_ci{performance_path.suffix}"
            if ci_performance_path.exists() and not args.regenerate:
                with open(ci_performance_path) as f:
                    ci_perf = json.load(f)
                existing_ci = ci_perf.get('metrics', {}).get('bootstrap_ci', {})
                if existing_ci.get('n_bootstrap') == args.n_bootstrap:
                    print(f"  ⏭️  {config_str} - already has {args.n_bootstrap:,} CIs")
                    skipped += 1
                    continue

            # Load performance
            with open(performance_path) as f:
                performance = json.load(f)

            # Load scores aligned to test set order
            violating_scores, compliant_scores = load_scores_from_reasoning(
                reasoning_path, test_data['mod_comment_ids']
            )

            # Check for missing scores
            n_missing = int(np.isnan(violating_scores).sum())
            warning = f" ⚠️ {n_missing} missing" if n_missing > 0 else ""

            # Compute CIs on GPU
            cis = compute_cis_gpu(
                violating_scores, compliant_scores,
                indices_gpu,
                rule_cluster_ids_gpu,
                subreddit_cluster_ids_gpu,
                language_ids_gpu,
                test_data['rule_cluster_names'],
                test_data['subreddit_cluster_names'],
                test_data['language_names']
            )

            # Merge into performance
            performance = merge_cis_into_performance(performance, cis, args.n_bootstrap)

            # Save to new file with _ci suffix
            if not args.dry_run:
                # performance_20251218_013627.json -> performance_20251218_013627_ci.json
                ci_performance_path = performance_path.parent / f"{performance_path.stem}_ci{performance_path.suffix}"
                with open(ci_performance_path, 'w') as f:
                    json.dump(performance, f, indent=2)

            overall_ci = cis['overall']['overall_accuracy_ci']
            overall_acc = performance['metrics']['overall']['overall_accuracy']
            print(f"  ✅ {config_str}: {overall_acc:.3f} [{overall_ci[0]:.3f}, {overall_ci[1]:.3f}]{warning}")
            processed += 1

        except Exception as e:
            print(f"  ❌ {config_str} - {e}")
            if args.dry_run:
                import traceback
                traceback.print_exc()
            errors += 1

    # Free GPU memory
    del indices_gpu, rule_cluster_ids_gpu, subreddit_cluster_ids_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {len(result_dirs)}")

    if args.dry_run:
        print("\n⚠️  DRY RUN - no files were modified")
    else:
        print(f"\n✅ Bootstrap indices saved to: {indices_path}")
        print(f"   (Reuse for future model comparisons)")


if __name__ == "__main__":
    main()
