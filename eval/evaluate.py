#!/usr/bin/env python3
"""
Reddit Moderation Evaluation Script

Evaluate Vision Language Models (VLMs) on Reddit moderation tasks using
clustered datasets with multimodal inputs (text + images).

Features:
- Multiple model support: vLLM (Qwen, LLaVA, Llama-Vision) and API (Claude, GPT-4V)
- Configurable context types: control what information is exposed in prompts
- Phrase variations: baseline, chain-of-thought, etc.
- Two-stage evaluation: reasoning generation + clean answer extraction
- Comprehensive metrics: overall, per-rule-cluster, per-subreddit-cluster accuracy

Usage:
    python evaluate.py --model qwen25-vl-7b --split test --context thread_with_rule --phrase cot --mode prefill
    python evaluate.py --model llava-onevision-7b --split val --context full --phrase baseline --mode prefill
    python evaluate.py --model qwen25-vl-7b --split test --context minimal --phrase analyze --mode prompt --debug
"""

import argparse
import json
import sys
import os
from pathlib import Path

# =============================================================================
# EARLY ENVIRONMENT SETUP (before any imports that might use GPU)
# =============================================================================

# Parse CUDA argument early to set CUDA_VISIBLE_DEVICES before importing vLLM
def _parse_cuda_arg() -> str:
    """Parse --cuda argument early, before heavy imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cuda', type=str, default='0')
    args, _ = parser.parse_known_args()
    return args.cuda

# Set CUDA devices BEFORE importing any GPU-related modules
os.environ['CUDA_VISIBLE_DEVICES'] = _parse_cuda_arg()

# Fix MKL threading and set vLLM environment variables
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add eval directory to path
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

# Now safe to import config and helpers (which may import vLLM)
import config
import helpers


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reddit Moderation Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=config.get_supported_models(),
        help='Model to evaluate'
    )

    parser.add_argument(
        '--split', '-s',
        type=str,
        required=True,
        choices=list(config.DATASET_FILES.keys()),
        help='Dataset split to evaluate on'
    )

    parser.add_argument(
        '--context', '-c',
        type=str,
        required=True,
        help='Context flags (dash-separated): none, subreddit, submission, media, discussion, user. Examples: "none", "submission-media", "subreddit-submission-media-discussion-user"'
    )

    parser.add_argument(
        '--phrase', '-p',
        type=str,
        required=True,
        choices=config.get_supported_phrases(),
        help='Phrase to use (e.g., cot, baseline)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='prefill',
        choices=config.PHRASE_MODES,
        help='Phrase mode: prefill (append after template) or prompt (append to question)'
    )

    parser.add_argument(
        '--cuda',
        type=str,
        default='1',
        help='CUDA device IDs for vLLM (e.g., "0,1")'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode with only 5 thread pairs'
    )

    parser.add_argument(
        '--override',
        action='store_true',
        help='Override existing results if they exist'
    )

    parser.add_argument(
        '--max-response-tokens',
        type=int,
        default=2048,
        help='Maximum tokens for response generation (default: 2048)'
    )

    return parser.parse_args()


def _log_section(logger, title: str) -> None:
    """Log a section header."""
    logger.info("\n" + "="*80)
    logger.info(title)
    logger.info("="*80)


def _display_top_clusters(logger, metrics: dict, cluster_type: str, n: int = 5) -> None:
    """
    Display top N clusters by count.

    Args:
        logger: Logger instance
        metrics: Metrics dictionary
        cluster_type: 'rule' or 'subreddit'
        n: Number of top clusters to display
    """
    cluster_key = f'per_{cluster_type}_cluster'
    logger.info(f"Top {n} {cluster_type.title()} Clusters by Count:")

    clusters = sorted(
        metrics[cluster_key].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:n]

    for cluster_label, stats in clusters:
        logger.info(f"  {cluster_label}:")
        logger.info(f"    Overall: {stats['overall_accuracy']:.4f} ({stats['count']} pairs)")
        logger.info(f"    Violating: {stats['violating_accuracy']:.4f}, Compliant: {stats['compliant_accuracy']:.4f}")


def main():
    """Main evaluation pipeline."""

    # Parse arguments
    args = parse_arguments()

    # Create logger
    logger, log_path = helpers.create_logger(
        args.split,
        args.model,
        args.context,
        args.phrase,
        args.mode
    )

    # Log CUDA setup (already set at module import time)
    logger.info(f"🖥️  CUDA_VISIBLE_DEVICES set to '{args.cuda}'")

    # Validate arguments
    try:
        config.validate_config_combination(
            args.model,
            args.split,
            args.context,
            args.phrase,
            args.mode
        )
    except ValueError as e:
        logger.error(f"❌ Invalid configuration: {e}")
        sys.exit(1)

    # Normalize mode for baseline phrase (mode is ignored for baseline)
    if args.phrase == 'baseline':
        if args.mode != 'prefill':
            logger.warning(
                f"⚠️  --mode is ignored for baseline phrase. "
                f"Normalizing to 'prefill' mode. Output will be in baseline/ directory."
            )
        args.mode = 'prefill'  # Normalize to prefill to avoid duplicate processing

    logger.info("="*80)
    logger.info("🚀 REDDIT MODERATION EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Context: {args.context}")
    logger.info(f"Phrase: {args.phrase} (mode: {args.mode})")
    logger.info(f"Debug mode: {args.debug}")
    logger.info("="*80)

    try:
        # Check if results already exist
        output_dir = config.get_dir(
            config.OUTPUT_DIR,
            args.split,
            args.model,
            args.context,
            args.phrase,
            args.mode
        )

        # If checkpoint has failed entries, clear stale results so retry flows naturally
        checkpoint_path = output_dir / "flex_checkpoint.json"
        if checkpoint_path.exists() and not args.override:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            failed_count = sum(
                1 for v in checkpoint_data.values()
                if (isinstance(v, dict) and not v.get('content'))
                or (isinstance(v, str) and not v)
            )
            if failed_count > 0:
                for stale in list(output_dir.glob("performance_*.json")) + list(output_dir.glob("reasoning_*.json")):
                    stale.unlink()
                logger.info(f"♻️  Checkpoint has {failed_count} failed entries — cleared stale results for retry")

        existing_performance_files = list(output_dir.glob("performance_*.json"))

        if existing_performance_files and not args.override:
            # Check if existing results are from debug mode (5 pairs)
            latest_perf_file = sorted(existing_performance_files)[-1]
            with open(latest_perf_file, 'r') as f:
                perf_data = json.load(f)
                total_pairs = perf_data.get('metrics', {}).get('overall', {}).get('total_pairs', 0)

            if total_pairs == 5:
                logger.warning(f"⚠️  Found debug results (5 pairs) in {output_dir}")
                logger.warning(f"   Current run will {'use debug mode' if args.debug else 'use full dataset'}")
                if not args.debug:
                    logger.info(f"♻️  Automatically overriding debug results with full evaluation")
                else:
                    logger.warning(f"   Use --override flag to overwrite existing debug results")
                    logger.info("❌ Evaluation skipped")
                    return
            else:
                logger.warning(f"⚠️  Results already exist in {output_dir}")
                logger.warning(f"   Found {len(existing_performance_files)} existing performance file(s) with {total_pairs} pairs")
                logger.warning(f"   Use --override flag to overwrite existing results")
                logger.info("❌ Evaluation skipped")
                return

        elif existing_performance_files and args.override:
            logger.info(f"♻️  Override mode: Will overwrite existing results in {output_dir}")

        # 1. Load dataset
        _log_section(logger, "STEP 1: LOADING DATASET")
        thread_pairs = helpers.load_dataset(args.split, logger, debug=args.debug)

        # 2. Build prompts
        _log_section(logger, "STEP 2: BUILDING PROMPTS")
        thread_pairs = helpers.build_prompts_for_thread_pairs(
            thread_pairs,
            args.context,
            args.phrase,
            args.model,
            args.mode,
            logger
        )

        # 3. Process evaluation
        model_config = config.get_model_config(args.model)
        batch_size = 275  # Batch size for local vLLM processing
        results = []

        if model_config['type'] == 'vllm':
            # vLLM models: Process in batches for GPU memory efficiency
            num_batches = (len(thread_pairs) + batch_size - 1) // batch_size
            logger.info(f"\nProcessing {len(thread_pairs)} pairs in {num_batches} batch(es) of {batch_size}")
            logger.info("="*80)

            for i in range(num_batches):
                start, end = i * batch_size, min((i + 1) * batch_size, len(thread_pairs))
                batch = thread_pairs[start:end]

                logger.info(f"\nBatch {i+1}/{num_batches}: pairs {start}-{end-1} ({len(batch)} pairs)")

                # Apply chat templates
                _log_section(logger, f"STEP 3.{i+1}: APPLYING CHAT TEMPLATES")
                batch, resource_stats = helpers.apply_chat_template(batch, args.model, logger)

                # Two-stage evaluation
                _log_section(logger, f"STEP 4.{i+1}: TWO-STAGE EVALUATION")
                num_gpus = len(args.cuda.split(','))
                batch_results = helpers.evaluate_two_stage_vllm(
                    batch, args.model, model_config, num_gpus,
                    resource_stats, args.max_response_tokens, args.context, logger
                )

                results.extend(batch_results)
                logger.info(f"✓ Batch {i+1}/{num_batches} complete ({len(results)}/{len(thread_pairs)} pairs)")
        else:
            # API models: Send all data at once for Stage 1 (OpenAI handles 190MB/50K limits)
            # Stage 2 uses batch_size for local vLLM processing
            logger.info(f"\nProcessing {len(thread_pairs)} pairs via API (Stage 2 batch_size={batch_size})")
            logger.info("="*80)

            _log_section(logger, "STEP 3: BUILDING PROMPTS FOR API")
            # Note: Chat templates for API models are applied inside evaluate_two_stage_api

            _log_section(logger, "STEP 4: TWO-STAGE EVALUATION (API)")
            results = helpers.evaluate_two_stage_api(
                thread_pairs, model_config, output_dir,
                args.context, args.max_response_tokens, logger,
                override=args.override,
                stage2_batch_size=batch_size
            )

        logger.info("\n" + "="*80)
        logger.info(f"✓ All batches complete! Total results: {len(results)}")
        logger.info("="*80)

        # 5. Calculate metrics
        _log_section(logger, "STEP 5: CALCULATING METRICS")
        metrics = helpers.calculate_metrics(results, logger)

        # 6. Save results
        _log_section(logger, "STEP 6: SAVING RESULTS")
        reasoning_path, performance_path = helpers.save_results(
            results,
            metrics,
            output_dir,
            args.model,
            args.split,
            args.context,
            args.phrase,
            args.mode,
            logger
        )

        # 7. Display final results
        _log_section(logger, "✅ EVALUATION COMPLETE")
        logger.info("Overall Metrics:")
        logger.info(f"  Total thread pairs: {metrics['overall']['total_pairs']}")
        logger.info(f"  Total threads: {metrics['overall']['total_threads']}")
        logger.info(f"  Overall accuracy: {metrics['overall']['overall_accuracy']:.4f}")
        logger.info(f"  Violating accuracy: {metrics['overall']['violating_accuracy']:.4f}")
        logger.info(f"  Compliant accuracy: {metrics['overall']['compliant_accuracy']:.4f}")
        logger.info("")

        # Display top clusters
        _display_top_clusters(logger, metrics, 'rule')
        logger.info("")
        _display_top_clusters(logger, metrics, 'subreddit')

        logger.info("")
        logger.info(f"📊 Reasoning traces: {reasoning_path}")
        logger.info(f"📊 Performance metrics: {performance_path}")
        logger.info(f"📝 Evaluation logs: {log_path}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"💥 Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
