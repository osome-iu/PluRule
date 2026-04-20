#!/usr/bin/env python3
"""
Label Clusters with Semantic Names using LLM

Reads clustered metadata, generates semantic labels for each cluster using vLLM,
and saves updated metadata with cluster_label column.

Usage:
    python label_clusters.py                         # Label both subreddits and rules
    python label_clusters.py --entity subreddit      # Label only subreddits
    python label_clusters.py --entity rule           # Label only rules

Input:
- output/embeddings/all_subreddit_metadata.tsv (with cluster_id column)
- output/embeddings/all_rule_metadata.tsv (with cluster_id column)

Output:
- output/embeddings/all_subreddit_metadata.tsv (updated with cluster_label column)
- output/embeddings/all_rule_metadata.tsv (updated with cluster_label column)
- output/clustering/subreddit_cluster_labels.json
- output/clustering/rule_cluster_labels.json
"""

import sys
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict
from datetime import datetime
import argparse
import time

# Disable vLLM's default logging configuration
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
# os.environ['TQDM_DISABLE'] = '1'

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PATHS
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


LABELING_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"


def build_cluster_prompts(metadata: pd.DataFrame, entity_type: str, tokenizer) -> tuple[Dict[int, str], int]:
    """Build prompts for each cluster and compute max token length.

    Args:
        metadata: Cluster metadata with cluster_id column
        entity_type: 'subreddit' or 'rule'
        tokenizer: Tokenizer for computing token lengths

    Returns:
        (prompts_dict, max_input_tokens)
    """
    prompts = {}
    max_input_tokens = 0

    # Get unique cluster IDs (excluding noise = -1)
    unique_clusters = sorted([c for c in metadata['cluster_id'].unique() if c != -1])

    for cluster_id in unique_clusters:
        # Get items in this cluster
        cluster_items = metadata[metadata['cluster_id'] == cluster_id]

        # Build prompt using full_text column (what was actually embedded)
        if entity_type == 'subreddit':
            items_text = "\n\n-------\n\n".join(row['full_text'].replace('Title', f"r/{row['subreddit']}") for _, row in cluster_items.iterrows())
            prompt = f"""Here are {len(cluster_items)} Reddit communities that were grouped together based on similarity:

{items_text}

What theme or category unites these communities? Provide a very short label (1-2 words) that captures their common theme."""

        else:  # rules
            items_text = "\n\n-------\n\n".join(row['full_text'] for _, row in cluster_items.iterrows())
            prompt = f"""Here are {len(cluster_items)} Reddit moderation rules that were grouped together based on similarity:

{items_text}

What theme or category unites these rules? Provide a very short label (1-2 words) that captures their common purpose."""

        prompts[cluster_id] = prompt

        # Tokenize to measure length
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(formatted)
        max_input_tokens = max(max_input_tokens, len(tokens))

    return prompts, max_input_tokens


def generate_cluster_labels(prompts: Dict[int, str], llm, tokenizer, response_budget: int, logger, n_samples: int = 10) -> tuple[Dict[int, str], Dict[int, str], Dict[int, Dict]]:
    """Generate cluster labels using LLM from prebuilt prompts with majority voting.

    Args:
        prompts: Dict mapping cluster_id to prompt text
        llm: vLLM model instance
        tokenizer: Tokenizer instance
        response_budget: Max tokens for response
        logger: Logger instance
        n_samples: Number of label samples to generate per cluster (default: 10)

    Returns:
        (cluster_labels, cluster_thinking, cluster_details): Three dicts mapping cluster_id to:
            - Best label (by majority vote)
            - Thinking process from best sample
            - Details dict with all samples and vote counts
    """
    cluster_labels = {}
    cluster_thinking = {}
    cluster_details = {}
    logger.info(f"Generating {n_samples} label samples for {len(prompts)} clusters (majority voting)...")

    for cluster_id, prompt in prompts.items():
        try:
            # Format with chat template
            messages = [{"role": "user", "content": prompt}]

            # Generate multiple samples (with enough tokens for thinking + response)
            # Using recommended parameters: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
            sampling_params = SamplingParams(temperature=0.6, top_p=0.92, top_k=20, min_p=0.0, max_tokens=response_budget, n=n_samples)
            outputs = llm.chat(messages, sampling_params=sampling_params)

            # Extract all responses
            all_labels = []
            all_thinking = []

            for output in outputs[0].outputs:
                # Find </think> token index (rindex for last occurrence)
                thinking = ""
                try:
                    think_idx = len(output.token_ids) - output.token_ids[::-1].index(151668) - 1
                    # Get tokens before and after </think>
                    thinking_tokens = output.token_ids[:think_idx]
                    response_tokens = output.token_ids[think_idx + 1:]
                    # Decode thinking process
                    thinking = tokenizer.decode(thinking_tokens, skip_special_tokens=True).strip()
                except ValueError:
                    # </think> token not found, use entire output as response
                    response_tokens = output.token_ids
                    thinking = ""

                # Decode only the response part (skip thinking)
                response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

                # Clean up label (take first line, remove quotes, lowercase)
                label = response.split('\n')[0].strip().strip('"').strip("'").lower()

                all_labels.append(label)
                all_thinking.append(thinking)

            # Count votes for each label
            from collections import Counter
            label_counts = Counter(all_labels)
            best_label, best_count = label_counts.most_common(1)[0]

            # Get thinking from first occurrence of best label
            best_idx = all_labels.index(best_label)
            best_thinking = all_thinking[best_idx]

            cluster_labels[cluster_id] = best_label
            cluster_thinking[cluster_id] = best_thinking
            cluster_details[cluster_id] = {
                'all_labels': all_labels,
                'vote_counts': dict(label_counts),
                'best_label': best_label,
                'best_count': best_count
            }

            logger.info(f"  Cluster {cluster_id}: {best_label} ({best_count}/{n_samples} votes)")

        except Exception as e:
            logger.error(f"  Error generating label for cluster {cluster_id}: {e}", exc_info=True)
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            cluster_thinking[cluster_id] = ""
            cluster_details[cluster_id] = {}

    return cluster_labels, cluster_thinking, cluster_details


def label_entity_type(entity_type: str, embeddings_dir: Path, clustering_dir: Path, llm, tokenizer, response_budget: int, logger) -> None:
    """Label clusters for a single entity type (subreddit or rule)."""
    logger.info("\n" + "="*80)
    logger.info(f"{entity_type.upper()} CLUSTERS")
    logger.info("="*80)

    # Load clustered metadata
    # Use 'all_rule' for rules, 'all_subreddit' for subreddits (both from train/val/test)
    prefix = 'all_rule' if entity_type == 'rule' else 'all_subreddit'
    metadata_file = embeddings_dir / f'{prefix}_metadata.tsv'
    logger.info(f"Loading clustered metadata from {metadata_file}...")
    metadata = pd.read_csv(metadata_file, sep='\t')

    # Check for cluster_id column
    if 'cluster_id' not in metadata.columns:
        logger.error(f"Error: cluster_id column not found in {metadata_file}")
        logger.error("Please run cluster_test_1k.py --apply-best first")
        return

    # Build prompts
    logger.info("Building cluster labeling prompts...")
    prompts, max_input_tokens = build_cluster_prompts(metadata, entity_type, tokenizer)
    logger.info(f"  Built {len(prompts)} prompts (max input tokens: {max_input_tokens})")

    # Generate labels with majority voting
    cluster_labels, cluster_thinking, cluster_details = generate_cluster_labels(prompts, llm, tokenizer, response_budget, logger, n_samples=10)

    # Add labels to metadata
    metadata['cluster_label'] = metadata['cluster_id'].map(lambda x: cluster_labels.get(x, 'Noise' if x == -1 else f'Cluster {x}'))

    # Save updated metadata
    metadata.to_csv(metadata_file, sep='\t', index=False)
    logger.info(f"✅ Saved labeled metadata to: {metadata_file}")

    # Save cluster labels JSON (compact, with voting details)
    labels_file = clustering_dir / f'{entity_type}_cluster_labels.json'
    cluster_data = {int(cid): {
        "label": label,
        "thinking": cluster_thinking.get(cid, ""),
        "voting": cluster_details.get(cid, {})
    } for cid, label in cluster_labels.items()}
    with open(labels_file, 'w') as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Saved cluster labels to: {labels_file}")

    # Save human-readable text file with prompts, thinking, labels, and voting
    readable_file = clustering_dir / f'{entity_type}_cluster_analysis.txt'
    with open(readable_file, 'w') as f:
        f.write(f"CLUSTER ANALYSIS: {entity_type.upper()}S\n")
        f.write(f"{'='*80}\n\n")

        for cluster_id in sorted(prompts.keys()):
            f.write(f"{'='*80}\n")
            f.write(f"CLUSTER {cluster_id}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"PROMPT:\n{'-'*80}\n")
            f.write(prompts[cluster_id])
            f.write(f"\n{'-'*80}\n\n")

            thinking = cluster_thinking.get(cluster_id, "")
            if thinking:
                f.write(f"THINKING:\n{'-'*80}\n")
                f.write(thinking)
                f.write(f"\n{'-'*80}\n\n")

            f.write(f"LABEL: {cluster_labels.get(cluster_id, 'Unknown')}, ID: {cluster_id}\n\n")

            # Add voting details
            details = cluster_details.get(cluster_id, {})
            if details:
                vote_counts = details.get('vote_counts', {})
                best_count = details.get('best_count', 0)
                if vote_counts:
                    f.write(f"VOTING RESULTS:\n{'-'*80}\n")
                    for label, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {count}/10: {label}\n")
                    f.write(f"{'-'*80}\n\n")

            f.write("\n")
    logger.info(f"✅ Saved readable analysis to: {readable_file}")


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Label clusters with semantic names')
    parser.add_argument('--entity', choices=['subreddit', 'rule'], help='Label only one entity type')
    args = parser.parse_args()

    logger = get_stage_logger("9c", "label_clusters")
    log_stage_start(logger, "9c", "Label Clusters with LLM")

    start_time = time.time()

    # Create directories using PATHS
    embeddings_dir = Path(PATHS['embeddings'])
    clustering_dir = Path(PATHS['clustering'])
    clustering_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load tokenizer
        logger.info(f"Loading tokenizer for {LABELING_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(LABELING_MODEL)
        logger.info("✅ Tokenizer loaded")

        # Determine entity types to process
        entity_types = [args.entity] if args.entity else ['subreddit', 'rule']

        # Pre-tokenize all prompts to determine max_model_len
        logger.info("\nPre-tokenizing prompts to determine max_model_len...")
        all_prompts = {}
        max_global_tokens = 0

        for entity_type in entity_types:
            # Use 'all_rule' for rules, 'all_subreddit' for subreddits (both from train/val/test)
            prefix = 'all_rule' if entity_type == 'rule' else 'all_subreddit'
            metadata_file = embeddings_dir / f'{prefix}_metadata.tsv'
            if not metadata_file.exists():
                logger.error(f"Error: {metadata_file} not found. Run cluster_test_1k.py --apply-best first.")
                return 1

            metadata = pd.read_csv(metadata_file, sep='\t')
            if 'cluster_id' not in metadata.columns:
                logger.error(f"Error: cluster_id column not found in {metadata_file}")
                logger.error("Please run cluster_test_1k.py --apply-best first")
                return 1

            prompts, max_tokens = build_cluster_prompts(metadata, entity_type, tokenizer)
            all_prompts[entity_type] = prompts
            max_global_tokens = max(max_global_tokens, max_tokens)

        # Add response budget (thinking + actual response)
        response_budget = 8192
        max_model_len = max_global_tokens + response_budget
        logger.info(f"Max input tokens across all prompts: {max_global_tokens}")
        logger.info(f"Setting max_model_len: {max_model_len} (input + {response_budget} response budget)")

        # Load LLM with optimized max_model_len
        logger.info(f"\nLoading LLM ({LABELING_MODEL})...")
        llm = LLM(model=LABELING_MODEL, gpu_memory_utilization=0.95, tensor_parallel_size=1, max_model_len=max_model_len, seed=0)
        logger.info("✅ LLM loaded")

        # Label each entity type
        for entity_type in entity_types:
            label_entity_type(entity_type, embeddings_dir, clustering_dir, llm, tokenizer, response_budget, logger)
        elapsed = time.time() - start_time
        logger.info(f"🎉 Stage 9c Complete!")
        log_stage_end(logger, "9c", success=True, elapsed_time=elapsed)

        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 9c execution")
        log_stage_end(logger, "9c", success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
