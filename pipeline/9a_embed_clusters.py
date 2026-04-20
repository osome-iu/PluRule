#!/usr/bin/env python3
"""
Embed All Communities and All Dataset Rules

Creates embeddings for:
1. Subreddit embeddings: Unique subreddits from train/val/test - based on title + public description from Stage 2
   - Deduplicated by subreddit name (each subreddit embedded once, even if in multiple splits)
2. Rule embeddings: ALL rules from train/val/test with cumulative violations (deduplicated by text)

Output:
- output/embeddings/all_subreddit_embeddings.tsv - Embedding vectors (one per line, tab-separated)
- output/embeddings/all_subreddit_metadata.tsv - Metadata (Subreddit, Language, Title, Description, FullText, Splits)
- output/embeddings/all_rule_embeddings.tsv - Embedding vectors (one per line, tab-separated)
- output/embeddings/all_rule_metadata.tsv - Metadata (Subreddit, ShortName, Description, FullText, TotalViolations, Splits, NumSubreddits)

Note: Both subreddits and rules are deduplicated. Subreddits by name, rules by text content.
Splits field shows comma-separated list of which datasets (train/val/test) contain each item.
"""

import sys
import os
import json
import time
import torch
import pandas as pd
import zstandard
from typing import Dict, List, Tuple
from tqdm import tqdm
from pathlib import Path

# Disable vLLM's default logging configuration
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
os.environ['TQDM_DISABLE'] = '1'

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PATHS, MIN_MATCHED_COMMENTS, EMBEDDING_MODEL
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import read_json_file
from transformers import AutoTokenizer
from vllm import LLM
from vllm.inputs import TokensPrompt


def load_all_datasets(logger) -> Dict:
    """Load train, val, and test datasets.

    Adds a 'split' field to each subreddit object to track which dataset it came from.
    """
    base_path = PATHS['data']
    datasets = [
        ('train_hydrated.json.zst', 'train'),
        ('val_hydrated.json.zst', 'val'),
        ('test_hydrated.json.zst', 'test')
    ]

    all_subreddits = []

    for dataset_file, split_name in datasets:
        file_path = os.path.join(base_path, dataset_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading {dataset_file}...")
        with open(file_path, 'rb') as f:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                data = json.loads(reader.read())

        subreddits = data.get('subreddits', [])
        # Add split information to each subreddit
        for subreddit in subreddits:
            subreddit['split'] = split_name
        all_subreddits.extend(subreddits)
        logger.info(f"  Loaded {len(subreddits)} subreddits from {dataset_file}")

    logger.info(f"  Total subreddits across all datasets: {len(all_subreddits)}")
    return {'subreddits': all_subreddits}


def load_stage2_data(logger) -> Dict[str, Dict]:
    """Load Stage 2 subreddit data and create mapping."""
    stage2_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')

    if not os.path.exists(stage2_file):
        raise FileNotFoundError(f"Stage 2 file not found: {stage2_file}")

    logger.info(f"Loading Stage 2 data from {stage2_file}...")
    data = read_json_file(stage2_file)

    # Create mapping: subreddit_name -> subreddit_data
    subreddit_map = {}
    for entry in data.get('subreddits', []):
        subreddit_data = entry.get('subreddit', {})
        name = subreddit_data.get('display_name', '').lower()
        if name:
            subreddit_map[name] = subreddit_data

    logger.info(f"  Loaded {len(subreddit_map)} subreddit descriptions")
    return subreddit_map


def prepare_subreddit_texts(all_data: Dict, stage2_map: Dict[str, Dict], logger) -> Tuple[List[str], List[Dict]]:
    """Prepare subreddit texts for embedding and metadata.

    Deduplicates by subreddit name - each unique subreddit is embedded only once.
    Tracks which splits the subreddit appears in.
    """
    texts = []
    metadata = []

    logger.info("Preparing subreddit texts from all datasets (train/val/test)...")

    # Deduplicate by subreddit name, track splits
    subreddit_dedup = {}

    for subreddit_obj in all_data.get('subreddits', []):
        subreddit_name = subreddit_obj.get('subreddit', '')
        language = subreddit_obj.get('language', 'unknown')
        split = subreddit_obj.get('split', 'unknown')

        if subreddit_name not in subreddit_dedup:
            subreddit_dedup[subreddit_name] = {
                'language': language,
                'splits': set()
            }

        subreddit_dedup[subreddit_name]['splits'].add(split)

    # Convert to lists for embedding
    for subreddit_name, info in subreddit_dedup.items():
        # Get title and description from stage2
        stage2_data = stage2_map.get(subreddit_name, {})
        title = stage2_data.get('title', subreddit_name)
        public_description = stage2_data.get('public_description', '')

        # Format: "{title}: {public_description}"
        text = f"{title}: {public_description}"

        texts.append(text)
        metadata.append({
            'subreddit': subreddit_name,
            'language': info['language'],
            'title': title,
            'description': public_description,
            'full_text': text,  # Store full embedded text for later use
            'splits': ','.join(sorted(info['splits']))  # Comma-separated list of splits
        })

    logger.info(f"  Prepared {len(texts)} unique subreddit texts (deduplicated)")
    return texts, metadata


def prepare_rule_texts(all_data: Dict, stage2_map: Dict[str, Dict], logger) -> Tuple[List[str], List[Dict]]:
    """Prepare rule texts for embedding and metadata.

    Processes all rules from train/val/test datasets that appear in thread pairs.
    Creates cumulative distribution across splits, automatically deduplicating rules.
    """
    logger.info("Preparing rule texts from all datasets (train/val/test)...")

    # Deduplicate rules by text content, accumulating violations and tracking usage
    rule_dedup = {}

    for subreddit_obj in all_data.get('subreddits', []):
        subreddit_name = subreddit_obj.get('subreddit', '')
        split = subreddit_obj.get('split', 'unknown')

        # Build rule lookup by short_name_clean
        rule_map = {r.get('short_name_clean', ''): r for r in subreddit_obj.get('rules', []) if r.get('short_name_clean')}

        # Count rule occurrences from thread pairs
        rule_counts = {}
        for pair in subreddit_obj.get('thread_pairs', []):
            rule_name = pair.get('metadata', {}).get('rule', '')
            if rule_name:
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

        # Process rules that appear in thread pairs
        for rule_name, count in rule_counts.items():
            if count > 0 and rule_name in rule_map:
                rule = rule_map[rule_name]

                # Use rule_comprehensive (same format as rule matching)
                text = rule.get('rule_comprehensive', '')

                # Initialize or update dedup entry
                if text not in rule_dedup:
                    rule_dedup[text] = {
                        'short_name': rule.get('short_name_clean', ''),
                        'description': rule.get('description_clean', ''),
                        'subreddits': set(),
                        'total_violations': 0,
                        'splits': set()
                    }

                rule_dedup[text]['subreddits'].add(subreddit_name)
                rule_dedup[text]['total_violations'] += count
                rule_dedup[text]['splits'].add(split)

    # Convert to lists for embedding
    texts = list(rule_dedup.keys())
    metadata = [{
        'subreddit': ','.join(sorted(info['subreddits'])),
        'short_name': info['short_name'],
        'description': info['description'],
        'full_text': text,
        'total_violations': info['total_violations'],
        'splits': ','.join(sorted(info['splits'])),
        'num_subreddits': len(info['subreddits'])
    } for text, info in rule_dedup.items()]

    logger.info(f"  Prepared {len(texts)} unique rule texts (deduplicated, cumulative violations)")
    if metadata:
        logger.info(f"  Average subreddits per rule: {sum(m['num_subreddits'] for m in metadata) / len(metadata):.2f}")

    return texts, metadata


def pretokenize_texts(texts: List[str], tokenizer, logger) -> Tuple[List[List[int]], int]:
    """Pretokenize texts and find max length (like stage 4).

    Returns:
        Tuple of (tokenized_texts, max_length)
    """
    logger.info(f"Tokenizing {len(texts)} texts...")
    tokenized_texts = []
    lengths = []

    for text in tqdm(texts, desc="Tokenizing"):
        # No instruction formatting - just tokenize the text directly (like rules in stage 4)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokenized_texts.append(tokens)
        lengths.append(len(tokens))

    max_length = max(lengths) if lengths else 0
    avg_length = sum(lengths) / len(lengths) if lengths else 0

    logger.info(f"  Max length: {max_length}")
    logger.info(f"  Avg length: {avg_length:.1f}")

    return tokenized_texts, max_length


def embed_with_vllm(tokenized_texts: List[List[int]], model_name: str, max_length: int, logger) -> List[List[float]]:
    """Embed pretokenized texts using vLLM (like stage 4).

    Returns:
        List of embeddings (one per text)
    """
    # Calculate optimal max_model_len with buffer
    optimal_max_len = max(max_length + 50, 512)
    logger.info(f"Initializing vLLM with max_model_len={optimal_max_len}...")

    # Initialize vLLM model
    model = LLM(model=model_name, task="embed", gpu_memory_utilization=0.92, enforce_eager=True, max_model_len=optimal_max_len, seed=0)
    logger.info("✅ vLLM model loaded")

    # Create TokensPrompt objects
    logger.info(f"Creating TokensPrompt objects for {len(tokenized_texts)} texts...")
    prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized_texts]

    # Embed in one batch (vLLM handles batching internally)
    logger.info(f"Embedding {len(prompts)} texts with vLLM...")
    outputs = model.embed(prompts)

    # Extract embeddings (vLLM already normalizes them)
    embeddings = [output.outputs.embedding for output in outputs]
    logger.info(f"  Generated {len(embeddings)} embeddings")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return embeddings


def write_tsv_files(embeddings: List[List[float]], metadata: List[Dict], embedding_file: Path,
                   metadata_file: Path, metadata_columns: List[str], logger):
    """Write embeddings and metadata to TSV files."""
    # Write embeddings
    logger.info(f"Writing embeddings to {embedding_file}...")
    with open(embedding_file, 'w') as f:
        for embedding in embeddings:
            line = '\t'.join(str(x) for x in embedding)
            f.write(line + '\n')

    # Write metadata using pandas to handle escaping properly
    logger.info(f"Writing metadata to {metadata_file}...")
    metadata_df = pd.DataFrame(metadata, columns=metadata_columns)
    metadata_df.to_csv(metadata_file, sep='\t', index=False)

    logger.info(f"  ✅ Wrote {len(embeddings)} rows to {embedding_file}")
    logger.info(f"  ✅ Wrote {len(metadata)} rows to {metadata_file}")


def main():
    """Main execution function."""
    logger = get_stage_logger("9a", "embed_clusters")
    log_stage_start(logger, "9a", "Embed Subreddits and Rules for Clustering")

    start_time = time.time()
    output_dir = Path(PATHS['embeddings'])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data (single function loads all datasets for both subreddits and rules)
        all_data = load_all_datasets(logger)
        stage2_map = load_stage2_data(logger)

        # Prepare texts
        subreddit_texts, subreddit_metadata = prepare_subreddit_texts(all_data, stage2_map, logger)
        rule_texts, rule_metadata = prepare_rule_texts(all_data, stage2_map, logger)

        # Set CUDA device 1 by default
        if torch.cuda.is_available():
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
            logger.info(f"🎯 Using CUDA device 1")
        else:
            logger.info(f"💻 Using CPU mode")

        # Load tokenizer once
        logger.info(f"\nLoading tokenizer for {EMBEDDING_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

        # Pretokenize and embed subreddits
        logger.info("\n" + "="*80)
        logger.info("EMBEDDING SUBREDDITS")
        logger.info("="*80)
        tokenized_subreddits, max_subreddit_len = pretokenize_texts(subreddit_texts, tokenizer, logger)
        subreddit_embeddings = embed_with_vllm(tokenized_subreddits, EMBEDDING_MODEL, max_subreddit_len, logger)

        # Pretokenize and embed rules
        logger.info("\n" + "="*80)
        logger.info("EMBEDDING RULES")
        logger.info("="*80)
        tokenized_rules, max_rule_len = pretokenize_texts(rule_texts, tokenizer, logger)
        rule_embeddings = embed_with_vllm(tokenized_rules, EMBEDDING_MODEL, max_rule_len, logger)

        # Write outputs
        logger.info("\n" + "="*80)
        logger.info("WRITING OUTPUTS")
        logger.info("="*80)

        write_tsv_files(subreddit_embeddings, subreddit_metadata, output_dir / 'all_subreddit_embeddings.tsv',
                       output_dir / 'all_subreddit_metadata.tsv', ['subreddit', 'language', 'title', 'description', 'full_text', 'splits'], logger)

        write_tsv_files(rule_embeddings, rule_metadata, output_dir / 'all_rule_embeddings.tsv',
                       output_dir / 'all_rule_metadata.tsv', ['subreddit', 'short_name', 'description', 'full_text', 'total_violations', 'splits', 'num_subreddits'], logger)

        elapsed = time.time() - start_time
        logger.info(f"Subreddit embeddings: {len(subreddit_embeddings)} x {len(subreddit_embeddings[0])}")
        logger.info(f"Rule embeddings: {len(rule_embeddings)} x {len(rule_embeddings[0])}")
        logger.info(f"🎉 Stage 9a Complete!")
        log_stage_end(logger, "9a", success=True, elapsed_time=elapsed)

        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 9a execution")
        log_stage_end(logger, "9a", success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
