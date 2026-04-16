#!/usr/bin/env python3
"""
Stage 8: Create Final Datasets with Train/Val/Test Splits

Creates train/val/test splits using an adaptive strategy per subreddit:
- n=1: 1 test, 0 val, 0 train
- n=2: 1 test, 0 val, 1 train
- 3≤n<10: 1 test, 1 val, (n-2) train
- n≥10: 10% test, 10% val, 80% train (rounded, min 1 each)

All three datasets are combined across subreddits (not per-subreddit splits).

Input:
- discussion_threads/{subreddit}_discussion_threads.pkl (from Stage 5)
- comment_trees/{subreddit}_comment_trees.pkl (from Stage 5)
- submissions/{subreddit}_submissions.zst (from Stage 6)
- media/{subreddit}/* (from Stage 7)
- stage7_successful_submission_ids.json
- stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json

Output:
- train_hydrated.json.zst / train_dehydrated.json.zst
- val_hydrated.json.zst / val_dehydrated.json.zst
- test_hydrated.json.zst / test_dehydrated.json.zst
- test_hydrated.json (uncompressed)
- stage8_final_datasets_stats.json
- stage8_thread_distribution_analysis.json
"""

import sys
import os
import time
import pickle
import json
import random
import hashlib
from typing import Dict, List, Tuple
from collections import defaultdict

# Disable vLLM's default logging configuration
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'

# Use only GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (PATHS, MIN_MATCHED_COMMENTS, create_directories)
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import read_json_file, write_json_file, read_zst_lines, json_loads, write_compressed_json
from utils.stats import calculate_jsd_from_uniform, rank_by_score

from vllm import LLM, SamplingParams

# ============================================================================
# Helper Functions
# ============================================================================

def stable_hash(value: str) -> int:
    """Create deterministic integer hash from string (reproducible across runs)."""
    return int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest(), 'big')


# ============================================================================
# LLM Judge Verification
# ============================================================================

def verify_mod_comments_with_llm(subreddit_data: Dict[str, Dict], subreddit_rules: Dict[str, List[Dict]], logger) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Use LLM judge to verify that mod comments actually cite the matched rule.

    Args:
        subreddit_data: Dict of {subreddit: {'submissions': {}, 'thread_pairs': []}}
        subreddit_rules: Dict of {subreddit: [rule_dicts]}
        logger: Logger instance

    Returns:
        (filtered_subreddit_data, verification_results): Filtered data and full LLM responses
    """
    logger.info("🤖 Initializing LLM judge for mod comment verification...")

    # Initialize vLLM
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    llm = LLM(
        seed=0,
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        max_model_len=8192  # Limit to reduce KV cache memory usage
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=10,
        stop=["(a)", "(b)", "(c)", "a)", "b)", "c)", "a", "b", "c", "A", "B", "C"],
        include_stop_str_in_output=True
    )

    # Collect all verification tasks
    verification_tasks = []
    total_pairs = sum(len(d['thread_pairs']) for d in subreddit_data.values())
    logger.info(f"  Preparing {total_pairs} mod comments for verification...")

    for subreddit, data in subreddit_data.items():
        rules = subreddit_rules.get(subreddit, [])
        rule_map = {r.get('short_name_clean', ''): r for r in rules}

        for pair_idx, pair in enumerate(data['thread_pairs']):
            mod_comment = pair.get('mod_comment', {})
            body_clean = mod_comment.get('body_clean', '')
            matched_rule = pair.get('metadata', {}).get('rule', '')

            # Get rule_comprehensive
            rule_obj = rule_map.get(matched_rule)
            rule_comprehensive = rule_obj.get('rule_comprehensive', matched_rule) if rule_obj else matched_rule

            # Build prompt
            prompt = f"""Moderator's Comment: {body_clean}

Target Rule: {rule_comprehensive}

Is the moderator's comment:
(a) Stating a violation of the target rule
(b) Discussing the target rule
(c) Unrelated to the target rule

Final Choice:"""

            verification_tasks.append({
                'subreddit': subreddit,
                'pair_idx': pair_idx,
                'mod_comment_id': pair.get('mod_comment_id'),
                'matched_rule': matched_rule,
                'rule_comprehensive': rule_comprehensive,
                'body_clean': body_clean,
                'prompt': prompt
            })

    # Batch inference
    logger.info(f"  Running LLM inference on {len(verification_tasks)} prompts...")
    prompts = [task['prompt'] for task in verification_tasks]
    outputs = llm.generate(prompts, sampling_params)

    # Parse results
    logger.info(f"  Parsing LLM responses...")
    verification_results = []
    pass_count = 0
    fail_count = 0

    for task, output in zip(verification_tasks, outputs):
        response = output.outputs[0].text.strip()

        # Parse (a)/(b)/(c) response - only (a) is valid
        is_valid = None
        response_lower = response.lower()
        if 'a' in response_lower and 'b' not in response_lower and 'c' not in response_lower:
            is_valid = True
            pass_count += 1
        elif 'b' in response_lower or 'c' in response_lower:
            is_valid = False
            fail_count += 1
        else:
            # Ambiguous response, mark as fail to be conservative
            is_valid = False
            fail_count += 1
            logger.warning(f"  Ambiguous response for {task['mod_comment_id']}: {response}")

        verification_results.append({
            'subreddit': task['subreddit'],
            'mod_comment_id': task['mod_comment_id'],
            'matched_rule': task['matched_rule'],
            'rule_comprehensive': task['rule_comprehensive'],
            'body_clean': task['body_clean'],
            'prompt': task['prompt'],
            'llm_response': response,
            'is_valid': is_valid,
            'pair_idx': task['pair_idx']
        })

    logger.info(f"  ✅ Verification complete: {pass_count} passed, {fail_count} failed ({pass_count / len(verification_tasks) * 100:.1f}% pass rate)")

    # Filter subreddit_data
    logger.info(f"  Filtering thread pairs based on verification...")
    filtered_data = {}

    for subreddit, data in subreddit_data.items():
        # Get verification results for this subreddit
        subreddit_results = [r for r in verification_results if r['subreddit'] == subreddit]
        valid_indices = {r['pair_idx'] for r in subreddit_results if r['is_valid']}

        # Filter thread_pairs
        original_count = len(data['thread_pairs'])
        filtered_pairs = [pair for idx, pair in enumerate(data['thread_pairs']) if idx in valid_indices]

        if filtered_pairs:
            filtered_data[subreddit] = {
                'submissions': data['submissions'],
                'thread_pairs': filtered_pairs
            }
            logger.info(f"    r/{subreddit}: {original_count} → {len(filtered_pairs)} pairs")

    logger.info(f"  Final: {len(filtered_data)} subreddits with verified data")

    return filtered_data, verification_results


# ============================================================================
# Data Loading
# ============================================================================

def load_and_filter_all_data(logger) -> Dict[str, Dict]:
    """Load submissions + threads, filter by Stage 8 success, [removed], and Pushshift only."""

    PUSHSHIFT_CUTOFF = 1677628800  # March 1, 2023 00:00:00 UTC

    # Load Stage 8 successful IDs
    success_file = os.path.join(PATHS['data'], 'stage7_successful_submission_ids.json')
    if not os.path.exists(success_file):
        logger.error(f"Stage 7 success file not found: {success_file}")
        return {}

    logger.info(f"📋 Loading Stage 7 successful submission IDs...")
    data = read_json_file(success_file)
    stage8_ids = {sub: set(ids) for sub, ids in data.get('subreddit_submission_ids', {}).items()}
    logger.info(f"  {sum(len(ids) for ids in stage8_ids.values())} IDs across {len(stage8_ids)} subreddits")

    # Load submissions from Stage 7 (filter by Stage 8 + [removed])
    logger.info(f"📄 Loading submissions from Stage 6...")
    subreddit_data = {}

    for subreddit, needed_ids in stage8_ids.items():
        submissions_file = os.path.join(PATHS['submissions'], f"{subreddit}_submissions.zst")
        if not os.path.exists(submissions_file):
            continue

        submissions = {}
        for line in read_zst_lines(submissions_file):
            sub = json_loads(line)
            sub_id = sub.get('id')
            if sub_id not in needed_ids:
                continue

            # Filter [removed]/[deleted]
            if any('[removed]' in str(sub.get(f, '')) or '[deleted]' in str(sub.get(f, ''))
                   for f in ['selftext', 'title', 'author', 'selftext_html']):
                continue

            # Filter submissions from distinguished moderators
            if sub.get('distinguished') in ['moderator', 'admin']:
                continue

            submissions[sub_id] = sub

        if submissions:
            subreddit_data[subreddit] = {'submissions': submissions, 'thread_pairs': []}

    logger.info(f"  ✅ {sum(len(d['submissions']) for d in subreddit_data.values())} submissions from {len(subreddit_data)} subreddits")

    # Load and filter thread pairs from Stage 6 (Pushshift only)
    logger.info(f"🧵 Loading and filtering thread pairs from Stage 5 (Pushshift only)...")
    total_original, total_filtered = 0, 0

    for subreddit, data in subreddit_data.items():
        threads_file = os.path.join(PATHS['discussion_threads'], f"{subreddit}_discussion_threads.pkl")
        if not os.path.exists(threads_file):
            continue

        try:
            with open(threads_file, 'rb') as f:
                threads = pickle.load(f)

            original = threads.get('thread_pairs', [])
            filtered = []
            for p in original:
                if p.get('metadata', {}).get('submission_id') not in data['submissions']:
                    continue

                # Only include Pushshift data (before March 1, 2023)
                created_utc = p.get('mod_comment', {}).get('created_utc', 0)
                if isinstance(created_utc, str):
                    created_utc = int(float(created_utc))

                if created_utc < PUSHSHIFT_CUTOFF:
                    filtered.append(p)

            data['thread_pairs'] = filtered
            total_original += len(original)
            total_filtered += len(filtered)
        except Exception as e:
            logger.warning(f"  Error loading threads for r/{subreddit}: {e}")

    logger.info(f"  ✅ Filtered {total_original} -> {total_filtered} pairs (Pushshift only)")

    # Remove subreddits with no thread pairs
    subreddit_data = {s: d for s, d in subreddit_data.items() if d['thread_pairs']}
    logger.info(f"  Final: {len(subreddit_data)} subreddits with data")

    return subreddit_data


def load_subreddit_metadata(logger) -> Tuple[Dict[str, List[Dict]], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load rules, languages, titles, and descriptions from Stage 2."""
    rules_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')
    if not os.path.exists(rules_file):
        logger.error(f"Stage 2 rules file not found: {rules_file}")
        return {}, {}, {}, {}

    data = read_json_file(rules_file)
    subreddit_rules = {}
    subreddit_languages = {}
    subreddit_titles = {}
    subreddit_descriptions = {}

    for entry in data.get('subreddits', []):
        sub_data = entry.get('subreddit', {})
        sub_name = sub_data.get('display_name', '').lower()
        if sub_name:
            subreddit_rules[sub_name] = entry.get('rules', [])
            subreddit_languages[sub_name] = sub_data.get('lang', 'unknown')
            subreddit_titles[sub_name] = sub_data.get('title', '')
            subreddit_descriptions[sub_name] = sub_data.get('public_description', '')

    logger.info(f"Loaded metadata for {len(subreddit_rules)} subreddits")
    return subreddit_rules, subreddit_languages, subreddit_titles, subreddit_descriptions


# ============================================================================
# Distribution Analysis
# ============================================================================

def analyze_thread_distribution(subreddit_data: Dict[str, Dict], logger) -> Dict:
    """Analyze distribution of thread pairs across subreddits."""
    logger.info(f"📊 Analyzing thread pair distribution...")

    counts = sorted([len(d['thread_pairs']) for d in subreddit_data.values()])
    n = len(counts)

    percentiles = {
        'min': counts[0] if n > 0 else 0,
        'p25': counts[int(n * 0.25)] if n > 0 else 0,
        'median': counts[int(n * 0.50)] if n > 0 else 0,
        'p75': counts[int(n * 0.75)] if n > 0 else 0,
        'p90': counts[int(n * 0.90)] if n > 0 else 0,
        'p95': counts[int(n * 0.95)] if n > 0 else 0,
        'max': counts[-1] if n > 0 else 0,
        'mean': sum(counts) / n if n > 0 else 0,
        'total': sum(counts)
    }

    # Histogram
    buckets = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, float('inf')]
    labels = ['0', '1-4', '5-9', '10-24', '25-49', '50-99', '100-249', '250-499', '500-999', '1000-2499', '2500-4999', '5000+']
    histogram = {label: 0 for label in labels}

    for count in counts:
        for i in range(len(buckets) - 1):
            if buckets[i] <= count < buckets[i + 1]:
                histogram[labels[i]] += 1
                break

    logger.info(f"  {n} subreddits, {percentiles['total']} pairs")
    logger.info(f"  Min: {percentiles['min']}, Median: {percentiles['median']}, Max: {percentiles['max']}, Mean: {percentiles['mean']:.1f}")
    logger.info(f"  Histogram: " + ", ".join(f"{k}:{v}" for k, v in histogram.items() if v > 0))

    return {
        'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_subreddits': n,
        'percentiles': percentiles,
        'histogram': histogram,
        'split_strategy': 'Adaptive: n=1→(1,0,0), n=2→(1,0,1), 3≤n<10→(1,1,n-2), n≥10→(10%,10%,80%)',
        'subreddit_details': sorted([
            {
                'subreddit': s,
                'thread_pairs': len(d['thread_pairs']),
                'submissions': len(d['submissions'])
            }
            for s, d in subreddit_data.items()
        ], key=lambda x: x['thread_pairs'], reverse=True)
    }


# ============================================================================
# Train/Val/Test Splitting
# ============================================================================

def create_shuffled_answer_options(rule_short_names: List[str], mod_comment_id: str, suffix: str) -> List[Dict]:
    """Create shuffled answer options including 'No rules broken', with labels (a), (b), (c), etc."""
    seed_str = mod_comment_id + suffix
    seed = int(mod_comment_id, 36) + stable_hash(seed_str)
    shuffle_rng = random.Random(seed)

    # Always include "No rules broken" as an option
    all_options = rule_short_names + ['No rules broken']
    shuffle_rng.shuffle(all_options)

    return [{'label': f'({chr(ord("a") + i)})', 'rule': rule} for i, rule in enumerate(all_options)]


def split_pairs(thread_pairs: List[Dict], subreddit: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split thread pairs into test/val/train using adaptive strategy:
    - n=1: 1 test, 0 val, 0 train
    - n=2: 1 test, 0 val, 1 train
    - 3≤n<10: 1 test, 1 val, (n-2) train
    - n≥10: 10% test, 10% val, 80% train (rounded, min 1 each)
    """
    random.seed(stable_hash(subreddit))  # Deterministic per subreddit
    pairs = thread_pairs[:]
    random.shuffle(pairs)

    n = len(pairs)

    if n == 1:
        return pairs, [], []
    elif n == 2:
        return [pairs[0]], [], [pairs[1]]
    elif 3 <= n < 10:
        return [pairs[0]], [pairs[1]], pairs[2:]
    else:  # n >= 10
        test_count = max(1, round(0.10 * n))
        val_count = max(1, round(0.10 * n))
        train_count = n - test_count - val_count

        # Ensure at least 1 train
        if train_count < 1:
            reduce_val = min(val_count - 1, 1 - train_count)
            val_count -= reduce_val
            train_count = n - test_count - val_count

        test_pairs = pairs[:test_count]
        val_pairs = pairs[test_count:test_count + val_count]
        train_pairs = pairs[test_count + val_count:]

        return test_pairs, val_pairs, train_pairs


def process_subreddit_split(subreddit: str, data: Dict, pairs_to_process: List[Dict],
                            rules: List[Dict], language: str, title: str, description: str,
                            trees_data: Dict) -> Dict:
    """Process one split (test/val/train) for a subreddit."""
    if not pairs_to_process:
        return None

    # Extract rule short names for answer options
    rule_short_names = [r.get('short_name_clean', '') for r in rules if r.get('short_name_clean')]

    # Process thread pairs and collect submission IDs used in this split
    processed_pairs = []
    split_submission_ids = set()
    split_tree_ids = set()

    for pair in pairs_to_process:
        submission_id = pair.get('metadata', {}).get('submission_id')
        if not submission_id or submission_id not in data['submissions']:
            continue

        # Make a copy to avoid mutating the original data
        pair_copy = pair.copy()

        # Add answer options for both threads
        mod_comment_id = pair_copy['mod_comment_id']
        matched_rule = pair_copy['metadata'].get('rule', '')

        # Violating thread: correct answer is the matched rule
        violating_options = create_shuffled_answer_options(rule_short_names, mod_comment_id, '_violating')
        pair_copy['violating_answer_options'] = violating_options
        pair_copy['violating_correct_answer'] = next(
            (opt['label'] for opt in violating_options if opt['rule'] == matched_rule), None
        )

        # Compliant thread: correct answer is "No rules broken"
        compliant_options = create_shuffled_answer_options(rule_short_names, mod_comment_id, '_compliant')
        pair_copy['compliant_answer_options'] = compliant_options
        pair_copy['compliant_correct_answer'] = next(
            (opt['label'] for opt in compliant_options if opt['rule'] == 'No rules broken'), None
        )

        processed_pairs.append(pair_copy)
        split_submission_ids.add(submission_id)
        split_tree_ids.add(submission_id)

    if not processed_pairs:
        return None

    # Get media info only for submissions used in this split
    media_dir = os.path.join(PATHS['media'], subreddit)
    submissions_with_media = {}

    for submission_id in split_submission_ids:
        submission_obj = data['submissions'][submission_id]
        media_files = []

        if os.path.exists(media_dir):
            for f in os.listdir(media_dir):
                if f.startswith(f"{submission_id}_"):
                    media_files.append(os.path.join(media_dir, f))

        submissions_with_media[submission_id] = {
            'id': submission_id,
            'submission_object': submission_obj,
            'num_media': len(media_files),
            'media_files': media_files
        }

    # Collect trees for submissions in this split
    split_trees = {}
    trees = trees_data.get('trees', {})
    for submission_id in split_tree_ids:
        if submission_id in trees:
            split_trees[submission_id] = trees[submission_id]

    # Calculate rule distribution
    rule_counts = defaultdict(int)
    for pair in processed_pairs:
        rule = pair.get('metadata', {}).get('rule', '')
        if rule:
            rule_counts[rule] += 1

    return {
        'subreddit': subreddit,
        'title': title,
        'description': description,
        'language': language,
        'data_version': '1.0',
        'last_updated': time.strftime('%Y-%m-%d'),
        'total_thread_pairs': len(processed_pairs),
        'jsd_from_uniform': calculate_jsd_from_uniform(dict(rule_counts)),
        'rules': rules,
        'submissions': submissions_with_media,
        'trees': split_trees,
        'thread_pairs': processed_pairs
    }


# ============================================================================
# Dehydration
# ============================================================================

def dehydrate_dataset(hydrated: Dict) -> Dict:
    """Strip to IDs only, preserving answer options and metadata."""
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

        # Dehydrate trees (replace tree content with placeholder)
        dehydrated_trees = {}
        for tree_id in sub_data.get('trees', {}):
            dehydrated_trees[tree_id] = '[NEEDS_HYDRATION]'

        dehydrated_pairs = []
        for pair in sub_data['thread_pairs']:
            dehydrated_pairs.append({
                'mod_comment_id': pair['mod_comment_id'],
                'mod_comment': '[NEEDS_HYDRATION]',
                'violating_thread': ['[NEEDS_HYDRATION]'] * len(pair['violating_thread']),
                'compliant_thread': ['[NEEDS_HYDRATION]'] * len(pair['compliant_thread']),
                'metadata': pair['metadata'],
                # Preserve answer options (they're deterministically generated, but useful to keep)
                'violating_answer_options': pair.get('violating_answer_options', []),
                'violating_correct_answer': pair.get('violating_correct_answer'),
                'compliant_answer_options': pair.get('compliant_answer_options', []),
                'compliant_correct_answer': pair.get('compliant_correct_answer')
            })

        dehydrated['subreddits'].append({
            'subreddit': sub_data['subreddit'],
            'title': sub_data['title'],
            'description': sub_data['description'],
            'language': sub_data['language'],
            'data_version': sub_data['data_version'],
            'last_updated': sub_data['last_updated'],
            'total_thread_pairs': sub_data['total_thread_pairs'],
            'jsd_from_uniform': sub_data['jsd_from_uniform'],
            'rules': sub_data['rules'],
            'submissions': dehydrated_subs,
            'trees': dehydrated_trees,
            'thread_pairs': dehydrated_pairs
        })

    dehydrated['metadata']['instructions'] = 'Use hydration script. All text fields contain [NEEDS_HYDRATION].'
    return dehydrated




# ============================================================================
# Main Execution
# ============================================================================

def main():
    logger = get_stage_logger(8, "create_final_datasets")
    log_stage_start(logger, 8, "Create Final Datasets (Train/Val/Test)")
    start_time = time.time()

    try:
        create_directories()

        # Load all data (submissions + threads, filtered)
        logger.info("📋 Loading and filtering all data...")
        subreddit_data = load_and_filter_all_data(logger)

        if not subreddit_data:
            logger.error("❌ No data loaded!")
            log_stage_end(logger, 8, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Load metadata
        logger.info("📚 Loading subreddit metadata...")
        subreddit_rules, subreddit_languages, subreddit_titles, subreddit_descriptions = load_subreddit_metadata(logger)

        # LLM Judge Verification
        logger.info("🔍 Verifying mod comments with LLM judge...")
        subreddit_data, verification_results = verify_mod_comments_with_llm(subreddit_data, subreddit_rules, logger)

        # Save verification results
        verification_file = os.path.join(PATHS['data'], 'stage8_llm_verification_results.json')
        verification_stats = {
            'metadata': {
                'stage': 8,
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
                'total_verified': len(verification_results),
                'passed': sum(1 for r in verification_results if r['is_valid']),
                'failed': sum(1 for r in verification_results if not r['is_valid']),
                'pass_rate': sum(1 for r in verification_results if r['is_valid']) / len(verification_results) * 100 if verification_results else 0
            },
            'per_subreddit_stats': {},
            'verification_results': verification_results
        }

        # Calculate per-subreddit stats
        for subreddit in set(r['subreddit'] for r in verification_results):
            sub_results = [r for r in verification_results if r['subreddit'] == subreddit]
            verification_stats['per_subreddit_stats'][subreddit] = {
                'total': len(sub_results),
                'passed': sum(1 for r in sub_results if r['is_valid']),
                'failed': sum(1 for r in sub_results if not r['is_valid']),
                'pass_rate': sum(1 for r in sub_results if r['is_valid']) / len(sub_results) * 100
            }

        write_json_file(verification_stats, verification_file, pretty=True)
        logger.info(f"  ✅ Saved verification results to: {verification_file}")

        if not subreddit_data:
            logger.error("❌ No data remaining after LLM verification!")
            log_stage_end(logger, 8, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Analyze distribution
        logger.info("📊 Analyzing distribution...")
        distribution = analyze_thread_distribution(subreddit_data, logger)
        dist_file = os.path.join(PATHS['data'], 'stage8_thread_distribution_analysis.json')
        write_json_file(distribution, dist_file, pretty=True)
        logger.info(f"  Saved to: {dist_file}")

        # Process each subreddit into test/val/train
        logger.info(f"🚀 Processing {len(subreddit_data)} subreddits...")
        test_subreddits, val_subreddits, train_subreddits = [], [], []

        for subreddit, data in subreddit_data.items():
            rules = subreddit_rules.get(subreddit, [])
            language = subreddit_languages.get(subreddit, 'unknown')
            title = subreddit_titles.get(subreddit, '')
            description = subreddit_descriptions.get(subreddit, '')

            # Load trees once per subreddit
            trees_file = os.path.join(PATHS['comment_trees'], f"{subreddit}_comment_trees.pkl")
            if not os.path.exists(trees_file):
                logger.warning(f"  Trees file not found for r/{subreddit}")
                continue

            try:
                with open(trees_file, 'rb') as f:
                    trees_data = pickle.load(f)
            except Exception as e:
                logger.warning(f"  Error loading trees for r/{subreddit}: {e}")
                continue

            test_pairs, val_pairs, train_pairs = split_pairs(data['thread_pairs'], subreddit)

            if test_pairs:
                test_data = process_subreddit_split(subreddit, data, test_pairs, rules, language, title, description, trees_data)
                if test_data:
                    test_subreddits.append(test_data)

            if val_pairs:
                val_data = process_subreddit_split(subreddit, data, val_pairs, rules, language, title, description, trees_data)
                if val_data:
                    val_subreddits.append(val_data)

            if train_pairs:
                train_data = process_subreddit_split(subreddit, data, train_pairs, rules, language, title, description, trees_data)
                if train_data:
                    train_subreddits.append(train_data)

        logger.info(f"  Test: {len(test_subreddits)}, Val: {len(val_subreddits)}, Train: {len(train_subreddits)}")

        if not test_subreddits:
            logger.error("❌ No data!")
            log_stage_end(logger, 8, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Rank by JSD
        test_subreddits = rank_by_score(test_subreddits, 'jsd_from_uniform', ascending=True)
        val_subreddits = rank_by_score(val_subreddits, 'jsd_from_uniform', ascending=True)
        train_subreddits = rank_by_score(train_subreddits, 'jsd_from_uniform', ascending=True)

        # Log thread pairs per split
        test_pairs = sum(s['total_thread_pairs'] for s in test_subreddits)
        val_pairs = sum(s['total_thread_pairs'] for s in val_subreddits)
        train_pairs = sum(s['total_thread_pairs'] for s in train_subreddits)
        logger.info(f"📊 Thread pairs per split: test={test_pairs}, val={val_pairs}, train={train_pairs}")

        # Load pipeline stats
        stage1_stats = read_json_file(os.path.join(PATHS['data'], 'stage1_subreddit_mod_comment_rankings.json'))
        stage4_stats = read_json_file(os.path.join(PATHS['data'], 'stage3_matching_summary.json'))
        stage6_stats = read_json_file(os.path.join(PATHS['data'], 'stage5_trees_and_threads_summary.json'))
        stage7_stats = read_json_file(os.path.join(PATHS['data'], 'stage6_submission_collection_stats.json'))

        # Create metadata
        def create_metadata(split: str, subs: List[Dict]) -> Dict:
            total_pairs = sum(s['total_thread_pairs'] for s in subs)
            total_submissions = sum(len(s['submissions']) for s in subs)
            total_media = sum(s['num_media'] for sub in subs for s in sub['submissions'].values())
            total_comments = sum(
                len(p['violating_thread']) + len(p['compliant_thread'])
                for sub in subs for p in sub['thread_pairs']
            )

            return {
                'version': '1.0',
                'split': split,
                'creation_date': time.strftime('%Y-%m-%d'),
                'total_subreddits': len(subs),
                'total_thread_pairs': total_pairs,
                'total_submissions': total_submissions,
                'total_submissions_with_media': sum(1 for sub in subs for s in sub['submissions'].values() if s['num_media'] > 0),
                'total_media_files': total_media,
                'total_comments': total_comments,
                'embedding_model': stage4_stats.get('embedding_model', 'unknown'),
                'gold_threshold': stage4_stats.get('gold_threshold', 0),
                'ambiguous_threshold': stage4_stats.get('ambiguous_threshold', 0),
                'date_range': stage4_stats.get('date_range', ('unknown', 'unknown')),
                'pipeline_statistics': {
                    'stage1_total_mod_comments': stage1_stats.get('summary', {}).get('total_mod_comments', 0),
                    'stage4_matched_comments': stage4_stats.get('total_matched', 0),
                    'stage6_successful_thread_pairs': stage6_stats.get('summary', {}).get('total_successful_pairs', 0),
                    'stage6_submissions_collected': stage7_stats.get('summary', {}).get('total_submissions_collected', 0),
                }
            }

        # Write datasets
        logger.info(f"💾 Writing datasets...")
        output_files = {}

        for split, subs in [('test', test_subreddits), ('val', val_subreddits), ('train', train_subreddits)]:
            if not subs:
                continue

            dataset = {'metadata': create_metadata(split, subs), 'subreddits': subs}

            # Hydrated
            hydrated_file = os.path.join(PATHS['data'], f'{split}_hydrated.json.zst')
            hydrated_size = write_compressed_json(dataset, hydrated_file, logger=logger)

            # Dehydrated
            dehydrated = dehydrate_dataset(dataset)
            dehydrated_file = os.path.join(PATHS['data'], f'{split}_dehydrated.json.zst')
            dehydrated_size = write_compressed_json(dehydrated, dehydrated_file, logger=logger)

            output_files[split] = {
                'hydrated': {'path': hydrated_file, 'size_mb': hydrated_size},
                'dehydrated': {'path': dehydrated_file, 'size_mb': dehydrated_size}
            }

            # Uncompressed test
            if split == 'test':
                uncompressed_file = os.path.join(PATHS['data'], 'test_hydrated.json')
                with open(uncompressed_file, 'w') as f:
                    json.dump(dataset, f, indent=2)
                uncompressed_size = os.path.getsize(uncompressed_file) / (1024 * 1024)
                logger.info(f"  ✅ {uncompressed_file} ({uncompressed_size:.1f} MB)")
                output_files[split]['uncompressed'] = {'path': uncompressed_file, 'size_mb': uncompressed_size}

        # Save stats
        overall_comments = sum(
            len(p['violating_thread']) + len(p['compliant_thread'])
            for subs in [test_subreddits, val_subreddits, train_subreddits]
            for sub in subs for p in sub['thread_pairs']
        )

        overall_thread_pairs = sum(
            sub['total_thread_pairs']
            for subs in [test_subreddits, val_subreddits, train_subreddits]
            for sub in subs
        )

        summary_stats = {
            'metadata': {
                'stage': 8,
                'stage_name': 'Create Final Datasets',
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time_seconds': time.time() - start_time
            },
            'dataset_statistics': {
                'test_subreddits': len(test_subreddits),
                'val_subreddits': len(val_subreddits),
                'train_subreddits': len(train_subreddits),
                'test_thread_pairs': test_pairs,
                'val_thread_pairs': val_pairs,
                'train_thread_pairs': train_pairs,
                'split_strategy': 'Adaptive: n=1→(1,0,0), n=2→(1,0,1), 3≤n<10→(1,1,n-2), n≥10→(10%,10%,80%)',
                'total_comments_overall': overall_comments,
                'total_thread_pairs_overall': overall_thread_pairs
            },
            'output_files': output_files
        }

        stats_file = os.path.join(PATHS['data'], 'stage8_final_datasets_stats.json')
        write_json_file(summary_stats, stats_file, pretty=True)

        elapsed = time.time() - start_time
        logger.info(f"🎉 Stage 8 Complete! ({elapsed:.1f}s)")
        log_stage_end(logger, 8, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 8 execution")
        log_stage_end(logger, 8, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
