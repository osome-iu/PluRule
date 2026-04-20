#!/usr/bin/env python3
"""
Stage 11a: Human Evaluation from Final Dataset

Creates Google Forms for human evaluation of rule matching quality using the
full Stage 10 clustered dataset (train+val+test). Samples 100 moderator comments
from 100 unique subreddits, stratified uniformly across rule clusters (best effort
on subreddit clusters).

Each form question shows:
- Subreddit name, title, description
- Subreddit cluster label
- All community rules
- Moderator comment (body_clean)
- MCQ with rule short names + Other field

Usage: python 11_human_evaluation.py
"""

import os
import sys
import json
import random
import hashlib
import time
from typing import List, Dict, Any
from collections import defaultdict, Counter

# Google Forms API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS
from utils.files import write_json_file, read_compressed_json
from utils.logging import get_stage_logger, log_stage_start, log_stage_end

# ============================================================================
# Configuration
# ============================================================================

TOTAL_SAMPLES = 100
QUESTIONS_PER_FORM = 50  # Split into 2 forms of 50 questions each
RANDOM_SEED = 42

# Google Forms API Configuration
SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/drive.file",
]
CLIENT_SECRETS = "/data3/zkachwal/reddit-mod-collection-pipeline/credentials/client_secret_795576073496-qo2r4ntgn1drrqo31p98it9bmtd2hvm4.apps.googleusercontent.com.json"
TOKEN_FILE = "/data3/zkachwal/reddit-mod-collection-pipeline/credentials/token.json"


# ============================================================================
# Helper Functions
# ============================================================================

def stable_hash(value: str) -> int:
    """Create deterministic integer hash from string (reproducible across runs)."""
    return int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest(), 'big')


def authenticate():
    """Authenticate with Google APIs using OAuth2."""
    creds = None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    except Exception:
        pass

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save for next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return creds


# ============================================================================
# Data Loading
# ============================================================================

def load_full_dataset(logger) -> Dict[str, Any]:
    """Load and merge all clustered splits (train/val/test) from Stage 10."""
    all_subreddits = []

    for split in ['train', 'val', 'test']:
        compressed_file = os.path.join(PATHS['data'], f'{split}_hydrated_clustered.json.zst')
        if not os.path.exists(compressed_file):
            logger.warning(f"{split} dataset not found: {compressed_file}, skipping...")
            continue

        logger.info(f"Loading {split} dataset from: {compressed_file}")
        dataset = read_compressed_json(compressed_file, logger)
        split_subreddits = dataset.get('subreddits', [])
        logger.info(f"  {split}: {len(split_subreddits)} subreddits")
        all_subreddits.extend(split_subreddits)

    if not all_subreddits:
        logger.error("No datasets found!")
        return None

    logger.info(f"Total subreddits across all splits: {len(all_subreddits)}")
    return {'subreddits': all_subreddits}


# ============================================================================
# Stratified Sampling
# ============================================================================

def build_sampling_pool(dataset: Dict) -> Dict[str, List[Dict]]:
    """Build a pool of candidate samples organized by rule cluster.

    Returns:
        Dict mapping rule_cluster_label -> list of candidate dicts
    """
    pool = defaultdict(list)

    for sub_data in dataset['subreddits']:
        # Only include English subreddits
        language = sub_data.get('language', 'unknown')
        lang_root = language.replace('_', '-').split('-')[0].lower()
        if lang_root != 'en':
            continue

        subreddit = sub_data['subreddit']
        subreddit_cluster = sub_data.get('subreddit_cluster_label', 'Other')
        title = sub_data.get('title', '')
        description = sub_data.get('description', '')
        rules = sub_data.get('rules', [])

        for pair_idx, pair in enumerate(sub_data.get('thread_pairs', [])):
            metadata = pair.get('metadata', {})
            rule_cluster = metadata.get('rule_cluster_label', 'Other')
            matched_rule = metadata.get('rule', '')

            # Get mod comment body_clean
            mod_comment = pair.get('mod_comment', {})
            body_clean = mod_comment.get('body_clean', mod_comment.get('body', ''))

            # Get answer options (excluding "No rules broken")
            answer_options = pair.get('violating_answer_options', [])
            rule_options = [opt['rule'] for opt in answer_options if opt['rule'] != 'No rules broken']

            candidate = {
                'subreddit': subreddit,
                'subreddit_cluster_label': subreddit_cluster,
                'title': title,
                'description': description,
                'rules': rules,
                'mod_comment_id': pair.get('mod_comment_id'),
                'body_clean': body_clean,
                'matched_rule': matched_rule,
                'rule_cluster_label': rule_cluster,
                'submission_id': metadata.get('submission_id'),
                'answer_options': rule_options,
                'pair_idx': pair_idx
            }

            pool[rule_cluster].append(candidate)

    return pool


def stratified_sample(pool: Dict[str, List[Dict]], total_samples: int,
                      seed: int, logger) -> List[Dict]:
    """Sample uniformly across rule clusters, ensuring unique subreddits.

    Strategy:
    1. Calculate target samples per rule cluster (uniform)
    2. For each cluster, randomly select candidates
    3. Ensure no subreddit is used twice
    4. Best effort on subreddit cluster distribution

    Returns:
        List of sampled candidate dicts
    """
    rng = random.Random(seed)

    # Get all rule clusters (excluding 'Other' for stratification, add back if needed)
    rule_clusters = sorted([c for c in pool.keys() if c != 'Other'])
    num_clusters = len(rule_clusters)

    logger.info(f"Rule clusters (excluding Other): {num_clusters}")

    # Calculate target per cluster
    base_per_cluster = total_samples // num_clusters
    remainder = total_samples % num_clusters

    # Assign targets (some clusters get +1 to use all samples)
    cluster_targets = {}
    shuffled_clusters = rule_clusters[:]
    rng.shuffle(shuffled_clusters)

    for i, cluster in enumerate(shuffled_clusters):
        cluster_targets[cluster] = base_per_cluster + (1 if i < remainder else 0)

    logger.info(f"Target per cluster: ~{base_per_cluster} (total: {total_samples})")

    # Track used subreddits
    used_subreddits = set()
    sampled = []
    cluster_actual = Counter()
    subreddit_cluster_counts = Counter()

    # First pass: sample from each rule cluster
    for cluster in rule_clusters:
        target = cluster_targets[cluster]
        candidates = pool[cluster][:]
        rng.shuffle(candidates)

        count = 0
        for candidate in candidates:
            if candidate['subreddit'] in used_subreddits:
                continue

            sampled.append(candidate)
            used_subreddits.add(candidate['subreddit'])
            cluster_actual[cluster] += 1
            subreddit_cluster_counts[candidate['subreddit_cluster_label']] += 1
            count += 1

            if count >= target:
                break

        if count < target:
            logger.warning(f"  {cluster}: only got {count}/{target} (not enough unique subreddits)")

    # If we didn't reach total_samples, try to fill from 'Other' or undersampled clusters
    if len(sampled) < total_samples:
        shortfall = total_samples - len(sampled)
        logger.info(f"Filling {shortfall} remaining samples from Other cluster...")

        other_candidates = pool.get('Other', [])[:]
        rng.shuffle(other_candidates)

        for candidate in other_candidates:
            if candidate['subreddit'] in used_subreddits:
                continue

            sampled.append(candidate)
            used_subreddits.add(candidate['subreddit'])
            cluster_actual['Other'] += 1
            subreddit_cluster_counts[candidate['subreddit_cluster_label']] += 1

            if len(sampled) >= total_samples:
                break

    # Log distribution
    logger.info(f"\nFinal sample: {len(sampled)} from {len(used_subreddits)} unique subreddits")
    logger.info(f"\nRule cluster distribution:")
    for cluster, count in sorted(cluster_actual.items(), key=lambda x: -x[1]):
        logger.info(f"  {cluster}: {count}")

    logger.info(f"\nSubreddit cluster distribution (best effort):")
    for cluster, count in sorted(subreddit_cluster_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cluster}: {count}")

    # Shuffle final order
    rng.shuffle(sampled)

    return sampled


# ============================================================================
# Google Form Creation
# ============================================================================

def create_rules_display(rules: List[Dict]) -> str:
    """Format rules for display in form."""
    lines = []
    for rule in rules:
        idx = rule.get('rule_index', 0)
        short_name = rule.get('short_name_clean', rule.get('short_name', ''))
        desc = rule.get('description_clean', rule.get('description', ''))

        if desc:
            lines.append(f"Rule {idx}: {short_name}\n   {desc}")
        else:
            lines.append(f"Rule {idx}: {short_name}")

    return "\n\n".join(lines)


def create_evaluation_form(service, samples: List[Dict], form_part: int,
                           total_forms: int, logger) -> str:
    """Create a Google Form with one question per page."""

    # Create the basic form
    import time as _time
    date_str = _time.strftime('%Y-%m-%d')
    if total_forms > 1:
        title = f"Part {form_part}/{total_forms} {date_str} - Reddit Moderation - Human Evaluation"
    else:
        title = f"{date_str} - Reddit Moderation - Human Evaluation"

    form = {"info": {"title": title}}
    result = service.forms().create(body=form).execute()
    form_id = result["formId"]
    logger.info(f"Created form with ID: {form_id}")

    # Build requests
    requests = [
        {
            "updateFormInfo": {
                "info": {
                    "title": title,
                    "description": (
                        f"Please evaluate {len(samples)} moderator comments.\n\n"
                        f"For each comment, select the rule that the moderator is citing. "
                        f"Use 'Other' if none of the rules apply or to add notes.\n\n"
                        f"Each page shows one subreddit with its rules and a moderator comment to evaluate."
                    )
                },
                "updateMask": "title,description"
            }
        }
    ]

    question_requests = []

    for idx, sample in enumerate(samples):
        subreddit = sample['subreddit']
        title = sample.get('title', '')
        description = sample.get('description', '')
        subreddit_cluster = sample['subreddit_cluster_label']
        rules = sample['rules']
        body_clean = sample['body_clean']
        answer_options = sample['answer_options']

        # Build page content
        rules_display = create_rules_display(rules)

        page_title = f"Question {idx + 1}/{len(samples)}: r/{subreddit}"

        page_description = f"""r/{subreddit} - {title}
{description}

Subreddit Cluster: {subreddit_cluster}

{'='*50}
COMMUNITY RULES
{'='*50}

{rules_display}

{'='*50}
MODERATOR COMMENT
{'='*50}

"{body_clean}"
"""

        # Create choice options from rule short names
        choice_options = [{"value": rule} for rule in answer_options]

        # Add "Other" option for notes
        choice_options.append({"isOther": True})

        # Add page break (except for first question)
        if idx > 0:
            page_break = {
                "createItem": {
                    "item": {
                        "title": page_title,
                        "pageBreakItem": {}
                    },
                    "location": {"index": len(question_requests)}
                }
            }
            question_requests.append(page_break)

        # Add the question
        question_request = {
            "createItem": {
                "item": {
                    "title": "Which rule is this comment referring to?",
                    "description": page_description if idx == 0 else page_description,
                    "questionItem": {
                        "question": {
                            "required": True,
                            "choiceQuestion": {
                                "type": "CHECKBOX",
                                "options": choice_options
                            }
                        }
                    }
                },
                "location": {"index": len(question_requests)}
            }
        }
        question_requests.append(question_request)

    # Add all requests
    requests.extend(question_requests)

    # Execute batch update
    body = {"requests": requests}
    service.forms().batchUpdate(formId=form_id, body=body).execute()

    logger.info(f"Added {len(samples)} questions to form")

    return form_id


# ============================================================================
# Metadata Saving
# ============================================================================

def save_evaluation_metadata(samples: List[Dict], form_data: List[Dict],
                             output_dir: str, logger) -> str:
    """Save evaluation metadata for later analysis."""

    # Build question metadata
    questions = []
    for idx, sample in enumerate(samples):
        questions.append({
            'question_index': idx + 1,
            'mod_comment_id': sample['mod_comment_id'],
            'subreddit': sample['subreddit'],
            'submission_id': sample['submission_id'],
            'predicted_answer': sample['matched_rule'],
            'rule_cluster_label': sample['rule_cluster_label'],
            'subreddit_cluster_label': sample['subreddit_cluster_label'],
            'answer_options': sample['answer_options']
        })

    # Compute distribution stats
    rule_cluster_dist = Counter(s['rule_cluster_label'] for s in samples)
    subreddit_cluster_dist = Counter(s['subreddit_cluster_label'] for s in samples)

    metadata = {
        'metadata': {
            'stage': 11,
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(samples),
            'unique_subreddits': len(set(s['subreddit'] for s in samples)),
            'random_seed': RANDOM_SEED,
            'sampling_strategy': 'Uniform across rule clusters, best effort on subreddit clusters'
        },
        'forms': form_data,
        'distributions': {
            'rule_clusters': dict(rule_cluster_dist),
            'subreddit_clusters': dict(subreddit_cluster_dist)
        },
        'questions': questions
    }

    # Save metadata
    os.makedirs(output_dir, exist_ok=True)
    metadata_file = os.path.join(output_dir, 'stage11_human_evaluation_metadata.json')
    write_json_file(metadata, metadata_file, pretty=True)
    logger.info(f"Saved metadata to: {metadata_file}")

    return metadata_file


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    logger = get_stage_logger(11, "human_evaluation")
    log_stage_start(logger, 11, "Human Evaluation Form Creation")

    start_time = time.time()

    try:
        print("=" * 60)
        print("Stage 11a: Human Evaluation Form Creation")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Total samples: {TOTAL_SAMPLES}")
        print(f"  Questions per form: {QUESTIONS_PER_FORM}")
        print(f"  Random seed: {RANDOM_SEED}")

        # Load full dataset (train+val+test)
        print(f"\n{'='*60}")
        print("LOADING DATA")
        print("=" * 60)
        dataset = load_full_dataset(logger)
        if not dataset:
            logger.error("Failed to load datasets!")
            return 1

        # Build sampling pool
        print(f"\n{'='*60}")
        print("BUILDING SAMPLING POOL")
        print("=" * 60)
        pool = build_sampling_pool(dataset)

        total_candidates = sum(len(v) for v in pool.values())
        print(f"Total candidates: {total_candidates}")
        print(f"Rule clusters: {len(pool)}")

        # Stratified sampling
        print(f"\n{'='*60}")
        print("STRATIFIED SAMPLING")
        print("=" * 60)
        samples = stratified_sample(pool, TOTAL_SAMPLES, RANDOM_SEED, logger)

        if len(samples) < TOTAL_SAMPLES:
            logger.warning(f"Only sampled {len(samples)}/{TOTAL_SAMPLES}")

        # Split into forms
        form_chunks = []
        for i in range(0, len(samples), QUESTIONS_PER_FORM):
            form_chunks.append(samples[i:i + QUESTIONS_PER_FORM])

        print(f"\n{'='*60}")
        print("CREATING GOOGLE FORMS")
        print("=" * 60)
        print(f"Creating {len(form_chunks)} form(s)...")

        # Authenticate
        print("\nAuthenticating with Google APIs...")
        creds = authenticate()
        service = build('forms', 'v1', credentials=creds)
        print("Authenticated successfully")

        # Create forms
        form_data = []
        for form_idx, chunk in enumerate(form_chunks):
            form_part = form_idx + 1
            total_forms = len(form_chunks)

            print(f"\nCreating form {form_part}/{total_forms} ({len(chunk)} questions)...")
            form_id = create_evaluation_form(service, chunk, form_part, total_forms, logger)

            form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
            public_url = f"https://docs.google.com/forms/d/{form_id}/viewform"

            form_data.append({
                'form_id': form_id,
                'form_url': form_url,
                'public_url': public_url,
                'form_part': form_part,
                'num_questions': len(chunk),
                'question_range': f"{form_idx * QUESTIONS_PER_FORM + 1}-{form_idx * QUESTIONS_PER_FORM + len(chunk)}"
            })

        # Save metadata
        print(f"\n{'='*60}")
        print("SAVING METADATA")
        print("=" * 60)
        output_dir = os.path.join(PATHS['data'], 'evaluation')
        metadata_file = save_evaluation_metadata(samples, form_data, output_dir, logger)

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(samples)}")
        print(f"Unique subreddits: {len(set(s['subreddit'] for s in samples))}")
        print(f"Forms created: {len(form_data)}")
        print(f"\nGoogle Forms URLs:")
        for fd in form_data:
            print(f"  Part {fd['form_part']}: {fd['public_url']}")
        print(f"\nMetadata saved to: {metadata_file}")

        elapsed = time.time() - start_time
        print(f"\nStage 11 completed in {elapsed:.1f}s")
        log_stage_end(logger, 11, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        logger.error(f"Stage 11 failed: {e}")
        import traceback
        traceback.print_exc()
        log_stage_end(logger, 11, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
