#!/usr/bin/env python3
"""
Stage 11c: Evaluate Model Predictions Against Human Annotations

Computes accuracy metrics by comparing predicted_answer against:
- majority_answers (for questions with ≥2 votes)
- adjudicated_answer (for questions manually adjudicated)

Usage: python 11c_evaluate_predictions.py
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from collections import defaultdict, Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS
from utils.files import write_json_file
from utils.logging import get_stage_logger, log_stage_start, log_stage_end


# ============================================================================
# Load Data
# ============================================================================

def load_annotations(logger) -> Dict[str, Any]:
    """Load human annotations with majority voting and adjudication."""
    annotations_file = os.path.join(PATHS['data'], 'evaluation', 'stage11_human_annotations.json')

    if not os.path.exists(annotations_file):
        logger.error(f"Annotations file not found: {annotations_file}")
        return None

    logger.info(f"Loading annotations from: {annotations_file}")

    with open(annotations_file, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data['questions'])} questions")
    return data


# ============================================================================
# Build Ground Truth
# ============================================================================

def build_ground_truth(data: Dict, logger) -> List[Dict]:
    """Build ground truth labels from majority_answers and adjudicated_answer.

    Returns:
        List of dicts with question metadata and ground_truth_answers
    """
    ground_truth = []

    for question in data['questions']:
        question_data = {
            'question_index': question['question_index'],
            'subreddit': question['subreddit'],
            'predicted_answer': question['predicted_answer'],
            'rule_cluster_label': question['rule_cluster_label'],
            'subreddit_cluster_label': question['subreddit_cluster_label'],
        }

        # Use majority_answers if available, otherwise use adjudicated_answers
        if question['majority_answers']:
            question_data['ground_truth_answers'] = question['majority_answers']
            question_data['ground_truth_source'] = 'majority'
            # Get vote count for the first majority answer
            question_data['agreement_level'] = question.get('answer_vote_counts', {}).get(question['majority_answers'][0], 0)
        elif 'adjudicated_answers' in question and question['adjudicated_answers']:
            # adjudicated_answers is a list
            question_data['ground_truth_answers'] = question['adjudicated_answers']
            question_data['ground_truth_source'] = 'adjudicated'
            question_data['agreement_level'] = 0  # No agreement
        else:
            # No ground truth available
            question_data['ground_truth_answers'] = []
            question_data['ground_truth_source'] = 'none'
            question_data['agreement_level'] = 0
            logger.warning(f"Question {question['question_index']} has no ground truth!")

        ground_truth.append(question_data)

    return ground_truth


# ============================================================================
# Compute Metrics
# ============================================================================

def evaluate_predictions(ground_truth: List[Dict], logger) -> Dict[str, Any]:
    """Evaluate model predictions against ground truth.

    A prediction is correct if it matches ANY of the ground truth answers.
    """

    total = len(ground_truth)
    correct = 0
    by_source = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_rule_cluster = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_subreddit_cluster = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_agreement_level = defaultdict(lambda: {'total': 0, 'correct': 0})

    errors = []

    for item in ground_truth:
        predicted = item['predicted_answer']
        ground_truth_answers = item['ground_truth_answers']
        source = item['ground_truth_source']
        rule_cluster = item['rule_cluster_label']
        subreddit_cluster = item['subreddit_cluster_label']
        agreement = item['agreement_level']

        # Skip if no ground truth
        if not ground_truth_answers:
            continue

        # Check if prediction matches any ground truth answer
        is_correct = predicted in ground_truth_answers

        if is_correct:
            correct += 1
            by_source[source]['correct'] += 1
            by_rule_cluster[rule_cluster]['correct'] += 1
            by_subreddit_cluster[subreddit_cluster]['correct'] += 1
            by_agreement_level[agreement]['correct'] += 1
        else:
            errors.append({
                'question_index': item['question_index'],
                'subreddit': item['subreddit'],
                'predicted': predicted,
                'ground_truth': ground_truth_answers,
                'source': source,
                'rule_cluster': rule_cluster
            })

        by_source[source]['total'] += 1
        by_rule_cluster[rule_cluster]['total'] += 1
        by_subreddit_cluster[subreddit_cluster]['total'] += 1
        by_agreement_level[agreement]['total'] += 1

    # Calculate percentages
    accuracy = correct / total if total > 0 else 0

    by_source_pct = {
        k: {**v, 'accuracy': v['correct'] / v['total'] if v['total'] > 0 else 0}
        for k, v in by_source.items()
    }

    by_rule_cluster_pct = {
        k: {**v, 'accuracy': v['correct'] / v['total'] if v['total'] > 0 else 0}
        for k, v in by_rule_cluster.items()
    }

    by_subreddit_cluster_pct = {
        k: {**v, 'accuracy': v['correct'] / v['total'] if v['total'] > 0 else 0}
        for k, v in by_subreddit_cluster.items()
    }

    by_agreement_level_pct = {
        str(k): {**v, 'accuracy': v['correct'] / v['total'] if v['total'] > 0 else 0}
        for k, v in by_agreement_level.items()
    }

    results = {
        'overall': {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        },
        'by_source': dict(by_source_pct),
        'by_rule_cluster': dict(by_rule_cluster_pct),
        'by_subreddit_cluster': dict(by_subreddit_cluster_pct),
        'by_agreement_level': dict(by_agreement_level_pct),
        'errors': errors
    }

    return results


# ============================================================================
# Save Results
# ============================================================================

def save_evaluation_results(results: Dict, output_dir: str, logger) -> str:
    """Save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'stage11_evaluation_results.json')
    write_json_file(results, output_file, pretty=True)
    logger.info(f"Saved evaluation results to: {output_file}")

    return output_file


def print_results(results: Dict):
    """Print evaluation results to console."""
    print("\n" + "=" * 60)
    print("OVERALL ACCURACY")
    print("=" * 60)
    print(f"Total questions: {results['overall']['total']}")
    print(f"Correct predictions: {results['overall']['correct']}")
    print(f"Accuracy: {results['overall']['accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("ACCURACY BY GROUND TRUTH SOURCE")
    print("=" * 60)
    for source, metrics in sorted(results['by_source'].items()):
        print(f"{source}:")
        print(f"  Total: {metrics['total']}")
        print(f"  Correct: {metrics['correct']}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("ACCURACY BY AGREEMENT LEVEL")
    print("=" * 60)
    for level, metrics in sorted(results['by_agreement_level'].items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0, reverse=True):
        level_int = int(level) if level.isdigit() else 0
        if level_int > 0:
            print(f"{level}/3 annotators agreed:")
            print(f"  Total: {metrics['total']}")
            print(f"  Correct: {metrics['correct']}")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("ACCURACY BY RULE CLUSTER (Top 10)")
    print("=" * 60)
    sorted_clusters = sorted(
        results['by_rule_cluster'].items(),
        key=lambda x: x[1]['total'],
        reverse=True
    )[:10]
    for cluster, metrics in sorted_clusters:
        print(f"{cluster}:")
        print(f"  Total: {metrics['total']}, Correct: {metrics['correct']}, Accuracy: {metrics['accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("ERRORS (First 10)")
    print("=" * 60)
    for error in results['errors'][:10]:
        print(f"\nQ{error['question_index']} - r/{error['subreddit']} ({error['rule_cluster']})")
        print(f"  Predicted: {error['predicted']}")
        print(f"  Ground truth: {error['ground_truth']}")
        print(f"  Source: {error['source']}")

    if len(results['errors']) > 10:
        print(f"\n... and {len(results['errors']) - 10} more errors")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    logger = get_stage_logger(11, "evaluate_predictions")
    log_stage_start(logger, 11, "Evaluate Model Predictions")

    start_time = time.time()

    try:
        print("=" * 60)
        print("Stage 11c: Evaluate Model Predictions")
        print("=" * 60)

        # Load annotations
        print(f"\n{'='*60}")
        print("LOADING ANNOTATIONS")
        print("=" * 60)
        data = load_annotations(logger)
        if not data:
            logger.error("Failed to load annotations!")
            return 1

        # Build ground truth
        print(f"\n{'='*60}")
        print("BUILDING GROUND TRUTH")
        print("=" * 60)
        ground_truth = build_ground_truth(data, logger)

        majority_count = sum(1 for item in ground_truth if item['ground_truth_source'] == 'majority')
        adjudicated_count = sum(1 for item in ground_truth if item['ground_truth_source'] == 'adjudicated')
        print(f"Questions with majority vote: {majority_count}")
        print(f"Questions with adjudicated answer: {adjudicated_count}")

        # Evaluate
        print(f"\n{'='*60}")
        print("EVALUATING PREDICTIONS")
        print("=" * 60)
        results = evaluate_predictions(ground_truth, logger)

        # Save
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print("=" * 60)
        output_dir = os.path.join(PATHS['data'], 'evaluation')
        output_file = save_evaluation_results(results, output_dir, logger)

        # Print summary
        print_results(results)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print("=" * 60)
        print(f"Results saved to: {output_file}")

        elapsed = time.time() - start_time
        print(f"\nStage 11c completed in {elapsed:.1f}s")
        log_stage_end(logger, 11, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        logger.error(f"Stage 11c failed: {e}")
        import traceback
        traceback.print_exc()
        log_stage_end(logger, 11, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
