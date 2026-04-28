"""
Configuration Module for Reddit Moderation Evaluation

This module contains all configuration settings for evaluating VLMs on Reddit
moderation tasks, including model configurations, context types, phrases,
and path utilities.
"""

from pathlib import Path
from typing import Dict, Any, List

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "eval"
LOGS_DIR = PROJECT_ROOT / "logs" / "eval"

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASET_FILES = {
    'train': DATA_DIR / 'train_hydrated_clustered.json.zst',
    'val': DATA_DIR / 'val_hydrated_clustered.json.zst',
    'test': DATA_DIR / 'test_hydrated_clustered.json',
    'delta': DATA_DIR / 'delta_hydrated_clustered.json.zst',
}

def get_dataset_path(split: str) -> Path:
    """Get dataset path for given split."""
    if split not in DATASET_FILES:
        raise ValueError(f"Invalid split: {split}. Must be one of {list(DATASET_FILES.keys())}")

    path = DATASET_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return path

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# vLLM Model Configurations
VLLM_MODELS = {
    'qwen3-vl-4b-instruct': {
        'hf_path': 'Qwen/Qwen3-VL-4B-Instruct',
        'gpu_memory_utilization': 0.95
    },
    'qwen3-vl-8b-instruct': {
        'hf_path': 'Qwen/Qwen3-VL-8B-Instruct',
        'gpu_memory_utilization': 0.95
    },
    'qwen3-vl-30b-instruct': {
        'hf_path': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'gpu_memory_utilization': 0.95
    },
    'qwen3-vl-4b-thinking': {
        'hf_path': 'Qwen/Qwen3-VL-4B-Thinking',
        'gpu_memory_utilization': 0.95
    },
    'qwen3-vl-8b-thinking': {
        'hf_path': 'Qwen/Qwen3-VL-8B-Thinking',
        'gpu_memory_utilization': 0.95
    },
    'qwen3-vl-30b-thinking': {
        'hf_path': 'Qwen/Qwen3-VL-30B-A3B-Thinking',
        'gpu_memory_utilization': 0.95
    }
}

# API Model Configurations (uses Flex API for OpenAI models)
API_MODELS = {
    'claude-sonnet-4': {
        'api_type': 'anthropic',
        'model_id': 'claude-sonnet-4-20250514',
        'max_tokens': 4096
    },
    'gpt-4o': {
        'api_type': 'openai',
        'model_id': 'gpt-4o',
        'max_tokens': 4096,
        'stage2_model': 'qwen3-vl-30b-instruct'
    },
    'gpt5.2-high': {
        'api_type': 'openai',
        'model_id': 'gpt-5.2',
        'max_tokens': 4096,
        'reasoning_effort': 'high',
        'stage2_model': 'qwen3-vl-30b-instruct'
    },
    'gpt5.2-low': {
        'api_type': 'openai',
        'model_id': 'gpt-5.2',
        'max_tokens': 4096,
        'reasoning_effort': 'low',
        'stage2_model': 'qwen3-vl-30b-instruct'
    }
}

# OpenAI Flex API Configuration
OPENAI_FLEX_CONFIG = {
    'timeout_seconds': 900,       # 15 minutes (Flex can be slow)
    'max_retries': 5,             # Retry on 429 Resource Unavailable
    'base_retry_delay': 30,       # Base delay for exponential backoff (seconds)
    'max_workers': 50,            # ThreadPoolExecutor workers (I/O-bound, not CPU-bound)
    'checkpoint_interval': 100,   # Save progress every N completions
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a model."""
    if model_name in VLLM_MODELS:
        return {'type': 'vllm', **VLLM_MODELS[model_name]}
    elif model_name in API_MODELS:
        return {'type': 'api', **API_MODELS[model_name]}
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_supported_models() -> List[str]:
    """Get list of all supported models."""
    return list(VLLM_MODELS.keys()) + list(API_MODELS.keys())

def is_vllm_model(model_name: str) -> bool:
    """Check if model uses vLLM."""
    return model_name in VLLM_MODELS

def is_api_model(model_name: str) -> bool:
    """Check if model uses API."""
    return model_name in API_MODELS

# =============================================================================
# CONTEXT CONFIGURATIONS
# =============================================================================

# Context flags that can be combined with dashes
# Format: "flag1-flag2-flag3" e.g., "submission-media-discussion-user"
#
# Available flags:
# - none: Just subreddit + rules + leaf comment (baseline)
# - submission: Include full submission (with media placeholder)
# - media: Include actual media (requires 'submission' flag)
# - discussion: Show all comments in thread (not just leaf)
# - user: Show anonymized author labels (USER1, USER2, etc.) for submissions and comments
#
# Note: Subreddit information is always included in prompts
#
# Examples:
# - "none" → subreddit + rules + leaf comment only
# - "submission" → subreddit + rules + leaf + submission (with placeholder)
# - "submission-media" → subreddit + rules + leaf + submission with images
# - "submission-discussion" → subreddit + rules + leaf + submission + all comments
# - "submission-media-discussion-user" → everything

VALID_CONTEXT_FLAGS = {
    'none', 'submission', 'media', 'discussion', 'user'
}

def parse_context_flags(context_string: str) -> Dict[str, bool]:
    """
    Parse dash-separated context string into flag dictionary.

    Args:
        context_string: Dash-separated flags like "submission-media-discussion"

    Returns:
        Dictionary with boolean flags for each context option

    Raises:
        ValueError: If invalid flags are provided
    """
    # Handle special case of "none"
    if context_string == 'none':
        return {
            'include_submission': False,
            'include_media': False,
            'include_discussion': False,
            'include_user': False
        }

    # Parse flags
    flags = context_string.split('-')

    # Validate flags
    invalid_flags = [f for f in flags if f not in VALID_CONTEXT_FLAGS and f != 'none']
    if invalid_flags:
        raise ValueError(f"Invalid context flags: {invalid_flags}. Valid flags: {VALID_CONTEXT_FLAGS}")

    # Validate dependencies
    if 'media' in flags and 'submission' not in flags:
        raise ValueError("'media' flag requires 'submission' flag to be set")

    # Build config dictionary
    return {
        'include_submission': 'submission' in flags,
        'include_media': 'media' in flags,
        'include_discussion': 'discussion' in flags,
        'include_user': 'user' in flags
    }

def get_supported_contexts() -> List[str]:
    """Get list of example context strings."""
    return [
        'none',
        'submission',
        'submission-media',
        'submission-discussion',
        'submission-discussion-user',
        'submission-media-discussion-user'
    ]

# =============================================================================
# PHRASE CONFIGURATIONS
# =============================================================================

PHRASES = {
    'baseline': '',
    'cot': 'Let\'s think step by step',
    'analyze': 'Let\'s carefully analyze this content',
    'artifacts': 'Let\'s look for rule violations',
    'rules': 'Let\'s compare this against the subreddit rules'
}

# Phrase modes (how to inject the phrase)
PHRASE_MODES = ['prefill', 'prompt']

# Answer extraction phrase for Stage 2
ANSWER_PHRASE = "\n\nFinal Choice:"

def get_supported_phrases() -> List[str]:
    """Get list of all supported phrases."""
    return list(PHRASES.keys())

# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_dir(base_dir: Path, split: str, model: str, context: str, phrase: str, mode: str) -> Path:
    """
    Get directory path for results or logs.

    Args:
        base_dir: Base directory (OUTPUT_DIR or LOGS_DIR)
        split: Dataset split
        model: Model name
        context: Context type
        phrase: Phrase name
        mode: Phrase mode

    Returns:
        Directory path (created if doesn't exist)

    Example:
        >>> get_dir(OUTPUT_DIR, 'test', 'qwen3-vl-8b', 'none', 'baseline', 'prefill')
        PosixPath('.../output/eval/qwen3-vl-8b/test/none/baseline')
    """
    # For baseline phrase, all modes are equivalent, so use just 'baseline'
    dir_name = 'baseline' if phrase == 'baseline' else f'{phrase}_{mode}'

    dir_path = base_dir / model / split / context / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config_combination(model: str, split: str, context: str, phrase: str, mode: str) -> None:
    """
    Validate that a configuration combination is valid.

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate model
    if model not in get_supported_models():
        raise ValueError(f"Invalid model: {model}")

    # Validate split
    if split not in DATASET_FILES:
        raise ValueError(f"Invalid split: {split}")

    # Validate context (parse to check validity)
    parse_context_flags(context)

    # Validate phrase
    if phrase not in PHRASES:
        raise ValueError(f"Invalid phrase: {phrase}. Must be one of {list(PHRASES.keys())}")

    # Validate mode
    if mode not in PHRASE_MODES:
        raise ValueError(f"Invalid mode: {mode}")

    # Check dataset file exists
    dataset_path = get_dataset_path(split)
    if not dataset_path.exists():
        raise ValueError(f"Dataset file not found: {dataset_path}")
