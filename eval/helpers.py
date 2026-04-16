"""
Helper Functions Module for Reddit Moderation Evaluation

This module contains all utility functions for the evaluation system, including:
- Data loading from clustered datasets
- Prompt building based on context types
- Two-stage model inference (reasoning + answer extraction)
- Answer extraction and validation
- Metrics calculation (accuracy, per-cluster stats)
- Result saving
"""

import json
import logging
import base64
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from PIL import Image

import config

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _normalize_language(lang_code: str) -> str:
    """
    Normalize language code by taking root (e.g., en-au → en, pt_BR → pt).

    Args:
        lang_code: Language code (e.g., 'en-au', 'pt_BR', 'en')

    Returns:
        Normalized language code (root only)
    """
    if not lang_code:
        return 'unknown'
    return lang_code.replace('_', '-').split('-')[0]


def _format_timestamp(created_utc) -> str:
    """Format Unix timestamp to human-readable date string."""
    # Handle both int and string timestamps
    if isinstance(created_utc, str):
        created_utc = int(created_utc)
    dt = datetime.fromtimestamp(created_utc)
    return dt.strftime("%a, %b %d, %Y, %I:%M%p").replace(" 0", " ")

def _clean_user_mentions(text: str) -> str:
    """
    Remove user mentions from text.
    Used for comment bodies.

    Removes patterns like:
    - u/username
    """
    import re

    # Remove user mentions: u/username -> ""
    text = re.sub(r'u/\w+', '', text)

    return text

# =============================================================================
# OPENAI BATCH API - IMAGE ENCODING
# =============================================================================

def _encode_image_base64(image_path: str) -> str:
    """
    Encode a local image file to base64 data URL for OpenAI API.

    Args:
        image_path: Path to local image file

    Returns:
        Base64 data URL string (e.g., "data:image/jpeg;base64,...")
    """
    path = Path(image_path)

    # Determine MIME type from extension
    extension = path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(extension, 'image/jpeg')

    # Read and encode
    with open(path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:{mime_type};base64,{image_data}"


def _build_openai_messages(pair: Dict[str, Any],
                           thread_type: str,
                           context_config: Dict[str, Any],
                           logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Build OpenAI-compatible messages with base64 images.

    Args:
        pair: Thread pair dictionary with prompts
        thread_type: 'violating' or 'compliant'
        context_config: Context configuration
        logger: Logger instance

    Returns:
        List of message dicts for OpenAI API
    """
    prompt_data = pair[f'{thread_type}_prompt']
    messages_in = prompt_data['messages']

    # Get media files from pair (for API models, image paths aren't in content dict)
    media_files = pair.get('submission', {}).get('media_files', [])
    image_idx = 0  # Track which image we're on

    messages_out = []
    for msg in messages_in:
        role = msg['role']
        content_in = msg['content']

        if isinstance(content_in, str):
            messages_out.append({"role": role, "content": content_in})
        elif isinstance(content_in, list):
            content_out = []
            for item in content_in:
                if item.get('type') == 'text':
                    # Responses API uses 'input_text' instead of 'text'
                    content_out.append({"type": "input_text", "text": item['text']})
                elif item.get('type') == 'image':
                    # Only encode images if media flag is set
                    if context_config.get('include_media', False):
                        # Try to get path from item (Qwen format) or from media_files list
                        image_path = item.get('image')
                        if not image_path and image_idx < len(media_files):
                            image_path = media_files[image_idx]
                            image_idx += 1

                        if image_path:
                            try:
                                base64_url = _encode_image_base64(image_path)
                                # Responses API uses 'input_image' with 'image_url' as direct string
                                content_out.append({
                                    "type": "input_image",
                                    "image_url": base64_url,
                                    "detail": "low"
                                })
                            except Exception as e:
                                logger.warning(f"Failed to encode image {image_path}: {e}")
            messages_out.append({"role": role, "content": content_out})

    return messages_out


# =============================================================================
# OPENAI FLEX API FUNCTIONS
# =============================================================================

def _call_openai_flex(
    messages: List[Dict],
    model_id: str,
    max_tokens: int,
    reasoning_effort: str = None,
    timeout: float = 900.0,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    Make a single OpenAI Flex API call using Responses API with reasoning.

    Handles:
    - 408 Request Timeout (auto-retry by SDK)
    - 429 Resource Unavailable (retry with exponential backoff)

    Args:
        messages: OpenAI-compatible messages list (input format)
        model_id: OpenAI model ID (e.g., 'gpt-5.2')
        max_tokens: Maximum output tokens for response
        reasoning_effort: Reasoning effort level ('low', 'medium', 'high') or None to skip reasoning
        timeout: Request timeout in seconds (default 15 min for Flex)
        max_retries: Maximum retry attempts for 429 errors

    Returns:
        Dict with 'content' (output text) and 'reasoning_summary' (reasoning summary)
    """
    from openai import OpenAI

    client = OpenAI(timeout=timeout)

    # Build request kwargs
    request_kwargs = {
        'model': model_id,
        'input': messages,
        'max_output_tokens': max_tokens,
        'service_tier': "flex"
    }
    if reasoning_effort:
        request_kwargs['reasoning'] = {"effort": reasoning_effort, "summary": "detailed"}

    for attempt in range(max_retries):
        try:
            response = client.responses.create(**request_kwargs)

            # Extract output text
            content = response.output_text or ''

            # Extract reasoning summary from output array
            reasoning_summary = ''
            for item in response.output:
                if item.type == 'reasoning' and hasattr(item, 'summary'):
                    # Concatenate all summary texts
                    summaries = [s.text for s in item.summary if hasattr(s, 'text')]
                    reasoning_summary = '\n'.join(summaries)
                    break

            return {
                'content': content,
                'reasoning_summary': reasoning_summary
            }
        except Exception as e:
            error_str = str(e)
            # Retry on 429 Resource Unavailable
            if "429" in error_str and attempt < max_retries - 1:
                delay = config.OPENAI_FLEX_CONFIG['base_retry_delay'] * (2 ** attempt)
                time.sleep(delay)
            else:
                raise

    return {'content': '', 'reasoning_summary': ''}


def _process_single_flex_request(args: Tuple) -> Tuple[str, Dict[str, str]]:
    """
    Worker function for ThreadPoolExecutor.

    Args:
        args: Tuple of (custom_id, messages, model_id, max_tokens, reasoning_effort)

    Returns:
        Tuple of (custom_id, response_dict) where response_dict has 'content' and 'reasoning_summary'
    """
    custom_id, messages, model_id, max_tokens, reasoning_effort = args
    response = _call_openai_flex(messages, model_id, max_tokens, reasoning_effort)
    return custom_id, response


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_dataset(split: str, logger: logging.Logger, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Load Reddit moderation dataset from clustered JSON file.

    Args:
        split: Dataset split ('train', 'val', 'test')
        logger: Logger instance
        debug: If True, load only first 5 thread pairs

    Returns:
        List of thread pair dictionaries with all necessary data
    """
    dataset_path = config.get_dataset_path(split)
    logger.info(f"📂 Loading dataset from {dataset_path}")

    # Load JSON (handle compressed and uncompressed)
    if dataset_path.suffix == '.zst':
        import zstandard as zstd
        with open(dataset_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                data = json.loads(reader.read())
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)

    # Extract thread pairs
    thread_pairs = []
    for subreddit_data in data['subreddits']:
        subreddit = subreddit_data['subreddit']
        subreddit_cluster_id = subreddit_data.get('subreddit_cluster_id', -1)
        subreddit_cluster_label = subreddit_data.get('subreddit_cluster_label', 'Other')
        subreddit_title = subreddit_data.get('title', '')
        subreddit_description = subreddit_data.get('description', '')
        subreddit_language = _normalize_language(subreddit_data.get('language', 'unknown'))

        for pair in subreddit_data['thread_pairs']:
            # Get submission data
            submission_id = pair['metadata']['submission_id']
            submission_data = subreddit_data['submissions'].get(submission_id, {})

            thread_pair = {
                'subreddit': subreddit,
                'subreddit_title': subreddit_title,
                'subreddit_description': subreddit_description,
                'subreddit_cluster_id': subreddit_cluster_id,
                'subreddit_cluster_label': subreddit_cluster_label,
                'subreddit_language': subreddit_language,
                'mod_comment_id': pair['mod_comment_id'],
                'submission_id': submission_id,
                'submission': submission_data,
                'rules': subreddit_data['rules'],
                'violating_thread': pair['violating_thread'],
                'compliant_thread': pair['compliant_thread'],
                'violating_answer_options': pair['violating_answer_options'],
                'violating_correct_answer': pair['violating_correct_answer'],
                'compliant_answer_options': pair['compliant_answer_options'],
                'compliant_correct_answer': pair['compliant_correct_answer'],
                'metadata': pair['metadata']
            }
            thread_pairs.append(thread_pair)

    # Debug mode: limit to first 5 pairs
    if debug:
        thread_pairs = thread_pairs[:5]
        logger.info(f"🐛 Debug mode: Using only {len(thread_pairs)} thread pairs")
    else:
        logger.info(f"✅ Loaded {len(thread_pairs)} thread pairs from {split} split")

    return thread_pairs

# =============================================================================
# PROMPT BUILDING FUNCTIONS
# =============================================================================

def build_prompts_for_thread_pairs(thread_pairs: List[Dict[str, Any]],
                                   context_type: str,
                                   phrase_name: str,
                                   model_name: str,
                                   mode: str,
                                   logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Build prompts for all thread pairs (both violating and compliant).

    Args:
        thread_pairs: List of thread pair dictionaries
        context_type: Context type (e.g., 'thread_with_rule')
        phrase_name: Phrase name (e.g., 'cot')
        model_name: Model name
        mode: Phrase mode ('prefill' or 'prompt')
        logger: Logger instance

    Returns:
        List of thread pairs with prompts added
    """
    context_config = config.parse_context_flags(context_type)
    phrase_text = config.PHRASES.get(phrase_name, '')
    model_config = config.get_model_config(model_name)

    processed_pairs = []
    for pair in thread_pairs:
        # Build prompts for both violating and compliant threads
        violating_prompt = _build_single_prompt(
            pair,
            thread_type='violating',
            context_config=context_config,
            phrase_text=phrase_text,
            model_config=model_config,
            mode=mode
        )

        compliant_prompt = _build_single_prompt(
            pair,
            thread_type='compliant',
            context_config=context_config,
            phrase_text=phrase_text,
            model_config=model_config,
            mode=mode
        )

        processed_pair = {
            **pair,
            'violating_prompt': violating_prompt,
            'compliant_prompt': compliant_prompt,
            'phrase_text': phrase_text if mode == 'prefill' else None,
            'mode': mode
        }
        processed_pairs.append(processed_pair)

    logger.info(f"🔨 Built prompts for {len(processed_pairs)} thread pairs ({len(processed_pairs) * 2} total threads)")
    return processed_pairs

def _build_single_prompt(pair: Dict[str, Any],
                        thread_type: str,
                        context_config: Dict[str, Any],
                        phrase_text: str,
                        model_config: Dict[str, Any],
                        mode: str) -> Dict[str, Any]:
    """
    Build prompt for a single thread (violating or compliant).

    Args:
        pair: Thread pair dictionary
        thread_type: 'violating' or 'compliant'
        context_config: Context configuration
        phrase_text: Phrase text to apply
        model_config: Model configuration
        mode: Phrase mode ('prefill' or 'prompt')

    Returns:
        Dictionary with 'messages' key
    """
    messages = []

    # Build question text based on context (pass phrase if mode is 'prompt')
    # Convert "Let's X" to "Please X." for prompt mode
    if mode == 'prompt' and phrase_text:
        prompt_phrase = phrase_text.replace("Let's", "Please")
        if not prompt_phrase.endswith('.'):
            prompt_phrase += '.'
    else:
        prompt_phrase = None
    question_text = _build_question_text(pair, thread_type, context_config, prompt_phrase)

    # Build user content with multimodal data
    content = _build_multimodal_content(
        pair,
        question_text,
        context_config,
        model_config
    )

    messages.append({"role": "user", "content": content})

    return {'messages': messages}

def _build_question_text(pair: Dict[str, Any],
                        thread_type: str,
                        context_config: Dict[str, Any],
                        prompt_phrase: str = None) -> str:
    """
    Build question text based on context configuration flags.

    Args:
        pair: Thread pair dictionary
        thread_type: 'violating' or 'compliant'
        context_config: Context configuration with boolean flags
        prompt_phrase: Optional phrase to integrate into the question (for 'prompt' mode)

    Returns:
        Formatted question text
    """
    # Get the appropriate thread and answer options
    thread = pair[f'{thread_type}_thread']
    answer_options = pair[f'{thread_type}_answer_options']
    submission = pair.get('submission', {}).get('submission_object', {})

    # Build context parts in order: subreddit, rules, submission, discussion, question
    context_parts = []

    # 1. Subreddit (always shown)
    subreddit_name = pair['subreddit']
    subreddit_parts = [f"Subreddit: r/{subreddit_name}"]

    # Add title and description if available
    subreddit_title = pair.get('subreddit_title', '')
    if subreddit_title:
        subreddit_parts.append(f"Title: {subreddit_title}")

    subreddit_description = pair.get('subreddit_description', '')
    if subreddit_description:
        subreddit_parts.append(f"Description: {subreddit_description}")

    context_parts.append("\n".join(subreddit_parts))

    # 2. Rules (always shown)
    rules_text = _format_rules(pair['rules'])
    context_parts.append(f"\nRules:\n{rules_text}")

    # 3. Submission (if submission flag is set)
    if context_config.get('include_submission', False):
        # Determine if we should show actual media or placeholder
        show_media = context_config.get('include_media', False)
        show_users = context_config.get('include_user', False)
        submission_text = _format_submission(submission, pair, show_media, show_users)
        context_parts.append(f"\n{submission_text}")

    # 4. Discussion thread (if discussion flag is set)
    # Otherwise, just show the leaf comment
    show_full_discussion = context_config.get('include_discussion', False)
    show_users = context_config.get('include_user', False)

    # Always build user anonymization map (users are always anonymized when shown)
    user_map = _build_user_map(submission, thread)

    if show_full_discussion:
        # Show all comments with the last one marked
        discussion_text = _format_discussion(thread, user_map, show_users, mark_last=True)
        context_parts.append(f"\n{discussion_text}")
    else:
        # Show only the leaf comment (last comment in thread)
        leaf_comment_text = _format_leaf_comment(thread[-1], user_map, show_users)
        context_parts.append(f"\nComment 1 [TARGET COMMENT]:\n{leaf_comment_text}")

    # 5. Question and answer options
    options_str = _format_answer_options(answer_options)

    # Build question with optional phrase integration
    if prompt_phrase:
        question_text = f"\nQuestion: Does the [TARGET COMMENT] violate a rule? {prompt_phrase}\n\n{options_str}"
    else:
        question_text = f"\nQuestion: Does the [TARGET COMMENT] violate a rule?\n\n{options_str}"

    context_parts.append(question_text)

    return "\n".join(context_parts)

def _format_rules(rules: List[Dict[str, Any]]) -> str:
    """Format rules list."""
    formatted = []
    for idx, rule in enumerate(rules, start=1):
        short_name = rule['short_name_clean']
        description = rule['description_clean']
        violation_reason = rule['violation_reason_clean']

        rule_text = (
            f"{idx}. {short_name}\n"
            f"Description: {description}\n"
            f"Violation: {violation_reason}"
        )
        formatted.append(rule_text)

    return "\n\n".join(formatted)

def _format_submission(submission: Dict[str, Any],
                       pair: Dict[str, Any],
                       show_media: bool,
                       show_user: bool = False) -> str:
    """Format submission content."""
    title = submission.get('title', '[No title]')
    selftext = submission.get('selftext', '').strip()
    url = submission.get('url', '').strip()
    link_flair = submission.get('link_flair_text')
    created_utc = submission.get('created_utc', 0)

    # Format datetime
    date_str = _format_timestamp(created_utc)

    # Build submission header
    parts = [f"Submission Title: {title}"]
    if link_flair:
        parts.append(f"Flair: {link_flair}")
    if show_user:
        parts.append(f"Author: USER1")  # Submission author is always USER1
    parts.append(f"Posted: {date_str}")

    header = "\n".join(parts)

    # Check if media exists
    media_files = pair.get('submission', {}).get('media_files', [])
    has_media = len(media_files) > 0

    # Build body content
    body_parts = []

    # Add URL if present
    if url:
        body_parts.append(url)

    # Add media placeholder if media exists but not included
    if has_media and not show_media:
        body_parts.append("[Image present but not shown]")

    # Add text content (always clean user mentions)
    if selftext:
        cleaned_text = _clean_user_mentions(selftext)
        body_parts.append(cleaned_text)
    elif not has_media and not url:  # Only show [No text] if there's no media and no URL
        body_parts.append("[No text]")

    body_text = "\n".join(body_parts)

    # Add Body: prefix (first occurrence allows splitting for media interleaving)
    return f"{header}\nBody: {body_text}"

def _format_leaf_comment(comment: Dict[str, Any],
                        user_map: Dict[str, str],
                        show_user: bool = False) -> str:
    """Format a single leaf comment (without section header)."""
    author = comment.get('author', '[deleted]')
    author_flair = comment.get('author_flair_text')
    body = comment.get('body', '[deleted]')
    created_utc = comment.get('created_utc', 0)

    # Format datetime
    date_str = _format_timestamp(created_utc)

    # Get anonymized username
    user_label = user_map.get(author, author)

    # Clean user mentions from body
    cleaned_body = _clean_user_mentions(body)

    # Build comment (no header - caller adds section header)
    parts = []
    if author_flair:
        parts.append(f"Flair: {author_flair}")
    if show_user:
        parts.append(f"Author: {user_label}")
    parts.append(f"Posted: {date_str}")
    parts.append(f"Body: {cleaned_body}")

    return "\n".join(parts)

def _format_discussion(thread: List[Dict[str, Any]],
                       user_map: Dict[str, str],
                       show_user: bool = False,
                       mark_last: bool = True) -> str:
    """Format discussion thread with comments."""
    formatted = []

    for idx, comment in enumerate(thread, start=1):
        author = comment.get('author', '[deleted]')
        author_flair = comment.get('author_flair_text')
        body = comment.get('body', '[deleted]')
        created_utc = comment.get('created_utc', 0)

        # Format datetime
        date_str = _format_timestamp(created_utc)

        # Get anonymized username
        user_label = user_map.get(author, author)

        # Clean user mentions from body
        cleaned_body = _clean_user_mentions(body)

        # Build comment header
        is_last = (idx == len(thread))
        comment_label = f"Comment {idx}" + (" [TARGET COMMENT]" if mark_last and is_last else "") + ":"

        parts = [comment_label]
        if author_flair:
            parts.append(f"Flair: {author_flair}")
        if show_user:
            parts.append(f"Author: {user_label}")
        parts.append(f"Posted: {date_str}")
        parts.append(f"Body: {cleaned_body}")

        formatted.append("\n".join(parts))

    return "\n\n".join(formatted)

def _build_user_map(submission: Dict[str, Any],
                    thread: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build mapping from real usernames to anonymized labels."""
    # Collect all unique authors
    submission_author = submission.get('author', '[deleted]')

    # Start with submission author as USER1
    user_map = {submission_author: 'USER1'}

    # Add other users in order of appearance
    user_counter = 2
    for comment in thread:
        author = comment.get('author', '[deleted]')
        if author not in user_map:
            user_map[author] = f'USER{user_counter}'
            user_counter += 1

    return user_map

def _format_answer_options(options: List[Dict[str, str]]) -> str:
    """
    Format answer options as multiple choice.

    Args:
        options: List of option dictionaries with 'label' and 'rule' keys

    Returns:
        Formatted options string
    """
    formatted = []
    for option in options:
        label = option['label']
        rule = option['rule']
        formatted.append(f"{label} {rule}")

    return "\n".join(formatted)

def _build_multimodal_content(pair: Dict[str, Any],
                              question_text: str,
                              context_config: Dict[str, Any],
                              model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build multimodal content with images and text.

    Images are interleaved in the submission by splitting on first "Body: " marker.
    This splits submission header from body, allowing images to appear between them.
    Comment bodies (which also have "Body: ") remain intact in the text after the split.

    Args:
        pair: Thread pair dictionary
        question_text: Question text (may contain "Body: " marker in submission)
        context_config: Context configuration
        model_config: Model configuration

    Returns:
        List of content items (text and images properly interleaved)
    """
    content = []

    # Check if we need to add media and interleave it
    if context_config.get('include_media', False) and context_config.get('include_submission', False):
        submission = pair.get('submission', {})
        media_files = submission.get('media_files', [])

        if media_files and "Body: " in question_text:
            # Split on first "Body: " only (submission body)
            # Everything after (including comment bodies) stays together
            parts = question_text.split("Body: ", 1)
            before_body = parts[0] + "Body: "  # Keep the "Body: " prefix
            after_body = parts[1] if len(parts) > 1 else ""

            # Add text before images (submission header)
            content.append({"type": "text", "text": before_body})

            # Add images
            for media_path in media_files:
                if model_config['type'] == 'vllm':
                    if model_config['hf_path'].startswith('Qwen'):
                        content.append({"type": "image", "image": media_path})
                    else:  # LLaVA, Llama
                        content.append({"type": "image"})
                else:  # API models
                    content.append({"type": "image"})

            # Add text after images (submission body + rest of prompt)
            content.append({"type": "text", "text": after_body})
        else:
            # No media or no Body: marker, just add text
            content.append({"type": "text", "text": question_text})
    else:
        # No media to interleave, just add text
        content.append({"type": "text", "text": question_text})

    return content

# =============================================================================
# CHAT TEMPLATE APPLICATION
# =============================================================================

def apply_chat_template(thread_pairs: List[Dict[str, Any]],
                       model_name: str,
                       logger: logging.Logger) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply model-specific chat template to all prompts and calculate resource requirements.

    Args:
        thread_pairs: List of thread pairs with prompts
        model_name: Model name
        logger: Logger instance

    Returns:
        Tuple of (thread_pairs with formatted prompts, resource_stats dict)
        resource_stats contains:
            - max_images_per_prompt: Maximum number of images in any single prompt
            - max_model_len: Maximum token length across all prompts (including image tokens)
    """
    model_config = config.get_model_config(model_name)

    # API models don't use AutoProcessor
    if model_config['type'] == 'api':
        logger.info("⏭️  Skipping chat template for API model (will be handled in API call)")
        return thread_pairs, {'max_images_per_prompt': 0, 'max_model_len': 0}

    # vLLM models use AutoProcessor
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_config['hf_path'],
        trust_remote_code=model_config.get('trust_remote_code', False)
    )

    max_images_per_prompt = 0
    max_token_length = 0

    for pair in thread_pairs:
        # Apply template to both violating and compliant threads
        for thread_type in ['violating', 'compliant']:
            messages = pair[f'{thread_type}_prompt']['messages']

            # Count images actually in the content (not just in media_files)
            content = messages[0]['content']
            num_images = sum(1 for item in content if isinstance(item, dict) and item.get('type') == 'image')
            max_images_per_prompt = max(max_images_per_prompt, num_images)

            # Generate prompt text (for vLLM)
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Add phrase text for prefill mode
            if pair.get('phrase_text') and pair['mode'] == 'prefill':
                prompt_text += pair['phrase_text']

            pair[f'{thread_type}_prompt_text'] = prompt_text

            # Tokenize with images to get accurate token count
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors='pt'
                )
                token_length = inputs['input_ids'].shape[1]
                max_token_length = max(max_token_length, token_length)
            except Exception as e:
                logger.warning(f"⚠️  Could not tokenize prompt for length calculation: {e}")

    resource_stats = {
        'max_images_per_prompt': max_images_per_prompt,
        'max_model_len': max_token_length
    }

    logger.info(f"✅ Applied chat template to {len(thread_pairs)} thread pairs")
    logger.info(f"📊 Resource requirements - Max images: {max_images_per_prompt}, Max tokens: {max_token_length}")

    return thread_pairs, resource_stats

# =============================================================================
# TWO-STAGE EVALUATION FUNCTIONS
# =============================================================================

def evaluate_two_stage_vllm(thread_pairs: List[Dict[str, Any]],
                             model_name: str,
                             model_config: Dict[str, Any],
                             num_gpus: int,
                             resource_stats: Dict[str, Any],
                             max_response_tokens: int,
                             context: str,
                             logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Two-stage evaluation using vLLM.

    Args:
        thread_pairs: Thread pairs with prompts
        model_name: Model name
        model_config: Model configuration
        num_gpus: Number of GPUs to use for tensor parallelism
        resource_stats: Resource statistics from chat template application
        max_response_tokens: Maximum tokens for response generation
        context: Context string (e.g., "submission-media")
        logger: Logger instance

    Returns:
        Evaluation results
    """
    from vllm import LLM

    # Initialize LLM engine
    logger.info(f"🚀 Initializing vLLM engine for {model_name}...")
    logger.info(f"📊 Using {num_gpus} GPU(s) for tensor parallelism")

    # For Qwen3-VL models, set multimodal limits based on actual data
    limit_mm_per_prompt = None
    if 'Qwen3-VL' in model_config['hf_path'] or 'Qwen/Qwen3-VL' in model_config['hf_path']:
        max_images = resource_stats.get('max_images_per_prompt', 50)
        limit_mm_per_prompt = {'image': max_images, 'video': 0}
        logger.info(f"📊 Setting limit_mm_per_prompt for Qwen3-VL: {limit_mm_per_prompt}")

    # Use actual max token length if available, otherwise fall back to config
    max_model_len = resource_stats.get('max_model_len', 0)
    if max_model_len == 0:
        max_model_len = model_config.get('max_model_len', 8192)
        logger.info(f"📊 Using configured max_model_len: {max_model_len}")
    else:
        # Add buffer for generation (response tokens) plus safety margin
        max_model_len = max_model_len + max_response_tokens + 50
        logger.info(f"📊 Using calculated max_model_len: {max_model_len} (including {max_response_tokens} token buffer + 50 token safety margin)")

    # Set max_num_seqs conservatively if media is included (images cause memory pressure)
    llm_kwargs = {
        'model': model_config['hf_path'],
        'tensor_parallel_size': num_gpus,
        'gpu_memory_utilization': model_config.get('gpu_memory_utilization', 0.9),
        'trust_remote_code': model_config.get('trust_remote_code', True),
        'max_model_len': max_model_len,
        'limit_mm_per_prompt': limit_mm_per_prompt,
        'seed': 0
    }

    if 'media' in context.split('-'):
        llm_kwargs['max_num_seqs'] = 32
        logger.info(f"📊 Setting max_num_seqs=32 due to media in context")
    else:
        logger.info(f"📊 Using vLLM auto max_num_seqs (no media in context)")

    llm_engine = LLM(**llm_kwargs)
    logger.info(f"✅ LLM engine initialized with tensor_parallel_size={num_gpus}")

    # Stage 1: Generate reasoning for all threads (violating + compliant)
    logger.info("📝 Stage 1: Generating reasoning responses...")
    stage1_responses = _generate_stage1_vllm(thread_pairs, llm_engine, max_response_tokens, context, logger)
    logger.info(f"✅ Generated {len(stage1_responses)} × 2 Stage 1 reasoning responses")

    # Stage 2: Extract clean answers
    logger.info("🎯 Stage 2: Extracting clean answers...")
    results = _generate_stage2_vllm(thread_pairs, stage1_responses, llm_engine, max_response_tokens, context, logger)

    logger.info(f"✅ Completed two-stage evaluation for {len(results)} thread pairs")
    return results

def _generate_stage1_vllm(thread_pairs: List[Dict[str, Any]],
                         llm_engine,
                         max_response_tokens: int,
                         context: str,
                         logger: logging.Logger,
                         violating_prompts: List[str] = None,
                         compliant_prompts: List[str] = None,
                         sampling_params = None,
                         uuid_suffix: str = '') -> List[Dict[str, str]]:
    """
    Generate responses for all threads using vLLM (base function for both Stage 1 and Stage 2).

    Args:
        thread_pairs: Thread pairs with prompts
        llm_engine: vLLM engine
        max_response_tokens: Maximum tokens for response generation
        context: Context string (e.g., "submission-media")
        logger: Logger instance
        violating_prompts: Optional custom prompts for violating threads (defaults to pair['violating_prompt_text'])
        compliant_prompts: Optional custom prompts for compliant threads (defaults to pair['compliant_prompt_text'])
        sampling_params: Optional custom sampling params (defaults to temperature=0, max_tokens from max_response_tokens)
        uuid_suffix: Optional suffix for multimodal UUIDs (e.g., '_s2' for stage 2)

    Returns:
        List of dicts with 'violating' and 'compliant' responses
    """
    from vllm import SamplingParams

    # Use default prompts if not provided
    if violating_prompts is None:
        violating_prompts = [pair['violating_prompt_text'] for pair in thread_pairs]
    if compliant_prompts is None:
        compliant_prompts = [pair['compliant_prompt_text'] for pair in thread_pairs]

    # Use default sampling params if not provided
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_response_tokens, stop=None)

    # Check if media should be included based on context flags
    include_media = 'media' in context.split('-')

    # Prepare inputs (flatten violating + compliant)
    inputs = []
    for pair in thread_pairs:
        # Build inputs for both violating and compliant threads
        for thread_type, prompts in [('violating', violating_prompts), ('compliant', compliant_prompts)]:
            idx = thread_pairs.index(pair)

            # Only load and pass images if media flag is set in context
            images = []
            if include_media:
                submission = pair.get('submission', {})
                media_files = submission.get('media_files', [])

                # Load images separately for each thread to avoid sharing image objects
                for media_path in media_files:
                    try:
                        img = Image.open(media_path).convert('RGB')
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load image {media_path}: {e}")

            input_dict = {
                "prompt": prompts[idx],
                "multi_modal_data": {"image": images} if images else {},
                "multi_modal_uuids": {"image": [f"uuid_{thread_type}{uuid_suffix}_{pair['mod_comment_id']}_{j}"
                                                for j in range(len(images))]} if images else {}
            }
            inputs.append(input_dict)

    # Generate responses
    outputs = llm_engine.generate(inputs, sampling_params=sampling_params)

    # Unflatten responses back to pairs
    responses = []
    for i in range(0, len(outputs), 2):
        violating_response = outputs[i].outputs[0].text
        compliant_response = outputs[i + 1].outputs[0].text

        responses.append({
            'violating': violating_response,
            'compliant': compliant_response
        })

    return responses

def _generate_stage2_vllm(thread_pairs: List[Dict[str, Any]],
                         stage1_responses: List[Dict[str, str]],
                         llm_engine,
                         max_response_tokens: int,
                         context: str,
                         logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Generate Stage 2 clean answer extraction by calling Stage 1 internally.

    Args:
        thread_pairs: Thread pairs
        stage1_responses: Stage 1 reasoning responses
        llm_engine: vLLM engine
        max_response_tokens: Maximum tokens for response generation (not used in Stage 2, uses fixed 10 tokens)
        context: Context string (e.g., "submission-media")
        logger: Logger instance

    Returns:
        Complete evaluation results
    """
    from vllm import SamplingParams

    # Prepare Stage 2 prompts (append reasoning + answer phrase)
    violating_prompts = [
        pair['violating_prompt_text'] + stage1['violating'] + config.ANSWER_PHRASE
        for pair, stage1 in zip(thread_pairs, stage1_responses)
    ]
    compliant_prompts = [
        pair['compliant_prompt_text'] + stage1['compliant'] + config.ANSWER_PHRASE
        for pair, stage1 in zip(thread_pairs, stage1_responses)
    ]

    # Generate clean answers using Stage 1 function (temperature=0, stop at newline to get single choice)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10
    )

    stage2_responses = _generate_stage1_vllm(
        thread_pairs=thread_pairs,
        llm_engine=llm_engine,
        max_response_tokens=max_response_tokens,
        context=context,
        logger=logger,
        violating_prompts=violating_prompts,
        compliant_prompts=compliant_prompts,
        sampling_params=sampling_params,
        uuid_suffix='_s2'
    )

    # Build final results
    results = []
    for pair, stage1, stage2 in zip(thread_pairs, stage1_responses, stage2_responses):
        violating_clean = stage2['violating']
        compliant_clean = stage2['compliant']

        # Extract predictions
        violating_prediction = _extract_answer_choice(violating_clean)
        compliant_prediction = _extract_answer_choice(compliant_clean)

        # Get ground truth
        violating_correct = pair['violating_correct_answer']
        compliant_correct = pair['compliant_correct_answer']

        # Calculate scores
        violating_score = 1 if violating_prediction == violating_correct else 0
        compliant_score = 1 if compliant_prediction == compliant_correct else 0

        result = {
            'mod_comment_id': pair['mod_comment_id'],
            'subreddit': pair['subreddit'],
            'submission_id': pair['submission_id'],

            'violating': {
                'input_prompt': pair['violating_prompt_text'],
                'reasoning_response': stage1['violating'],
                'clean_answer_response': violating_clean,
                'extracted_prediction': violating_prediction,
                'correct_answer': violating_correct,
                'score': violating_score,
                'answer_options': pair['violating_answer_options']
            },

            'compliant': {
                'input_prompt': pair['compliant_prompt_text'],
                'reasoning_response': stage1['compliant'],
                'clean_answer_response': compliant_clean,
                'extracted_prediction': compliant_prediction,
                'correct_answer': compliant_correct,
                'score': compliant_score,
                'answer_options': pair['compliant_answer_options']
            },

            'metadata': {
                'rule': pair['metadata']['rule'],
                'rule_cluster_id': pair['metadata']['rule_cluster_id'],
                'rule_cluster_label': pair['metadata']['rule_cluster_label'],
                'subreddit_cluster_id': pair['subreddit_cluster_id'],
                'subreddit_cluster_label': pair['subreddit_cluster_label'],
                'subreddit_language': pair.get('subreddit_language', 'unknown')
            }
        }
        results.append(result)

    logger.info(f"Generated {len(results)} × 2 Stage 2 clean answers")
    return results

def evaluate_two_stage_api(thread_pairs: List[Dict[str, Any]],
                           model_config: Dict[str, Any],
                           output_dir: Path,
                           context: str,
                           max_response_tokens: int,
                           logger: logging.Logger,
                           override: bool = False,
                           stage2_batch_size: int = 275) -> List[Dict[str, Any]]:
    """
    Two-stage evaluation using OpenAI Flex API (Stage 1) + local vLLM (Stage 2).

    Stage 1: OpenAI Flex API generates reasoning responses (50% cost savings)
             Uses ThreadPoolExecutor for concurrent requests with checkpointing
    Stage 2: Local Qwen3-VL-30B via vLLM extracts clean answers (fast, free)
             Processes in batches to manage GPU memory

    Args:
        thread_pairs: Thread pairs with prompts
        model_config: Model configuration from config.py
        output_dir: Output directory for checkpoint and results
        context: Context string (e.g., "submission-media")
        max_response_tokens: Max tokens for Stage 1 response
        logger: Logger instance
        override: If True, re-run even if results exist
        stage2_batch_size: Batch size for Stage 2 local vLLM processing (default: 275)

    Returns:
        Evaluation results list
    """
    # =========================================================================
    # CHECK FOR EXISTING RESULTS (skip if reasoning_*.json exists and not override)
    # =========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_reasoning_files = list(output_dir.glob("reasoning_*.json"))

    if existing_reasoning_files and not override:
        latest_reasoning_file = sorted(existing_reasoning_files)[-1]
        logger.info(f"✅ Found existing results: {latest_reasoning_file}")
        logger.info(f"   Skipping Stage 1 and Stage 2 (use --override to re-run)")

        with open(latest_reasoning_file, 'r') as f:
            results = json.load(f)

        logger.info(f"✅ Loaded {len(results)} existing results")
        return results

    if existing_reasoning_files and override:
        logger.info(f"♻️  Override mode: Will re-run (found {len(existing_reasoning_files)} existing result file(s))")

    # =========================================================================
    # STAGE 1: OpenAI Flex API with ThreadPool + Checkpointing
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: OpenAI Flex API - Generating Reasoning")
    logger.info("=" * 60)

    context_config = config.parse_context_flags(context)
    checkpoint_path = output_dir / "flex_checkpoint.json"

    # Load checkpoint if exists (for crash recovery)
    stage1_responses = {}
    failed_in_checkpoint = 0
    if checkpoint_path.exists() and not override:
        with open(checkpoint_path, 'r') as f:
            raw_checkpoint = json.load(f)
        # Remove failed entries (empty content) so they get retried
        for cid, resp in raw_checkpoint.items():
            has_content = (isinstance(resp, dict) and resp.get('content')) or (isinstance(resp, str) and resp)
            if has_content:
                stage1_responses[cid] = resp
            else:
                failed_in_checkpoint += 1
        logger.info(f"📂 Loaded {len(stage1_responses)} successful responses from checkpoint ({failed_in_checkpoint} failed entries removed for retry)")

    # Build work items (skip already completed requests)
    work_items = []
    reasoning_effort = model_config.get('reasoning_effort')
    for pair in thread_pairs:
        for thread_type in ['violating', 'compliant']:
            custom_id = f"{pair['mod_comment_id']}_{thread_type}"
            if custom_id not in stage1_responses:
                messages = _build_openai_messages(pair, thread_type, context_config, logger)
                work_items.append((custom_id, messages, model_config['model_id'], max_response_tokens, reasoning_effort))

    total_requests = len(thread_pairs) * 2
    logger.info(f"📊 Stage 1: {len(work_items)} requests remaining (of {total_requests} total)")
    logger.info(f"🧠 Reasoning effort: {reasoning_effort or 'none'}")

    if work_items:
        # Process with ThreadPoolExecutor
        max_workers = config.OPENAI_FLEX_CONFIG['max_workers']
        checkpoint_interval = config.OPENAI_FLEX_CONFIG['checkpoint_interval']
        completed = 0
        failed = failed_in_checkpoint

        logger.info(f"🚀 Starting ThreadPoolExecutor with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single_flex_request, item): item[0] for item in work_items}

            for future in as_completed(futures):
                custom_id = futures[future]
                try:
                    result_id, response_dict = future.result()
                    stage1_responses[result_id] = response_dict  # Dict with 'content' and 'reasoning_summary'
                    completed += 1

                    # Checkpoint periodically
                    if completed % checkpoint_interval == 0:
                        with open(checkpoint_path, 'w') as f:
                            json.dump(stage1_responses, f)
                        logger.info(f"📊 Stage 1: {completed}/{len(work_items)} complete (checkpoint saved)")

                except Exception as e:
                    logger.error(f"❌ Request {custom_id} failed: {e}")
                    stage1_responses[custom_id] = {'content': '', 'reasoning_summary': ''}
                    failed += 1

        # Final checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump(stage1_responses, f)

        logger.info(f"✅ Stage 1 complete: {completed} succeeded, {failed} failed")

    else:
        logger.info("✅ Stage 1: All requests already in checkpoint")

    logger.info(f"📊 Stage 1 total: {len(stage1_responses)} responses collected")

    # Helper to extract content from response (handles both dict and legacy string format)
    def _get_response_field(resp, field='content'):
        if isinstance(resp, dict):
            return resp.get(field, '') or ''
        return resp if field == 'content' else ''  # Legacy string format has no reasoning

    # Organize Stage 1 responses by pair (both content and reasoning_summary)
    stage1_by_pair = [
        {
            'violating_content': _get_response_field(
                stage1_responses.get(f"{pair['mod_comment_id']}_violating", {}), 'content'),
            'violating_reasoning': _get_response_field(
                stage1_responses.get(f"{pair['mod_comment_id']}_violating", {}), 'reasoning_summary'),
            'compliant_content': _get_response_field(
                stage1_responses.get(f"{pair['mod_comment_id']}_compliant", {}), 'content'),
            'compliant_reasoning': _get_response_field(
                stage1_responses.get(f"{pair['mod_comment_id']}_compliant", {}), 'reasoning_summary'),
        }
        for pair in thread_pairs
    ]

    # =========================================================================
    # FILTER: Drop pairs where Stage 1 failed (empty content)
    # =========================================================================
    original_count = len(thread_pairs)
    filtered = [
        (pair, stage1) for pair, stage1 in zip(thread_pairs, stage1_by_pair)
        if stage1['violating_content'] and stage1['compliant_content']
    ]

    if len(filtered) < original_count:
        dropped = original_count - len(filtered)
        logger.warning(f"⚠️  Dropping {dropped}/{original_count} pairs with empty Stage 1 responses ({len(filtered)} remaining)")
        thread_pairs, stage1_by_pair = zip(*filtered) if filtered else ([], [])
        thread_pairs = list(thread_pairs)
        stage1_by_pair = list(stage1_by_pair)

        if not thread_pairs:
            logger.error("❌ All Stage 1 responses are empty. Aborting evaluation.")
            raise RuntimeError("All Stage 1 responses are empty — cannot proceed to Stage 2.")
    else:
        logger.info(f"✅ All {original_count} pairs have valid Stage 1 responses")

    # =========================================================================
    # STAGE 2: Local Qwen3-VL-30B via vLLM for Answer Extraction
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: Local vLLM - Extracting Clean Answers")
    logger.info("=" * 60)

    stage2_model = model_config.get('stage2_model', 'qwen3-vl-30b-instruct')
    stage2_config = config.get_model_config(stage2_model)

    # Apply chat template for Stage 2 model (gets base prompt with assistant turn prefix)
    thread_pairs, _ = apply_chat_template(thread_pairs, stage2_model, logger)

    # Build Stage 2 prompts FIRST (Stage 1 content only + answer phrase as prefill)
    # NOTE: We use 'content' (output text), not 'reasoning_summary' for Stage 2
    violating_prompts = [
        pair['violating_prompt_text'] + stage1['violating_content'] + config.ANSWER_PHRASE
        for pair, stage1 in zip(thread_pairs, stage1_by_pair)
    ]
    compliant_prompts = [
        pair['compliant_prompt_text'] + stage1['compliant_content'] + config.ANSWER_PHRASE
        for pair, stage1 in zip(thread_pairs, stage1_by_pair)
    ]

    # Calculate actual max token length for Stage 2 prompts (including reasoning)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        stage2_config['hf_path'],
        trust_remote_code=True
    )

    max_stage2_tokens = 0
    for violating_prompt, compliant_prompt in zip(violating_prompts, compliant_prompts):
        violating_tokens = len(tokenizer.encode(violating_prompt))
        compliant_tokens = len(tokenizer.encode(compliant_prompt))
        max_stage2_tokens = max(max_stage2_tokens, violating_tokens, compliant_tokens)

    # Add buffer for generation (10 tokens) + safety margin
    stage2_max_model_len = max_stage2_tokens + 50
    logger.info(f"📊 Stage 2 max tokens (with reasoning): {max_stage2_tokens}, using max_model_len: {stage2_max_model_len}")

    # Initialize vLLM engine for Stage 2
    from vllm import LLM, SamplingParams

    logger.info(f"Initializing vLLM engine for Stage 2 ({stage2_model})...")

    # Determine GPU count from CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    num_gpus = len(cuda_devices.split(','))

    llm_engine = LLM(
        model=stage2_config['hf_path'],
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=stage2_config.get('gpu_memory_utilization', 0.95),
        trust_remote_code=True,
        max_model_len=stage2_max_model_len,
        seed=0
    )
    logger.info(f"✅ Stage 2 LLM engine initialized with {num_gpus} GPU(s)")

    # Generate Stage 2 answers in batches (to manage GPU memory)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    # Use text-only context for Stage 2 (no media needed)
    stage2_context = context.replace('-media', '') if 'media' in context else context

    # Process Stage 2 in batches for GPU memory efficiency
    num_stage2_batches = (len(thread_pairs) + stage2_batch_size - 1) // stage2_batch_size
    logger.info(f"📊 Processing Stage 2 in {num_stage2_batches} batch(es) of {stage2_batch_size}")

    stage2_responses = []
    for batch_idx in range(num_stage2_batches):
        start = batch_idx * stage2_batch_size
        end = min((batch_idx + 1) * stage2_batch_size, len(thread_pairs))

        batch_pairs = thread_pairs[start:end]
        batch_violating_prompts = violating_prompts[start:end]
        batch_compliant_prompts = compliant_prompts[start:end]

        logger.info(f"Stage 2 Batch {batch_idx + 1}/{num_stage2_batches}: pairs {start}-{end-1} ({len(batch_pairs)} pairs)")

        batch_responses = _generate_stage1_vllm(
            thread_pairs=batch_pairs,
            llm_engine=llm_engine,
            max_response_tokens=10,
            context=stage2_context,  # No media for answer extraction
            logger=logger,
            violating_prompts=batch_violating_prompts,
            compliant_prompts=batch_compliant_prompts,
            sampling_params=sampling_params,
            uuid_suffix=f'_s2_api_b{batch_idx}'
        )
        stage2_responses.extend(batch_responses)
        logger.info(f"✓ Stage 2 Batch {batch_idx + 1}/{num_stage2_batches} complete")

    # =========================================================================
    # BUILD FINAL RESULTS
    # =========================================================================
    results = []
    for idx, (pair, stage1, stage2) in enumerate(zip(thread_pairs, stage1_by_pair, stage2_responses)):
        violating_clean = stage2['violating']
        compliant_clean = stage2['compliant']

        violating_prediction = _extract_answer_choice(violating_clean)
        compliant_prediction = _extract_answer_choice(compliant_clean)

        violating_correct = pair['violating_correct_answer']
        compliant_correct = pair['compliant_correct_answer']

        violating_score = 1 if violating_prediction == violating_correct else 0
        compliant_score = 1 if compliant_prediction == compliant_correct else 0

        result = {
            'mod_comment_id': pair['mod_comment_id'],
            'subreddit': pair['subreddit'],
            'submission_id': pair['submission_id'],

            'violating': {
                'stage2_prompt': violating_prompts[idx],
                'reasoning_response': stage1['violating_content'],
                'reasoning_summary': stage1['violating_reasoning'],
                'clean_answer_response': violating_clean,
                'extracted_prediction': violating_prediction,
                'correct_answer': violating_correct,
                'score': violating_score,
                'answer_options': pair['violating_answer_options']
            },

            'compliant': {
                'stage2_prompt': compliant_prompts[idx],
                'reasoning_response': stage1['compliant_content'],
                'reasoning_summary': stage1['compliant_reasoning'],
                'clean_answer_response': compliant_clean,
                'extracted_prediction': compliant_prediction,
                'correct_answer': compliant_correct,
                'score': compliant_score,
                'answer_options': pair['compliant_answer_options']
            },

            'metadata': {
                'rule': pair['metadata']['rule'],
                'rule_cluster_id': pair['metadata']['rule_cluster_id'],
                'rule_cluster_label': pair['metadata']['rule_cluster_label'],
                'subreddit_cluster_id': pair['subreddit_cluster_id'],
                'subreddit_cluster_label': pair['subreddit_cluster_label'],
                'subreddit_language': pair.get('subreddit_language', 'unknown')
            }
        }
        results.append(result)

    # Save results to batch output directory for caching
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reasoning_path = output_dir / f"reasoning_{timestamp}.json"
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Batch reasoning saved to: {reasoning_path}")

    logger.info(f"✅ Completed hybrid two-stage evaluation for {len(results)} thread pairs")
    return results

# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def _extract_answer_choice(text: str) -> str:
    """
    Extract answer choice (a-h) from model response.

    Args:
        text: Model response text

    Returns:
        Extracted choice like "(a)", "(b)", etc., or empty string if not found
    """
    import re

    text = text.strip().lower()

    # Look for patterns like "(a)", "(b)", "a)", "a.", "option a", etc.
    # Match single letter a-z
    match = re.search(r'\(?([a-z])\)?', text)
    if match:
        letter = match.group(1)
        return f"({letter})"

    return ""

# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(results: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.

    Args:
        results: Evaluation results
        logger: Logger instance

    Returns:
        Metrics dictionary with overall and per-cluster accuracy
    """
    total_pairs = len(results)

    # Overall accuracy
    violating_correct = sum(r['violating']['score'] for r in results)
    compliant_correct = sum(r['compliant']['score'] for r in results)
    total_correct = violating_correct + compliant_correct
    total_threads = total_pairs * 2

    overall_accuracy = total_correct / total_threads if total_threads > 0 else 0
    violating_accuracy = violating_correct / total_pairs if total_pairs > 0 else 0
    compliant_accuracy = compliant_correct / total_pairs if total_pairs > 0 else 0

    # Per-rule-cluster accuracy
    rule_cluster_stats = _calculate_cluster_accuracy(
        results,
        cluster_key='rule_cluster_label'
    )

    # Per-subreddit-cluster accuracy
    subreddit_cluster_stats = _calculate_cluster_accuracy(
        results,
        cluster_key='subreddit_cluster_label'
    )

    # Per-language accuracy
    language_stats = _calculate_cluster_accuracy(
        results,
        cluster_key='subreddit_language'
    )

    metrics = {
        'overall': {
            'total_pairs': total_pairs,
            'total_threads': total_threads,
            'overall_accuracy': overall_accuracy,
            'violating_accuracy': violating_accuracy,
            'compliant_accuracy': compliant_accuracy,
            'violating_correct': violating_correct,
            'compliant_correct': compliant_correct,
            'total_correct': total_correct
        },
        'per_rule_cluster': rule_cluster_stats,
        'per_subreddit_cluster': subreddit_cluster_stats,
        'per_language': language_stats
    }

    logger.info(f"📊 Metrics calculated - Overall: {overall_accuracy:.4f}, Violating: {violating_accuracy:.4f}, Compliant: {compliant_accuracy:.4f}")
    return metrics

def _calculate_cluster_accuracy(results: List[Dict[str, Any]],
                                cluster_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Calculate per-cluster accuracy statistics.

    Args:
        results: Evaluation results
        cluster_key: Key to group by ('rule_cluster_label' or 'subreddit_cluster_label')

    Returns:
        Dictionary mapping cluster labels to accuracy stats
    """
    cluster_stats = defaultdict(lambda: {
        'violating_correct': 0,
        'compliant_correct': 0,
        'total_correct': 0,
        'count': 0
    })

    for result in results:
        cluster_label = result['metadata'][cluster_key]

        cluster_stats[cluster_label]['violating_correct'] += result['violating']['score']
        cluster_stats[cluster_label]['compliant_correct'] += result['compliant']['score']
        cluster_stats[cluster_label]['total_correct'] += result['violating']['score'] + result['compliant']['score']
        cluster_stats[cluster_label]['count'] += 1

    # Calculate accuracies
    final_stats = {}
    for cluster, stats in cluster_stats.items():
        count = stats['count']
        total_threads = count * 2

        final_stats[cluster] = {
            'overall_accuracy': stats['total_correct'] / total_threads if total_threads > 0 else 0,
            'violating_accuracy': stats['violating_correct'] / count if count > 0 else 0,
            'compliant_accuracy': stats['compliant_correct'] / count if count > 0 else 0,
            'count': count,
            'total_threads': total_threads
        }

    return final_stats

# =============================================================================
# RESULT SAVING
# =============================================================================

def save_results(results: List[Dict[str, Any]],
                metrics: Dict[str, Any],
                output_dir: Path,
                model_name: str,
                split: str,
                context: str,
                phrase: str,
                mode: str,
                logger: logging.Logger) -> Tuple[Path, Path]:
    """
    Save evaluation results and metrics.

    Args:
        results: Evaluation results
        metrics: Metrics dictionary
        output_dir: Output directory
        model_name: Model name
        split: Dataset split
        context: Context type
        phrase: Phrase name
        mode: Phrase mode
        logger: Logger instance

    Returns:
        Tuple of (reasoning_path, performance_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save reasoning traces
    reasoning_path = output_dir / f"reasoning_{timestamp}.json"
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Save performance metrics
    performance_data = {
        'model': model_name,
        'split': split,
        'context': context,
        'phrase': phrase,
        'mode': mode,
        'metrics': metrics
    }

    performance_path = output_dir / f"performance_{timestamp}.json"
    with open(performance_path, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2)

    logger.info(f"💾 Reasoning traces saved to: {reasoning_path}")
    logger.info(f"💾 Performance metrics saved to: {performance_path}")

    return reasoning_path, performance_path

# =============================================================================
# LOGGING
# =============================================================================

def create_logger(split: str, model: str, context: str, phrase: str, mode: str) -> Tuple[logging.Logger, Path]:
    """
    Create logger with file and console handlers.

    Args:
        split: Dataset split
        model: Model name
        context: Context type
        phrase: Phrase name
        mode: Phrase mode

    Returns:
        Tuple of (logger, log_file_path)
    """
    import sys

    logs_dir = config.get_dir(config.LOGS_DIR, split, model, context, phrase, mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"evaluation_{timestamp}.log"

    logger = logging.getLogger('reddit_mod_eval')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"📝 Logging to: {log_path}")
    return logger, log_path
