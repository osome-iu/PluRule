# Evaluating PluRule

Evaluate vision-language and API models on the PluRule benchmark using
clustered datasets with multimodal Reddit discussion context.

## Features

- **Multiple Model Support**: Qwen3-VL vLLM models and OpenAI API models configured in `eval/config.py`
- **Configurable Contexts**: Control whether prompts include submission text, media, discussion context, and user labels
- **Prompt Variants**: Evaluate baseline and optional prompt-injection variants
- **Two-Stage Evaluation**: Generate reasoning first, then extract clean answers
- **Comprehensive Metrics**: Overall, per-rule-cluster, and per-subreddit-cluster accuracy

## Directory Structure

```
PluRule/
├── eval/
│   ├── config.py           # Model, context, and phrase configurations
│   ├── helpers.py          # Utility functions for data loading, prompting, evaluation
│   └── evaluate.py         # Main evaluation script
├── output/
│   └── eval/               # Evaluation results (reasoning + performance JSONs)
│       └── {model}/
│           └── {split}/
│               └── {context}/
│                   └── {phrase}_{mode}/
│                       ├── reasoning_TIMESTAMP.json
│                       └── performance_TIMESTAMP.json
└── logs/
    └── eval/               # Evaluation logs
        └── {model}/
            └── {split}/
                └── {context}/
                    └── {phrase}_{mode}/
                        └── evaluation_TIMESTAMP.log
```

## Installation

Use the bundled evaluation environment:

```bash
conda env create -f environment-eval.yml
conda activate plurule-eval
```

## Usage

### Basic Usage

```bash
# Evaluate Qwen3-VL-8B-Instruct on the test set with submission + discussion context
python eval/evaluate.py \
    --model qwen3-vl-8b-instruct \
    --split test \
    --context submission-discussion \
    --phrase cot \
    --mode prefill

# Debug mode (only 5 thread pairs)
python eval/evaluate.py \
    --model qwen3-vl-8b-instruct \
    --split test \
    --context none \
    --phrase baseline \
    --mode prefill \
    --debug
```

### Arguments

- `--model, -m`: Model to evaluate
  - vLLM: `qwen3-vl-4b-instruct`, `qwen3-vl-8b-instruct`, `qwen3-vl-30b-instruct`, `qwen3-vl-4b-thinking`, `qwen3-vl-8b-thinking`, `qwen3-vl-30b-thinking`
  - OpenAI API path: `gpt-4o`, `gpt5.2-low`, `gpt5.2-high`
  - `claude-sonnet-4` is present in `API_MODELS`, but the current API runner in `helpers.evaluate_two_stage_api()` is wired to OpenAI Flex

- `--split, -s`: Dataset split (`train`, `val`, `test`, `delta`)

- `--context, -c`: Dash-separated context flags. Subreddit metadata, rules, and the target comment are always included.
  - `none`: no optional context
  - `submission`: include submission text
  - `submission-media`: include submission text and media
  - `submission-discussion`: include submission text and full discussion thread
  - `submission-discussion-user`: include submission, discussion, and anonymized user labels
  - `submission-media-discussion-user`: include all optional context

- `--phrase, -p`: Prompting phrase
  - `baseline`: No additional phrase
  - `cot`: "Let's think step by step"
  - `analyze`: "Let's carefully analyze this content"
  - `artifacts`: "Let's look for rule violations"
  - `rules`: "Let's compare this against the subreddit rules"

- `--mode`: Phrase injection mode
  - `prefill`: Append phrase after chat template (default)
  - `prompt`: Append a prompt-mode rewrite to the question text

- `--cuda`: CUDA device IDs (default: `"1"`)
- `--debug`: Run with only 5 thread pairs for testing
- `--override`: Overwrite existing results
- `--max-response-tokens`: Maximum generation length for Stage 1 responses (default: 2048)

## Output Format

### Reasoning JSON

Each thread pair produces two predictions (violating and compliant):

```json
{
  "mod_comment_id": "fdzc60l",
  "subreddit": "excel",
  "submission_id": "en8cqn",

  "violating": {
    "reasoning_response": "Let me analyze this comment...",
    "clean_answer_response": "(b)",
    "extracted_prediction": "(b)",
    "correct_answer": "(b)",
    "score": 1,
    "answer_options": [...]
  },

  "compliant": {
    "reasoning_response": "Looking at this thread...",
    "clean_answer_response": "(e)",
    "extracted_prediction": "(e)",
    "correct_answer": "(e)",
    "score": 1,
    "answer_options": [...]
  },

  "metadata": {
    "rule": "Close your post by replying...",
    "rule_cluster_id": 5,
    "rule_cluster_label": "civility rules",
    "subreddit_cluster_id": 2,
    "subreddit_cluster_label": "tech communities"
  }
}
```

### Performance JSON

```json
{
  "model": "qwen3-vl-8b-instruct",
  "split": "test",
  "context": "submission-discussion",
  "phrase": "cot",
  "mode": "prefill",

  "metrics": {
    "overall": {
      "total_pairs": 4166,
      "total_threads": 8332,
      "overall_accuracy": 0.815,
      "violating_accuracy": 0.85,
      "compliant_accuracy": 0.78,
      ...
    },

    "per_rule_cluster": {
      "civility rules": {
        "overall_accuracy": 0.88,
        "violating_accuracy": 0.9,
        "compliant_accuracy": 0.85,
        "count": 500
      },
      ...
    },

    "per_subreddit_cluster": {
      "tech communities": {
        "overall_accuracy": 0.82,
        "violating_accuracy": 0.85,
        "compliant_accuracy": 0.79,
        "count": 1000
      },
      ...
    }
  }
}
```

## Extending the Framework

### Adding New Models

Edit `config.py` and add to `VLLM_MODELS` or `API_MODELS`:

```python
VLLM_MODELS = {
    'my-new-model': {
        'hf_path': 'org/model-name',
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': 0.95,
        'trust_remote_code': True,
        'max_model_len': 8192,
        'prefill_mode': 'append'
    }
}
```

### Adding New Context Types

Context strings are parsed as dash-separated flags in `config.parse_context_flags()`.
To add a new flag, edit `VALID_CONTEXT_FLAGS`, `parse_context_flags()`, and the
prompt formatting in `helpers._build_question_text()`:

```python
VALID_CONTEXT_FLAGS = {
    'none', 'submission', 'media', 'discussion', 'user', 'my_flag'
}
```

### Adding New Phrases

Edit `config.py` and add to `PHRASES`:

```python
PHRASES = {
    'my_phrase': 'Custom instruction text here.'
}
```

## Notes

- API evaluation currently uses OpenAI Flex for Stage 1 reasoning and a local
  Qwen3-VL model for Stage 2 answer extraction.
- Results and logs are grouped by `{model}/{split}/{context}/{phrase}_{mode}`;
  baseline runs always use a `baseline/` directory because `--mode` is ignored
  for the empty baseline phrase.
