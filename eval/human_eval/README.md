# Human Evaluation Protocol

Reproduces the 100-instance human audit reported in the paper (§5.4), which
showed **96% agreement** between PluRule's automated labels and human
annotators.

> In the paper, three authors independently labeled 100 moderator comments.
> 85 received unanimous labels, 12 had 2-of-3 majority agreement, and 3 were
> adjudicated. The pipeline's labels matched on **100% of unanimous cases,
> 66.7% of majority cases, and 100% of adjudicated cases**.

## Prerequisites

- The **clustered hydrated datasets** from Stage 10 must exist in `./data/`:
  `{train,val,test}_hydrated_clustered.json.zst`.
- **Google OAuth2 client secrets** at
  `credentials/client_secret_<…>.apps.googleusercontent.com.json`.
  Create one in the Google Cloud Console with *Forms API* + *Drive API*
  enabled. The form scripts auto-detect exactly one
  `credentials/client_secret_*.json` file and fail clearly if none or multiple
  are present.
- On first run, each script opens a browser window for OAuth consent and
  caches the token at `credentials/token.json`.

## Scripts (run in order)

### 1. `1_create_forms.py` — sample + generate forms

Loads all three clustered splits, samples **100 moderator comments** that
are:

- From **English** subreddits (`lang == en` after normalization)
- Stratified **uniformly across rule clusters** (best-effort across subreddit
  clusters)
- From **unique subreddits** (no subreddit used twice)

Creates **two Google Forms of 50 questions each** (title prefixed with the
current date). Each question shows the subreddit, its rules, the moderator
comment, and a CHECKBOX with the shuffled rule options + an "Other" field
for notes.

```bash
python eval/human_eval/1_create_forms.py
```

**Seed**: `RANDOM_SEED = 42` (paper). Change in the script if you need a
different sample.

**Output**:
`data/evaluation/stage11_human_evaluation_metadata.json` — records the form
IDs, public URLs, sampled candidates, and the ground-truth rule mapped to
each question index. Keep this file; steps 2 and 3 need it.

Share the two public form URLs (printed at the end) with your annotators.

### 2. `2_retrieve_responses.py` — pull responses back

Once annotators have submitted, fetch every response from both forms,
reconcile them by annotator order (response #1 in Form 1 and Form 2 are
assumed to be the same person), and compute majority agreement.

```bash
python eval/human_eval/2_retrieve_responses.py
```

**Output**:
`data/evaluation/stage11_human_annotations.json` — per-question
`majority_answers` (labels with ≥ 2/3 votes), per-annotator answer dumps,
and raw `answer_vote_counts`.

### Adjudication (manual, optional)

For questions where no label got a majority (i.e. all three annotators
disagreed), open
`data/evaluation/stage11_human_annotations.json` and add an
`adjudicated_answers` list next to those questions — the label(s) chosen
after a second reading by one annotator:

```jsonc
{
  "question_index": 42,
  "majority_answers": [],
  "adjudicated_answers": ["No low-effort posts"],
  …
}
```

In the paper, 3 of 100 questions needed this pass.

### 3. `3_evaluate_predictions.py` — score the pipeline

Builds ground truth from `majority_answers` (falling back to
`adjudicated_answers`), compares each question's `predicted_answer` (the
pipeline's label) against it, and reports accuracy overall + by agreement
level + by rule cluster + by subreddit cluster.

```bash
python eval/human_eval/3_evaluate_predictions.py
```

**Output**:
`data/evaluation/stage11_evaluation_results.json` plus a console summary
matching the paper's breakdown.

## Notes

- Question filenames still begin with `stage11_*` for continuity with earlier
  pipeline outputs — these scripts used to live at `pipeline/11{a,b,c}_*.py`.
- The Google API scripts use the cached `credentials/token.json` after the
  initial OAuth consent flow.
- Forms API writes are rate-limited; creating two 50-question forms takes
  a minute or two.
