# PluRule

**PluRule** is a multilingual, multimodal benchmark for Reddit rule-violation
detection: 13,371 discussion instances drawn from the Pushshift archives,
each pairing a rule-violating thread with a compliant thread from the same
submission, labeled against the community's own rules.

This repository contains the full construction pipeline, the scripts used to
hydrate the released dataset from IDs, and the evaluation harness used in the
paper.

> **Paper:** *PluRule: A Multilingual, Multimodal Benchmark for Rule-Aware
> Moderation on Reddit* <!-- TODO: arXiv link when processing completes -->

## At a glance

| Split | Instances | Comments | Images | Subreddits / Clusters | Rules / Clusters | Languages |
|---|---:|---:|---:|---:|---:|---:|
| Train | 9,155 | 51,968 | 2,077 | 861 / 25 | 1,336 / 27 | 9 |
| Val   | 1,382 |  7,631 |   376 | 537 / 25 |   586 / 27 | 9 |
| Test  | 2,834 | 13,076 | 1,190 | **1,989 / 25** | 2,039 / 27 | 9 |
| **Total** | **13,371** | **72,675** | **3,643** | **1,989 / 25** | **2,885 / 27** | **9** |

Every instance contains (a) a root-to-leaf discussion thread where a
moderator cited a rule on the leaf comment, (b) a compliant sibling thread
from the same submission, (c) the submission itself with any images, and
(d) the subreddit's full rule set.

## What do you want to do?

### ▶︎ Run the benchmark on the released dataset

Start here if you want to evaluate a model on PluRule.

1. Grab the three dehydrated split files from
   <!-- TODO: HuggingFace link -->`<HF repo>` and place them under `./data/`.
2. Follow **[`hydrate/README.md`](hydrate/README.md)** to fill in comments,
   submissions, and media from the Pushshift archives (~a few hours, no GPU).
3. Run your model through **[`eval/README.md`](eval/README.md)** — supports
   vLLM (Qwen-VL, LLaVA, Llama-Vision) and API models (Claude, GPT-4V) out
   of the box.

### ▶︎ Rebuild PluRule from scratch

Start here if you want to reproduce the dataset end to end, tweak
thresholds, or extend the pipeline.

Follow **[`pipeline/README.md`](pipeline/README.md)**. Budget 1–2 days and
multiple GPUs: embedding matcher (Qwen3-Embedding-8B), LLM judge
(Qwen3-30B-A3B-Instruct), and cluster labeler (Qwen3-30B-A3B-Thinking) are
all run locally via vLLM.

### ▶︎ Reproduce the human evaluation

See **[`eval/human_eval/`](eval/human_eval/)** for the Google Forms
annotation protocol used in Section 5.4 of the paper (96% overall
agreement with the pipeline's labels on a 100-instance audit).

## Install

```bash
git clone https://github.com/<org>/PluRule.git
cd PluRule

# Pick the env that matches your goal:
conda env create -f environment-hydrate.yml   # minimal, hydration only (no GPU)
conda env create -f environment-pipeline.yml  # end-to-end reconstruction (GPUs)
conda env create -f environment-eval.yml      # benchmark evaluation (GPU or API keys)
```

For API-model evaluation, copy `credentials/.env.template` to
`credentials/.env` and fill in your `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

## Repo layout

```
PluRule/
├── hydrate/          # 3 scripts to reconstitute the released dataset
├── pipeline/         # end-to-end reconstruction from Pushshift (paper §5)
├── eval/             # benchmark evaluation harness
│   └── human_eval/   # human annotation reproduction
├── utils/            # shared helpers (zst I/O, Pushshift torrent, media, …)
├── config.py         # base paths + thresholds (edit before running)
├── credentials/      # API key templates (.env, Reddit, Google)
├── environment-hydrate.yml    # hydration-only conda env
├── environment-pipeline.yml   # reconstruction conda env
└── environment-eval.yml       # evaluation conda env
```

## Citing

```bibtex
@misc{plurule2025,
  title  = {PluRule: A Multilingual, Multimodal Benchmark for Rule-Aware
            Moderation on Reddit},
  author = {TODO},
  year   = {2025},
  note   = {arXiv preprint},
}
```
<!-- TODO: replace with final BibTeX once arXiv ID is assigned -->

## License

TBA. Pipeline code will be released under an OSS license; the dataset is
derived from publicly archived Pushshift data and will be distributed under
terms consistent with that source.
