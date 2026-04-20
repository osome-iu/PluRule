"""
Simple configuration for Reddit mod collection pipeline.

Edit the base directories below for your environment.
All other paths are generated automatically based on data flow.
"""

import os
import multiprocessing

# =============================================================================
# BASE CONFIGURATION - Override via environment variables or edit here.
# =============================================================================
# Defaults: BASE_DATA = repo root; PUSHSHIFT_DATA = <BASE_DATA>/data/pushshift.
# Override with:
#   export PLURULE_BASE_DATA=/your/working/dir
#   export PLURULE_PUSHSHIFT_DATA=/your/pushshift/mirror

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_DATA = os.environ.get("PLURULE_BASE_DATA", _REPO_ROOT)
PUSHSHIFT_DATA = os.environ.get(
    "PLURULE_PUSHSHIFT_DATA", os.path.join(BASE_DATA, "data", "pushshift")
)
# Legacy Pushshift-dumps location (Stage 0 when fetching RC_*/RS_* style dumps).
# Keep consistent with PUSHSHIFT_DATA by default.
REDDIT_DATA = os.environ.get("PLURULE_REDDIT_DATA", PUSHSHIFT_DATA)

CREDENTIALS_DIR = os.path.join(_REPO_ROOT, "credentials")

# Processing settings
DATE_RANGE = ("2005-12", "2023-02")  # (start, end) inclusive PushshiftDumps
MIN_RULES_FOR_MATCHING = 2  # Minimum rules needed for semantic matching (skip subreddits with ≤1 rule)
GOLD_PERCENTILE = 99.2  # Top 0.8% of similarity scores considered gold matches (Stage 3 Phase 2)
AMBIGUOUS_PERCENTILE = 98  # Top 2% of similarity scores considered ambiguous matches (Stage 3 Phase 2)
MIN_MATCHED_COMMENTS = 1 # Minimum matched comments for subreddit inclusion in Stage 3
MAX_MATCHED_COMMENTS = 500  # Max sample size for matched comments in Stage 3

# Stage 8: Dataset split configuration
# Note: No minimum threshold - all subreddits with ≥1 pair are included
# Split logic per subreddit:
#   n=1: 1 test, 0 val, 0 train
#   n=2: 1 test, 0 val, 1 train
#   3≤n<10: 1 test, 1 val, (n-2) train
#   n≥10: 10% test, 10% val, 80% train (rounded, min 1 each)

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"  # Model used in Stage 3 for semantic matching
# Auto-detect number of CPU cores (use all available cores)
PROCESSES = multiprocessing.cpu_count()

# Alternative: Use 75% of available cores to leave some for system
# PROCESSES = max(1, int(multiprocessing.cpu_count() * 0.75))

# =============================================================================
# DATA FLOW MAPPING - Shows what each stage produces and consumes
# =============================================================================

DATA_FLOW = {
    # Phase 1: Data Collection
    'stage0_download_data': {
        'name': 'Download Reddit Data from Internet Archive',
        'script': '0_download_data.py',
        'input_paths': [],  # No inputs - downloads from internet
        'output_dir': 'reddit_data',
        'produces': [
            'comments/YYYY/RC_*.zst',  # Reddit comment files organized by year
            'submissions/YYYY/RS_*.zst',  # Reddit submission files organized by year
            '../logs/stage0_download_log.json'  # actually written to PATHS['logs']
        ]
    },

    'stage1_mod_comments': {
        'name': 'Collect Moderator Comments from Pushshift',
        'script': '1_collect_mod_comments.py',
        'input_paths': [],  # Uses Pushshift data directly
        'output_dir': 'top_subreddits',
        'produces': [
            '{subreddit}_mod_comments.jsonl.zst',  # in PATHS['top_subreddits'], one per subreddit
            '../../data/stage1_subreddit_mod_comment_rankings.json'  # actually written to PATHS['data']
        ],
        'notes': 'Reads Pushshift subreddit files, filters mod comments. Replaces old Stage 1 + Stage 3.'
    },

    'stage2_top_sfw': {
        'name': 'Get SFW Subreddits with Minimum Mod Comments',
        'script': '2_get_top_sfw_subreddits.py',
        'input_files': ['stage1_subreddit_mod_comment_rankings.json'],
        'output_dir': 'data',
        'produces': ['stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json'],
        'notes': 'Uses Reddit API to filter NSFW, collect subreddit metadata and rules'
    },

    # Phase 2: Comment Matching
    # NOTE: Stage 3 (filter_and_consolidate) is now obsolete - Stage 1 directly outputs to top_subreddits/

    'stage3_match_rules': {
        'name': 'Match Comments to Rules (2-Phase: Similarity Matrices + Global Thresholds)',
        'script': '3_match_rules.py',
        'helper_scripts': ['utils/match_rules_bucket.py'],
        'input_paths': ['top_subreddits'],
        'input_files': [
            'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json'
        ],
        'output_dir': 'matched_comments',
        'produces': [
            '{subreddit}_match.jsonl.zst',  # in PATHS['matched_comments']
            '{subreddit}_stats.json',  # in PATHS['matched_comments']
            '{subreddit}_similarity_matrix.pt',  # in PATHS['matched_comments']
            'cosine_similarity_distribution_all_percentiles.png',  # in PATHS['matched_comments']
            '../../data/stage3_matching_summary.json',  # actually written to PATHS['data']
            '../../data/stage3_subreddit_submission_ids.json'  # actually written to PATHS['data']
        ],
        'notes': 'Phase 1: Create similarity matrices using vLLM embeddings. Phase 2: Apply global percentile thresholds for matching. Filters ambiguous matches, ranks by JSD.'
    },

    # Phase 3: Thread Construction
    'stage4_collect_submission_comments': {
        'name': 'Collect and Organize Submission Comments from Pushshift',
        'script': '4_collect_submission_comments.py',
        'input_paths': [],  # Uses Pushshift data directly
        'input_files': ['stage3_subreddit_submission_ids.json'],
        'output_dir': 'organized_comments',
        'produces': [
            '{subreddit}/submission_{submission_id}.pkl',  # one file per submission, inside per-subreddit subdir
            '../../data/stage4_submission_comment_collection_stats.json'  # actually written to PATHS['data']
        ],
        'notes': '2-pass per subreddit: filter with process_zst_file_multi → deduplicate with [removed]/[deleted] preservation'
    },

    'stage5_build_trees_and_threads': {
        'name': 'Build Comment Trees and Discussion Threads',
        'script': '5_build_trees_and_threads.py',
        'input_paths': ['organized_comments', 'matched_comments'],
        'input_files': [
            'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json',
            'stage3_matching_summary.json'
        ],
        'output_dir': 'comment_trees',
        'alternate_output_dirs': ['discussion_threads'],  # Also outputs here
        'produces': [
            'comment_trees/{subreddit}_comment_trees.pkl',  # in PATHS['comment_trees']
            'discussion_threads/{subreddit}_discussion_threads.pkl',  # in PATHS['discussion_threads']
            '../../data/stage5_trees_and_threads_summary.json'  # actually written to PATHS['data']
        ],
        'notes': 'Builds trees (parent-child, depth levels), creates moderated/unmoderated pairs, requires 500+ pairs, ranks by JSD'
    },

    # Phase 4: Dataset Finalization
    'stage6_collect_submissions': {
        'name': 'Collect Submissions from Discussion Threads',
        'script': '6_collect_submissions.py',
        'input_paths': ['reddit_submissions'],  # Pushshift submissions
        'input_files': ['stage5_trees_and_threads_summary.json'],
        'output_dir': 'submissions',
        'produces': [
            '{subreddit}_submissions.zst',  # in PATHS['submissions']
            '../../data/stage6_submission_collection_stats.json'  # actually written to PATHS['data']
        ],
        'notes': '3-phase: extract IDs from stage 5 summary → process RS files from Pushshift → consolidate by subreddit'
    },

    'stage7_collect_media': {
        'name': 'Collect Media for Submissions',
        'script': '7_collect_media.py',
        'input_paths': ['submissions'],
        'input_files': ['stage6_submission_collection_stats.json'],
        'output_dir': 'media',
        'produces': [
            '{subreddit}/{submission_id}_{media_id}_{source}.{ext}',  # Downloaded media files in PATHS['media']
            '../../data/stage7_media_collection_stats.json',  # actually written to PATHS['data']
            '../../data/stage7_successful_submission_ids.json'  # actually written to PATHS['data']
        ],
        'notes': 'Priority: media_metadata → url → oembed → preview. Skips NSFW/crosspost/URL-only selfposts. Validates file types.'
    },

    'stage8_create_datasets': {
        'name': 'Create Final Datasets (Hydrated + Dehydrated splits)',
        'script': '8_create_dehydrated_dataset.py',
        'input_paths': ['discussion_threads', 'comment_trees', 'submissions', 'media'],
        'input_files': [
            'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json',
            'stage7_successful_submission_ids.json',
            'stage1_subreddit_mod_comment_rankings.json',
            'stage3_matching_summary.json',
            'stage5_trees_and_threads_summary.json',
            'stage6_submission_collection_stats.json',
            'stage7_media_collection_stats.json'
        ],
        'output_dir': 'data',
        'produces': [
            'train_hydrated.json.zst',
            'val_hydrated.json.zst',
            'test_hydrated.json.zst',
            'train_dehydrated.json.zst',
            'val_dehydrated.json.zst',
            'test_dehydrated.json.zst',
            'test_hydrated.json',  # uncompressed test split
            'stage8_final_datasets_stats.json',
            'stage8_llm_verification_results.json',
            'stage8_thread_distribution_analysis.json'
        ],
        'notes': 'Adaptive train/val/test splits per subreddit + Qwen3-30B LLM judge verification. Hydrated: full objects. Dehydrated: IDs with [NEEDS_HYDRATION] placeholders.'
    },

    # Phase 5: Clustering
    'stage9a_embed_clusters': {
        'name': 'Embed Subreddits and Rules for Clustering',
        'script': '9a_embed_clusters.py',
        'input_files': [
            'train_hydrated.json.zst',
            'val_hydrated.json.zst',
            'test_hydrated.json.zst',
            'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json'
        ],
        'output_dir': 'embeddings',
        'produces': [
            'all_subreddit_embeddings.tsv',
            'all_subreddit_metadata.tsv',
            'all_rule_embeddings.tsv',
            'all_rule_metadata.tsv'
        ],
        'notes': 'Creates embeddings using vLLM for subreddits (title+description) and rules (rule_comprehensive text).'
    },

    'stage9b_cluster_embeddings': {
        'name': 'Cluster Embeddings with UMAP + HDBSCAN',
        'script': '9b_cluster_embeddings.py',
        'input_paths': ['embeddings'],
        'output_dir': 'clustering',
        'alternate_output_dirs': ['embeddings'],  # reduced TSVs + updated metadata go here
        'produces': [
            'clustering/subreddit_grid_search_results.json',
            'clustering/rule_grid_search_results.json',
            'embeddings/all_subreddit_embeddings_reduced.tsv',
            'embeddings/all_rule_embeddings_reduced.tsv',
            'embeddings/all_subreddit_metadata.tsv',  # MUTATED in place: cluster_id, cluster_label columns added
            'embeddings/all_rule_metadata.tsv'  # MUTATED in place: cluster_id, cluster_label columns added
        ],
        'notes': 'Grid search for optimal UMAP + HDBSCAN parameters. Reduced embeddings + augmented metadata are written back into PATHS["embeddings"]; only grid_search results live in PATHS["clustering"].'
    },

    'stage9c_label_clusters': {
        'name': 'Label Clusters with LLM',
        'script': '9c_label_clusters.py',
        'input_paths': ['embeddings', 'clustering'],
        'output_dir': 'clustering',
        'produces': [
            'subreddit_cluster_labels.json',
            'rule_cluster_labels.json',
            'subreddit_cluster_analysis.txt',
            'rule_cluster_analysis.txt'
        ],
        'notes': 'Uses LLM to generate semantic labels for each cluster via majority voting.'
    },

    'stage9d_reapply_cluster_labels': {
        'name': 'Reapply/Override Cluster Labels',
        'script': '9d_reapply_cluster_labels.py',
        'input_paths': ['embeddings', 'clustering'],
        'output_dir': 'clustering',
        'produces': [
            'subreddit_cluster_labels.json',
            'rule_cluster_labels.json'
        ],
        'notes': 'Optional manual step to apply label overrides and merge clusters.'
    },

    # Phase 6: Final Assignment and Evaluation
    'stage10_assign_cluster_labels': {
        'name': 'Assign Cluster Labels to Dataset',
        'script': '10_assign_cluster_labels.py',
        'input_paths': ['embeddings'],
        'input_files': [
            'train_hydrated.json.zst',
            'val_hydrated.json.zst',
            'test_hydrated.json.zst'
        ],
        'output_dir': 'data',
        'produces': [
            'train_hydrated_clustered.json.zst',
            'val_hydrated_clustered.json.zst',
            'test_hydrated_clustered.json.zst',
            'train_dehydrated_clustered.json.zst',
            'val_dehydrated_clustered.json.zst',
            'test_dehydrated_clustered.json.zst',
            'test_hydrated_clustered.json',  # uncompressed
            'stage10_cluster_assignment_stats.json',
            'stage10_dataset_stats_table.tex'  # LaTeX table for paper
        ],
        'notes': 'Assigns cluster labels to all thread pairs in the dataset based on embedding metadata.'
    }

    # Human evaluation scripts live in eval/human_eval/ (not pipeline stages).
}

# =============================================================================
# AUTO-GENERATED PATHS - Don't edit these
# =============================================================================

def _generate_paths():
    """Generate all paths based on base directories and data flow."""
    paths = {
        # Input data sources
        'reddit_comments': f"{REDDIT_DATA}/comments",
        'reddit_submissions': f"{REDDIT_DATA}/submissions",
        'reddit_data': f"{REDDIT_DATA}",  # Base directory for downloaded data

        # Base output directories
        'data': f"{BASE_DATA}/data",
        'logs': f"{BASE_DATA}/logs",

        # Stage output directories (auto-generated from DATA_FLOW)
        'mod_comments': f"{BASE_DATA}/data/mod_comments",
        'top_subreddits': f"{BASE_DATA}/output/top_subreddits",
        'matched_comments': f"{BASE_DATA}/output/matched_comments",
        'matched_comments_sample': f"{BASE_DATA}/output/matched_comments_sample",
        'submission_comments': f"{BASE_DATA}/data/submission_comments",
        'organized_comments': f"{BASE_DATA}/output/organized_comments",
        'comment_trees': f"{BASE_DATA}/output/comment_trees",
        'discussion_threads': f"{BASE_DATA}/output/discussion_threads",
        'submissions': f"{BASE_DATA}/output/submissions",
        'media': f"{BASE_DATA}/output/media",
        'final_dataset': f"{BASE_DATA}/output/final_dataset",
        'embeddings': f"{BASE_DATA}/output/embeddings",
        'clustering': f"{BASE_DATA}/output/clustering",
        'evaluation': f"{BASE_DATA}/data/evaluation"
    }

    return paths

PATHS = _generate_paths()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_stage_info(stage_num):
    """Get information for a specific stage number (0-13)."""
    stage_key = f"stage{stage_num}_" + list(DATA_FLOW.keys())[stage_num].split('_', 1)[1]
    return DATA_FLOW.get(stage_key)

def get_input_paths_for_stage(stage_num):
    """Get resolved input paths for a stage."""
    stage_info = get_stage_info(stage_num)
    if not stage_info:
        return []

    input_paths = []

    # Add directory paths
    for path_key in stage_info.get('input_paths', []):
        input_paths.append(PATHS[path_key])

    # Add specific files
    for file_name in stage_info.get('input_files', []):
        # Substitute template variables
        resolved_file_name = file_name.format(
            MIN_MATCHED_COMMENTS=MIN_MATCHED_COMMENTS
        )
        input_paths.append(os.path.join(PATHS['data'], resolved_file_name))

    return input_paths

def get_output_path_for_stage(stage_num):
    """Get resolved output path for a stage."""
    stage_info = get_stage_info(stage_num)
    if not stage_info:
        return None

    output_dir = stage_info.get('output_dir')
    return PATHS.get(output_dir)

def create_directories():
    """Create necessary output directories (excludes read-only input paths)."""
    # Skip input directories that should already exist
    skip_paths = {'reddit_comments', 'reddit_submissions', 'reddit_data'}

    for name, path in PATHS.items():
        if name not in skip_paths:
            os.makedirs(path, exist_ok=True)

def validate_stage_inputs(stage_num):
    """Check if inputs exist for a stage."""
    input_paths = get_input_paths_for_stage(stage_num)

    for path in input_paths:
        if os.path.isfile(path):
            if not os.path.exists(path):
                return False, f"Missing file: {path}"
        elif os.path.isdir(path):
            if not os.path.exists(path) or not os.listdir(path):
                return False, f"Missing or empty directory: {path}"
        else:
            return False, f"Path doesn't exist: {path}"

    return True, "All inputs available"

def print_pipeline_status():
    """Print status of entire pipeline."""
    print("Reddit Mod Collection Pipeline Status")
    print("=" * 80)
    print()

    for i in range(0, 11):  # Now 0-10 stages (including stage 10)
        stage_info = get_stage_info(i)
        if stage_info:
            valid, msg = validate_stage_inputs(i)
            output_path = get_output_path_for_stage(i)
            output_exists = os.path.exists(output_path) if output_path else False

            status = "✓" if valid else "✗"
            output_status = "✓" if output_exists else "✗"

            print(f"Stage {i:2d}: {stage_info['name']}")
            print(f"         Script: {stage_info.get('script', 'N/A')}")
            print(f"         Input: {status} | Output: {output_status}")
            if not valid:
                print(f"         Issue: {msg}")
            if stage_info.get('notes'):
                print(f"         Notes: {stage_info['notes']}")
            print()

    print("=" * 80)

if __name__ == "__main__":
    # When run directly, show pipeline status
    create_directories()
    print_pipeline_status()