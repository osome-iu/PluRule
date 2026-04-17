#!/bin/bash
# Pipeline stages 3-8: Rule matching, comment collection, tree building, submission collection, media collection, and dataset creation

# set -e  # Exit on any error

# echo "=================================="
# echo "Running Pipeline Stages 3-8"
# echo "=================================="
# echo ""

# echo "Stage 3: Matching rules (phase 2 only)..."
# python pipeline/3_match_rules.py --phase2-only
# echo "Stage 3 complete"
# echo ""

# echo "Stage 4: Collecting submission comments..."
# python pipeline/4_collect_submission_comments.py
# echo "Stage 4 complete"
# echo ""

# echo "Stage 5: Building trees and threads..."
# python pipeline/5_build_trees_and_threads.py
# echo "Stage 5 complete"
# echo ""

# echo "Stage 6: Collecting submissions..."
# python pipeline/6_collect_submissions.py
# echo "Stage 6 complete"
# echo ""

# echo "Stage 7: Collecting media..."
# python pipeline/7_collect_media.py
# echo "Stage 7 complete"
# echo ""

# echo "Stage 8: Creating final datasets..."
# python pipeline/8_create_dehydrated_dataset.py
# echo "Stage 8 complete"
# echo ""

echo "=================================="
echo "Pipeline stages 3-8 completed successfully!"
echo "=================================="


# Full Analysis Pipeline: Embed -> Cluster -> Label -> Plot

# Usage:
#   ./run_pipeline_stages_3to8.sh              # Run full pipeline
#   ./run_pipeline_stages_3to8.sh --skip-embed # Skip embedding step
#   ./run_pipeline_stages_3to8.sh --skip-cluster # Skip clustering step

set -e  # Exit on any error

# Parse arguments
SKIP_EMBED=false
SKIP_CLUSTER=false
SKIP_LABEL=false
SKIP_PLOT=false

for arg in "$@"; do
    case $arg in
        --skip-embed)
            SKIP_EMBED=true
            ;;
        --skip-cluster)
            SKIP_CLUSTER=true
            ;;
        --skip-label)
            SKIP_LABEL=true
            ;;
        --skip-plot)
            SKIP_PLOT=true
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "REDDIT MOD COLLECTION PIPELINE - CLUSTERING ANALYSIS"
echo "================================================================================"
echo ""

# Step 1: Embed (Stage 9a)
if [ "$SKIP_EMBED" = false ]; then
    echo "================================================================================"
    echo "STEP 1/4: EMBEDDING (Stage 9a)"
    echo "================================================================================"
    conda run -n reddit-mod-pipeline python pipeline/9a_embed_clusters.py
    if [ $? -ne 0 ]; then
        echo "Embedding failed!"
        exit 1
    fi
    echo "Embedding complete"
    echo ""
else
    echo "Skipping embedding step"
    echo ""
fi

# Step 2: Cluster (Stage 9b - grid search + apply best)
if [ "$SKIP_CLUSTER" = false ]; then
    echo "================================================================================"
    echo "STEP 2/4: CLUSTERING (Stage 9b - Grid Search + Apply Best)"
    echo "================================================================================"
    conda run -n reddit-mod-pipeline python pipeline/9b_cluster_embeddings.py
    if [ $? -ne 0 ]; then
        echo "Clustering failed!"
        exit 1
    fi
    echo "Clustering complete"
    echo ""
else
    echo "Skipping clustering step"
    echo ""
fi

# Step 3: Label clusters (Stage 9c)
if [ "$SKIP_LABEL" = false ]; then
    echo "================================================================================"
    echo "STEP 3/4: LABELING CLUSTERS (Stage 9c)"
    echo "================================================================================"
    conda run -n reddit-mod-pipeline python pipeline/9c_label_clusters.py
    if [ $? -ne 0 ]; then
        echo "Labeling failed!"
        exit 1
    fi
    echo "Labeling complete"
    echo ""
else
    echo "Skipping labeling step"
    echo ""
fi

# Step 4: Plot
if [ "$SKIP_PLOT" = false ]; then
    echo "================================================================================"
    echo "STEP 4/4: PLOTTING"
    echo "================================================================================"
    conda run -n reddit-mod-pipeline python eval/plot_clusters.py
    if [ $? -ne 0 ]; then
        echo "Plotting failed!"
        exit 1
    fi
    echo "Plotting complete"
    echo ""
else
    echo "Skipping plotting step"
    echo ""
fi

echo "================================================================================"
echo "FULL PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Outputs:"
echo "  - Subreddit Embeddings: output/embeddings/all_subreddit_embeddings.tsv"
echo "  - Rule Embeddings: output/embeddings/all_rule_embeddings.tsv"
echo "  - Metadata: output/embeddings/all_subreddit_metadata.tsv, output/embeddings/all_rule_metadata.tsv"
echo "  - Grid search: output/clustering/*_grid_search_results.json"
echo "  - Labels: output/clustering/*_cluster_labels.json"
echo "  - Analysis: output/clustering/*_cluster_analysis.txt"
echo "  - Plots: output/clustering/*_clusters_2d.png"
echo ""
