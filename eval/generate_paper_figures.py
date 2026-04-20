#!/usr/bin/env python3
"""
Generate paper figures from evaluation results.

Subcommands:
    forest          Cluster forest plot (accuracy + 95% CI per cluster)
    stacked         Cluster stacked bar plot (violating/overall/compliant)
    correlation     Cluster correlation scatter (accuracy vs. cluster size)
    language-grid   Appendix language grid: distribution + 4 model pairs
    all             Generate all figures

Usage:
    python eval/generate_paper_figures.py all
    python eval/generate_paper_figures.py forest --model gpt5.2-high --split test
    python eval/generate_paper_figures.py correlation --show-regression
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plotting_config import (
    create_two_column_figure, save_figure, style_clean_axis, PUBLICATION_DPI
)
from eval.eval_utils import (
    get_eval_dir, load_performance, extract_cluster_metrics,
    extract_three_metrics, load_cluster_size_stats,
)


# ============================================================================
# COLORS
# ============================================================================

COLOR_SUBREDDIT = '#FF4500'  # Orange Red
COLOR_RULE = '#336699'       # Lapis Lazuli


# ============================================================================
# FOREST PLOT
# ============================================================================

def _plot_forest_panel(ax, labels, values, cis, color, xlabel):
    """Draw one panel of the forest plot (shared between subreddit and rule)."""
    y_pos = np.arange(len(labels))

    # Faint horizontal lines at each cluster position
    for i in y_pos:
        ax.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

    ax.scatter(values, y_pos, color=color, s=40, marker='s', zorder=3)

    # Accuracy values inside squares
    for i, val in enumerate(values):
        ax.text(val, i, f'{val:.0f}', ha='center', va='center',
                fontsize=5.5, color='white', fontweight='bold', zorder=4)

    # CI error bars with caps
    for i, (ci_low, ci_high) in enumerate(cis):
        ax.hlines(i, ci_low, ci_high, color=color, linewidth=1, zorder=2)
        ax.vlines([ci_low, ci_high], i - 0.25, i + 0.25, color=color, linewidth=1, zorder=2)

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    style_clean_axis(ax, grid_axis='x')


def plot_forest(model, split, context, metric, phrase='baseline', mode='prefill'):
    """Forest plot showing accuracy with 95% CI by cluster."""
    perf_data = load_performance(model, split, context, phrase, mode)

    sub_data = extract_cluster_metrics(perf_data, 'subreddit', metric)
    rule_data = extract_cluster_metrics(perf_data, 'rule', metric)

    if not sub_data or not rule_data:
        print("No cluster metrics found")
        return 1

    sub_labels, sub_accs, sub_ci_low, sub_ci_high = zip(*sub_data)
    sub_cis = list(zip(sub_ci_low, sub_ci_high))
    rule_labels, rule_accs, rule_ci_low, rule_ci_high = zip(*rule_data)
    rule_cis = list(zip(rule_ci_low, rule_ci_high))

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='barplot')

    _plot_forest_panel(ax_left, sub_labels, sub_accs, sub_cis, COLOR_SUBREDDIT, 'Accuracy (%)')
    _plot_forest_panel(ax_right, rule_labels, rule_accs, rule_cis, COLOR_RULE, 'Accuracy (%)')

    # Subplot labels
    for ax, label in zip([ax_left, ax_right], ['a', 'b']):
        ax.text(0.98, 0.02, f'({label})', transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    fig.subplots_adjust(left=0.11, right=0.98, top=0.99, bottom=0.1, wspace=0.33)

    phrase_suffix = 'baseline' if phrase == 'baseline' else f'{phrase}_{mode}'
    filename = f"cluster_forest_{model}_{split}_{context}_{phrase_suffix}_{metric}"
    plots_dir = get_eval_dir() / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print("Done: cluster forest plot")
    return 0


# ============================================================================
# STACKED BAR PLOT
# ============================================================================

def _plot_stacked_panel(ax, labels, violating, overall, compliant, color, xlabel):
    """Draw one panel of the stacked bar plot."""
    y_pos = np.arange(len(labels))

    # Draw bars back to front: compliant (lightest) -> overall -> violating (darkest)
    ax.barh(y_pos, compliant, height=0.8, color=color, alpha=0.2, edgecolor='none', zorder=2)
    ax.barh(y_pos, overall, height=0.8, color=color, alpha=0.55, edgecolor='none', zorder=3)
    ax.barh(y_pos, violating, height=0.8, color=color, alpha=1.0, edgecolor='none', zorder=4)

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    style_clean_axis(ax, grid_axis='x')


def plot_stacked(model, split, context, phrase='baseline', mode='prefill'):
    """Stacked bar plot showing violating, overall, and compliant accuracy by cluster."""
    perf_data = load_performance(model, split, context, phrase, mode)

    sub_data = extract_three_metrics(perf_data, 'subreddit')
    rule_data = extract_three_metrics(perf_data, 'rule')

    if not sub_data or not rule_data:
        print("No cluster metrics found")
        return 1

    sub_labels, sub_vio, sub_overall, sub_comp = zip(*sub_data)
    rule_labels, rule_vio, rule_overall, rule_comp = zip(*rule_data)

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='barplot')

    _plot_stacked_panel(ax_left, sub_labels, sub_vio, sub_overall, sub_comp, COLOR_SUBREDDIT, '%')
    _plot_stacked_panel(ax_right, rule_labels, rule_vio, rule_overall, rule_comp, COLOR_RULE, '%')

    # Subplot labels
    for ax, label in zip([ax_left, ax_right], ['a', 'b']):
        ax.text(0.98, 0.02, f'({label})', transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    # Legend with gray swatches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=1.0, label='violating recall'),
        Patch(facecolor='gray', alpha=0.55, label='accuracy'),
        Patch(facecolor='gray', alpha=0.2, label='compliant recall'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 1.0))

    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.09, wspace=0.33)

    phrase_suffix = 'baseline' if phrase == 'baseline' else f'{phrase}_{mode}'
    filename = f"cluster_stacked_{model}_{split}_{context}_{phrase_suffix}"
    plots_dir = get_eval_dir() / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print("Done: cluster stacked plot")
    return 0


# ============================================================================
# CORRELATION SCATTER PLOT
# ============================================================================

def plot_correlation(model, split, context, metric='overall_accuracy',
                     phrase='baseline', mode='prefill',
                     show_regression=True, cluster_split='all', cluster_type='rule'):
    """Correlation scatter plots: accuracy vs. cluster size metrics."""
    from adjustText import adjust_text

    perf_data = load_performance(model, split, context, phrase, mode)

    print(f"Loading cluster size stats: split={cluster_split}, type={cluster_type}")
    size_stats = load_cluster_size_stats(split=cluster_split, cluster_type=cluster_type)

    # Merge performance + size data
    per_cluster = perf_data['metrics'].get(f'per_{cluster_type}_cluster', {})
    cluster_size_data = size_stats['cluster_stats']
    merged = []

    for name, info in per_cluster.items():
        if metric not in info or name.lower() == 'other':
            continue
        acc = info[metric] * 100
        ci_key = f'{metric}_ci'
        if ci_key in info:
            ci_low, ci_high = info[ci_key]
            ci_low *= 100
            ci_high *= 100
        else:
            ci_low, ci_high = acc, acc
        if name not in cluster_size_data:
            print(f"  Warning: Cluster '{name}' not found in size stats, skipping")
            continue
        n_sub = cluster_size_data[name]['n_subreddits']
        n_rules = cluster_size_data[name]['n_rules']
        merged.append((name, acc, ci_low, ci_high, n_sub, n_rules))

    if not merged:
        print("No cluster data available")
        return 1

    print(f"  Merged {len(merged)} clusters (excluded 'Other')")

    names, accs, _, _, n_subs, n_rules = zip(*merged)
    accs = np.array(accs)
    n_subs = np.array(n_subs)
    n_rules = np.array(n_rules)

    # Axis limits with 5% padding
    acc_range = accs.max() - accs.min()
    xlim = (accs.min() - 0.05 * acc_range, accs.max() + 0.05 * acc_range)
    sub_range = n_subs.max() - n_subs.min()
    ylim_sub = (n_subs.min() - 0.05 * sub_range, n_subs.max() + 0.05 * sub_range)
    rules_range = n_rules.max() - n_rules.min()
    ylim_rules = (n_rules.min() - 0.05 * rules_range, n_rules.max() + 0.05 * rules_range)

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='barplot')

    # LEFT: Accuracy vs. # Subreddits
    ax_left.scatter(accs, n_subs, color=COLOR_SUBREDDIT, s=20, alpha=0.7, marker='o', zorder=3)
    texts_left = [ax_left.text(accs[i], n_subs[i], name, fontsize=4.5, alpha=0.8)
                  for i, name in enumerate(names)]
    adjust_text(texts_left, ax=ax_left,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))

    rho_sub, p_sub = stats.spearmanr(accs, n_subs)
    ax_left.text(0.05, 0.88, f'\u03c1 = {rho_sub:.3f}\np = {p_sub:.3f}',
                 transform=ax_left.transAxes, fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    if show_regression:
        slope, intercept, *_ = stats.linregress(accs, n_subs)
        x_line = np.linspace(accs.min(), accs.max(), 100)
        ax_left.plot(x_line, slope * x_line + intercept, color=COLOR_SUBREDDIT,
                     linewidth=1.5, alpha=0.6, linestyle='--', zorder=1)

    ax_left.set_xlabel('Accuracy (%)', fontsize=8)
    ax_left.set_ylabel('Number of Subreddits', fontsize=8)
    ax_left.axvline(x=50, color='lightgray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    style_clean_axis(ax_left, grid_axis='both')
    ax_left.set_xlim(xlim)
    ax_left.set_ylim(ylim_sub)

    # RIGHT: Accuracy vs. # Rules
    ax_right.scatter(accs, n_rules, color=COLOR_RULE, s=20, alpha=0.7, marker='o', zorder=3)
    texts_right = [ax_right.text(accs[i], n_rules[i], name, fontsize=4.5, alpha=0.8)
                   for i, name in enumerate(names)]
    adjust_text(texts_right, ax=ax_right,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))

    rho_rules, p_rules = stats.spearmanr(accs, n_rules)
    ax_right.text(0.05, 0.88, f'\u03c1 = {rho_rules:.3f}\np = {p_rules:.3f}',
                  transform=ax_right.transAxes, fontsize=7, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    if show_regression:
        slope, intercept, *_ = stats.linregress(accs, n_rules)
        x_line = np.linspace(accs.min(), accs.max(), 100)
        ax_right.plot(x_line, slope * x_line + intercept, color=COLOR_RULE,
                      linewidth=1.5, alpha=0.6, linestyle='--', zorder=1)

    ax_right.set_xlabel('Accuracy (%)', fontsize=8)
    ax_right.set_ylabel('Number of Rules', fontsize=8)
    ax_right.axvline(x=50, color='lightgray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    style_clean_axis(ax_right, grid_axis='both')
    ax_right.set_xlim(xlim)
    ax_right.set_ylim(ylim_rules)

    # Subplot labels
    ax_left.text(0.02, 0.98, '(a)', transform=ax_left.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='left')
    ax_right.text(0.02, 0.98, '(b)', transform=ax_right.transAxes,
                  fontsize=10, verticalalignment='top', horizontalalignment='left')

    fig.subplots_adjust(left=0.075, right=0.98, top=0.99, bottom=0.10, wspace=0.15)

    phrase_suffix = 'baseline' if phrase == 'baseline' else f'{phrase}_{mode}'
    filename = f"{cluster_type}_cluster_correlation_{model}_{split}_{context}_{phrase_suffix}_{metric}"
    plots_dir = get_eval_dir() / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)

    print(f"Done: cluster correlation plot")
    print(f"  Accuracy vs. Subreddits: \u03c1 = {rho_sub:.3f}, p = {p_sub:.3f}")
    print(f"  Accuracy vs. Rules:      \u03c1 = {rho_rules:.3f}, p = {p_rules:.3f}")
    return 0


# ============================================================================
# LANGUAGE GRID (appendix: 5 rows × 2 columns)
# ============================================================================

LANGUAGE_NAMES = {
    'en': 'English', 'fr': 'French', 'de': 'German', 'pt': 'Portuguese',
    'es': 'Spanish', 'nl': 'Dutch', 'it': 'Italian', 'pl': 'Polish',
    'tr': 'Turkish', 'sv': 'Swedish', 'da': 'Danish', 'el': 'Greek',
    'uk': 'Ukrainian', 'ro': 'Romanian', 'eo': 'Esperanto', 'hu': 'Hungarian',
    'hr': 'Croatian', 'sk': 'Slovak', 'zh': 'Chinese', 'fi': 'Finnish',
    'cs': 'Czech', 'ru': 'Russian', 'no': 'Norwegian', 'sl': 'Slovenian'
}


def _load_language_distribution():
    """Load language distribution from Stage 10 stats."""
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    stats_file = data_dir / 'stage10_cluster_assignment_stats.json'

    if not stats_file.exists():
        raise FileNotFoundError(
            f"Stage 10 stats not found: {stats_file}\n"
            f"Please run: python pipeline/10_assign_cluster_labels.py"
        )

    with open(stats_file) as f:
        stage10_data = json.load(f)

    language_counts = stage10_data['overall_totals']['language_counts']
    sorted_data = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_data


LANGUAGE_GRID_MODELS = [
    ('Qwen3-VL-4B',  'qwen3-vl-4b-instruct',  'qwen3-vl-4b-thinking',  'Instruct', 'Thinking'),
    ('Qwen3-VL-8B',  'qwen3-vl-8b-instruct',  'qwen3-vl-8b-thinking',  'Instruct', 'Thinking'),
    ('Qwen3-VL-30B', 'qwen3-vl-30b-instruct', 'qwen3-vl-30b-thinking', 'Instruct', 'Thinking'),
    ('GPT-5.2',      'gpt5.2-low',            'gpt5.2-high',           'Low',      'High'),
]


def _get_language_accuracies(model, languages):
    """Return (accs, ci_lows, ci_highs) arrays aligned to `languages`. NaN for missing."""
    perf = load_performance(model, 'test', 'submission-media-discussion-user')
    per_lang = perf['metrics'].get('per_language', {})
    accs, ci_lows, ci_highs = [], [], []
    for lang in languages:
        if lang in per_lang and 'overall_accuracy' in per_lang[lang]:
            acc = per_lang[lang]['overall_accuracy'] * 100
            if 'overall_accuracy_ci' in per_lang[lang]:
                cl, ch = per_lang[lang]['overall_accuracy_ci']
                cl *= 100
                ch *= 100
            else:
                cl, ch = acc, acc
        else:
            acc, cl, ch = np.nan, np.nan, np.nan
        accs.append(acc)
        ci_lows.append(cl)
        ci_highs.append(ch)
    return np.array(accs), np.array(ci_lows), np.array(ci_highs)


def _draw_language_forest_panel(ax, accs, ci_lows, ci_highs, color):
    """Draw forest markers + CI whiskers on a shared-y axis (no y-label handling)."""
    n = len(accs)
    y_pos = np.arange(n)
    for i in y_pos:
        ax.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

    valid = ~np.isnan(accs)
    ax.scatter(accs[valid], y_pos[valid], color=color, s=40, marker='s', zorder=3)
    for i in np.where(valid)[0]:
        ax.text(accs[i], i, f'{accs[i]:.0f}', ha='center', va='center',
                fontsize=5.5, color='white', fontweight='bold', zorder=4)
        ax.hlines(i, ci_lows[i], ci_highs[i], color=color, linewidth=1, zorder=2)
        ax.vlines([ci_lows[i], ci_highs[i]], i - 0.25, i + 0.25, color=color, linewidth=1, zorder=2)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    style_clean_axis(ax, grid_axis='x')


def plot_language_grid():
    """5-row appendix figure: distribution + 4 model pairs (instruct/thinking)."""
    import matplotlib.gridspec as gridspec

    print("Loading language distribution...")
    lang_distribution = _load_language_distribution()

    # Collect languages with data in at least one model
    per_model_langs = {}
    for _, left_model, right_model, *_ in LANGUAGE_GRID_MODELS:
        for m in (left_model, right_model):
            perf = load_performance(m, 'test', 'submission-media-discussion-user')
            per_model_langs[m] = set(perf['metrics'].get('per_language', {}).keys())
    all_model_langs = set().union(*per_model_langs.values())

    # Final language list: frequency >= 10 AND present in at least one model
    counts_map = dict(lang_distribution)
    languages = [lang for lang, count in lang_distribution
                 if count >= 10 and lang in all_model_langs]
    counts_arr = [counts_map[l] for l in languages]
    language_labels = [LANGUAGE_NAMES.get(l, l) for l in languages]
    n_langs = len(languages)

    if n_langs == 0:
        print("No languages to plot")
        return 1

    print(f"  {n_langs} languages, {len(LANGUAGE_GRID_MODELS)} model pairs")

    # Figure
    fig = plt.figure(figsize=(6.3, 7))
    gs = gridspec.GridSpec(5, 2, figure=fig,
                           height_ratios=[1.1, 1.0, 1.0, 1.0, 1.0],
                           hspace=0.15, wspace=0.05)

    y_pos = np.arange(n_langs)

    # ---- Row 1: distribution (spans both columns) ----
    ax_dist = fig.add_subplot(gs[0, :])
    ax_dist.barh(y_pos, counts_arr, height=0.8, color=COLOR_SUBREDDIT, edgecolor='none')
    ax_dist.set_xlabel('Number of Instances', fontsize=8)
    ax_dist.set_yticks(y_pos)
    ax_dist.set_yticklabels(language_labels, fontsize=7)
    ax_dist.set_ylim(-0.5, n_langs - 0.5)
    ax_dist.invert_yaxis()
    ax_dist.set_xscale('log')
    style_clean_axis(ax_dist, grid_axis='x')

    # ---- Rows 2-5: forest pairs ----
    forest_left_axes = []
    n_rows = len(LANGUAGE_GRID_MODELS)
    for row_idx, (model_label, left_model, right_model, left_panel_label, right_panel_label) in enumerate(LANGUAGE_GRID_MODELS, start=1):
        ax_left = fig.add_subplot(gs[row_idx, 0])
        ax_right = fig.add_subplot(gs[row_idx, 1], sharey=ax_left)
        forest_left_axes.append((ax_left, model_label))

        accs_l, cl_l, ch_l = _get_language_accuracies(left_model, languages)
        accs_r, cl_r, ch_r = _get_language_accuracies(right_model, languages)

        _draw_language_forest_panel(ax_left, accs_l, cl_l, ch_l, COLOR_SUBREDDIT)
        _draw_language_forest_panel(ax_right, accs_r, cl_r, ch_r, COLOR_RULE)

        # Language labels only on left panel
        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels(language_labels, fontsize=7)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        ax_right.tick_params(axis='y', length=0)

        # Panel label in top-right corner of each panel
        ax_left.text(0.98, 0.92, left_panel_label, transform=ax_left.transAxes,
                     fontsize=8, ha='right', va='top', fontweight='bold')
        ax_right.text(0.98, 0.92, right_panel_label, transform=ax_right.transAxes,
                      fontsize=8, ha='right', va='top', fontweight='bold')

        # X-axis label only on bottom row
        is_bottom = (row_idx == n_rows)
        if is_bottom:
            ax_left.set_xlabel('Accuracy (%)', fontsize=8)
            ax_right.set_xlabel('Accuracy (%)', fontsize=8)
        else:
            plt.setp(ax_left.get_xticklabels(), visible=False)
            plt.setp(ax_right.get_xticklabels(), visible=False)

    fig.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.05,
                        hspace=0.15, wspace=0.05)

    # Shrink distribution row from the bottom so its x-axis/label has breathing room
    dist_gap = 0.03  # figure fraction
    pos = ax_dist.get_position()
    ax_dist.set_position([pos.x0, pos.y0 + dist_gap, pos.width, pos.height - dist_gap])

    # ---- Row labels on far-left margin (after subplots_adjust settles positions) ----
    for ax_left, model_label in forest_left_axes:
        bbox = ax_left.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.01, y_center, model_label, fontsize=9, ha='left', va='center',
                 rotation=90, fontweight='bold')

    # ---- Faint horizontal separator between distribution row and forest block ----
    # Place below the distribution's x-axis label by using its tight bbox
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dist_tb = ax_dist.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    first_forest_ax = forest_left_axes[0][0]
    forest_top = first_forest_ax.get_position().y1
    y_sep = (dist_tb.y0 + forest_top) / 2
    fig.add_artist(Line2D([0, 1], [y_sep, y_sep], transform=fig.transFigure,
                          color='black', linestyle='-', linewidth=0.25, alpha=0.2))

    filename = "language_grid"
    plots_dir = get_eval_dir() / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print(f"Done: language grid plot")
    return 0


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate paper figures from evaluation results'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Shared args for plot commands
    def add_common_args(p):
        p.add_argument('--model', default='gpt5.2-high', help='Model name')
        p.add_argument('--split', default='test', help='Dataset split')
        p.add_argument('--context', default='submission-media-discussion-user', help='Context')
        p.add_argument('--phrase', default='baseline', help='Phrase type')
        p.add_argument('--mode', default='prefill', help='Mode')

    # forest
    p_forest = subparsers.add_parser('forest', help='Cluster forest plot (accuracy + 95%% CI)')
    add_common_args(p_forest)
    p_forest.add_argument('--metric', default='overall_accuracy', help='Accuracy metric')

    # stacked
    p_stacked = subparsers.add_parser('stacked', help='Cluster stacked bar plot')
    add_common_args(p_stacked)

    # correlation
    p_corr = subparsers.add_parser('correlation', help='Cluster correlation scatter')
    add_common_args(p_corr)
    p_corr.add_argument('--metric', default='overall_accuracy', help='Accuracy metric')
    p_corr.add_argument('--cluster-split', default='all', help='Split for cluster sizes')
    p_corr.add_argument('--cluster-type', default='rule', choices=['rule', 'subreddit'])
    p_corr.add_argument('--no-regression', action='store_true', help='Disable regression line')

    # language-grid
    subparsers.add_parser('language-grid', help='Appendix language grid: 5 rows × 2 columns')

    # all
    p_all = subparsers.add_parser('all', help='Generate all figures')
    add_common_args(p_all)
    p_all.add_argument('--metric', default='overall_accuracy', help='Accuracy metric')

    args = parser.parse_args()

    if args.command == 'forest':
        return plot_forest(args.model, args.split, args.context, args.metric, args.phrase, args.mode)
    elif args.command == 'stacked':
        return plot_stacked(args.model, args.split, args.context, args.phrase, args.mode)
    elif args.command == 'correlation':
        return plot_correlation(args.model, args.split, args.context, args.metric,
                                args.phrase, args.mode,
                                show_regression=not args.no_regression,
                                cluster_split=args.cluster_split,
                                cluster_type=args.cluster_type)
    elif args.command == 'language-grid':
        return plot_language_grid()
    elif args.command == 'all':
        results = []
        print("=" * 60)
        print("Generating all paper figures")
        print("=" * 60)

        print("\n--- Forest plot ---")
        results.append(plot_forest(args.model, args.split, args.context, args.metric, args.phrase, args.mode))

        print("\n--- Stacked plot ---")
        results.append(plot_stacked(args.model, args.split, args.context, args.phrase, args.mode))

        print("\n--- Correlation plot ---")
        results.append(plot_correlation(args.model, args.split, args.context, args.metric, args.phrase, args.mode))

        print("\n--- Language grid ---")
        results.append(plot_language_grid())

        failures = sum(1 for r in results if r != 0)
        print(f"\n{'=' * 60}")
        print(f"Complete: {len(results) - failures}/{len(results)} succeeded")
        if failures:
            print(f"  {failures} failed")
        return 1 if failures else 0


if __name__ == '__main__':
    exit(main())
