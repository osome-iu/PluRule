#!/usr/bin/env python3
"""
Generate LaTeX results tables from evaluation performance files.

Subcommands:
    main        Main results table (contexts x models, symmetric 1:1 accuracy)
    weighted    Main results table with asymmetric 2:1 accuracy (violating weighted 2x)
    appendix    Appendix table (violating recall / compliant recall / accuracy)
    all         Generate all three tables

Usage:
    python eval/generate_paper_tables.py all
    python eval/generate_paper_tables.py main
    python eval/generate_paper_tables.py weighted
    python eval/generate_paper_tables.py appendix
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_utils import find_performance_file_by_parts


# ============================================================================
# CONFIGURATION
# ============================================================================

SPLIT = "test"

CONTEXTS = [
    ("none", "Comment Only", False),
    ("discussion", "\\quad +Discussion", True),
    ("submission-discussion", "\\quad\\quad +Submission", True),
    ("submission-discussion-user", "\\quad\\quad\\quad +User", True),
    ("submission-media-discussion-user", "\\quad\\quad\\quad\\quad +Images", True),
]

MODELS = [
    ("qwen3-vl-4b", "Qwen3-VL-4B", [("instruct", "Instruct"), ("thinking", "Thinking")]),
    ("qwen3-vl-8b", "Qwen3-VL-8B", [("instruct", "Instruct"), ("thinking", "Thinking")]),
    ("qwen3-vl-30b", "Qwen3-VL-30B", [("instruct", "Instruct"), ("thinking", "Thinking")]),
    ("gpt5.2", "GPT-5.2", [("low", "Low"), ("high", "High")]),
]


# ============================================================================
# HELPERS
# ============================================================================

def _load_perf(model_base, variant, split, context):
    """Load performance data. Returns (full_data, overall_metrics) or (None, None)."""
    perf_file = find_performance_file_by_parts(model_base, variant, split, context)
    if not perf_file:
        return None, None
    with open(perf_file) as f:
        data = json.load(f)
    return data, data.get("metrics", {}).get("overall", {})


def _compute_accuracy(overall, weights=(1, 1)):
    """Return accuracy under the given (violating, compliant) weighting.

    weights=(1,1) -> overall_accuracy (stored directly).
    weights=(2,1) -> (2*V + C) / 3 from violating/compliant recall.
    """
    if overall is None:
        return None
    w_v, w_c = weights
    if w_v == w_c == 1:
        return overall.get("overall_accuracy")
    v = overall.get("violating_accuracy")
    c = overall.get("compliant_accuracy")
    if v is None or c is None:
        return None
    return (w_v * v + w_c * c) / (w_v + w_c)


def _baseline_accuracy(weights=(1, 1)):
    """Always-'no violation' baseline: V=0, C=1."""
    w_v, w_c = weights
    return (w_v * 0.0 + w_c * 1.0) / (w_v + w_c)


def _format_accuracy(value, decimals=1):
    return f"{value * 100:.{decimals}f}"


def _format_accuracy_with_delta(value, prev_value, decimals=1, bold=False):
    acc = value * 100
    prev_acc = prev_value * 100
    acc_rounded = round(acc, decimals)
    prev_rounded = round(prev_acc, decimals)
    delta = acc_rounded - prev_rounded
    sign = "+" if delta >= 0 else ""
    acc_str = f"{acc_rounded:.{decimals}f}"
    if bold:
        acc_str = f"\\textbf{{{acc_str}}}"
    return f"{acc_str} {{\\tiny ({sign}{delta:.{decimals}f})}}"


# ============================================================================
# MAIN TABLE
# ============================================================================

def generate_main_table(weights=(1, 1)):
    """Generate main results LaTeX table under the given (V, C) weighting."""
    is_symmetric = weights == (1, 1)
    all_columns = []
    for model_base, _, variants in MODELS:
        for variant_name, _ in variants:
            all_columns.append((model_base, variant_name))

    data_grid = {}
    prev_row_values = {col: None for col in all_columns}
    max_per_col = {col: -1 for col in all_columns}
    max_ci = 0.0

    for context_name, _, _ in CONTEXTS:
        current_row = {}
        for model_base, variant_name in all_columns:
            _, overall = _load_perf(model_base, variant_name, SPLIT, context_name)
            acc = _compute_accuracy(overall, weights)
            if is_symmetric and overall:
                ci = overall.get("overall_accuracy_ci")
                if ci and len(ci) == 2:
                    max_ci = max(max_ci, (ci[1] - ci[0]) / 2 * 100)
            prev = prev_row_values[(model_base, variant_name)]
            data_grid[(context_name, model_base, variant_name)] = (acc, prev)
            current_row[(model_base, variant_name)] = acc if acc else prev
            if acc and acc > max_per_col[(model_base, variant_name)]:
                max_per_col[(model_base, variant_name)] = acc
        prev_row_values = current_row

    total_cols = sum(len(variants) for _, _, variants in MODELS)

    table = []
    table.append("\\begin{table*}[t]")
    table.append("\\centering")
    table.append("\\setlength{\\tabcolsep}{3.75pt}")
    table.append(f"\\begin{{tabular}}{{l{'l' * total_cols}}}")
    table.append("\\toprule")

    # Header row 1: Model names
    header_parts = ["\\textbf{Models}"]
    col_idx = 2
    cmidrules = []
    for model_base, display_name, variants in MODELS:
        num_variants = len(variants)
        header_parts.append(f"\\multicolumn{{{num_variants}}}{{c}}{{\\textbf{{{display_name}}}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + num_variants - 1}}}")
        col_idx += num_variants
    table.append(" & ".join(header_parts) + " \\\\")
    table.append(" ".join(cmidrules))

    # Header row 2: Variant names
    subheader_parts = ["\\textbf{Variants}"]
    for model_base, _, variants in MODELS:
        for _, variant_display in variants:
            subheader_parts.append(f"\\multicolumn{{1}}{{c}}{{{variant_display}}}")
    table.append(" & ".join(subheader_parts) + " \\\\")
    table.append("\\midrule")

    # Data rows
    for context_name, context_label, show_delta in CONTEXTS:
        cells = [context_label]
        for model_base, variant_name in all_columns:
            acc, prev = data_grid[(context_name, model_base, variant_name)]
            if acc is not None:
                is_best = abs(acc - max_per_col[(model_base, variant_name)]) < 0.0001
                if show_delta and prev is not None:
                    cell = _format_accuracy_with_delta(acc, prev, bold=is_best)
                else:
                    acc_str = _format_accuracy(acc)
                    cell = f"\\textbf{{{acc_str}}}" if is_best else acc_str
            else:
                cell = "\u2014"
            cells.append(cell)
        table.append(" & ".join(cells) + " \\\\")

    baseline_pct = _baseline_accuracy(weights) * 100
    table.append("\\midrule")
    table.append(f"No Moderation & \\multicolumn{{{total_cols}}}{{c}}{{{baseline_pct:.1f}}} \\\\")
    table.append("\\bottomrule")
    table.append("\\end{tabular}")

    if is_symmetric:
        ci_str = f"{max_ci:.1f}" if max_ci > 0 else "1.1"
        caption = (f"Overall accuracy (\\%) across different models and contexts on the {SPLIT} set. "
                   "Numbers in parentheses indicate differences compared to accuracy values in the previous row. "
                   f"All values have 95\\% CI of $\\pm {ci_str}\\%$. "
                   "See the Appendix for a breakdown of moderated and unmoderated accuracy.")
        label = "tab:results-across-contexts"
    else:
        w_v, w_c = weights
        caption = (f"Weighted accuracy (\\%) with violating:compliant weighting of {w_v}:{w_c} "
                   f"across different models and contexts on the {SPLIT} set. "
                   "Numbers in parentheses indicate differences compared to accuracy values in the previous row. "
                   f"The always-compliant baseline drops to {baseline_pct:.1f}\\% under this weighting.")
        label = f"tab:results-across-contexts-weighted-{w_v}-{w_c}"

    table.append(f"\\caption{{{caption}}}")
    table.append(f"\\label{{{label}}}")
    table.append("\\end{table*}")

    return "\n".join(table)


# ============================================================================
# APPENDIX TABLE
# ============================================================================

def generate_appendix_table():
    """Generate appendix LaTeX table with violating/compliant/overall breakdown."""
    all_rows = []
    for model_base, display_name, variants in MODELS:
        for variant_name, variant_display in variants:
            all_rows.append((model_base, variant_name, display_name, variant_display))

    num_context_cols = len(CONTEXTS)

    data_grid = {}
    max_ci = 0.0
    max_mod_per_row = {(mb, vn): -1 for mb, vn, _, _ in all_rows}
    max_unmod_per_row = {(mb, vn): -1 for mb, vn, _, _ in all_rows}
    max_overall_per_row = {(mb, vn): -1 for mb, vn, _, _ in all_rows}

    for model_base, variant_name, _, _ in all_rows:
        for context_name, _, _ in CONTEXTS:
            _, overall_metrics = _load_perf(model_base, variant_name, SPLIT, context_name)
            if overall_metrics:
                mod_acc = overall_metrics.get("violating_accuracy")
                unmod_acc = overall_metrics.get("compliant_accuracy")
                overall_acc = overall_metrics.get("overall_accuracy")
                for ci_key in ["violating_accuracy_ci", "compliant_accuracy_ci", "overall_accuracy_ci"]:
                    ci = overall_metrics.get(ci_key)
                    if ci and len(ci) == 2:
                        max_ci = max(max_ci, (ci[1] - ci[0]) / 2 * 100)
            else:
                mod_acc, unmod_acc, overall_acc = None, None, None

            data_grid[(model_base, variant_name, context_name)] = (mod_acc, unmod_acc, overall_acc)
            if mod_acc is not None and mod_acc > max_mod_per_row[(model_base, variant_name)]:
                max_mod_per_row[(model_base, variant_name)] = mod_acc
            if unmod_acc is not None and unmod_acc > max_unmod_per_row[(model_base, variant_name)]:
                max_unmod_per_row[(model_base, variant_name)] = unmod_acc
            if overall_acc is not None and overall_acc > max_overall_per_row[(model_base, variant_name)]:
                max_overall_per_row[(model_base, variant_name)] = overall_acc

    table = []
    table.append("\\begin{table*}[t]")
    table.append("\\centering")
    table.append("\\setlength{\\tabcolsep}{4.5pt}")
    table.append(f"\\begin{{tabular}}{{lll{'l' * num_context_cols}}}")
    table.append("\\toprule")

    context_headers = []
    for _, ctx_label, _ in CONTEXTS:
        clean = ctx_label.replace("\\quad", "").strip()
        context_headers.append(clean)
    header_parts = ["\\textbf{Model}", "\\textbf{Variant}", "\\textbf{Metric}"] + \
                   [f"\\textbf{{{h}}}" for h in context_headers]
    table.append(" & ".join(header_parts) + " \\\\")
    table.append("\\midrule")

    current_model = None
    variant_idx_in_model = 0

    for i, (model_base, variant_name, display_name, variant_display) in enumerate(all_rows):
        if display_name != current_model:
            current_model = display_name
            variant_idx_in_model = 0
            num_variants = len([r for r in all_rows if r[2] == display_name])
            model_row_span = num_variants * 3
        else:
            variant_idx_in_model += 1

        def _build_cells(metric_idx, max_per_row_dict):
            cells = []
            prev = None
            for context_name, _, show_delta in CONTEXTS:
                val = data_grid[(model_base, variant_name, context_name)][metric_idx]
                if val is not None:
                    is_best = abs(val - max_per_row_dict[(model_base, variant_name)]) < 0.0001
                    if show_delta and prev is not None:
                        cell = _format_accuracy_with_delta(val, prev, bold=is_best)
                    else:
                        acc_str = _format_accuracy(val)
                        cell = f"\\textbf{{{acc_str}}}" if is_best else acc_str
                    prev = val
                else:
                    cell = "\u2014"
                cells.append(cell)
            return cells

        mod_cells = _build_cells(0, max_mod_per_row)
        unmod_cells = _build_cells(1, max_unmod_per_row)
        overall_cells = _build_cells(2, max_overall_per_row)

        if variant_idx_in_model == 0:
            model_cell = f"\\multirow{{{model_row_span}}}{{*}}{{{display_name}}}"
        else:
            model_cell = ""

        variant_cell = f"\\multirow{{3}}{{*}}{{{variant_display}}}"

        table.append(f"{model_cell} & {variant_cell} & Vio Rec & " + " & ".join(mod_cells) + " \\\\")
        table.append(" & & Com Rec & " + " & ".join(unmod_cells) + " \\\\")
        table.append(" & & Acc & " + " & ".join(overall_cells) + " \\\\")

        if i < len(all_rows) - 1:
            next_is_new_model = all_rows[i + 1][2] != display_name
            if next_is_new_model:
                table.append("\\midrule")
            else:
                table.append(f"\\cmidrule(lr){{2-{3 + num_context_cols}}}")

    table.append("\\midrule")
    table.append(f"\\multirow{{3}}{{*}}{{No Moderation}} & & Vio Rec & \\multicolumn{{{num_context_cols}}}{{c}}{{0.0}} \\\\")
    table.append(f" & & Com Rec & \\multicolumn{{{num_context_cols}}}{{c}}{{100.0}} \\\\")
    table.append(f" & & Acc & \\multicolumn{{{num_context_cols}}}{{c}}{{50.0}} \\\\")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    ci_str = f"{max_ci:.1f}" if max_ci > 0 else "1.5"
    table.append(f"\\caption{{Violating recall, compliant recall, and accuracy (\\%) across different models and contexts on the {SPLIT} set. Numbers in parentheses indicate differences compared to accuracy values in the previous column. All values have 95\\% CI of $\\pm {ci_str}\\%$.}}")
    table.append("\\label{tab:results-mod-unmod}")
    table.append("\\end{table*}")

    return "\n".join(table)


# ============================================================================
# CLI
# ============================================================================

def write_table(table_type='main'):
    """Write LaTeX table to file."""
    tables_dir = Path(__file__).resolve().parent.parent / 'output' / 'eval' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    if table_type == 'main':
        latex = generate_main_table(weights=(1, 1))
        output_file = tables_dir / f"results_table_{SPLIT}.tex"
    elif table_type == 'weighted':
        latex = generate_main_table(weights=(2, 1))
        output_file = tables_dir / f"results_table_{SPLIT}_weighted_2_1.tex"
    else:
        latex = generate_appendix_table()
        output_file = tables_dir / f"results_table_appendix_{SPLIT}.tex"

    with open(output_file, 'w') as f:
        f.write(latex)
    print(f"Done: {output_file.name}")
    print(latex)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX results tables from evaluation performance files'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('main', help='Main results table (symmetric 1:1)')
    subparsers.add_parser('weighted', help='Main results table (asymmetric 2:1)')
    subparsers.add_parser('appendix', help='Appendix results table')
    subparsers.add_parser('all', help='Generate all three tables')

    args = parser.parse_args()

    if args.command == 'main':
        return write_table('main')
    elif args.command == 'weighted':
        return write_table('weighted')
    elif args.command == 'appendix':
        return write_table('appendix')
    elif args.command == 'all':
        results = []
        print("--- Main table (1:1) ---")
        results.append(write_table('main'))
        print("\n--- Weighted table (2:1) ---")
        results.append(write_table('weighted'))
        print("\n--- Appendix table ---")
        results.append(write_table('appendix'))
        failures = sum(1 for r in results if r != 0)
        print(f"\nComplete: {len(results) - failures}/{len(results)} succeeded")
        return 1 if failures else 0


if __name__ == '__main__':
    exit(main())
