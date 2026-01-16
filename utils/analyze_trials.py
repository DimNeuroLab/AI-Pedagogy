# utils/analyze_trials.py

import json
import os
import matplotlib
matplotlib.use('Agg')  # To avoid display issues when running headless
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config_loader import load_config
import numpy as np
import statsmodels.stats.multitest as smm
import matplotlib.transforms as mtransforms
from scipy.stats import ttest_ind


# -----------------------
# Analysis Helpers
# -----------------------

def analyze_trials(trials):
    if not trials:
        return (0, [], 0, [])

    question_list = [t['questions_used'] for t in trials]
    correctness_list = [t['correct'] for t in trials]

    average_question = sum(question_list) / len(question_list)
    average_correctness = sum(correctness_list) / len(correctness_list)

    return average_question, question_list, average_correctness, correctness_list

def load_trials(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_grouped_boxplots(questions_dict, label_map, output_dir, ontology, model):
    """
    Create grouped boxplots of number of questions per strategy, comparing them with the Expert.

    - Data is grouped into categories (Directionality, Questions, Mixed TD/BU/Qs, Expert).
    - Each subplot shows a family of strategies.
    - The Expert median is drawn as a magenta dashed line across all plots.
    - Pairwise statistical tests (Welch's t-test, two-sided) are run for each strategy vs. Expert.
    - P-values are corrected for multiple comparisons using FDR (Benjamini–Hochberg).
    - Significance levels (ns, *, **, ***) are displayed in bold below the x-axis labels.
    - A CSV summary with raw/adjusted p-values, effect sizes (Cohen's d / Hedges' g),
      and bootstrap 95% CIs for the mean difference is saved to disk.

    Parameters
    ----------
    questions_dict : dict
        Mapping from agent/strategy name -> list/array of "number of questions".
    label_map : dict
        Mapping from agent/strategy code to display label.
    output_dir : str
        Directory where the resulting plot and CSV will be saved.
    ontology : str
        Name of the ontology (used in output filename).
    model : str
        Name of the model (used in output filename).
    """

    n_items = ontology.split("_")[-1].split(".")[0]
    # ---------------------------
    # 1) Group definitions & palettes
    # ---------------------------
    groups = {
        "Directionality": ["top_down", "bottom_up"],
        "Questions": ["learner_questions", "teacher_questions"],
        "Mixed TD": ["mixed_top-down_learner_questions", "mixed_top-down_teacher_questions"],
        "Mixed BU": ["mixed_bottom-up_learner_questions", "mixed_bottom-up_teacher_questions"],
        "Mixed Qs": ["mixed_learner_questions", "mixed_teacher_questions"],
        "Expert": ["expert"]
    }

    # Colour palette families (distinct hue by group; tonal variants within group)
    family_by_group = {
        "Directionality": ["#1f77b4", "#6baed6"],   # blues
        "Questions":      ["#2ca02c", "#98df8a"],   # greens
        "Mixed TD":       ["#ff7f0e", "#ffbb78"],   # oranges
        "Mixed BU":       ["#9467bd", "#c5b0d5"],   # purples
        "Mixed Qs":       ["#d62728", "#ff9896"],   # reds
        "Expert":         ["#e377c2"]               # magenta
    }

    # ---------------------------
    # 2) Expert values
    # ---------------------------
    expert_values = np.array(questions_dict.get("expert", []), dtype=float)
    expert_median = np.median(expert_values) if expert_values.size > 0 else None
    expert_mean   = float(np.mean(expert_values)) if expert_values.size > 0 else None
    expert_var    = float(np.var(expert_values, ddof=1)) if expert_values.size > 1 else None

    # ---------------------------
    # 3) Welch's t-tests (two-sided) vs Expert for ALL non-Experts
    #    + FDR correction + effect sizes + bootstrap CIs
    # ---------------------------
    # Settings for bootstrap (kept internal to avoid changing the function signature)
    N_BOOT = 10000           # number of bootstrap resamples
    RNG_SEED = 42            # fixed seed for reproducibility
    rng = np.random.default_rng(RNG_SEED)

    def hedges_g(x, y):
        """Cohen's d with Hedges' small-sample correction (g)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        sx2 = np.var(x, ddof=1)
        sy2 = np.var(y, ddof=1)
        # Pooled SD (classic formulation; acceptable with unequal variances for effect sizing)
        sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)) if (nx + ny - 2) > 0 else np.nan
        if sp == 0 or np.isnan(sp):
            return np.nan
        d = (np.mean(x) - np.mean(y)) / sp
        # Hedges' correction
        J = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0)) if (nx + ny) > 2 else 1.0
        return d * J

    def bootstrap_mean_diff_CI(x, y, alpha=0.05):
        """Percentile bootstrap CI for mean(x) - mean(y)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            return (np.nan, np.nan)
        diffs = np.empty(N_BOOT, dtype=float)
        nx, ny = x.size, y.size
        for b in range(N_BOOT):
            xb = x[rng.integers(0, nx, nx)]
            yb = y[rng.integers(0, ny, ny)]
            diffs[b] = xb.mean() - yb.mean()
        lo = np.percentile(diffs, 100 * (alpha / 2))
        hi = np.percentile(diffs, 100 * (1 - alpha / 2))
        return (float(lo), float(hi))

    stats_rows = []     # will be written to CSV
    agents = []
    pvals_raw = []

    for agent_key, vals in questions_dict.items():
        if agent_key == "expert":
            continue
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0 or expert_values.size == 0:
            continue

        # Welch's t-test (two-sided)
        t_stat, p_two = ttest_ind(arr, expert_values, equal_var=False)

        # Effect size (Hedges' g)
        g = hedges_g(arr, expert_values)

        # Bootstrap 95% CI for mean difference (agent - expert)
        ci_lo, ci_hi = bootstrap_mean_diff_CI(arr, expert_values, alpha=0.05)

        # Collect for correction and CSV
        agents.append(agent_key)
        pvals_raw.append(p_two)
        stats_rows.append({
            "agent": agent_key,
            "n_agent": int(arr.size),
            "mean_agent": float(np.mean(arr)),
            "sd_agent": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
            "n_expert": int(expert_values.size),
            "mean_expert": expert_mean if expert_mean is not None else np.nan,
            "sd_expert": np.sqrt(expert_var) if expert_var is not None else np.nan,
            "t_stat": float(t_stat),
            "p_raw_two_sided": float(p_two),
            "hedges_g": float(g) if g is not None else np.nan,
            "boot_mean_diff_lo": ci_lo,
            "boot_mean_diff_hi": ci_hi
        })

    # FDR correction across ALL strategy-vs-expert comparisons
    p_corr_map = {}
    if pvals_raw:
        _, pvals_corr, _, _ = smm.multipletests(pvals_raw, method="fdr_bh")
        for ak, p_corr in zip(agents, pvals_corr):
            p_corr_map[ak] = float(p_corr)

        # write adjusted p back into rows
        for row in stats_rows:
            row["p_adj_fdr_bh"] = p_corr_map.get(row["agent"], np.nan)
    else:
        for row in stats_rows:
            row["p_adj_fdr_bh"] = np.nan

    # Save CSV summary
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"stats_summary_{model}_{ontology}.csv")
    pd.DataFrame(stats_rows).to_csv(csv_path, index=False)

    # ---------------------------
    # 4) Plotting
    # ---------------------------
    fig, axes = plt.subplots(1, 6, figsize=(24, 5), sharey=True)
    fig.suptitle("Number of Questions per Grouped Strategy (model="+model+", n_items="+n_items+")", fontsize=20)

    # Extra bottom margin so significance marks below x labels are visible
    fig.subplots_adjust(bottom=0.4)

    for ax, (title, agent_keys) in zip(axes, groups.items()):
        group_data, group_labels = [], []

        # Gather data for this group
        for agent in agent_keys:
            if agent in questions_dict:
                group_data.extend(questions_dict[agent])
                group_labels.extend([agent] * len(questions_dict[agent]))

        if group_data:
            # DataFrame for seaborn
            df = pd.DataFrame({
                "Agent": [label_map.get(agent, agent) for agent in group_labels],
                "Questions": group_data
            })

            # Category order and palette mapping
            order = [label_map.get(a, a) for a in agent_keys if a in questions_dict]
            cols = family_by_group[title]
            palette_map = {cat: cols[i % len(cols)] for i, cat in enumerate(order)}

            # Draw the boxplot
            sns.boxplot(
                data=df, x="Agent", y="Questions",
                ax=ax, order=order, palette=palette_map
            )

            # Axes cosmetics
            ax.set_title(title, fontsize=20)
            ax.set_xlabel("")
            ax.set_ylabel("Questions", fontsize=18)
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Expert median (magenta) — legend only on Expert panel
            if expert_median is not None:
                if title == "Expert":
                    ax.axhline(
                        expert_median, color="magenta", linestyle="--", linewidth=2,
                        label=f"Expert median = {expert_median:.1f}"
                    )
                    ax.legend(loc="upper right", fontsize=16, handlelength=2.5, handleheight=1.5)
                else:
                    ax.axhline(expert_median, color="magenta", linestyle="--", linewidth=2)

            # Significance marks below x-axis labels (bold): based on FDR-adjusted p
            if expert_values.size > 0 and title != "Expert":
                trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
                y_axes_pos = -0.45  # further below the x-axis (negative means below)
                for agent in agent_keys:
                    if agent == "expert" or agent not in questions_dict:
                        continue
                    cat = label_map.get(agent, agent)
                    if cat not in order:
                        continue
                    p_adj = p_corr_map.get(agent, None)
                    if p_adj is None:
                        text = "ns"
                    else:
                        if p_adj < 0.001:
                            text = "***"
                        elif p_adj < 0.01:
                            text = "**"
                        elif p_adj < 0.05:
                            text = "*"
                        else:
                            text = "ns"

                    ax.text(
                        x=order.index(cat),
                        y=y_axes_pos,
                        s=text,
                        transform=trans,
                        ha="center", va="top",
                        fontsize=14, fontweight="bold", color="black"
                    )
        else:
            ax.set_title(title + "\n(no data)", fontsize=12)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"boxplot_grouped_{model}_{ontology}.pdf"))
    plt.close()


def plot_grouped_boxplots_anova(questions_dict, label_map, output_dir, ontology, model):
    """
    Create grouped boxplots of number of questions per strategy, comparing them with the Expert.

    - Data is grouped into categories (Directionality, Questions, Mixed TD/BU/Qs, Expert).
    - Each subplot shows a family of strategies.
    - The Expert median is drawn as a magenta dashed line across all plots.
    - Pairwise statistical tests (Welch's t-test, two-sided) are run for each strategy vs. Expert.
    - P-values are corrected for multiple comparisons using FDR (Benjamini–Hochberg).
    - Significance levels (ns, *, **, ***) are displayed in bold below the x-axis labels.
    - A CSV summary with raw/adjusted p-values, effect sizes (Cohen's d / Hedges' g),
      and bootstrap 95% CIs for the mean difference is saved to disk.

    Additionally (Expert excluded):
    - One-way omnibus ANOVA across strategies (diagnostic).
    - Pairwise Welch t-tests among non-Expert strategies with FDR (BH).
    - Effect sizes (Hedges’ g) and bootstrap CIs for mean differences.
    - A ranking table (by mean questions and by 'wins' = significant improvements).
    - Saves a heatmap of pairwise significance and a ranking barplot.

    Parameters
    ----------
    questions_dict : dict
        Mapping from agent/strategy name -> list/array of "number of questions".
    label_map : dict
        Mapping from agent/strategy code to display label.
    output_dir : str
        Directory where the resulting plot and CSV will be saved.
    ontology : str
        Name of the ontology (used in output filename).
    model : str
        Name of the model (used in output filename).
    """

    # Extract a compact count to print context in titles
    n_items = ontology.split("_")[-1].split(".")[0]

    # ---------------------------
    # 1) Group definitions & palettes (visual only)
    # ---------------------------
    groups = {
        "Directionality": ["top_down", "bottom_up"],
        "Questions": ["learner_questions", "teacher_questions"],
        "Mixed TD": ["mixed_top-down_learner_questions", "mixed_top-down_teacher_questions"],
        "Mixed BU": ["mixed_bottom-up_learner_questions", "mixed_bottom-up_teacher_questions"],
        "Mixed Qs": ["mixed_learner_questions", "mixed_teacher_questions"],
        "Expert": ["expert"]
    }

    # Colour palette families (distinct hue by group; tonal variants within group)
    family_by_group = {
        "Directionality": ["#1f77b4", "#6baed6"],   # blues
        "Questions":      ["#2ca02c", "#98df8a"],   # greens
        "Mixed TD":       ["#ff7f0e", "#ffbb78"],   # oranges
        "Mixed BU":       ["#9467bd", "#c5b0d5"],   # purples
        "Mixed Qs":       ["#d62728", "#ff9896"],   # reds
        "Expert":         ["#e377c2"]               # magenta
    }

    # ---------------------------
    # 2) Expert values
    # ---------------------------
    expert_values = np.array(questions_dict.get("expert", []), dtype=float)
    expert_median = np.median(expert_values) if expert_values.size > 0 else None
    expert_mean   = float(np.mean(expert_values)) if expert_values.size > 0 else None
    expert_var    = float(np.var(expert_values, ddof=1)) if expert_values.size > 1 else None

    # ---------------------------
    # 3) Helper stats functions
    # ---------------------------
    N_BOOT = 10000           # bootstrap samples for CIs
    RNG_SEED = 42            # fixed seed for reproducibility
    rng = np.random.default_rng(RNG_SEED)

    def hedges_g(x, y):
        """Cohen's d with Hedges' small-sample correction (g)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        sx2 = np.var(x, ddof=1)
        sy2 = np.var(y, ddof=1)
        if (nx + ny - 2) <= 0:
            return np.nan
        sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
        if sp == 0 or np.isnan(sp):
            return np.nan
        d = (np.mean(x) - np.mean(y)) / sp
        # Hedges' correction
        J = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0)) if (nx + ny) > 2 else 1.0
        return d * J

    def bootstrap_mean_CI(x, alpha=0.05):
        """Percentile bootstrap CI for mean(x)."""
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return (np.nan, np.nan)
        nx = x.size
        means = np.empty(N_BOOT, dtype=float)
        for b in range(N_BOOT):
            xb = x[rng.integers(0, nx, nx)]
            means[b] = xb.mean()
        lo = np.percentile(means, 100 * (alpha / 2))
        hi = np.percentile(means, 100 * (1 - alpha / 2))
        return (float(lo), float(hi))

    def bootstrap_mean_diff_CI(x, y, alpha=0.05):
        """Percentile bootstrap CI for mean(x) - mean(y)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            return (np.nan, np.nan)
        diffs = np.empty(N_BOOT, dtype=float)
        nx, ny = x.size, y.size
        for b in range(N_BOOT):
            xb = x[rng.integers(0, nx, nx)]
            yb = y[rng.integers(0, ny, ny)]
            diffs[b] = xb.mean() - yb.mean()
        lo = np.percentile(diffs, 100 * (alpha / 2))
        hi = np.percentile(diffs, 100 * (1 - alpha / 2))
        return (float(lo), float(hi))

    # ---------------------------
    # 4) Expert vs each strategy (Welch two-sided + FDR); CSV as before
    # ---------------------------
    stats_rows = []
    agents = []
    pvals_raw = []

    for agent_key, vals in questions_dict.items():
        if agent_key == "expert":
            continue
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0 or expert_values.size == 0:
            continue

        # Welch's t-test (two-sided)
        t_stat, p_two = ttest_ind(arr, expert_values, equal_var=False)

        # Effect size (Hedges' g)
        g = hedges_g(arr, expert_values)

        # Bootstrap 95% CI for mean difference (agent - expert)
        ci_lo, ci_hi = bootstrap_mean_diff_CI(arr, expert_values, alpha=0.05)

        agents.append(agent_key)
        pvals_raw.append(p_two)
        stats_rows.append({
            "agent": agent_key,
            "n_agent": int(arr.size),
            "mean_agent": float(np.mean(arr)),
            "sd_agent": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
            "n_expert": int(expert_values.size),
            "mean_expert": expert_mean if expert_mean is not None else np.nan,
            "sd_expert": np.sqrt(expert_var) if expert_var is not None else np.nan,
            "t_stat": float(t_stat),
            "p_raw_two_sided": float(p_two),
            "hedges_g": float(g) if g is not None else np.nan,
            "boot_mean_diff_lo": ci_lo,
            "boot_mean_diff_hi": ci_hi
        })

    # FDR correction across ALL strategy-vs-expert comparisons
    p_corr_map = {}
    if pvals_raw:
        _, pvals_corr, _, _ = smm.multipletests(pvals_raw, method="fdr_bh")
        for ak, p_corr in zip(agents, pvals_corr):
            p_corr_map[ak] = float(p_corr)
        for row in stats_rows:
            row["p_adj_fdr_bh"] = p_corr_map.get(row["agent"], np.nan)
    else:
        for row in stats_rows:
            row["p_adj_fdr_bh"] = np.nan

    # Save CSV summary (Expert vs others)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"stats_summary_{model}_{ontology}.csv")
    pd.DataFrame(stats_rows).to_csv(csv_path, index=False)

    # ---------------------------
    # 5) NON-EXPERT analysis: omnibus + pairwise + ranking
    # ---------------------------
    non_expert_keys = [k for k in questions_dict.keys() if k != "expert" and len(questions_dict[k]) > 0]
    non_expert_means = {k: float(np.mean(questions_dict[k])) for k in non_expert_keys}

    # 5a) Omnibus ANOVA across non-Experts (diagnostic).
    #     Caveat: scipy.stats.f_oneway is classic (homoscedastic). For heteroscedastic data,
    #     Welch ANOVA (Brown–Forsythe) would be preferable (e.g., pingouin.welch_anova).
    #     We keep this as a quick diagnostic and rely on pairwise Welch+FDR for decisions.
    anova_F, anova_p = (np.nan, np.nan)
    if len(non_expert_keys) >= 2:
        group_data = [np.asarray(questions_dict[k], dtype=float) for k in non_expert_keys]
        try:
            anova_F, anova_p = f_oneway(*group_data)
        except Exception:
            pass  # keep NaN if something odd happens

    # 5b) Pairwise Welch t-tests among non-Experts, with FDR
    pair_rows = []
    raw_ps = []
    pair_ids = []
    for i in range(len(non_expert_keys)):
        for j in range(i + 1, len(non_expert_keys)):
            a, b = non_expert_keys[i], non_expert_keys[j]
            xa = np.asarray(questions_dict[a], dtype=float)
            xb = np.asarray(questions_dict[b], dtype=float)
            t_stat, p_two = ttest_ind(xa, xb, equal_var=False)
            g = hedges_g(xa, xb)
            ci_lo, ci_hi = bootstrap_mean_diff_CI(xa, xb)
            pair_rows.append({
                "A": a, "B": b,
                "n_A": int(xa.size), "mean_A": float(xa.mean()), "sd_A": float(np.std(xa, ddof=1)) if xa.size > 1 else np.nan,
                "n_B": int(xb.size), "mean_B": float(xb.mean()), "sd_B": float(np.std(xb, ddof=1)) if xb.size > 1 else np.nan,
                "t_stat": float(t_stat),
                "p_raw_two_sided": float(p_two),
                "hedges_g": float(g) if g is not None else np.nan,
                "boot_mean_diff_lo": ci_lo,
                "boot_mean_diff_hi": ci_hi
            })
            raw_ps.append(p_two)
            pair_ids.append((a, b))

    if raw_ps:
        _, p_adj_pairs, _, _ = smm.multipletests(raw_ps, method="fdr_bh")
    else:
        p_adj_pairs = []

    # attach adjusted p to rows
    for row, p_adj in zip(pair_rows, p_adj_pairs):
        row["p_adj_fdr_bh"] = float(p_adj)

    # Save pairwise CSV (non-Experts only)
    pairwise_csv = os.path.join(output_dir, f"nonexpert_pairwise_{model}_{ontology}.csv")
    pd.DataFrame(pair_rows).to_csv(pairwise_csv, index=False)

    # 5c) Build a 'wins' metric (lower mean questions is better).
    #     Count how many opponents a strategy beats significantly (mean lower and p_adj<.05).
    wins = {k: 0 for k in non_expert_keys}
    sig_matrix = pd.DataFrame(index=non_expert_keys, columns=non_expert_keys, data="")
    for (a, b), p_adj in zip(pair_ids, p_adj_pairs):
        ma, mb = non_expert_means[a], non_expert_means[b]
        if np.isnan(p_adj):
            mark = "ns"
        elif p_adj < 0.001:
            mark = "***"
        elif p_adj < 0.01:
            mark = "**"
        elif p_adj < 0.05:
            mark = "*"
        else:
            mark = "ns"

        # Fill symmetric significance matrix with direction arrows
        if mark == "ns":
            sig_matrix.loc[a, b] = "ns"
            sig_matrix.loc[b, a] = "ns"
        else:
            if ma < mb:
                sig_matrix.loc[a, b] = mark  # a better than b
                sig_matrix.loc[b, a] = ""    # leave opposite empty
                wins[a] += 1
            elif mb < ma:
                sig_matrix.loc[b, a] = mark  # b better than a
                sig_matrix.loc[a, b] = ""
                wins[b] += 1
            else:
                sig_matrix.loc[a, b] = "ns"
                sig_matrix.loc[b, a] = "ns"

    # Ranking table: sort by mean (ascending), then by wins (descending)
    ranking_rows = []
    for k in non_expert_keys:
        m = non_expert_means[k]
        lo, hi = bootstrap_mean_CI(np.asarray(questions_dict[k], dtype=float))
        ranking_rows.append({
            "strategy": k,
            "mean_questions": m,
            "boot_mean_lo": lo,
            "boot_mean_hi": hi,
            "wins_vs_others": wins.get(k, 0),
            "n": int(len(questions_dict[k]))
        })
    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        by=["mean_questions", "wins_vs_others"], ascending=[True, False]
    )
    ranking_csv = os.path.join(output_dir, f"nonexpert_ranking_{model}_{ontology}.csv")
    ranking_df.to_csv(ranking_csv, index=False)

    # ---------------------------
    # 6) Plotting
    # ---------------------------
    # 6a) Your grouped boxplots with Expert line and star marks (unchanged, just title tweak)
    fig, axes = plt.subplots(1, 6, figsize=(24, 5), sharey=True)
    fig.suptitle(f"Number of Questions per Grouped Strategy (model={model}, n_items={n_items})", fontsize=20)

    # Extra bottom margin so significance marks below x labels are visible
    fig.subplots_adjust(bottom=0.4)

    for ax, (title, agent_keys) in zip(axes, groups.items()):
        group_data, group_labels = [], []

        for agent in agent_keys:
            if agent in questions_dict:
                group_data.extend(questions_dict[agent])
                group_labels.extend([agent] * len(questions_dict[agent]))

        if group_data:
            df = pd.DataFrame({
                "Agent": [label_map.get(agent, agent) for agent in group_labels],
                "Questions": group_data
            })

            order = [label_map.get(a, a) for a in agent_keys if a in questions_dict]
            cols = family_by_group[title]
            palette_map = {cat: cols[i % len(cols)] for i, cat in enumerate(order)}

            sns.boxplot(
                data=df, x="Agent", y="Questions",
                ax=ax, order=order, palette=palette_map
            )

            ax.set_title(title, fontsize=20)
            ax.set_xlabel("")
            ax.set_ylabel("Questions", fontsize=18)
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Expert median (magenta); legend only on Expert panel
            if expert_median is not None:
                if title == "Expert":
                    ax.axhline(
                        expert_median, color="magenta", linestyle="--", linewidth=2,
                        label=f"Expert median = {expert_median:.1f}"
                    )
                    ax.legend(loc="upper right", fontsize=16, handlelength=2.5, handleheight=1.5)
                else:
                    ax.axhline(expert_median, color="magenta", linestyle="--", linewidth=2)

            # Significance marks below x-axis labels (bold) for Expert comparisons
            if expert_values.size > 0 and title != "Expert":
                trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
                y_axes_pos = -0.45  # below the x-axis
                for agent in agent_keys:
                    if agent == "expert" or agent not in questions_dict:
                        continue
                    cat = label_map.get(agent, agent)
                    if cat not in order:
                        continue
                    p_adj = p_corr_map.get(agent, None)
                    if p_adj is None:
                        text = "ns"
                    else:
                        if p_adj < 0.001:   text = "***"
                        elif p_adj < 0.01:  text = "**"
                        elif p_adj < 0.05:  text = "*"
                        else:               text = "ns"

                    ax.text(
                        x=order.index(cat),
                        y=y_axes_pos,
                        s=text,
                        transform=trans,
                        ha="center", va="top",
                        fontsize=14, fontweight="bold", color="black"
                    )
        else:
            ax.set_title(title + "\n(no data)", fontsize=12)

    grouped_pdf = os.path.join(output_dir, f"boxplot_grouped_{model}_{ontology}.pdf")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(grouped_pdf)
    plt.close()

    # 6b) Heatmap of non-Expert pairwise significance (FDR-adjusted p as stars)
    if len(non_expert_keys) >= 2 and len(pair_rows) > 0:
        # Build matrix of star marks and numeric p_adj for colour scale
        names = [label_map.get(k, k) for k in non_expert_keys]
        star_mat = pd.DataFrame(index=names, columns=names, data="")
        p_mat = pd.DataFrame(index=names, columns=names, data=np.nan)
        name_map = {k: label_map.get(k, k) for k in non_expert_keys}

        for row in pair_rows:
            a, b = name_map[row["A"]], name_map[row["B"]]
            p = row["p_adj_fdr_bh"]
            if np.isnan(p):
                mark = "ns"
            elif p < 0.001: mark = "***"
            elif p < 0.01:  mark = "**"
            elif p < 0.05:  mark = "*"
            else:           mark = "ns"
            # Fill both directions with the same p for colour; stars on upper
            p_mat.loc[a, b] = p
            p_mat.loc[b, a] = p
            # Directional star goes to the cell of the BETTER (lower mean) vs the other
            mean_a = non_expert_means[row["A"]]
            mean_b = non_expert_means[row["B"]]
            if mark == "ns":
                star_mat.loc[a, b] = "ns"
                star_mat.loc[b, a] = "ns"
            else:
                if mean_a < mean_b:
                    star_mat.loc[a, b] = mark  # a better than b
                    star_mat.loc[b, a] = ""
                elif mean_b < mean_a:
                    star_mat.loc[b, a] = mark  # b better than a
                    star_mat.loc[a, b] = ""
                else:
                    star_mat.loc[a, b] = "ns"
                    star_mat.loc[b, a] = "ns"

        # Convert p to -log10(p) for colour (higher = stronger evidence)
        with np.errstate(invalid='ignore'):
            logp = -np.log10(p_mat.astype(float))

        plt.figure(figsize=(8, 7))
        ax = sns.heatmap(
            logp, cmap="YlGnBu", annot=star_mat, fmt="s",
            cbar_kws={"label": "-log10(FDR-adjusted p)"},
            linewidths=0.5, linecolor="white"
        )
        ax.set_title("Non-Expert pairwise differences (FDR-adjusted)", fontsize=16)
        plt.tight_layout()
        heatmap_pdf = os.path.join(output_dir, f"nonexpert_heatmap_{model}_{ontology}.pdf")
        plt.savefig(heatmap_pdf)
        plt.close()

    # 6c) Ranking barplot (means with bootstrap 95% CI)
    if len(non_expert_keys) >= 1:
        plot_df = ranking_df.copy()
        plot_df["label"] = plot_df["strategy"].map(lambda k: label_map.get(k, k))
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            data=plot_df, x="label", y="mean_questions", order=plot_df["label"],
            edgecolor="black"
        )
        # Add error bars (bootstrap CI)
        for i, row in plot_df.iterrows():
            ax.errorbar(
                x=i, y=row["mean_questions"],
                yerr=[[row["mean_questions"] - row["boot_mean_lo"]],
                      [row["boot_mean_hi"] - row["mean_questions"]]],
                fmt="none", capsize=4, linewidth=1
            )
        ax.set_title("Non-Expert ranking (lower is better)", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("Mean questions")
        ax.tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        rank_pdf = os.path.join(output_dir, f"nonexpert_ranking_{model}_{ontology}.pdf")
        plt.savefig(rank_pdf)
        plt.close()
# -----------------------
# MAIN
# -----------------------

if __name__ == "__main__":
    all_strategies = [
        "top_down", "bottom_up",
        "learner_questions", "teacher_questions",
        "mixed_top-down_learner_questions", "mixed_top-down_teacher_questions",
        "mixed_bottom-up_learner_questions", "mixed_bottom-up_teacher_questions",
        "mixed_learner_questions", "mixed_teacher_questions",
        "expert"
    ]

    label_map = {
        "top_down": "TD",
        "bottom_up": "BU",
        "learner_questions": "LQ",
        "teacher_questions": "TQ",
        "mixed_top-down_learner_questions": "mix-TD-LQ",
        "mixed_top-down_teacher_questions": "mix-TD-TQ",
        "mixed_bottom-up_learner_questions": "mix-BU-LQ",
        "mixed_bottom-up_teacher_questions": "mix-BU-TQ",
        "mixed_learner_questions": "mix-LQ",
        "mixed_teacher_questions": "mix-TQ",
        "expert": "Expert"
    }

    config_path = os.path.join(os.path.dirname(__file__), "../config.yml")
    config = load_config(config_path)

    ontology = os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]
    model = config["model"]["name"]

    questions_dict = {}
    question_steps = []
    correct = []


    for strategy in all_strategies:
        print(f"Analyzing trials for strategy: {strategy}")
        run_name = f"{strategy}_test.json"
        path = os.path.join(os.path.dirname(__file__), "../results", model, ontology, "tests", "20q_game", run_name)    

        if not os.path.exists(path):
            print(f"Warning: file {path} does not exist. Skipping.")
            continue

        trials = load_trials(path)
        avg_question, question_list, avg_correctness, correct_trial = analyze_trials(trials)
        questions_dict[strategy] = question_list
        question_steps.append(question_list)
        correct.append(sum(correct_trial))

        print(f"Average Question Used: {avg_question}")
        print(f"Average Correctness: {avg_correctness}\n")

    questions_dict = {}

    for strategy in all_strategies:
        run_name = f"{strategy}_test.json"
        path = os.path.join(os.path.dirname(__file__), "../results", model, ontology, "tests", "20q_game", run_name)

        if not os.path.exists(path):
            print(f"Warning: file {path} does not exist. Skipping.")
            continue

        trials = load_trials(path)
        _, question_list, _, _ = analyze_trials(trials)
        questions_dict[strategy] = question_list

    output_dir = os.path.join(os.path.dirname(__file__), "../results", "plots")

    # Generate figures
    #plot_grouped_boxplots(questions_dict, label_map, output_dir, ontology, model)
    plot_grouped_boxplots_anova(questions_dict, label_map, output_dir, ontology, model)
