"""
Optimization Proficiency Level (OPL): measuring optimization capability scaling.

Adapts METR's time-horizon methodology to optimization. METR asks: "at 50%
success, how long a task can the model handle?" We ask: "at 50% success,
what fraction of expert speedup can the model achieve?"

    P(optimization_level >= t) = sigma(alpha + beta * t)
    50% OPL horizon = -alpha/beta

where optimization_level = model_speedup / expert_speedup.

Higher OPL = better optimizer. Track over model generations to measure
optimization capability scaling (distinct from general coding ability).
"""

import json
import os
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
with open("/tmp/logistic/all_results.json") as f:
    all_results = json.load(f)

from datasets import load_dataset
ds = load_dataset("gso-bench/gso", split="test")
hf_data = {row["instance_id"]: row for row in ds}

# ---------------------------------------------------------------------------
# Load ALL models from report files + cached results
# ---------------------------------------------------------------------------
REPORT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results", "reports")

def load_model_results(model_name):
    """Load per-instance results from report JSON."""
    report_path = os.path.join(REPORT_DIR, f"{model_name}.json")
    if not os.path.exists(report_path):
        return {}
    with open(report_path) as f:
        report = json.load(f)
    opt_stats = report.get("opt_stats", {})
    passed_ids = set(report.get("instance_sets", {}).get("passed_ids", []))
    opt_commit_ids = set(report.get("instance_sets", {}).get("opt_commit_ids", []))
    results = {}
    for inst_id, stats in opt_stats.items():
        results[inst_id] = {
            "test_passed": inst_id in passed_ids,
            "opt_commit": inst_id in opt_commit_ids,
            "gm_speedup_patch_base": stats.get("gm_speedup_patch_base"),
            "gm_speedup_commit_base": stats.get("gm_speedup_commit_base"),
        }
    return results

# All models from report files
ALL_MODEL_NAMES = sorted([
    f.replace(".json", "") for f in os.listdir(REPORT_DIR)
    if f.endswith(".json")
])

# Load all results (use cached all_results.json where available, fall back to reports)
all_results = {}
cached_path = "/tmp/logistic/all_results.json"
if os.path.exists(cached_path):
    with open(cached_path) as f:
        all_results = json.load(f)

for model_name in ALL_MODEL_NAMES:
    if model_name not in all_results:
        all_results[model_name] = load_model_results(model_name)

MODELS = ALL_MODEL_NAMES

MODEL_LABELS = {
    "claude-opus-4.6": "Claude Opus 4.6",
    "claude-opus-4.5": "Claude Opus 4.5",
    "claude-opus-4": "Claude Opus 4",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "claude-sonnet-4": "Claude Sonnet 4",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.1": "GPT-5.1",
    "gpt-5": "GPT-5",
    "gemini-3-pro": "Gemini 3 Pro",
    "gemini-3-flash": "Gemini 3 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "o3": "o3",
    "qwen3-coder": "Qwen3-Coder",
    "kimi-k2": "Kimi K2",
    "glm-4.5": "GLM-4.5",
}

MODEL_COLORS = {
    "claude-opus-4.6": "#B45309",
    "claude-opus-4.5": "#D97706",
    "claude-opus-4": "#F59E0B",
    "claude-sonnet-4.5": "#FBBF24",
    "claude-sonnet-4": "#FCD34D",
    "gpt-5.2": "#047857",
    "gpt-5.1": "#059669",
    "gpt-5": "#10B981",
    "gemini-3-pro": "#2563EB",
    "gemini-3-flash": "#60A5FA",
    "gemini-2.5-pro": "#93C5FD",
    "o3": "#6366F1",
    "qwen3-coder": "#DC2626",
    "kimi-k2": "#9333EA",
    "glm-4.5": "#64748B",
}

# ---------------------------------------------------------------------------
# Expert speedup per instance
# ---------------------------------------------------------------------------
instance_expert_speedup = {}
for model in MODELS:
    for inst_id, r in all_results[model].items():
        if r and r.get("gm_speedup_commit_base") and inst_id not in instance_expert_speedup:
            t = r["gm_speedup_commit_base"]
            if t and t > 1.0:
                instance_expert_speedup[inst_id] = t

# ---------------------------------------------------------------------------
# Instance metadata (for technique categorization)
# ---------------------------------------------------------------------------
compiled_ext = {'.c', '.cpp', '.cxx', '.h', '.hpp', '.pyx', '.pxd', '.rs', '.go', '.src'}
instance_meta = {}
for row in ds:
    inst_id = row["instance_id"]
    msg = (row.get("gt_commit_message", "") or "").lower()
    diff = row["gt_diff"]
    diff_lines = max(1, sum(1 for l in diff.split('\n') if l.startswith('+') or l.startswith('-')))
    exts = set()
    for l in diff.split('\n'):
        if l.startswith('diff --git'):
            parts = l.split()
            if len(parts) >= 3:
                fname = parts[2].lstrip('a/')
                if '.' in fname:
                    exts.add('.' + fname.rsplit('.', 1)[-1])
    has_compiled = bool(exts & compiled_ext)

    if 'simd' in msg or 'sse' in msg or 'avx' in msg or 'vectorize' in msg or 'svml' in msg:
        technique = 'SIMD/vectorization'
    elif 'ufunc' in msg or 'indexed loop' in msg or 'fast iter' in msg:
        technique = 'ufunc/C-loop'
    elif '.pyx' in str(exts):
        technique = 'Cython'
    elif '.rs' in exts:
        technique = 'Rust rewrite'
    elif has_compiled and '.pyx' not in str(exts):
        technique = 'C/C++ optimization'
    elif 'cache' in msg or 'lazy' in msg or 'avoid' in msg or 'skip' in msg:
        technique = 'caching/avoidance'
    else:
        technique = 'Python-level'

    instance_meta[inst_id] = {
        "diff_lines": diff_lines,
        "compiled": has_compiled,
        "technique": technique,
        "repo": inst_id.split("__")[0],
    }

# ---------------------------------------------------------------------------
# Compute optimization level (= model_speedup / expert_speedup) per model
# ---------------------------------------------------------------------------
model_ratios = {}  # model -> list of (inst_id, ratio)
for model in MODELS:
    ratios = []
    for inst_id, r in all_results[model].items():
        if r is None or not r.get("test_passed"):
            continue
        expert_sp = instance_expert_speedup.get(inst_id)
        model_sp = r.get("gm_speedup_patch_base")
        if not expert_sp or not model_sp:
            continue
        ratios.append((inst_id, model_sp / expert_sp))
    model_ratios[model] = ratios

# ---------------------------------------------------------------------------
# Fit logistic: P(ratio >= t) = sigma(alpha + beta * t)
# ---------------------------------------------------------------------------
def fit_opl(ratios, thresholds=np.arange(0.05, 1.5, 0.05)):
    """Fit logistic to the survival function of optimization ratios."""
    X, y = [], []
    for _, ratio in ratios:
        for t in thresholds:
            X.append(t)
            y.append(1 if ratio >= t else 0)
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000)
    clf.fit(X, y)
    alpha, beta = clf.intercept_[0], clf.coef_[0, 0]
    level_50 = -alpha / beta if beta != 0 else float("inf")
    return alpha, beta, level_50


def bootstrap_opl(ratios, repos_list, n_boot=2000, seed=42):
    """Hierarchical bootstrap for 50% optimization level CI."""
    rng = np.random.RandomState(seed)
    repo_arr = np.array(repos_list)
    unique_repos = list(set(repos_list))
    levels = []
    for _ in range(n_boot):
        boot_repos = rng.choice(unique_repos, size=len(unique_repos), replace=True)
        boot_idx = []
        for repo in boot_repos:
            idx = np.where(repo_arr == repo)[0]
            boot_idx.extend(rng.choice(idx, size=len(idx), replace=True))
        boot_ratios = [ratios[i] for i in boot_idx]
        if len(boot_ratios) < 10:
            continue
        try:
            _, beta, level = fit_opl(boot_ratios)
            if beta < 0 and 0.1 <= level <= 2.0:
                levels.append(level)
        except Exception:
            continue
    if len(levels) < 100:
        return None, None
    return np.percentile(levels, 2.5), np.percentile(levels, 97.5)


print("=" * 70)
print("Optimization Proficiency Level (OPL) Analysis")
print("=" * 70)
print()

fit_results = {}
print(f"{'Model':20s} {'n':>4s} {'alpha':>7s} {'beta':>7s} {'50% OPL':>10s} {'95% CI':>20s}")
print("-" * 65)

for model in MODELS:
    ratios = model_ratios[model]
    repos = [instance_meta.get(iid, {}).get("repo", "?") for iid, _ in ratios]
    alpha, beta, level_50 = fit_opl(ratios)
    ci_lo, ci_hi = bootstrap_opl(ratios, repos)
    fit_results[model] = {
        "alpha": alpha, "beta": beta, "level_50": level_50,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n": len(ratios),
    }
    ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo else "N/A"
    print(f"{MODEL_LABELS[model]:20s} {len(ratios):4d} {alpha:7.3f} {beta:7.3f} {level_50:10.1%} {ci_str:>20s}")

# ---------------------------------------------------------------------------
# Figure 1: Optimization Proficiency Curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

thresholds = np.linspace(0.0, 1.3, 200)
sorted_models = sorted(MODELS, key=lambda m: -fit_results[m]["level_50"])

for model in sorted_models:
    fr = fit_results[model]
    color = MODEL_COLORS[model]
    label = MODEL_LABELS[model]

    # Fitted sigmoid
    p_curve = 1.0 / (1.0 + np.exp(-(fr["alpha"] + fr["beta"] * thresholds)))
    ci_str = ""
    if fr["ci_lo"]:
        ci_str = f" [{fr['ci_lo']:.0%}, {fr['ci_hi']:.0%}]"
    ax.plot(thresholds, p_curve, color=color, linewidth=2.5,
            label=f"{label}: {fr['level_50']:.0%}{ci_str}", zorder=2)

    # Mark 50% point
    lv = fr["level_50"]
    if 0 <= lv <= 1.3:
        ax.plot(lv, 0.5, 'o', color=color, markersize=8,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)

    # Empirical points
    ratios_arr = np.array([r for _, r in model_ratios[model]])
    empirical_thresholds = np.arange(0.1, 1.2, 0.1)
    emp_rates = [(ratios_arr >= t).mean() for t in empirical_thresholds]
    ax.scatter(empirical_thresholds, emp_rates, color=color, s=40, alpha=0.3,
               edgecolors="none", zorder=3)

ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.4, linewidth=1)
ax.text(1.01, 0.95, "expert level", color="gray", fontsize=8, rotation=90,
        va="top", ha="left")
ax.text(1.25, 0.52, "50%", color="gray", fontsize=9, ha="right", va="bottom")

ax.set_xlabel("Optimization Level (model speedup / expert speedup)", fontsize=12)
ax.set_ylabel("P(achieving this level)", fontsize=12)
ax.set_title("Optimization Proficiency Level (OPL)\n"
             "P(model_speedup / expert_speedup ≥ t) = σ(α + β·t)",
             fontsize=13)
ax.set_xlim(-0.02, 1.3)
ax.set_ylim(-0.03, 1.03)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.legend(loc="upper right", fontsize=8.5,
          title="Model: 50% OPL [95% CI]", title_fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "optimization_proficiency.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: optimization_proficiency.png")

# ---------------------------------------------------------------------------
# Figure 2: Technique breakdown — what types of optimization can models do?
# ---------------------------------------------------------------------------
from collections import Counter

tech_order = ['caching/avoidance', 'Python-level', 'Cython',
              'ufunc/C-loop', 'C/C++ optimization', 'Rust rewrite', 'SIMD/vectorization']

fig2, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: solve rate by technique (top 4 models)
ax = axes[0]
top_models = sorted(MODELS, key=lambda m: -fit_results[m]["level_50"])[:4]
bar_width = 0.18
techniques_with_data = []
for tech in tech_order:
    inst_ids = [iid for iid, m in instance_meta.items() if m["technique"] == tech]
    if len(inst_ids) >= 3:
        techniques_with_data.append(tech)

x_pos = np.arange(len(techniques_with_data))
for i, model in enumerate(top_models):
    rates = []
    for tech in techniques_with_data:
        inst_ids = [iid for iid, m in instance_meta.items() if m["technique"] == tech]
        solved = sum(1 for iid in inst_ids
                     if (all_results[model].get(iid) or {}).get("opt_commit"))
        total = sum(1 for iid in inst_ids
                    if all_results[model].get(iid) is not None and
                    all_results[model][iid] is not None and
                    all_results[model][iid].get("test_passed") is not None)
        rates.append(solved / total if total > 0 else 0)
    ax.bar(x_pos + i * bar_width, rates, bar_width, color=MODEL_COLORS[model],
           label=MODEL_LABELS[model], alpha=0.85)

ax.set_xticks(x_pos + bar_width * 1.5)
ax.set_xticklabels(techniques_with_data, rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("Solve Rate", fontsize=11)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.set_title("Solve Rate by Optimization Technique", fontsize=12)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.2, axis="y")

# Right: median optimization level by technique (top 4 models)
ax = axes[1]
for i, model in enumerate(top_models):
    med_ratios = []
    for tech in techniques_with_data:
        inst_ids = {iid for iid, m in instance_meta.items() if m["technique"] == tech}
        tech_ratios = [ratio for iid, ratio in model_ratios[model] if iid in inst_ids]
        med_ratios.append(np.median(tech_ratios) if tech_ratios else 0)
    ax.bar(x_pos + i * bar_width, med_ratios, bar_width, color=MODEL_COLORS[model],
           label=MODEL_LABELS[model], alpha=0.85)

ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4, linewidth=1)
ax.set_xticks(x_pos + bar_width * 1.5)
ax.set_xticklabels(techniques_with_data, rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("Median Optimization Level", fontsize=11)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.set_title("Median Optimization Level by Technique", fontsize=12)
ax.legend(fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.2, axis="y")

fig2.suptitle("Optimization Capability by Technique Type", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "optimization_by_technique.png"), dpi=150, bbox_inches="tight")
print(f"Saved: optimization_by_technique.png")

# ---------------------------------------------------------------------------
# Figure 3: The "partial optimization" story
# ---------------------------------------------------------------------------
fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: distribution of optimization levels for top model
model = "claude-opus-4.6"
ratios_arr = np.array([r for _, r in model_ratios[model]])
ax_left.hist(ratios_arr, bins=25, range=(0, 1.5), color=MODEL_COLORS[model],
             alpha=0.7, edgecolor="white", linewidth=0.5)
ax_left.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                label="Expert level (100%)")
ax_left.axvline(x=np.median(ratios_arr), color="black", linestyle="-", linewidth=1.5,
                label=f"Median: {np.median(ratios_arr):.0%}")
ax_left.set_xlabel("Optimization Level (model / expert)", fontsize=11)
ax_left.set_ylabel("Count", fontsize=11)
ax_left.set_title(f"Distribution of Optimization Levels\n({MODEL_LABELS[model]})", fontsize=12)
ax_left.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax_left.legend(fontsize=9)
ax_left.grid(True, alpha=0.2, axis="y")

# Right: median optimization level by difficulty tier (all models)
tiers = [(0, 50, '<50'), (50, 150, '50-150'), (150, 500, '150-500'), (500, 10000, '500+')]
tier_labels = [t[2] for t in tiers]
x_pos = np.arange(len(tiers))
bar_width = 0.18

for i, model in enumerate(top_models):
    med_by_tier = []
    for lo, hi, _ in tiers:
        tier_ratios = [ratio for iid, ratio in model_ratios[model]
                       if instance_meta.get(iid, {}).get("diff_lines", 0) >= lo
                       and instance_meta.get(iid, {}).get("diff_lines", 0) < hi]
        med_by_tier.append(np.median(tier_ratios) if tier_ratios else 0)
    ax_right.bar(x_pos + i * bar_width, med_by_tier, bar_width,
                 color=MODEL_COLORS[model], label=MODEL_LABELS[model], alpha=0.85)

ax_right.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4, linewidth=1)
ax_right.set_xticks(x_pos + bar_width * 1.5)
ax_right.set_xticklabels([f"{l} lines" for l in tier_labels], fontsize=9)
ax_right.set_xlabel("Expert Diff Size", fontsize=11)
ax_right.set_ylabel("Median Optimization Level", fontsize=11)
ax_right.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax_right.set_title("Optimization Level by Task Complexity", fontsize=12)
ax_right.legend(fontsize=8, framealpha=0.9)
ax_right.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "optimization_partial.png"), dpi=150, bbox_inches="tight")
print(f"Saved: optimization_partial.png")

# ---------------------------------------------------------------------------
# Compute p50 and p80 OPL (empirical, from logistic fit)
# ---------------------------------------------------------------------------
# p50 OPL: optimization level at which P(achieving) = 50% → -alpha/beta
# p80 OPL: optimization level at which P(achieving) = 80% → -(alpha - log(4))/beta
#   since sigma(x)=0.8 → x=log(4)≈1.386

import datetime

for model in MODELS:
    fr = fit_results[model]
    alpha, beta = fr["alpha"], fr["beta"]
    fr["level_80"] = -(alpha - np.log(4)) / beta if beta != 0 else float("inf")

# Model release dates (from METR eval-analysis-public + web search)
MODEL_RELEASE_DATES = {
    "claude-opus-4.6": "2026-02-05",
    "claude-opus-4.5": "2025-11-24",
    "claude-opus-4": "2025-05-22",
    "claude-sonnet-4.5": "2025-09-29",
    "claude-sonnet-4": "2025-05-22",
    "gpt-5.2": "2025-12-11",
    "gpt-5.1": "2025-11-19",
    "gpt-5": "2025-08-07",
    "gemini-3-pro": "2025-11-18",
    "gemini-3-flash": "2025-12-17",
    "gemini-2.5-pro": "2025-06-17",
    "o3": "2025-04-16",
    "qwen3-coder": "2025-07-22",
    "kimi-k2": "2025-07-11",
    "glm-4.5": "2025-07-28",
}

# ---------------------------------------------------------------------------
# Figures 4a & 4b: OPL horizon over time — separate p50 and p80 plots
# ---------------------------------------------------------------------------
import datetime
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from adjustText import adjust_text

MIN_N_FOR_PLOT = 20  # need enough data for a meaningful logistic fit

plot_data = []
for model in MODELS:
    if model not in fit_results or model not in MODEL_RELEASE_DATES:
        continue
    fr = fit_results[model]
    if fr["n"] < MIN_N_FOR_PLOT:
        print(f"  Skipping {MODEL_LABELS.get(model, model)} for temporal plot (n={fr['n']} < {MIN_N_FOR_PLOT})")
        continue
    dt = datetime.datetime.strptime(MODEL_RELEASE_DATES[model], "%Y-%m-%d")
    plot_data.append({
        "model": model, "date": dt, "label": MODEL_LABELS.get(model, model),
        "p50": fr["level_50"], "p80": fr["level_80"],
        "ci_lo": fr.get("ci_lo"), "ci_hi": fr.get("ci_hi"),
        "color": MODEL_COLORS.get(model, "#888888"),
    })

# Convert dates to ordinal for trend fitting
date_ordinals = np.array([d["date"].toordinal() for d in plot_data])

for metric, metric_label, filename in [
    ("p50", "p50 OPL", "opl_over_time_p50.png"),
    ("p80", "p80 OPL", "opl_over_time_p80.png"),
]:
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6))
    vals = np.array([d[metric] for d in plot_data])

    for ax, title_suffix, is_log in [(ax_lin, "(linear scale)", False),
                                      (ax_log, "(log scale)", True)]:
        texts = []
        for d in plot_data:
            # CI error bars (light gray)
            if metric == "p50" and d["ci_lo"] and d["ci_hi"]:
                yerr_lo = d["p50"] - d["ci_lo"]
                yerr_hi = d["ci_hi"] - d["p50"]
                ax.errorbar(d["date"], d[metric], yerr=[[yerr_lo], [yerr_hi]],
                            fmt='none', ecolor='lightgray', capsize=3,
                            capthick=1, elinewidth=1.5, zorder=2)

            # Point
            ax.scatter(d["date"], d[metric], s=80, color=d["color"],
                       edgecolors='white', linewidths=1.2, zorder=5)

            # Label
            texts.append(ax.text(d["date"], d[metric], d["label"],
                                 fontsize=6.5, color=d["color"], fontweight='bold'))

        # Trend line (linear fit on log values vs time)
        valid = vals > 0
        if valid.sum() >= 3:
            log_vals = np.log(vals[valid])
            ords = date_ordinals[valid]
            coeffs = np.polyfit(ords, log_vals, 1)
            trend_dates = np.linspace(ords.min() - 30, ords.max() + 30, 200)
            trend_vals = np.exp(np.polyval(coeffs, trend_dates))
            trend_dt = [datetime.datetime.fromordinal(int(o)) for o in trend_dates]
            ax.plot(trend_dt, trend_vals, color='gray', linestyle='--',
                    linewidth=1.5, alpha=0.5, zorder=1)

        # Expert level reference
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.3, linewidth=1.5)
        ax.text(plot_data[0]["date"], 1.01, "expert level",
                color="red", fontsize=8, alpha=0.5, va="bottom")

        if is_log:
            ax.set_yscale("log", base=2)
            if metric == "p50":
                yticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            else:
                yticks = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{v:.0%}" for v in yticks])
            if metric == "p50":
                ax.set_ylim(0.35, 1.15)
            else:
                ax.set_ylim(0.08, 1.15)
        else:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
            if metric == "p50":
                ax.set_ylim(0.3, 1.1)
            else:
                ax.set_ylim(0.0, 1.1)

        ax.set_xlabel("Model Release Date", fontsize=11)
        ax.set_ylabel("Optimization Level (model / expert speedup)", fontsize=11)
        ax.set_title(f"{metric_label} Over Time {title_suffix}", fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', rotation=30)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        try:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray',
                        alpha=0.3, lw=0.5), force_text=(0.3, 0.3))
        except Exception:
            pass

    fig.suptitle(f"Optimization Proficiency Level — {metric_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")

# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------
summary = {
    "description": (
        "Optimization Proficiency Level (OPL): at 50% success rate, what optimization "
        "level (model_speedup / expert_speedup) can the model achieve? "
        "Analogous to METR's p50 time-horizon."
    ),
    "methodology": (
        "optimization_level = model_speedup / expert_speedup. "
        "Fit logistic: P(level >= t) = sigma(alpha + beta * t). "
        "p50 OPL = -alpha/beta. p80 OPL = -(alpha - ln4)/beta. "
        "CIs via hierarchical bootstrap."
    ),
    "models": {},
}

for model in sorted_models:
    if model not in fit_results:
        continue
    fr = fit_results[model]
    summary["models"][model] = {
        "label": MODEL_LABELS.get(model, model),
        "release_date": MODEL_RELEASE_DATES.get(model, "unknown"),
        "n_instances": fr["n"],
        "p50_opl": round(fr["level_50"], 4),
        "p80_opl": round(fr["level_80"], 4),
        "logistic_alpha": round(fr["alpha"], 4),
        "logistic_beta": round(fr["beta"], 4),
    }
    if fr["ci_lo"]:
        summary["models"][model]["p50_opl_95ci"] = [
            round(fr["ci_lo"], 4), round(fr["ci_hi"], 4)
        ]

with open(os.path.join(OUT_DIR, "optimization_proficiency.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: optimization_proficiency.json")
print(f"\n{json.dumps(summary, indent=2)}")
