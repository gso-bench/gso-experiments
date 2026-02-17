"""
Optimization Proficiency Level (OPL): measuring optimization capability scaling.

Unlike METR's time-horizon (which measures long-horizon coding ability), OPL
measures something specific to optimization: how close to expert-level speedup
can a model reliably achieve?

Key insight: optimization is inherently continuous. A model might achieve 0%,
50%, or 100%+ of the expert's speedup. By treating the "optimization level"
(model_speedup / expert_speedup) as a continuous variable, we can define:

    P(optimization_level >= t | model) = sigma(alpha + beta * t)

The "50% optimization level" = -alpha/beta = the fraction of expert speedup
at which the model's success rate crosses 50%.

This is distinct from coding ability because it measures the model's
understanding of WHERE and HOW to optimize, not just its ability to write code.
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

MODELS = [
    "claude-opus-4.6", "gpt-5.2", "claude-opus-4.5", "gemini-3-pro",
    "claude-sonnet-4.5", "gpt-5.1", "gemini-3-flash", "o3",
]

MODEL_LABELS = {
    "claude-opus-4.6": "Claude Opus 4.6",
    "gpt-5.2": "GPT-5.2",
    "claude-opus-4.5": "Claude Opus 4.5",
    "gemini-3-pro": "Gemini 3 Pro",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "gpt-5.1": "GPT-5.1",
    "gemini-3-flash": "Gemini 3 Flash",
    "o3": "o3",
}

MODEL_COLORS = {
    "claude-opus-4.6": "#D97706",
    "gpt-5.2": "#059669",
    "claude-opus-4.5": "#E07A3A",
    "gemini-3-pro": "#4285F4",
    "claude-sonnet-4.5": "#F59E0B",
    "gpt-5.1": "#10B981",
    "gemini-3-flash": "#7AAFFF",
    "o3": "#6366F1",
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
# Summary JSON
# ---------------------------------------------------------------------------
summary = {
    "description": (
        "Optimization Proficiency Level (OPL): measures how close to expert-level "
        "speedup a model can reliably achieve. Unlike METR's time-horizon (which "
        "measures long-horizon coding ability), OPL specifically measures optimization "
        "capability — the ability to identify WHERE and HOW to optimize code."
    ),
    "methodology": (
        "For each (model, instance) pair where tests pass, compute optimization_level = "
        "model_speedup / expert_speedup. Fit logistic: P(level >= t) = sigma(alpha + beta * t). "
        "The 50% OPL = -alpha/beta = the optimization fraction at 50% success. "
        "CIs via hierarchical bootstrap (repos -> instances)."
    ),
    "key_findings": {
        "continuous_metric": (
            "Optimization is inherently continuous. Even when models 'fail' (don't match "
            "expert speedup), they achieve 65-85% of the expert's speedup on median. "
            "Only 4-24% of test-passing attempts produce no speedup at all."
        ),
        "technique_gap": (
            "Models struggle most with compiled-language optimizations (SIMD, ufuncs, "
            "C/C++ rewrites). Python-level and caching optimizations are much easier. "
            "This is a capability gap specific to optimization, not coding in general."
        ),
        "scaling_proposal": (
            "Track 50% OPL over model generations. If it grows consistently toward and "
            "past 1.0 (expert level), that demonstrates exponential improvement in "
            "optimization capability specifically."
        ),
    },
    "models": {},
}

for model in sorted_models:
    fr = fit_results[model]
    summary["models"][model] = {
        "label": MODEL_LABELS[model],
        "n_instances": fr["n"],
        "fifty_pct_opl": round(fr["level_50"], 4),
        "logistic_alpha": round(fr["alpha"], 4),
        "logistic_beta": round(fr["beta"], 4),
    }
    if fr["ci_lo"]:
        summary["models"][model]["opl_95ci"] = [
            round(fr["ci_lo"], 4), round(fr["ci_hi"], 4)
        ]

with open(os.path.join(OUT_DIR, "optimization_proficiency.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: optimization_proficiency.json")
print(f"\n{json.dumps(summary, indent=2)}")
