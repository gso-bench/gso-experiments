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
import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from adjustText import adjust_text

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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

# Model release dates
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

# Minimum instances with test-passing results for meaningful logistic fit
MIN_N = 20

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
# Instance metadata (repo info for hierarchical bootstrap)
# ---------------------------------------------------------------------------
instance_meta = {}
for row in ds:
    inst_id = row["instance_id"]
    instance_meta[inst_id] = {"repo": inst_id.split("__")[0]}

# ---------------------------------------------------------------------------
# Compute optimization level (= model_speedup / expert_speedup) per model
# Uses ALL instances as denominator — test failures count as ratio=0.
# This makes the logistic directly comparable to leaderboard opt@1.
# ---------------------------------------------------------------------------
all_instance_ids = sorted(instance_expert_speedup.keys())

model_ratios = {}  # model -> list of (inst_id, ratio)
for model in MODELS:
    ratios = []
    for inst_id in all_instance_ids:
        r = all_results[model].get(inst_id)
        if r and r.get("test_passed"):
            model_sp = r.get("gm_speedup_patch_base")
            expert_sp = instance_expert_speedup[inst_id]
            if model_sp and expert_sp:
                ratios.append((inst_id, model_sp / expert_sp))
            else:
                ratios.append((inst_id, 0.0))
        else:
            # Test failure or missing result → ratio = 0
            ratios.append((inst_id, 0.0))
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
    level_80 = -(alpha - np.log(4)) / beta if beta != 0 else float("inf")
    return alpha, beta, level_50, level_80


def bootstrap_opl(ratios, repos_list, n_boot=2000, seed=42):
    """Hierarchical bootstrap for p50 and p80 OPL CIs."""
    rng = np.random.RandomState(seed)
    repo_arr = np.array(repos_list)
    unique_repos = list(set(repos_list))
    p50_levels, p80_levels = [], []
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
            _, beta, lv50, lv80 = fit_opl(boot_ratios)
            if beta < 0:
                if 0.1 <= lv50 <= 2.0:
                    p50_levels.append(lv50)
                if 0.01 <= lv80 <= 2.0:
                    p80_levels.append(lv80)
        except Exception:
            continue

    p50_ci = (np.percentile(p50_levels, 2.5), np.percentile(p50_levels, 97.5)) \
        if len(p50_levels) >= 100 else (None, None)
    p80_ci = (np.percentile(p80_levels, 2.5), np.percentile(p80_levels, 97.5)) \
        if len(p80_levels) >= 100 else (None, None)
    return p50_ci, p80_ci


print("=" * 70)
print("Optimization Proficiency Level (OPL) Analysis")
print("=" * 70)
print()

fit_results = {}
print(f"{'Model':20s} {'n':>4s} {'pass':>5s} {'alpha':>7s} {'beta':>7s} {'p50 OPL':>10s} {'p80 OPL':>10s} {'p50 95% CI':>20s} {'p80 95% CI':>20s}")
print("-" * 105)

for model in MODELS:
    ratios = model_ratios[model]
    n_nonzero = sum(1 for _, r in ratios if r > 0)
    repos = [instance_meta.get(iid, {}).get("repo", "?") for iid, _ in ratios]
    alpha, beta, level_50, level_80 = fit_opl(ratios)
    p50_ci, p80_ci = bootstrap_opl(ratios, repos)
    fit_results[model] = {
        "alpha": alpha, "beta": beta,
        "level_50": level_50, "level_80": level_80,
        "p50_ci_lo": p50_ci[0], "p50_ci_hi": p50_ci[1],
        "p80_ci_lo": p80_ci[0], "p80_ci_hi": p80_ci[1],
        "n": len(ratios), "n_nonzero": n_nonzero,
    }
    p50_ci_str = f"[{p50_ci[0]:.3f}, {p50_ci[1]:.3f}]" if p50_ci[0] else "N/A"
    p80_ci_str = f"[{p80_ci[0]:.3f}, {p80_ci[1]:.3f}]" if p80_ci[0] else "N/A"
    label = MODEL_LABELS.get(model, model)
    print(f"{label:20s} {len(ratios):4d} {n_nonzero:5d} {alpha:7.3f} {beta:7.3f} {level_50:10.1%} {level_80:10.1%} {p50_ci_str:>20s} {p80_ci_str:>20s}")

# Models with enough non-zero (test-passing) instances for reliable logistic fit
plotable_models = [m for m in MODELS if fit_results[m]["n_nonzero"] >= MIN_N]
sorted_models = sorted(plotable_models, key=lambda m: -fit_results[m]["level_50"])

print(f"\nModels with >= {MIN_N} test-passing instances for plots: {len(sorted_models)}")
for m in MODELS:
    if fit_results[m]["n_nonzero"] < MIN_N:
        print(f"  Skipping {MODEL_LABELS.get(m, m)} (n_pass={fit_results[m]['n_nonzero']})")

# ---------------------------------------------------------------------------
# Figure 1: Optimization Proficiency Curves (only models with enough data)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

thresholds = np.linspace(0.0, 1.3, 200)

for model in sorted_models:
    fr = fit_results[model]
    color = MODEL_COLORS.get(model, "#888888")
    label = MODEL_LABELS.get(model, model)

    # Fitted sigmoid
    p_curve = 1.0 / (1.0 + np.exp(-(fr["alpha"] + fr["beta"] * thresholds)))
    ci_str = ""
    if fr["p50_ci_lo"]:
        ci_str = f" [{fr['p50_ci_lo']:.0%}, {fr['p50_ci_hi']:.0%}]"
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
             "P(model_speedup / expert_speedup \u2265 t) = \u03c3(\u03b1 + \u03b2\u00b7t)",
             fontsize=13)
ax.set_xlim(-0.02, 1.3)
ax.set_ylim(-0.03, 1.03)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.legend(loc="upper right", fontsize=8.5,
          title="Model: p50 OPL [95% CI]", title_fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "optimization_proficiency.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: optimization_proficiency.png")

# ---------------------------------------------------------------------------
# Figures 2a & 2b: OPL over time — separate p50 and p80 plots
# ---------------------------------------------------------------------------
plot_data = []
for model in sorted_models:
    if model not in MODEL_RELEASE_DATES:
        continue
    fr = fit_results[model]
    dt = datetime.datetime.strptime(MODEL_RELEASE_DATES[model], "%Y-%m-%d")
    plot_data.append({
        "model": model, "date": dt, "label": MODEL_LABELS.get(model, model),
        "p50": fr["level_50"], "p80": fr["level_80"],
        "p50_ci_lo": fr["p50_ci_lo"], "p50_ci_hi": fr["p50_ci_hi"],
        "p80_ci_lo": fr["p80_ci_lo"], "p80_ci_hi": fr["p80_ci_hi"],
        "color": MODEL_COLORS.get(model, "#888888"),
    })

date_ordinals = np.array([d["date"].toordinal() for d in plot_data])

for metric, metric_label, filename in [
    ("p50", "p50 OPL", "opl_over_time_p50.png"),
    ("p80", "p80 OPL", "opl_over_time_p80.png"),
]:
    # Filter to models with positive metric values (negative = undefined)
    metric_data = [d for d in plot_data if d[metric] > 0]
    metric_ordinals = np.array([d["date"].toordinal() for d in metric_data])

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6))
    vals = np.array([d[metric] for d in metric_data])
    ci_lo_key = f"{metric}_ci_lo"
    ci_hi_key = f"{metric}_ci_hi"

    for ax, title_suffix, is_log in [(ax_lin, "(linear scale)", False),
                                      (ax_log, "(log scale)", True)]:
        texts = []
        for d in metric_data:
            # CI error bars (light gray)
            if d[ci_lo_key] is not None and d[ci_hi_key] is not None:
                yerr_lo = max(0, d[metric] - d[ci_lo_key])
                yerr_hi = max(0, d[ci_hi_key] - d[metric])
                if yerr_lo > 0 or yerr_hi > 0:
                    ax.errorbar(d["date"], d[metric], yerr=[[yerr_lo], [yerr_hi]],
                                fmt='none', ecolor='lightgray', capsize=3,
                                capthick=1, elinewidth=1.5, zorder=2)

            # Point
            ax.scatter(d["date"], d[metric], s=80, color=d["color"],
                       edgecolors='white', linewidths=1.2, zorder=5)

            # Label
            texts.append(ax.text(d["date"], d[metric], d["label"],
                                 fontsize=6.5, color=d["color"], fontweight='bold'))

        # Exponential trend: fit linear in log-space → straight on log, curved on linear
        valid = np.isfinite(vals) & (vals > 0)
        if valid.sum() >= 3:
            ords = metric_ordinals[valid]
            log_y = np.log(vals[valid])
            coeffs = np.polyfit(ords, log_y, 1)  # log(y) = a*t + b
            trend_dates = np.linspace(ords.min() - 30, ords.max() + 30, 200)
            trend_vals = np.exp(np.polyval(coeffs, trend_dates))
            trend_dt = [datetime.datetime.fromordinal(int(o)) for o in trend_dates]
            ax.plot(trend_dt, trend_vals, color='gray', linestyle='--',
                    linewidth=1.5, alpha=0.5, zorder=1)

        # Expert level reference
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.3, linewidth=1.5)
        ax.text(metric_data[0]["date"], 1.01, "expert level",
                color="red", fontsize=8, alpha=0.5, va="bottom")

        if is_log:
            ax.set_yscale("log", base=2)
            if metric == "p50":
                yticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ax.set_ylim(0.35, 1.15)
            else:
                yticks = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
                ax.set_ylim(0.08, 1.15)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{v:.0%}" for v in yticks])
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

    fig.suptitle(f"Optimization Proficiency Level \u2014 {metric_label}",
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

all_sorted = sorted(MODELS, key=lambda m: -fit_results[m]["level_50"])
for model in all_sorted:
    fr = fit_results[model]
    entry = {
        "label": MODEL_LABELS.get(model, model),
        "release_date": MODEL_RELEASE_DATES.get(model, "unknown"),
        "n_instances": fr["n"],
        "n_test_passing": fr["n_nonzero"],
        "p50_opl": round(fr["level_50"], 4),
        "p80_opl": round(fr["level_80"], 4),
        "logistic_alpha": round(fr["alpha"], 4),
        "logistic_beta": round(fr["beta"], 4),
    }
    if fr["p50_ci_lo"]:
        entry["p50_opl_95ci"] = [round(fr["p50_ci_lo"], 4), round(fr["p50_ci_hi"], 4)]
    if fr["p80_ci_lo"]:
        entry["p80_opl_95ci"] = [round(fr["p80_ci_lo"], 4), round(fr["p80_ci_hi"], 4)]
    summary["models"][model] = entry

with open(os.path.join(OUT_DIR, "optimization_proficiency.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: optimization_proficiency.json")
print(f"\n{json.dumps(summary, indent=2)}")
