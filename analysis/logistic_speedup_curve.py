"""
GSO Speedup Horizon: METR-style logistic curve analysis.

Adapts METR's time-horizon methodology (arXiv:2503.14499) to the GSO benchmark.
METR fits:  P(success) = sigma(alpha + beta * log2(human_minutes))
We fit:     P(solve)   = sigma(alpha + beta * log2(expert_speedup))

The 50% horizon is the expert speedup at which the model's predicted
solve rate crosses 50%:  horizon = 2^(-alpha/beta)

Fitting: standard logistic regression via MLE (sklearn LogisticRegression
with no regularization). Confidence intervals via hierarchical bootstrap
over task families (repos) and instances, following METR's approach.
"""

import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
with open("/tmp/logistic/all_results.json") as f:
    all_results = json.load(f)

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
# Compute instance metadata
# ---------------------------------------------------------------------------
from datasets import load_dataset
ds = load_dataset("gso-bench/gso", split="test")
hf_data = {row["instance_id"]: row for row in ds}

# Target speedups (consensus across models)
instance_target = {}
for model in MODELS:
    for inst_id, r in all_results[model].items():
        if r and r.get("gm_speedup_commit_base") and inst_id not in instance_target:
            target = r["gm_speedup_commit_base"]
            if target and target > 0:
                instance_target[inst_id] = target

# Filter anomalous (target < 1x or extreme outliers)
instance_target = {k: v for k, v in instance_target.items() if 1.0 <= v <= 100.0}

# Get repo (task family) for hierarchical bootstrap
instance_repo = {}
for inst_id in instance_target:
    instance_repo[inst_id] = inst_id.split("__")[0]  # e.g. "numpy__numpy-ba89ef9" -> "numpy"

print(f"Instances: {len(instance_target)}")
print(f"Repos: {len(set(instance_repo.values()))}: {sorted(set(instance_repo.values()))}")

# Compiled-code classification
compiled_extensions = {'.c', '.cpp', '.cxx', '.h', '.hpp', '.pyx', '.pxd', '.rs', '.go'}
instance_compiled = {}
for inst_id in instance_target:
    row = hf_data.get(inst_id)
    if not row:
        continue
    diff = row["gt_diff"]
    has_compiled = False
    for line in diff.split('\n'):
        if line.startswith('diff --git'):
            parts = line.split()
            if len(parts) >= 3:
                fname = parts[2].lstrip('a/')
                ext = '.' + fname.rsplit('.', 1)[-1] if '.' in fname else ''
                if ext in compiled_extensions:
                    has_compiled = True
                    break
    instance_compiled[inst_id] = has_compiled

# ---------------------------------------------------------------------------
# Build per-model datasets
# ---------------------------------------------------------------------------
model_data = {}
for model in MODELS:
    instances, log2_targets, outcomes, repos, compiled = [], [], [], [], []
    for inst_id, target in instance_target.items():
        r = all_results[model].get(inst_id)
        if r is None:
            continue
        if r.get("test_passed") is not None:
            instances.append(inst_id)
            log2_targets.append(np.log2(target))
            outcomes.append(1 if r.get("opt_commit") else 0)
            repos.append(instance_repo[inst_id])
            compiled.append(instance_compiled.get(inst_id, False))
    model_data[model] = {
        "instances": instances,
        "log2_target": np.array(log2_targets),
        "outcome": np.array(outcomes),
        "repo": repos,
        "compiled": np.array(compiled),
    }
    n = len(outcomes)
    s = sum(outcomes)
    print(f"{MODEL_LABELS[model]:20s}: {s:2d}/{n} solved ({100*s/n:.1f}%)")


# ---------------------------------------------------------------------------
# Logistic regression via MLE (no regularization)
# ---------------------------------------------------------------------------
def fit_logistic_mle(log2_x, y):
    """Fit P(y=1) = sigma(alpha + beta * log2_x) via MLE.
    Returns alpha, beta, horizon_50pct."""
    X = log2_x.reshape(-1, 1)
    # C=1e10 effectively disables regularization
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
    clf.fit(X, y)
    alpha = clf.intercept_[0]
    beta = clf.coef_[0, 0]

    # 50% horizon: sigma(alpha + beta * log2(h)) = 0.5
    # => alpha + beta * log2(h) = 0 => log2(h) = -alpha/beta
    if beta != 0:
        log2_horizon = -alpha / beta
        horizon = 2 ** log2_horizon
    else:
        horizon = float("inf")

    return alpha, beta, horizon


def bootstrap_horizon(log2_x, y, repos, n_boot=2000, seed=42):
    """Hierarchical bootstrap: resample repos, then instances within repos."""
    rng = np.random.RandomState(seed)
    unique_repos = list(set(repos))
    repo_arr = np.array(repos)
    horizons = []

    for _ in range(n_boot):
        # Level 1: resample repos with replacement
        boot_repos = rng.choice(unique_repos, size=len(unique_repos), replace=True)

        # Level 2: for each sampled repo, resample its instances
        boot_indices = []
        for repo in boot_repos:
            repo_indices = np.where(repo_arr == repo)[0]
            sampled = rng.choice(repo_indices, size=len(repo_indices), replace=True)
            boot_indices.extend(sampled)

        boot_indices = np.array(boot_indices)
        bx, by = log2_x[boot_indices], y[boot_indices]

        # Need both classes present
        if by.sum() == 0 or by.sum() == len(by):
            continue

        try:
            _, _, h = fit_logistic_mle(bx, by)
            if 0.5 <= h <= 1000:  # filter degenerate fits
                horizons.append(h)
        except Exception:
            continue

    if len(horizons) < 100:
        return None, None
    horizons = np.array(horizons)
    return np.percentile(horizons, 2.5), np.percentile(horizons, 97.5)


# ---------------------------------------------------------------------------
# Fit each model
# ---------------------------------------------------------------------------
fit_results = {}
print("\n=== Logistic Regression Fits (METR-style) ===")
print(f"{'Model':20s} {'alpha':>7s} {'beta':>7s} {'50% horizon':>12s}  {'95% CI':>20s}")
print("-" * 75)

for model in MODELS:
    d = model_data[model]
    alpha, beta, horizon = fit_logistic_mle(d["log2_target"], d["outcome"])

    ci_lo, ci_hi = bootstrap_horizon(d["log2_target"], d["outcome"], d["repo"])

    fit_results[model] = {
        "alpha": alpha, "beta": beta, "horizon": horizon,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }

    ci_str = f"[{ci_lo:.1f}x, {ci_hi:.1f}x]" if ci_lo else "N/A"
    print(f"{MODEL_LABELS[model]:20s} {alpha:7.3f} {beta:7.3f} {horizon:10.2f}x  {ci_str:>20s}")

# ---------------------------------------------------------------------------
# Binned empirical rates (for validation overlay)
# ---------------------------------------------------------------------------
bin_edges = np.array([1.0, 1.3, 1.6, 2.0, 2.5, 3.5, 5.0, 8.0, 100.0])


def bin_solve_rates(targets, outcomes, edges):
    centers, rates, counts, ci_lo, ci_hi = [], [], [], [], []
    for i in range(len(edges) - 1):
        mask = (targets >= edges[i]) & (targets < edges[i + 1])
        n = mask.sum()
        if n >= 3:
            r = outcomes[mask].mean()
            # Wilson CI
            z = 1.96
            p = outcomes[mask].sum() / n
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
            centers.append(np.sqrt(edges[i] * edges[i + 1]))
            rates.append(r)
            counts.append(n)
            ci_lo.append(max(0, center - spread))
            ci_hi.append(min(1, center + spread))
    return (np.array(centers), np.array(rates), np.array(counts),
            np.array(ci_lo), np.array(ci_hi))


# ---------------------------------------------------------------------------
# Figure 1: Per-model logistic curves with empirical overlay
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

x_plot = np.linspace(1.0, 60.0, 500)
log2_x_plot = np.log2(x_plot)

# Sort models by horizon (best first)
sorted_models = sorted(MODELS, key=lambda m: -fit_results[m]["horizon"])

for model in sorted_models:
    d = model_data[model]
    fr = fit_results[model]
    color = MODEL_COLORS[model]
    label = MODEL_LABELS[model]

    # Fitted sigmoid
    p_plot = 1.0 / (1.0 + np.exp(-(fr["alpha"] + fr["beta"] * log2_x_plot)))
    horizon_label = f'{label}: {fr["horizon"]:.1f}x'
    if fr["ci_lo"]:
        horizon_label += f' [{fr["ci_lo"]:.1f}, {fr["ci_hi"]:.1f}]'
    ax.plot(x_plot, p_plot, color=color, linewidth=2, label=horizon_label, zorder=2)

    # Empirical binned points
    raw_targets = 2 ** d["log2_target"]
    centers, rates, counts, _, _ = bin_solve_rates(raw_targets, d["outcome"], bin_edges)
    ax.scatter(centers, rates, color=color, s=counts * 3.5, alpha=0.35, edgecolors="none", zorder=3)

# 50% reference line
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
ax.text(55, 0.52, "50%", color="gray", fontsize=9, ha="right", va="bottom")

ax.set_xscale("log", base=2)
ax.set_xlabel("Expert Speedup (log₂ scale)", fontsize=12)
ax.set_ylabel("P(solve)", fontsize=12)
ax.set_title("GSO Speedup Horizon\nP(solve) = σ(α + β · log₂(expert_speedup))", fontsize=13)
ax.set_xlim(1.0, 60.0)
ax.set_ylim(-0.03, 1.03)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.4g}x"))
ax.xaxis.set_major_locator(ticker.FixedLocator([1, 1.5, 2, 3, 4, 6, 8, 16, 32]))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.legend(loc="upper right", fontsize=8.5, title="Model: 50% horizon [95% CI]",
          title_fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "speedup_horizon.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "speedup_horizon.pdf"), bbox_inches="tight")
print("\nSaved: speedup_horizon.png")

# ---------------------------------------------------------------------------
# Figure 2: Split by compiled vs Python-only
# ---------------------------------------------------------------------------
fig2, (ax_py, ax_comp) = plt.subplots(1, 2, figsize=(15, 6))

for ax, comp_flag, panel_title in [(ax_py, False, "Python-Only Tasks"), (ax_comp, True, "Compiled-Code Tasks")]:
    for model in sorted_models:
        d = model_data[model]
        mask = d["compiled"] == comp_flag
        if mask.sum() < 10:
            continue

        log2_x = d["log2_target"][mask]
        y = d["outcome"][mask]
        repos_sub = [d["repo"][i] for i in range(len(d["repo"])) if mask[i]]

        try:
            alpha, beta, horizon = fit_logistic_mle(log2_x, y)
        except Exception:
            continue

        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]

        p_plot = 1.0 / (1.0 + np.exp(-(alpha + beta * log2_x_plot)))
        ax.plot(x_plot, p_plot, color=color, linewidth=1.8,
                label=f'{label}: {horizon:.1f}x ({100*y.mean():.0f}%)', zorder=2)

        raw_t = 2 ** log2_x
        centers, rates, counts, _, _ = bin_solve_rates(raw_t, y, bin_edges)
        ax.scatter(centers, rates, color=color, s=counts * 3, alpha=0.3, edgecolors="none", zorder=3)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Expert Speedup (log₂ scale)", fontsize=11)
    ax.set_ylabel("P(solve)", fontsize=11)
    ax.set_title(panel_title, fontsize=12)
    ax.set_xlim(1.0, 60.0)
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.4g}x"))
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 1.5, 2, 3, 4, 8, 16, 32]))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=7.5, title="Model: 50% horizon (solve rate)",
              title_fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)

fig2.suptitle("GSO Speedup Horizon by Optimization Type", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "speedup_horizon_by_type.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "speedup_horizon_by_type.pdf"), bbox_inches="tight")
print("Saved: speedup_horizon_by_type.png")

# ---------------------------------------------------------------------------
# Figure 3: Difficulty predictors bar chart
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Pool all models for aggregate stats
all_targets = np.concatenate([2 ** model_data[m]["log2_target"] for m in MODELS])
all_outcomes = np.concatenate([model_data[m]["outcome"] for m in MODELS])
all_compiled = np.concatenate([model_data[m]["compiled"] for m in MODELS])

# Diff metadata per data point
from datasets import load_dataset
instance_diff_lines = {}
instance_n_files = {}
for inst_id in instance_target:
    row = hf_data.get(inst_id)
    if not row:
        continue
    diff = row["gt_diff"]
    files = set()
    for line in diff.split('\n'):
        if line.startswith('diff --git'):
            parts = line.split()
            if len(parts) >= 3:
                files.add(parts[2].lstrip('a/'))
    instance_diff_lines[inst_id] = sum(1 for l in diff.split('\n') if l.startswith('+') or l.startswith('-'))
    instance_n_files[inst_id] = len(files)

all_diff_lines = np.concatenate([
    np.array([instance_diff_lines.get(iid, 0) for iid in model_data[m]["instances"]])
    for m in MODELS
])
all_n_files = np.concatenate([
    np.array([instance_n_files.get(iid, 0) for iid in model_data[m]["instances"]])
    for m in MODELS
])

# Panel A: compiled vs python-only
ax = axes3[0]
categories = ["Python-only", "Compiled"]
rates_cat = [all_outcomes[~all_compiled].mean(), all_outcomes[all_compiled].mean()]
counts_cat = [(~all_compiled).sum(), all_compiled.sum()]
colors_cat = ["#2563EB", "#DC2626"]
ax.bar(categories, rates_cat, color=colors_cat, alpha=0.7, width=0.5)
for i, (r, n) in enumerate(zip(rates_cat, counts_cat)):
    ax.text(i, r + 0.01, f"{r:.0%}\n(n={n})", ha="center", fontsize=9)
ax.set_ylabel("Solve Rate", fontsize=10)
ax.set_title("By Language Type", fontsize=11)
ax.set_ylim(0, 0.45)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.grid(True, alpha=0.2, axis="y")

# Panel B: by diff size
ax = axes3[1]
size_bins = [("< 100", all_diff_lines < 100),
             ("100-500", (all_diff_lines >= 100) & (all_diff_lines < 500)),
             ("500+", all_diff_lines >= 500)]
for i, (label, mask) in enumerate(size_bins):
    r = all_outcomes[mask].mean()
    n = mask.sum()
    ax.bar(i, r, color="#8B5CF6", alpha=0.6 + 0.15 * (2 - i), width=0.5)
    ax.text(i, r + 0.01, f"{r:.0%}\n(n={n})", ha="center", fontsize=9)
ax.set_xticks(range(len(size_bins)))
ax.set_xticklabels([s[0] for s in size_bins])
ax.set_xlabel("Expert Diff Size (lines)", fontsize=10)
ax.set_title("By Diff Size", fontsize=11)
ax.set_ylim(0, 0.45)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.grid(True, alpha=0.2, axis="y")

# Panel C: by file count
ax = axes3[2]
file_bins = [("1 file", all_n_files == 1),
             ("2-5 files", (all_n_files >= 2) & (all_n_files <= 5)),
             ("6+ files", all_n_files >= 6)]
for i, (label, mask) in enumerate(file_bins):
    r = all_outcomes[mask].mean()
    n = mask.sum()
    ax.bar(i, r, color="#059669", alpha=0.6 + 0.15 * (2 - i), width=0.5)
    ax.text(i, r + 0.01, f"{r:.0%}\n(n={n})", ha="center", fontsize=9)
ax.set_xticks(range(len(file_bins)))
ax.set_xticklabels([f[0] for f in file_bins])
ax.set_xlabel("Files Changed in Expert Diff", fontsize=10)
ax.set_title("By Diff Scope", fontsize=11)
ax.set_ylim(0, 0.45)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.grid(True, alpha=0.2, axis="y")

fig3.suptitle("What Predicts GSO Difficulty? (Solve Rate by Task Property)",
              fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "difficulty_predictors.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "difficulty_predictors.pdf"), bbox_inches="tight")
print("Saved: difficulty_predictors.png")

# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------
summary = {
    "description": (
        "METR-style logistic analysis for GSO. P(solve) = sigma(alpha + beta * log2(expert_speedup)). "
        "The 50% horizon is 2^(-alpha/beta): the expert speedup at which the model has a 50% predicted solve rate."
    ),
    "methodology": (
        "Standard logistic regression via MLE (sklearn, no regularization). "
        "95% CI via hierarchical bootstrap (resample repos, then instances within repos, 2000 iterations). "
        "Following METR arXiv:2503.14499."
    ),
    "models": {},
}

for model in sorted_models:
    fr = fit_results[model]
    d = model_data[model]
    entry = {
        "label": MODEL_LABELS[model],
        "n_instances": len(d["outcome"]),
        "solved": int(d["outcome"].sum()),
        "solve_rate": round(float(d["outcome"].mean()), 3),
        "logistic_fit": {
            "alpha": round(fr["alpha"], 4),
            "beta": round(fr["beta"], 4),
            "horizon_50pct": round(fr["horizon"], 2),
        },
    }
    if fr["ci_lo"]:
        entry["logistic_fit"]["horizon_95ci"] = [round(fr["ci_lo"], 2), round(fr["ci_hi"], 2)]
    summary["models"][model] = entry

summary["interpretation"] = {
    "beta_meaning": (
        "beta < 0 means higher speedup predicts lower solve rate (expected). "
        "beta ≈ 0 means speedup magnitude does not predict solvability."
    ),
    "horizon_meaning": (
        "The 50% horizon is the expert speedup level where the model is predicted to "
        "solve half the tasks. Higher horizon = model handles harder optimizations. "
        "Infinite/very large horizon means the model never drops below 50% (or is always below)."
    ),
}

with open(os.path.join(OUT_DIR, "logistic_speedup_curve.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + json.dumps(summary, indent=2))
