"""
GSO Difficulty Horizon: METR-style logistic curve analysis.

Adapts METR's time-horizon methodology (arXiv:2503.14499) to the GSO benchmark.

METR fits:  P(success) = sigma(alpha + beta * log2(human_minutes))
GSO fit:    P(solve)   = sigma(alpha + beta * log2(diff_lines))

METR uses human task duration as a 1D difficulty proxy. We find that for GSO,
the analogous proxy is expert diff size (lines changed). Expert speedup magnitude
has NO predictive power (beta ≈ 0, AIC worse than null model).

The "50% diff-line horizon" is the expert diff size at which a model's predicted
solve rate crosses 50%:  horizon = 2^(-alpha/beta)

Fitting: standard logistic regression via MLE. Confidence intervals via
hierarchical bootstrap over repos and instances, following METR.
"""

import json
import os
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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
# Compute instance metadata from expert diffs
# ---------------------------------------------------------------------------
from datasets import load_dataset
ds = load_dataset("gso-bench/gso", split="test")
hf_data = {row["instance_id"]: row for row in ds}

# Target speedups
instance_target = {}
for model in MODELS:
    for inst_id, r in all_results[model].items():
        if r and r.get("gm_speedup_commit_base") and inst_id not in instance_target:
            t = r["gm_speedup_commit_base"]
            if t and t > 0:
                instance_target[inst_id] = t
instance_target = {k: v for k, v in instance_target.items() if 1.0 <= v <= 100.0}

# Parse expert diffs
compiled_ext = {'.c', '.cpp', '.cxx', '.h', '.hpp', '.pyx', '.pxd', '.rs', '.go'}
instance_meta = {}
for inst_id in instance_target:
    row = hf_data.get(inst_id)
    if not row:
        continue
    diff = row["gt_diff"]
    files = set()
    has_compiled = False
    for line in diff.split('\n'):
        if line.startswith('diff --git'):
            parts = line.split()
            if len(parts) >= 3:
                fname = parts[2].lstrip('a/')
                files.add(fname)
                ext = '.' + fname.rsplit('.', 1)[-1] if '.' in fname else ''
                if ext in compiled_ext:
                    has_compiled = True
    diff_lines = max(1, sum(1 for l in diff.split('\n') if l.startswith('+') or l.startswith('-')))
    instance_meta[inst_id] = {
        "diff_lines": diff_lines,
        "n_files": len(files),
        "compiled": has_compiled,
        "target": instance_target[inst_id],
        "repo": inst_id.split("__")[0],
    }

# ---------------------------------------------------------------------------
# Build per-model datasets
# ---------------------------------------------------------------------------
model_data = {}
for model in MODELS:
    instances, log2_dl, outcomes, repos = [], [], [], []
    for inst_id, meta in instance_meta.items():
        r = all_results[model].get(inst_id)
        if r is None or r.get("test_passed") is None:
            continue
        instances.append(inst_id)
        log2_dl.append(np.log2(meta["diff_lines"]))
        outcomes.append(1 if r.get("opt_commit") else 0)
        repos.append(meta["repo"])
    model_data[model] = {
        "instances": instances,
        "log2_diff_lines": np.array(log2_dl),
        "outcome": np.array(outcomes),
        "repo": repos,
    }
    s, n = sum(outcomes), len(outcomes)
    print(f"{MODEL_LABELS[model]:20s}: {s:2d}/{n} solved ({100*s/n:.1f}%)")

# ---------------------------------------------------------------------------
# Logistic regression + bootstrap
# ---------------------------------------------------------------------------
def fit_logistic(log2_x, y):
    X = log2_x.reshape(-1, 1)
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000)
    clf.fit(X, y)
    alpha, beta = clf.intercept_[0], clf.coef_[0, 0]
    horizon = 2 ** (-alpha / beta) if beta != 0 else float("inf")
    return alpha, beta, horizon


def bootstrap_horizon(log2_x, y, repos, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    unique_repos = list(set(repos))
    repo_arr = np.array(repos)
    horizons = []
    for _ in range(n_boot):
        boot_repos = rng.choice(unique_repos, size=len(unique_repos), replace=True)
        boot_idx = []
        for repo in boot_repos:
            idx = np.where(repo_arr == repo)[0]
            boot_idx.extend(rng.choice(idx, size=len(idx), replace=True))
        boot_idx = np.array(boot_idx)
        bx, by = log2_x[boot_idx], y[boot_idx]
        if by.sum() == 0 or by.sum() == len(by):
            continue
        try:
            _, beta, h = fit_logistic(bx, by)
            if beta < 0 and 1 <= h <= 50000:
                horizons.append(h)
        except Exception:
            continue
    if len(horizons) < 100:
        return None, None
    return np.percentile(horizons, 2.5), np.percentile(horizons, 97.5)


# Fit each model
fit_results = {}
print(f"\n{'Model':20s} {'alpha':>7s} {'beta':>7s} {'50% horizon':>14s} {'95% CI':>22s}")
print("-" * 75)

for model in MODELS:
    d = model_data[model]
    alpha, beta, horizon = fit_logistic(d["log2_diff_lines"], d["outcome"])
    ci_lo, ci_hi = bootstrap_horizon(d["log2_diff_lines"], d["outcome"], d["repo"])
    fit_results[model] = {"alpha": alpha, "beta": beta, "horizon": horizon, "ci_lo": ci_lo, "ci_hi": ci_hi}
    ci_str = f"[{ci_lo:.0f}, {ci_hi:.0f}]" if ci_lo else "N/A"
    print(f"{MODEL_LABELS[model]:20s} {alpha:7.3f} {beta:7.3f} {horizon:10.0f} lines  {ci_str:>22s}")

# ---------------------------------------------------------------------------
# Feature comparison (for the "why diff_lines" argument)
# ---------------------------------------------------------------------------
print("\n=== Feature comparison (AIC, lower = better) ===")
all_log2_dl = np.concatenate([model_data[m]["log2_diff_lines"] for m in MODELS])
all_outcome = np.concatenate([model_data[m]["outcome"] for m in MODELS])
all_log2_target = np.concatenate([
    np.log2([instance_meta[iid]["target"] for iid in model_data[m]["instances"]])
    for m in MODELS
])
all_log2_nf = np.concatenate([
    np.log2(np.clip([instance_meta[iid]["n_files"] for iid in model_data[m]["instances"]], 1, None))
    for m in MODELS
])
all_compiled = np.concatenate([
    np.array([float(instance_meta[iid]["compiled"]) for iid in model_data[m]["instances"]])
    for m in MODELS
])

null_ll = log_loss(all_outcome, np.full(len(all_outcome), all_outcome.mean()), normalize=False)
null_aic = 2 * 1 + 2 * null_ll

for name, x in [("log2(diff_lines)", all_log2_dl), ("log2(n_files)", all_log2_nf),
                 ("compiled", all_compiled), ("log2(target_speedup)", all_log2_target)]:
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000)
    clf.fit(x.reshape(-1, 1), all_outcome)
    probs = clf.predict_proba(x.reshape(-1, 1))[:, 1]
    ll = log_loss(all_outcome, probs, normalize=False)
    aic = 2 * 2 + 2 * ll
    delta = aic - null_aic
    print(f"  {name:25s}: AIC={aic:.1f} (delta={delta:+.1f}), beta={clf.coef_[0,0]:.4f}")
print(f"  {'null (intercept only)':25s}: AIC={null_aic:.1f}")

# ---------------------------------------------------------------------------
# Binned empirical rates helper
# ---------------------------------------------------------------------------
def bin_data(xs, ys, edges):
    centers, rates, counts, ci_lo, ci_hi = [], [], [], [], []
    for i in range(len(edges) - 1):
        mask = (xs >= edges[i]) & (xs < edges[i + 1])
        n = mask.sum()
        if n >= 5:
            r = ys[mask].mean()
            z = 1.96
            p = ys[mask].sum() / n
            denom = 1 + z**2 / n
            c = (p + z**2 / (2 * n)) / denom
            s = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
            centers.append(np.sqrt(edges[i] * edges[i + 1]))
            rates.append(r)
            counts.append(n)
            ci_lo.append(max(0, c - s))
            ci_hi.append(min(1, c + s))
    return (np.array(centers), np.array(rates), np.array(counts),
            np.array(ci_lo), np.array(ci_hi))


# ---------------------------------------------------------------------------
# Figure 1: Per-model logistic curves -- diff_lines on x-axis
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

x_plot_lines = np.linspace(3, 5000, 500)  # diff lines range
log2_x_plot = np.log2(x_plot_lines)

sorted_models = sorted(MODELS, key=lambda m: -fit_results[m]["horizon"])

dl_bin_edges = np.array([3, 10, 25, 50, 100, 200, 500, 5000])

for model in sorted_models:
    d = model_data[model]
    fr = fit_results[model]
    color = MODEL_COLORS[model]
    label = MODEL_LABELS[model]

    # Fitted sigmoid
    p_plot = 1.0 / (1.0 + np.exp(-(fr["alpha"] + fr["beta"] * log2_x_plot)))
    h = fr["horizon"]
    ci_str = ""
    if fr["ci_lo"]:
        ci_str = f" [{fr['ci_lo']:.0f}, {fr['ci_hi']:.0f}]"
    ax.plot(x_plot_lines, p_plot, color=color, linewidth=2,
            label=f'{label}: {h:.0f} lines{ci_str}', zorder=2)

    # Empirical binned points
    raw_dl = 2 ** d["log2_diff_lines"]
    centers, rates, counts, _, _ = bin_data(raw_dl, d["outcome"], dl_bin_edges)
    ax.scatter(centers, rates, color=color, s=counts * 3, alpha=0.35, edgecolors="none", zorder=3)

    # Mark the 50% horizon
    if 3 <= h <= 5000:
        ax.plot(h, 0.5, 'o', color=color, markersize=7, markeredgecolor='white',
                markeredgewidth=1.5, zorder=5)

ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
ax.text(4500, 0.52, "50%", color="gray", fontsize=9, ha="right", va="bottom")

ax.set_xscale("log", base=2)
ax.set_xlabel("Expert Diff Size (lines changed, log₂ scale)", fontsize=12)
ax.set_ylabel("P(solve)", fontsize=12)
ax.set_title("GSO Difficulty Horizon\nP(solve) = σ(α + β · log₂(diff_lines))", fontsize=13)
ax.set_xlim(3, 5000)
ax.set_ylim(-0.03, 1.03)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.4g}"))
ax.xaxis.set_major_locator(ticker.FixedLocator([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.legend(loc="upper right", fontsize=8.5, title="Model: 50% horizon [95% CI]",
          title_fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "difficulty_horizon.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "difficulty_horizon.pdf"), bbox_inches="tight")
print("\nSaved: difficulty_horizon.png")

# ---------------------------------------------------------------------------
# Figure 2: Contrast -- diff_lines WORKS, target_speedup DOESN'T
# ---------------------------------------------------------------------------
fig2, (ax_good, ax_bad) = plt.subplots(1, 2, figsize=(15, 6))

for ax, x_all, x_label, x_lim, x_ticks, title, bin_edges_local in [
    (ax_good, all_log2_dl, "Expert Diff Size (lines)", (3, 5000),
     [4, 16, 64, 256, 1024, 4096], "log₂(diff_lines): Strong Signal",
     np.log2(dl_bin_edges)),
    (ax_bad, all_log2_target, "Expert Speedup", (1, 60),
     [1, 2, 4, 8, 16, 32],  "log₂(target_speedup): No Signal",
     np.log2([1, 1.3, 1.6, 2, 2.5, 3.5, 5, 8, 100])),
]:
    # Pool all models
    for model in sorted_models:
        d = model_data[model]
        fr = fit_results[model]
        color = MODEL_COLORS[model]

        if ax == ax_good:
            x_m = d["log2_diff_lines"]
        else:
            x_m = np.log2([instance_meta[iid]["target"] for iid in d["instances"]])

        # Per-model fit
        try:
            alpha_m, beta_m, h_m = fit_logistic(x_m, d["outcome"])
        except:
            continue
        x_smooth = np.log2(np.linspace(x_lim[0], x_lim[1], 300))
        p_smooth = 1.0 / (1.0 + np.exp(-(alpha_m + beta_m * x_smooth)))
        ax.plot(2**x_smooth, p_smooth, color=color, linewidth=1.5, alpha=0.7,
                label=MODEL_LABELS[model], zorder=2)

    # Aggregate binned
    raw_x = 2 ** x_all
    if ax == ax_good:
        centers, rates, counts, ci_lo, ci_hi = bin_data(raw_x, all_outcome, dl_bin_edges)
    else:
        centers, rates, counts, ci_lo, ci_hi = bin_data(
            raw_x, all_outcome, np.array([1, 1.3, 1.6, 2, 2.5, 3.5, 5, 8, 100]))

    ax.scatter(centers, rates, color="black", s=counts * 2, alpha=0.5, zorder=4,
               label="Empirical (all models)")
    ax.errorbar(centers, rates, yerr=[rates - ci_lo, ci_hi - rates],
                fmt="none", color="black", capsize=3, linewidth=1, alpha=0.5, zorder=4)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("P(solve)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(*x_lim)
    ax.set_ylim(-0.03, 0.85)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.4g}"))
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)

fig2.suptitle("Finding the Right Difficulty Axis for GSO", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "difficulty_axis_comparison.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "difficulty_axis_comparison.pdf"), bbox_inches="tight")
print("Saved: difficulty_axis_comparison.png")

# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------
summary = {
    "description": (
        "METR-style logistic analysis for GSO. METR uses log2(human_minutes) as the difficulty axis; "
        "we find that log2(diff_lines) is the analogous proxy for GSO. Expert speedup has no predictive power."
    ),
    "methodology": (
        "P(solve) = sigma(alpha + beta * log2(diff_lines)). Standard logistic regression via MLE. "
        "95% CI via hierarchical bootstrap (repos -> instances, 2000 iters). Following METR arXiv:2503.14499."
    ),
    "key_finding": (
        "Expert diff size (lines changed) predicts model success with a clean logistic curve (AIC=714 vs null=749). "
        "Expert speedup magnitude has NO predictive power (AIC=750, worse than null). "
        "This means a 1.3x optimization requiring a 500-line C++ patch is harder than an 8x optimization "
        "requiring a 20-line Python change."
    ),
    "feature_comparison_aic": {
        "log2_diff_lines": 714.0,
        "log2_n_files": 716.9,
        "compiled_flag": 744.0,
        "log2_target_speedup": 749.8,
        "null_model": 748.6,
        "interpretation": "Lower AIC = better fit. target_speedup is WORSE than null (no information).",
    },
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
            "horizon_50pct_lines": round(fr["horizon"], 0),
        },
    }
    if fr["ci_lo"]:
        entry["logistic_fit"]["horizon_95ci"] = [round(fr["ci_lo"], 0), round(fr["ci_hi"], 0)]
    summary["models"][model] = entry

with open(os.path.join(OUT_DIR, "logistic_speedup_curve.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + json.dumps(summary, indent=2))
