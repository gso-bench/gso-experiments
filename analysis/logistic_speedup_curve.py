"""
GSO Speedup Horizon: METR-style logistic curve analysis.

Inspired by METR's time-horizon benchmark, which fits a logistic curve predicting
model success probability as a function of human task completion time.

We adapt the idea using **expert speedup** on the x-axis. The key finding is that
the naive METR analogy (one logistic curve) does NOT fit -- solve rate is flat
across speedup levels when all tasks are pooled. But when we split by optimization
type (compiled-code vs Python-only), the picture becomes clear:

  - Python-only tasks follow a METR-like sigmoid: ~50% at low speedup -> ~15% at high
  - Compiled-code tasks are flat at ~15% regardless of speedup magnitude
  - Diff size and file count are stronger predictors than speedup alone

The "compiled code barrier" creates a floor effect that masks the difficulty gradient.
"""

import json
import os
import numpy as np
from scipy.optimize import curve_fit
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
            target = r["gm_speedup_commit_base"]
            if target and target > 0:
                instance_target[inst_id] = target

instance_target = {k: v for k, v in instance_target.items() if 1.0 <= v <= 100.0}

# Parse expert diffs for metadata
compiled_extensions = {'.c', '.cpp', '.cxx', '.h', '.hpp', '.pyx', '.pxd', '.rs', '.go'}

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
                if ext in compiled_extensions:
                    has_compiled = True
    diff_lines = sum(1 for l in diff.split('\n') if l.startswith('+') or l.startswith('-'))
    instance_meta[inst_id] = {
        "compiled": has_compiled,
        "diff_lines": diff_lines,
        "n_files": len(files),
    }

# ---------------------------------------------------------------------------
# Collect (target, solved) data points, tagged by category
# ---------------------------------------------------------------------------
data_points = []  # list of (inst_id, model, target, solved, compiled, diff_lines, n_files)
for model in MODELS:
    for inst_id, target in instance_target.items():
        if inst_id not in instance_meta:
            continue
        r = all_results[model].get(inst_id)
        if r is None:
            continue
        if r.get("test_passed") is not None:
            solved = 1 if r.get("opt_commit") else 0
            meta = instance_meta[inst_id]
            data_points.append({
                "inst_id": inst_id, "model": model, "target": target,
                "solved": solved, "compiled": meta["compiled"],
                "diff_lines": meta["diff_lines"], "n_files": meta["n_files"],
            })

print(f"Total data points: {len(data_points)}")

# Convert to arrays for easy slicing
targets = np.array([d["target"] for d in data_points])
solved = np.array([d["solved"] for d in data_points])
is_compiled = np.array([d["compiled"] for d in data_points])
diff_lines = np.array([d["diff_lines"] for d in data_points])
n_files = np.array([d["n_files"] for d in data_points])

# ---------------------------------------------------------------------------
# Logistic fitting helpers
# ---------------------------------------------------------------------------
def logistic_ceiling(log_x, L, k, log_x0):
    return L / (1.0 + np.exp(k * (log_x - log_x0)))


def fit_logistic(xs, ys, L_hint=None):
    log_xs = np.log(xs)
    rate = ys.mean()
    L0 = L_hint or min(rate * 2.5, 0.95)
    try:
        popt, pcov = curve_fit(
            logistic_ceiling, log_xs, ys,
            p0=[L0, 1.5, np.log(2.0)],
            bounds=([0.05, 0.05, np.log(1.0)], [1.0, 20.0, np.log(100.0)]),
            maxfev=20000,
        )
        L, k, log_x0 = popt
        perr = np.sqrt(np.diag(pcov))
        return {"L": L, "k": k, "x0": np.exp(log_x0),
                "L_err": perr[0], "k_err": perr[1], "x0_err": perr[2] * np.exp(log_x0)}
    except Exception as e:
        return None


def bin_data(xs, ys, edges):
    """Bin data and compute Wilson CIs."""
    centers, rates, counts, ci_lo, ci_hi = [], [], [], [], []
    for i in range(len(edges) - 1):
        mask = (xs >= edges[i]) & (xs < edges[i + 1])
        n = mask.sum()
        if n >= 3:
            r = ys[mask].mean()
            s = ys[mask].sum()
            # Wilson CI
            z = 1.96
            p = s / n
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
            lo, hi = max(0, center - spread), min(1, center + spread)

            centers.append(np.sqrt(edges[i] * edges[i + 1]))
            rates.append(r)
            counts.append(n)
            ci_lo.append(lo)
            ci_hi.append(hi)
    return (np.array(centers), np.array(rates), np.array(counts),
            np.array(ci_lo), np.array(ci_hi))


bin_edges = np.array([1.0, 1.3, 1.6, 2.0, 2.5, 3.5, 5.0, 8.0, 100.0])

# ---------------------------------------------------------------------------
# Figure 1: The main finding -- compiled vs python-only speedup horizons
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

x_smooth = np.linspace(1.0, 50.0, 300)
log_x_smooth = np.log(x_smooth)

for ax_idx, (mask, title, color) in enumerate([
    (~is_compiled, "Python-Only Tasks", "#2563EB"),
    (is_compiled, "Compiled-Code Tasks", "#DC2626"),
    (np.ones(len(targets), dtype=bool), "All Tasks (pooled)", "#6B7280"),
]):
    ax = axes[ax_idx]
    t, s = targets[mask], solved[mask]
    centers, rates, counts, ci_lo, ci_hi = bin_data(t, s, bin_edges)

    # Bars
    bar_x = np.arange(len(centers))
    ax.bar(bar_x, rates, width=0.65, color=color, alpha=0.6, zorder=2)
    ax.errorbar(bar_x, rates, yerr=[rates - ci_lo, ci_hi - rates],
                fmt="none", color="black", capsize=3, linewidth=1, zorder=3)
    for i, (r, n) in enumerate(zip(rates, counts)):
        ax.text(i, r + 0.03, f"n={int(n)}", ha="center", fontsize=7, color="gray")

    # Logistic fit overlay
    fit = fit_logistic(t, s)
    if fit:
        y_fit = logistic_ceiling(log_x_smooth, fit["L"], fit["k"], np.log(fit["x0"]))
        # Map curve onto bar positions via secondary axis
        ax2 = ax.twiny()
        ax2.plot(x_smooth, y_fit, color=color, linewidth=2.5, alpha=0.8, zorder=4,
                 label=f'L={fit["L"]:.0%}, k={fit["k"]:.1f}')
        ax2.set_xlim(1.0, 50.0)
        ax2.set_xscale("log")
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%gx'))
        ax2.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 5, 10, 20, 50]))
        ax2.tick_params(labelsize=7)
        ax2.legend(loc="upper right", fontsize=8)
        if ax_idx == 1:
            ax2.set_xlabel("Expert Speedup (log scale)", fontsize=9)

    # X labels
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        m = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
        if m.sum() >= 3:
            if bin_edges[i + 1] >= 100:
                bin_labels.append(f"{bin_edges[i]:.0f}x+")
            else:
                bin_labels.append(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}x".replace(".0", ""))
    ax.set_xticks(bar_x)
    ax.set_xticklabels(bin_labels, fontsize=7, rotation=30)
    ax.set_xlabel("Expert Speedup Bin", fontsize=9)
    ax.set_ylabel("Solve Rate" if ax_idx == 0 else "", fontsize=10)
    ax.set_title(f"{title}\n({int(s.sum())}/{len(s)} solved, {100*s.mean():.0f}%)", fontsize=11)
    ax.set_ylim(-0.03, 0.85)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.2, axis="y")

    rate_desc = f"  {title}: {int(s.sum())}/{len(s)} = {100*s.mean():.1f}%"
    if fit:
        rate_desc += f"  | logistic: L={fit['L']:.0%}, k={fit['k']:.2f}, x0={fit['x0']:.1f}x"
    print(rate_desc)

fig.suptitle("GSO Speedup Horizon: Does Optimization Magnitude Predict Model Success?",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "speedup_horizon_compiled_split.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "speedup_horizon_compiled_split.pdf"), bbox_inches="tight")
print("\nSaved: speedup_horizon_compiled_split.png")

# ---------------------------------------------------------------------------
# Figure 2: Solve rate by multiple difficulty dimensions
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: compiled vs python-only
ax = axes2[0]
categories = ["Python-only", "Compiled"]
rates_cat = [solved[~is_compiled].mean(), solved[is_compiled].mean()]
counts_cat = [(~is_compiled).sum(), is_compiled.sum()]
colors_cat = ["#2563EB", "#DC2626"]
bars = ax.bar(categories, rates_cat, color=colors_cat, alpha=0.7, width=0.5)
for i, (r, n) in enumerate(zip(rates_cat, counts_cat)):
    ax.text(i, r + 0.01, f"{r:.0%}\n(n={n})", ha="center", fontsize=9)
ax.set_ylabel("Solve Rate", fontsize=10)
ax.set_title("By Language Type", fontsize=11)
ax.set_ylim(0, 0.45)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.grid(True, alpha=0.2, axis="y")

# Panel B: by diff size
ax = axes2[1]
size_bins = [("< 100", diff_lines < 100), ("100-500", (diff_lines >= 100) & (diff_lines < 500)),
             ("500+", diff_lines >= 500)]
for i, (label, mask) in enumerate(size_bins):
    r = solved[mask].mean()
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
ax = axes2[2]
file_bins = [("1 file", n_files == 1), ("2-5 files", (n_files >= 2) & (n_files <= 5)),
             ("6+ files", n_files >= 6)]
for i, (label, mask) in enumerate(file_bins):
    r = solved[mask].mean()
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

fig2.suptitle("What Predicts GSO Difficulty? (Solve Rate by Task Property)",
              fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "difficulty_predictors.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "difficulty_predictors.pdf"), bbox_inches="tight")
print("Saved: difficulty_predictors.png")

# ---------------------------------------------------------------------------
# Figure 3: Per-model curves (compiled vs Python-only)
# ---------------------------------------------------------------------------
fig3, (ax_py, ax_comp) = plt.subplots(1, 2, figsize=(14, 5.5))

MODEL_COLORS = {
    "claude-opus-4.6": "#D97706", "gpt-5.2": "#059669", "claude-opus-4.5": "#E07A3A",
    "gemini-3-pro": "#4285F4", "claude-sonnet-4.5": "#F59E0B", "gpt-5.1": "#10B981",
    "gemini-3-flash": "#7AAFFF", "o3": "#6366F1",
}

for ax, comp_flag, panel_title in [(ax_py, False, "Python-Only Tasks"), (ax_comp, True, "Compiled-Code Tasks")]:
    for model in MODELS:
        model_mask = np.array([d["model"] == model for d in data_points])
        comp_mask = is_compiled if comp_flag else ~is_compiled
        mask = model_mask & comp_mask
        if mask.sum() < 5:
            continue

        t_m, s_m = targets[mask], solved[mask]
        centers, rates, counts, _, _ = bin_data(t_m, s_m, bin_edges)

        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]
        rate = s_m.mean()
        ax.scatter(centers, rates, color=color, s=counts * 3, alpha=0.4, zorder=3)

        # Fit per-model
        fit = fit_logistic(t_m, s_m)
        if fit and fit["k"] > 0.1:
            y_fit = logistic_ceiling(log_x_smooth, fit["L"], fit["k"], np.log(fit["x0"]))
            ax.plot(x_smooth, y_fit, color=color, linewidth=1.5, alpha=0.8,
                    label=f'{label} ({rate:.0%})', zorder=2)
        else:
            # Flat line if no sigmoid detected
            ax.axhline(y=rate, color=color, linewidth=1, alpha=0.4, linestyle="--")
            ax.plot([], [], color=color, linewidth=1.5, label=f'{label} ({rate:.0%})')

    ax.set_xscale("log")
    ax.set_xlabel("Expert Speedup (log scale)", fontsize=10)
    ax.set_ylabel("Solve Rate", fontsize=10)
    ax.set_title(panel_title, fontsize=11)
    ax.set_xlim(1.0, 50.0)
    ax.set_ylim(-0.05, 0.85)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%gx'))
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 1.5, 2, 3, 5, 10, 20, 50]))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig3.suptitle("Per-Model Speedup Horizons (split by optimization type)",
              fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "speedup_horizon_per_model.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "speedup_horizon_per_model.pdf"), bbox_inches="tight")
print("Saved: speedup_horizon_per_model.png")

# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------
py_fit = fit_logistic(targets[~is_compiled], solved[~is_compiled])
comp_fit = fit_logistic(targets[is_compiled], solved[is_compiled])

summary = {
    "description": (
        "METR-style logistic analysis adapted for GSO. The key finding: speedup "
        "magnitude alone is NOT a strong predictor of model success (unlike METR's "
        "task duration). Splitting by optimization type reveals the real structure: "
        "Python-only tasks show a METR-like difficulty gradient, while compiled-code "
        "tasks have a flat ~15% solve rate regardless of speedup."
    ),
    "methodology": (
        "P(solve) = L / (1 + exp(k * (log(x) - log(x0)))). L = ceiling solve rate, "
        "k = steepness, x0 = half-max horizon (speedup at which solve rate = L/2). "
        "Data: 8 models x ~97 instances = ~750 (model, instance) pairs."
    ),
    "key_findings": [
        "Speedup magnitude is NOT a strong predictor of solve rate when all tasks are pooled (solve rate ~20% across all speedup levels)",
        f"Python-only tasks: {100*solved[~is_compiled].mean():.0f}% solve rate, with a clear declining trend as speedup increases",
        f"Compiled-code tasks: {100*solved[is_compiled].mean():.0f}% solve rate, flat across speedup levels",
        f"Diff size is a strong predictor: <100 lines = {100*solved[diff_lines<100].mean():.0f}%, 100-500 = {100*solved[(diff_lines>=100)&(diff_lines<500)].mean():.0f}%, 500+ = {100*solved[diff_lines>=500].mean():.0f}%",
        f"File count is a strong predictor: 1 file = {100*solved[n_files==1].mean():.0f}%, 2-5 = {100*solved[(n_files>=2)&(n_files<=5)].mean():.0f}%, 6+ = {100*solved[n_files>=6].mean():.0f}%",
    ],
    "comparison_to_metr": (
        "METR's time-horizon works because task duration is a reliable difficulty proxy: "
        "models approach 100% on short tasks and 0% on long ones. In GSO, speedup magnitude "
        "is NOT such a proxy because a 1.3x optimization requiring Cython can be harder than "
        "a 8x optimization in pure Python. The compiled-code barrier creates a floor effect "
        "that masks the difficulty gradient visible in Python-only tasks."
    ),
    "logistic_fits": {
        "python_only": {
            "n_datapoints": int((~is_compiled).sum()),
            "solve_rate": round(float(solved[~is_compiled].mean()), 3),
            "fit": {"L": round(py_fit["L"], 3), "k": round(py_fit["k"], 3), "x0": round(py_fit["x0"], 2)} if py_fit else None,
        },
        "compiled_code": {
            "n_datapoints": int(is_compiled.sum()),
            "solve_rate": round(float(solved[is_compiled].mean()), 3),
            "fit": {"L": round(comp_fit["L"], 3), "k": round(comp_fit["k"], 3), "x0": round(comp_fit["x0"], 2)} if comp_fit else None,
        },
    },
    "solve_rates_by_dimension": {
        "language_type": {
            "python_only": round(float(solved[~is_compiled].mean()), 3),
            "compiled": round(float(solved[is_compiled].mean()), 3),
        },
        "diff_size": {
            "small_lt100": round(float(solved[diff_lines < 100].mean()), 3),
            "medium_100_500": round(float(solved[(diff_lines >= 100) & (diff_lines < 500)].mean()), 3),
            "large_500plus": round(float(solved[diff_lines >= 500].mean()), 3),
        },
        "file_count": {
            "single_file": round(float(solved[n_files == 1].mean()), 3),
            "multi_2_5": round(float(solved[(n_files >= 2) & (n_files <= 5)].mean()), 3),
            "many_6plus": round(float(solved[n_files >= 6].mean()), 3),
        },
    },
}

with open(os.path.join(OUT_DIR, "logistic_speedup_curve.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + json.dumps(summary, indent=2))
