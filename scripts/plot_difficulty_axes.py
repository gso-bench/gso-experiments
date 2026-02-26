#!/usr/bin/env python3
"""
Systematically try every plausible difficulty axis for GSO tasks
and see which one produces the cleanest METR-style plot.

METR gets R²=0.83 with human time-to-complete on x-axis.
Can we find an x-axis that produces a similar relationship for optimization?
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

ROOT = Path(__file__).parent.parent

def main():
    data = json.loads((ROOT / "results" / "insight_density_data.json").read_text())
    diff_cache = json.loads((ROOT / "results" / "diff_stats_cache.json").read_text())

    tasks = [t for t in data["tasks"] if t["insight_density"] is not None]
    models = data["models"]

    # Compute all candidate difficulty axes per task
    for t in tasks:
        iid = t["instance_id"]
        dc = diff_cache.get(iid, {})

        t["log_diff_lines"] = np.log10(t["diff_lines"])
        t["log_expert_speedup"] = np.log10(t["expert_speedup"])
        t["log_insight_density"] = np.log10(t["insight_density"])
        t["num_files"] = t["num_files"]
        t["additions"] = dc.get("additions", 0)
        t["deletions"] = dc.get("deletions", 0)

        # Composite: files × lines (locality × scope)
        t["scope"] = t["num_files"] * t["diff_lines"]
        t["log_scope"] = np.log10(max(t["scope"], 1))

        # New code ratio: additions / total (high = mostly new code)
        if t["diff_lines"] > 0:
            t["new_code_ratio"] = t["additions"] / t["diff_lines"]
        else:
            t["new_code_ratio"] = 0.5

        # "Implementation complexity": diff_lines / expert_speedup
        # (lines needed per unit speedup — high = mechanically complex)
        t["impl_complexity"] = t["diff_lines"] / t["expert_speedup"]
        t["log_impl_complexity"] = np.log10(t["impl_complexity"])

        # Check if any files are non-Python (C, C++, Cython, Rust)
        files = dc.get("files", [])
        compiled_exts = {'.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.pyx', '.pxd', '.rs'}
        t["has_compiled"] = any(
            any(f.endswith(ext) for ext in compiled_exts) for f in files
        )
        t["n_languages"] = len(set(
            'compiled' if any(f.endswith(ext) for ext in compiled_exts) else 'python'
            for f in files
        ))

        # Average solve rate across all models
        t["solve_rate"] = np.mean(list(t["model_solved"].values()))

        # Number of models that solved it
        t["n_solved"] = sum(t["model_solved"].values())

    # Define all candidate axes (name, key, direction_label)
    # direction: "higher_harder" means we expect higher x → lower solve rate
    candidates = [
        ("log₁₀(Diff Lines)", "log_diff_lines", True),
        ("log₁₀(Expert Speedup)", "log_expert_speedup", False),
        ("log₁₀(Insight Density)", "log_insight_density", False),
        ("Num Files", "num_files", True),
        ("log₁₀(Scope = Files × Lines)", "log_scope", True),
        ("New Code Ratio", "new_code_ratio", False),
        ("log₁₀(Impl Complexity)", "log_impl_complexity", True),
        ("Has Compiled Code", "has_compiled", True),
    ]

    # First, print correlation table
    print("=" * 80)
    print(f"{'Candidate Axis':<35} {'Spearman ρ':>12} {'p-value':>10} {'Pearson r':>12} {'p-value':>10}")
    print("=" * 80)

    solve_rates = np.array([t["solve_rate"] for t in tasks])

    results = []
    for name, key, higher_harder in candidates:
        x = np.array([float(t[key]) for t in tasks])
        rho, p_rho = spearmanr(x, solve_rates)
        r, p_r = pearsonr(x, solve_rates)
        print(f"  {name:<33} {rho:>+10.3f}   {p_rho:>8.4f}   {r:>+10.3f}   {p_r:>8.4f}")
        results.append((name, key, rho, p_rho, r, p_r))

    # Now make the METR-style plot for each candidate
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, (name, key, higher_harder) in enumerate(candidates):
        ax = axes[idx]
        x = np.array([float(t[key]) for t in tasks])
        y = solve_rates

        # Scatter
        ax.scatter(x, y, alpha=0.5, s=35, c='steelblue', edgecolors='white', linewidth=0.5)

        # Binned means (METR style)
        if key == "has_compiled":
            # Binary: just show two bars
            for val in [0, 1]:
                mask = x == val
                if mask.sum() > 0:
                    mean_y = np.mean(y[mask])
                    ax.plot(val, mean_y, 'rs', markersize=12, zorder=5)
                    ax.annotate(f'n={mask.sum()}\nμ={mean_y:.2f}', (val, mean_y),
                              textcoords="offset points", xytext=(15, 5), fontsize=8)
        else:
            # Quantile bins
            n_bins = 6
            try:
                bin_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
                # Remove duplicate edges
                bin_edges = np.unique(bin_edges)
                bin_centers = []
                bin_means = []
                bin_stds = []
                bin_ns = []
                for i in range(len(bin_edges) - 1):
                    if i == len(bin_edges) - 2:
                        mask = (x >= bin_edges[i]) & (x <= bin_edges[i+1])
                    else:
                        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
                    if mask.sum() > 0:
                        bin_centers.append(np.mean(x[mask]))
                        bin_means.append(np.mean(y[mask]))
                        bin_stds.append(np.std(y[mask]) / np.sqrt(mask.sum()))
                        bin_ns.append(mask.sum())

                ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                           fmt='rs-', markersize=8, linewidth=2, capsize=4, zorder=5,
                           label='Binned mean ± SE')
            except Exception:
                pass

        # Linear fit for R²
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 2:
            coeffs = np.polyfit(x[valid], y[valid], 1)
            x_fit = np.linspace(x[valid].min(), x[valid].max(), 100)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, 'k--', alpha=0.3, linewidth=1)

            r, p = pearsonr(x[valid], y[valid])
            rho, p_rho = spearmanr(x[valid], y[valid])

        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('Mean Model Solve Rate', fontsize=9)
        ax.set_title(f'{name}\nρ={rho:+.3f} (p={p_rho:.3f}), R²={r**2:.3f}', fontsize=10)
        ax.set_ylim(-0.05, 0.85)

    plt.suptitle('GSO: Searching for a METR-style Difficulty Axis\n(METR achieves R²=0.83 with human time-to-complete)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = ROOT / "results" / "difficulty_axis_search.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {outpath}")

    # ================================================================
    # BONUS: Try the "best" axis in METR's exact format
    # Sort tasks by the best axis, bin into ~10 groups, plot like METR
    # ================================================================
    best_key = "log_scope"  # files × lines seems most principled

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.array([float(t[best_key]) for t in tasks])
    y = solve_rates

    # Sort and bin
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    n_bins = 10
    bin_size = len(tasks) // n_bins
    bin_centers = []
    bin_means = []
    bin_stds = []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(tasks)
        bx = x_sorted[start:end]
        by = y_sorted[start:end]
        bin_centers.append(np.mean(bx))
        bin_means.append(np.mean(by))
        bin_stds.append(np.std(by) / np.sqrt(len(by)))

    # Scatter individual points
    ax.scatter(x, y, alpha=0.3, s=30, c='#7fb3d8', edgecolors='none', label='Individual tasks')

    # Binned means
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
               fmt='o-', color='#c0392b', markersize=8, linewidth=2, capsize=5,
               label='Binned mean ± SE', zorder=5)

    r, p = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)

    ax.set_xlabel('log₁₀(Optimization Scope: Files × Lines Changed)', fontsize=12)
    ax.set_ylabel('Mean Model Success Rate', fontsize=12)
    ax.set_title(f'Model Success Rate vs Optimization Scope\nR²={r**2:.2f}, Spearman ρ={rho:+.3f} (p={p_rho:.4f})', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 0.85)

    # Add METR reference
    ax.text(0.98, 0.98, 'cf. METR Time Horizons: R²=0.83',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            style='italic', alpha=0.5)

    plt.tight_layout()
    outpath2 = ROOT / "results" / "metr_style_optimization_scope.png"
    plt.savefig(outpath2, dpi=150, bbox_inches='tight')
    print(f"Saved METR-style plot to {outpath2}")

if __name__ == "__main__":
    main()
