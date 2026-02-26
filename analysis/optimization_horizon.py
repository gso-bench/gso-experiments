"""
IRT-based Optimization Horizon analysis for GSO benchmark.

Uses a 1PL (Rasch) Item Response Theory model to infer:
  - θ_m: each model's latent optimization ability
  - α_j: each task's latent optimization difficulty

Binary outcome per (model, task): did the model achieve ≥95% of expert speedup
AND pass correctness tests? (i.e., is the instance in opt_commit_ids?)

With only 15 models, the 2PL (per-task discrimination) is overparameterized.
The 1PL/Rasch model is the appropriate choice: P(success) = σ(θ_m - α_j).

We plot θ_m vs. release date to track optimization ability over time,
analogous to how METR tracks time horizon vs. release date for autonomy.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "results" / "reports"
MANIFEST_PATH = REPO_ROOT / "results" / "manifest.json"
OUTPUT_DIR = Path(__file__).resolve().parent

# Model release dates (public announcement dates)
RELEASE_DATES = {
    "claude-opus-4":     "2025-05-14",
    "claude-sonnet-4":   "2025-05-14",
    "claude-sonnet-4.5": "2025-09-29",
    "claude-opus-4.5":   "2025-11-01",
    "claude-opus-4.6":   "2026-02-04",
    "gemini-2.5-pro":    "2025-03-25",
    "gemini-3-flash":    "2025-12-10",
    "gemini-3-pro":      "2025-12-10",
    "glm-4.5":           "2025-06-27",
    "gpt-5":             "2025-08-07",
    "gpt-5.1":           "2025-11-13",
    "gpt-5.2":           "2025-12-11",
    "kimi-k2":           "2025-07-10",
    "o3":                "2025-04-16",
    "qwen3-coder":       "2025-07-22",
}

MODEL_COLORS = {
    "claude-opus-4":     "#D97706",
    "claude-opus-4.5":   "#B45309",
    "claude-opus-4.6":   "#92400E",
    "claude-sonnet-4":   "#FBBF24",
    "claude-sonnet-4.5": "#F59E0B",
    "gemini-2.5-pro":    "#6366F1",
    "gemini-3-flash":    "#818CF8",
    "gemini-3-pro":      "#4F46E5",
    "glm-4.5":           "#10B981",
    "gpt-5":             "#60A5FA",
    "gpt-5.1":           "#3B82F6",
    "gpt-5.2":           "#1D4ED8",
    "kimi-k2":           "#EC4899",
    "o3":                "#8B5CF6",
    "qwen3-coder":       "#EF4444",
}

MODEL_FAMILIES = {
    "claude-opus-4": "Claude", "claude-opus-4.5": "Claude", "claude-opus-4.6": "Claude",
    "claude-sonnet-4": "Claude", "claude-sonnet-4.5": "Claude",
    "gemini-2.5-pro": "Gemini", "gemini-3-flash": "Gemini", "gemini-3-pro": "Gemini",
    "glm-4.5": "GLM", "gpt-5": "GPT", "gpt-5.1": "GPT", "gpt-5.2": "GPT",
    "kimi-k2": "Kimi", "o3": "OpenAI", "qwen3-coder": "Qwen",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_response_matrix():
    """Build binary response matrix: models × tasks."""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    model_names = sorted(manifest["models"].keys())

    first_report_path = REPORTS_DIR / f"{model_names[0]}.json"
    with open(first_report_path) as f:
        first_report = json.load(f)
    task_ids = sorted(first_report["instance_sets"]["completed_ids"])

    n_models = len(model_names)
    n_tasks = len(task_ids)
    response_matrix = np.zeros((n_models, n_tasks), dtype=int)
    task_to_idx = {t: i for i, t in enumerate(task_ids)}

    for m_idx, model_name in enumerate(model_names):
        report_path = REPORTS_DIR / f"{model_name}.json"
        with open(report_path) as f:
            report = json.load(f)
        opt_commit_ids = set(report["instance_sets"].get("opt_commit_ids", []))
        for task_id in opt_commit_ids:
            if task_id in task_to_idx:
                response_matrix[m_idx, task_to_idx[task_id]] = 1

    return response_matrix, model_names, task_ids


# ---------------------------------------------------------------------------
# 1PL (Rasch) IRT fitting via MLE
# ---------------------------------------------------------------------------

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def fit_rasch(response_matrix, reg_lambda=0.001):
    """
    Fit 1PL (Rasch) model via MLE: P(success) = σ(θ_m - α_j)

    Identification: mean(θ) = 0 enforced post-hoc.
    Light L2 regularization for numerical stability.
    """
    n_models, n_tasks = response_matrix.shape

    # Initialize from marginal rates
    model_rates = np.clip(response_matrix.mean(axis=1), 0.01, 0.99)
    theta_init = np.log(model_rates / (1 - model_rates))
    task_rates = np.clip(response_matrix.mean(axis=0), 0.01, 0.99)
    alpha_init = -np.log(task_rates / (1 - task_rates))
    params_init = np.concatenate([theta_init, alpha_init])

    def objective(params):
        theta = params[:n_models]
        alpha = params[n_models:]
        logit = theta[:, None] - alpha[None, :]
        p = sigmoid(logit)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = response_matrix * np.log(p) + (1 - response_matrix) * np.log(1 - p)
        reg = reg_lambda * (np.sum(theta ** 2) + np.sum(alpha ** 2))
        return -np.sum(ll) + reg

    result = minimize(objective, params_init, method="L-BFGS-B",
                      options={"maxiter": 5000, "ftol": 1e-12})
    if not result.success:
        print(f"  Warning: {result.message}", file=sys.stderr)

    theta = result.x[:n_models]
    alpha = result.x[n_models:]

    # Center θ
    shift = theta.mean()
    theta -= shift
    alpha -= shift

    return theta, alpha


def bootstrap_rasch(response_matrix, n_bootstrap=2000, reg_lambda=0.001):
    """Hierarchical bootstrap over tasks for CIs on θ."""
    n_models, n_tasks = response_matrix.shape
    theta_samples = np.zeros((n_bootstrap, n_models))

    for b in range(n_bootstrap):
        task_idx = np.random.choice(n_tasks, size=n_tasks, replace=True)
        resampled = response_matrix[:, task_idx]
        try:
            theta_b, _ = fit_rasch(resampled, reg_lambda=reg_lambda)
            theta_samples[b] = theta_b
        except Exception:
            theta_samples[b] = np.nan

    return theta_samples


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ability_vs_date(theta, model_names, theta_ci):
    """Plot θ vs release date with CIs and trend line."""
    fig, ax = plt.subplots(figsize=(12, 7))

    dates, abilities, labels, colors = [], [], [], []
    for i, name in enumerate(model_names):
        if name not in RELEASE_DATES:
            continue
        d = datetime.strptime(RELEASE_DATES[name], "%Y-%m-%d")
        dates.append(d)
        abilities.append(theta[i])
        labels.append(name)
        colors.append(MODEL_COLORS.get(name, "#666666"))

    dates_arr = np.array(dates)
    abilities_arr = np.array(abilities)

    # Error bars
    for i, name in enumerate(model_names):
        if name not in RELEASE_DATES:
            continue
        idx = labels.index(name)
        lo, hi = theta_ci[i]
        ax.errorbar(
            dates_arr[idx], abilities_arr[idx],
            yerr=[[abilities_arr[idx] - lo], [hi - abilities_arr[idx]]],
            fmt="o", color=colors[idx], markersize=10,
            capsize=4, capthick=1.5, linewidth=1.5, zorder=5,
        )

    # Labels with collision avoidance
    for i, label in enumerate(labels):
        ax.annotate(
            label, (dates_arr[i], abilities_arr[i]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=7.5, color=colors[i], fontweight="bold",
        )

    # OLS trend line
    date_numeric = np.array([(d - datetime(2025, 1, 1)).days for d in dates_arr], dtype=float)
    coeffs = np.polyfit(date_numeric, abilities_arr, 1)
    slope_per_day = coeffs[0]
    trend_x = np.linspace(date_numeric.min() - 20, date_numeric.max() + 20, 100)
    trend_y = np.polyval(coeffs, trend_x)
    trend_dates = [datetime(2025, 1, 1) + timedelta(days=float(d)) for d in trend_x]
    ax.plot(trend_dates, trend_y, "k--", alpha=0.3, linewidth=1,
            label=f"OLS trend: {slope_per_day*30:.2f} θ/month")

    ax.set_xlabel("Release Date", fontsize=12)
    ax.set_ylabel("Optimization Ability (θ, Rasch)", fontsize=12)
    ax.set_title("GSO: IRT-Inferred Optimization Ability vs. Release Date", fontsize=14)
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = OUTPUT_DIR / "optimization_ability_vs_date.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)

    return slope_per_day


def plot_task_difficulty(alpha, task_ids, response_matrix):
    """Plot task difficulty distribution colored by project."""
    projects = [tid.split("__")[0] for tid in task_ids]
    unique_projects = sorted(set(projects))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(unique_projects)))
    proj_color = {p: c for p, c in zip(unique_projects, cmap)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: histogram by project
    ax = axes[0]
    for proj in unique_projects:
        mask = np.array([p == proj for p in projects])
        if mask.sum() > 0:
            ax.hist(alpha[mask], bins=12, alpha=0.6, label=proj, color=proj_color[proj])
    ax.set_xlabel("Task Difficulty (α)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Inferred Task Difficulty by Project", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: difficulty vs solve rate (sanity check — should be monotonic)
    ax = axes[1]
    solve_rates = response_matrix.mean(axis=0)
    task_colors = [proj_color[p] for p in projects]
    ax.scatter(alpha, solve_rates, c=task_colors, s=50, alpha=0.7,
               edgecolors="white", linewidths=0.5)
    # Overlay logistic fit
    alpha_range = np.linspace(alpha.min() - 0.5, alpha.max() + 0.5, 200)
    # For Rasch at θ=0 (mean model): P = σ(-α)
    ax.plot(alpha_range, sigmoid(-alpha_range), "k--", alpha=0.4, linewidth=1.5,
            label="Rasch prediction\n(mean-ability model)")
    ax.set_xlabel("Task Difficulty (α)", fontsize=11)
    ax.set_ylabel("Solve Rate", fontsize=11)
    ax.set_title("Difficulty vs. Solve Rate (sanity check)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "task_difficulty_distribution.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_response_heatmap(response_matrix, model_names, task_ids, theta, alpha):
    """
    Heatmap of response matrix, sorted by θ (rows) and α (columns).
    This is the core diagnostic: does the Guttman pattern hold?
    """
    # Sort models by θ descending, tasks by α ascending
    model_order = np.argsort(-theta)
    task_order = np.argsort(alpha)

    sorted_matrix = response_matrix[model_order][:, task_order]
    sorted_models = [model_names[i] for i in model_order]
    sorted_tasks = [task_ids[i] for i in task_order]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(sorted_matrix, cmap="RdYlGn", aspect="auto", interpolation="nearest")

    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels([f"{m} (θ={theta[i]:.1f})" for m, i in
                        zip(sorted_models, model_order)], fontsize=8)

    # Only label every nth task
    n_tasks = len(sorted_tasks)
    step = max(1, n_tasks // 20)
    ax.set_xticks(range(0, n_tasks, step))
    ax.set_xticklabels([sorted_tasks[i].split("__")[-1][:10] for i in range(0, n_tasks, step)],
                       fontsize=6, rotation=90)
    ax.set_xlabel("Tasks (sorted by difficulty α →)", fontsize=11)
    ax.set_ylabel("Models (sorted by ability θ ↓)", fontsize=11)
    ax.set_title("Response Matrix: Green=solved, Red=unsolved\n"
                 "(Guttman pattern = upper-left green, lower-right red)", fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.5, label="Solved")
    fig.tight_layout()

    out_path = OUTPUT_DIR / "response_heatmap.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading response matrix...")
    response_matrix, model_names, task_ids = load_response_matrix()

    n_models, n_tasks = response_matrix.shape
    print(f"  {n_models} models × {n_tasks} tasks = {n_models * n_tasks} observations")
    print(f"  Overall solve rate: {response_matrix.mean():.3f}")

    # Remove degenerate tasks
    task_means = response_matrix.mean(axis=0)
    degen_mask = (task_means == 0) | (task_means == 1)
    n_degen = degen_mask.sum()
    if n_degen > 0:
        print(f"  Removing {n_degen} degenerate tasks (all-0 or all-1)")
        keep = ~degen_mask
        response_matrix = response_matrix[:, keep]
        task_ids = [t for t, k in zip(task_ids, keep) if k]
        n_tasks = len(task_ids)
        print(f"  Remaining: {n_models} models × {n_tasks} tasks")
    print()

    # Fit Rasch model
    print("Fitting 1PL (Rasch) IRT model...")
    theta, alpha = fit_rasch(response_matrix)

    print("\n=== Model Optimization Abilities (θ) ===")
    sorted_models = sorted(zip(model_names, theta), key=lambda x: -x[1])
    for name, t in sorted_models:
        date_str = RELEASE_DATES.get(name, "unknown")
        score = response_matrix[model_names.index(name)].mean() * 100
        print(f"  {name:25s}  θ={t:+.3f}  score={score:.1f}%  released={date_str}")

    print(f"\n=== Task Difficulty Summary ===")
    print(f"  Range: [{alpha.min():.2f}, {alpha.max():.2f}]")
    print(f"  Mean:  {alpha.mean():.2f}, Std: {alpha.std():.2f}")

    print("\n  Easiest tasks:")
    for idx in np.argsort(alpha)[:5]:
        n_solved = response_matrix[:, idx].sum()
        print(f"    {task_ids[idx]:50s}  α={alpha[idx]:+.2f}  solved_by={n_solved}/{n_models}")
    print("\n  Hardest tasks:")
    for idx in np.argsort(alpha)[-5:]:
        n_solved = response_matrix[:, idx].sum()
        print(f"    {task_ids[idx]:50s}  α={alpha[idx]:+.2f}  solved_by={n_solved}/{n_models}")

    # Bootstrap CIs
    print("\nBootstrapping confidence intervals (2000 samples)...")
    np.random.seed(42)
    theta_samples = bootstrap_rasch(response_matrix, n_bootstrap=2000)
    valid = ~np.any(np.isnan(theta_samples), axis=1)
    theta_samples = theta_samples[valid]
    print(f"  {valid.sum()} / 2000 valid bootstrap samples")
    theta_ci = np.percentile(theta_samples, [2.5, 97.5], axis=0).T

    print("\n=== θ with 95% CIs ===")
    for name, t in sorted_models:
        i = model_names.index(name)
        lo, hi = theta_ci[i]
        print(f"  {name:25s}  θ={t:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    # Plots
    print("\nGenerating plots...")
    slope = plot_ability_vs_date(theta, model_names, theta_ci)
    plot_task_difficulty(alpha, task_ids, response_matrix)
    plot_response_heatmap(response_matrix, model_names, task_ids, theta, alpha)

    print(f"\n=== Trend ===")
    print(f"  Slope: {slope:.4f} θ/day = {slope*30:.2f} θ/month")

    # Save results
    results = {
        "method": "1PL (Rasch) IRT",
        "models": {
            name: {
                "theta": round(float(theta[i]), 4),
                "theta_95ci": [round(float(theta_ci[i, 0]), 4),
                               round(float(theta_ci[i, 1]), 4)],
                "opt_commit_rate": round(float(response_matrix[i].mean()), 4),
                "release_date": RELEASE_DATES.get(name),
            }
            for i, name in enumerate(model_names)
        },
        "tasks": {
            tid: {
                "alpha": round(float(alpha[j]), 4),
                "solve_rate": round(float(response_matrix[:, j].mean()), 4),
            }
            for j, tid in enumerate(task_ids)
        },
        "summary": {
            "n_models": n_models,
            "n_tasks_used": n_tasks,
            "n_tasks_removed_degenerate": int(n_degen),
            "overall_solve_rate": round(float(response_matrix.mean()), 4),
            "theta_range": [round(float(theta.min()), 4), round(float(theta.max()), 4)],
            "alpha_range": [round(float(alpha.min()), 4), round(float(alpha.max()), 4)],
            "trend_theta_per_day": round(float(slope), 6),
            "trend_theta_per_month": round(float(slope * 30), 4),
        },
    }

    out_json = OUTPUT_DIR / "optimization_horizon.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
