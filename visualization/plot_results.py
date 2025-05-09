# visualization/plot_results.py
"""
Reads progress data (progress.csv or result.json) from Ray/Tune experiment
directories and generates plots for key metrics like reward, loss, etc.
Focuses on the Cleanup scenario.
"""
import os
import json
import argparse
from math import sqrt
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from visualization.utility_funcs import get_all_subdirs

# --- Configuration ---
DEFAULT_RAY_RESULTS_DIR = os.path.expanduser("~/ray_results")
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/ray_results_plots") # Separate dir for plots
# Define metrics to plot and their appearance
# Adapt these keys based on actual metrics logged by your RLlib setup
METRICS_TO_PLOT = {
    "episode_reward_mean": {"label": "Mean Episode Reward", "color": "blue"},
    "policy_loss": {"label": "Mean Policy Loss", "color": "red"},
    # Example: Add losses per policy if available
    # "info/learner/agent_0/learner_stats/policy_loss": {"label": "Agent 0 Policy Loss", "color": "purple"},
    # "info/learner/agent_1/learner_stats/policy_loss": {"label": "Agent 1 Policy Loss", "color": "orange"},
    "episode_len_mean": {"label": "Mean Episode Length", "color": "green"},
    # Add other relevant metrics here
    # "policy_entropy": {"label": "Mean Policy Entropy", "color": "cyan"},
    # "vf_loss": {"label": "Mean VF Loss", "color": "magenta"},
}
X_AXIS_METRIC = "timesteps_total" # Usually timesteps or iterations

def smooth(y: np.ndarray, box_pts: int = 10) -> np.ndarray:
    """Smooths data using a simple moving average."""
    if len(y) < box_pts:
        return y # Not enough points to smooth
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    # Handle boundaries where convolution uses fewer points
    y_smooth[:box_pts//2] = y_smooth[box_pts//2]
    y_smooth[-box_pts//2:] = y_smooth[-box_pts//2-1]
    return y_smooth

def change_color_luminosity(color, amount=0.5):
    """
    Lightens/darkens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Lifted from: https://stackoverflow.com/a/49601444
    """
    import colorsys
    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    # Adjust luminosity (c[1]) - higher amount means lighter
    new_luminosity = c[1] + (1 - c[1]) * amount if amount > 0 else c[1] * (1 + amount)
    new_luminosity = max(0, min(1, new_luminosity)) # Clamp between 0 and 1
    return colorsys.hls_to_rgb(c[0], new_luminosity, c[2])


def plot_single_metric(
    ax: plt.Axes,
    x_data: np.ndarray,
    y_datas: List[np.ndarray], # List of arrays, one per trial/seed
    metric_key: str,
    plot_config: Dict[str, Any],
    smoothing_window: int = 10,
    confidence_interval: float = 0.95,
    plot_individual: bool = False
):
    """Plots a single metric with mean and confidence interval."""
    if not y_datas:
        print(f"Warning: No data provided for metric {metric_key}")
        return

    color = plot_config.get("color", "black")
    label = plot_config.get("label", metric_key)

    # Interpolate all trials to a common x-axis (using the longest trial's x)
    # Find the trial with the most steps
    max_len = max(len(x) for x in x_data)
    longest_x = x_data[np.argmax([len(x) for x in x_data])] if x_data else np.array([]) # Use the x-axis of the longest run
    if len(longest_x) == 0:
        print(f"Warning: No x-axis data for metric {metric_key}")
        return

    interpolated_ys = []
    for x_trial, y_trial in zip(x_data, y_datas):
        if len(x_trial) > 1 and len(y_trial) > 1: # Need at least 2 points to interpolate
            interp_y = np.interp(longest_x, x_trial, y_trial, left=np.nan, right=np.nan)
            interpolated_ys.append(interp_y)
        elif len(x_trial) == 1 and len(y_trial) == 1: # Handle single point case
             # Extend the single point across the whole range? Or just plot the point?
             # Let's pad with NaN for interpolation consistency
             interp_y = np.full_like(longest_x, np.nan, dtype=float)
             # Find index closest to the single x_trial point
             idx = np.argmin(np.abs(longest_x - x_trial[0]))
             interp_y[idx] = y_trial[0]
             interpolated_ys.append(interp_y)

    if not interpolated_ys:
        print(f"Warning: Could not interpolate data for metric {metric_key}")
        return

    interpolated_ys = np.array(interpolated_ys) # Shape: (num_trials, num_steps_longest)

    # Calculate mean and confidence intervals, ignoring NaNs
    with np.errstate(invalid='ignore'): # Suppress warnings from rows with all NaNs
         mean_y = np.nanmean(interpolated_ys, axis=0)
         std_y = np.nanstd(interpolated_ys, axis=0, ddof=1) # ddof=1 for sample std dev

    # Filter out NaNs before smoothing and CI calculation
    valid_indices = ~np.isnan(mean_y)
    if not np.any(valid_indices):
         print(f"Warning: All mean values are NaN for metric {metric_key}")
         return

    mean_y_valid = mean_y[valid_indices]
    std_y_valid = std_y[valid_indices]
    longest_x_valid = longest_x[valid_indices]
    num_trials_valid = np.sum(~np.isnan(interpolated_ys[:, valid_indices]), axis=0) # Count non-NaN trials per step

    # Smoothing
    if smoothing_window > 1 and len(mean_y_valid) >= smoothing_window:
        mean_y_smooth = smooth(mean_y_valid, smoothing_window)
        # Smooth std dev? Less common, let's smooth the mean only for now.
    else:
        mean_y_smooth = mean_y_valid

    # Confidence Interval
    if confidence_interval > 0 and len(mean_y_valid) > 1:
        # Use t-distribution critical value
        alpha = 1.0 - confidence_interval
        # Degrees of freedom: number of valid trials at each step - 1
        dof = np.maximum(1, num_trials_valid - 1) # Ensure dof >= 1
        t_crit = t.ppf(1.0 - alpha / 2.0, dof)
        # Standard error of the mean
        sem = std_y_valid / np.sqrt(np.maximum(1, num_trials_valid)) # Avoid sqrt(0)
        ci_margin = t_crit * sem

        lower_bound = mean_y_smooth - ci_margin
        upper_bound = mean_y_smooth + ci_margin

        # Plot confidence interval fill
        fill_color = change_color_luminosity(color, 0.8) # Lighter fill
        ax.fill_between(longest_x_valid, lower_bound, upper_bound, color=fill_color, alpha=0.3)

    # Plot individual trials (optional, usually very noisy)
    if plot_individual:
        light_color = change_color_luminosity(color, 0.6)
        for i, y_interp in enumerate(interpolated_ys):
            ax.plot(longest_x, y_interp, color=light_color, alpha=0.2, linewidth=0.5, label='_nolegend_') # No legend for individual

    # Plot the smoothed mean line
    ax.plot(longest_x_valid, mean_y_smooth, color=color, label=label, linewidth=1.5)

    # Set labels and potentially limits
    ax.set_ylabel(label)
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # Use scientific notation for x-axis if large


def load_experiment_data(experiment_dir: str) -> Optional[pd.DataFrame]:
    """Loads progress data from progress.csv or result.json."""
    progress_csv_path = os.path.join(experiment_dir, "progress.csv")
    result_json_path = os.path.join(experiment_dir, "result.json")

    if os.path.exists(progress_csv_path):
        try:
            return pd.read_csv(progress_csv_path, sep=",")
        except Exception as e:
            print(f"Error reading CSV {progress_csv_path}: {e}")
            return None
    elif os.path.exists(result_json_path):
        try:
            data = []
            with open(result_json_path, 'r') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {result_json_path}")
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error reading JSON {result_json_path}: {e}")
            return None
    else:
        print(f"Warning: No progress.csv or result.json found in {experiment_dir}")
        return None


def plot_experiment_group(
    experiment_dirs: List[str],
    output_dir: str,
    group_name: str,
    metrics_to_plot: Dict[str, Dict],
    x_axis_metric: str,
    smoothing_window: int = 10,
    confidence_interval: float = 0.95,
    plot_individual: bool = False
):
    """Plots metrics for a group of related experiments (e.g., multiple seeds)."""
    print(f"\n--- Plotting Group: {group_name} ---")
    all_data = {} # {metric_key: {'x': [trial1_x, ...], 'y': [trial1_y, ...]}, ...}

    # Load data for all experiments in the group
    valid_dfs = []
    for exp_dir in experiment_dirs:
        df = load_experiment_data(exp_dir)
        if df is not None and not df.empty and x_axis_metric in df.columns:
            valid_dfs.append(df)
        else:
            print(f"Skipping invalid or empty data from: {exp_dir}")

    if not valid_dfs:
        print(f"No valid data found for group {group_name}. Skipping plot.")
        return

    # Prepare data structure for plotting
    for metric_key in metrics_to_plot.keys():
        all_data[metric_key] = {'x': [], 'y': []}

    for df in valid_dfs:
        for metric_key in metrics_to_plot.keys():
            if metric_key in df.columns:
                # Extract x and y, drop NaNs specifically for this metric pair
                metric_df = df[[x_axis_metric, metric_key]].dropna()
                if not metric_df.empty:
                    all_data[metric_key]['x'].append(metric_df[x_axis_metric].values)
                    all_data[metric_key]['y'].append(metric_df[metric_key].values)
            # else:
                # print(f"Metric '{metric_key}' not found in data from one trial.")


    # Create plot
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes] # Make it iterable
    fig.suptitle(f"Training Results: {group_name} ({len(valid_dfs)} trials)", fontsize=16)

    # Plot each metric
    for i, (metric_key, plot_config) in enumerate(metrics_to_plot.items()):
        if all_data[metric_key]['x'] and all_data[metric_key]['y']: # Check if data exists for this metric
            plot_single_metric(
                ax=axes[i],
                x_data=all_data[metric_key]['x'],
                y_datas=all_data[metric_key]['y'],
                metric_key=metric_key,
                plot_config=plot_config,
                smoothing_window=smoothing_window,
                confidence_interval=confidence_interval,
                plot_individual=plot_individual
            )
        else:
             print(f"No data to plot for metric: {metric_key}")
             axes[i].set_ylabel(plot_config.get('label', metric_key)) # Still label the axis

        axes[i].grid(True, linestyle='--', alpha=0.6)
        if i == num_metrics - 1: # Only show x-label on the bottom plot
             axes[i].set_xlabel(f"{x_axis_metric.replace('_', ' ').title()} (Total)")
             axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        if i==0: # Add legend to the top plot only
             axes[i].legend(loc='best')


    # Save plot
    output_filename = os.path.join(output_dir, f"{group_name}_plot.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
    plt.savefig(output_filename)
    print(f"Plot saved to: {output_filename}")
    plt.close(fig) # Close the figure to free memory


def main(args):
    """Finds experiment directories and generates plots."""
    base_ray_dir = args.ray_dir
    output_dir = args.output_dir
    experiment_filter = args.filter
    group_by_parent = args.group # Group by parent directory name

    if not os.path.isdir(base_ray_dir):
        print(f"Error: Ray results directory not found: {base_ray_dir}")
        return

    # Find potential experiment directories
    all_items = os.listdir(base_ray_dir)
    potential_exp_dirs = [os.path.join(base_ray_dir, item) for item in all_items
                          if os.path.isdir(os.path.join(base_ray_dir, item))]

    # Filter by name if provided
    if experiment_filter:
        potential_exp_dirs = [d for d in potential_exp_dirs if experiment_filter in os.path.basename(d)]

    # Filter for cleanup env experiments (assuming 'cleanup' is in the name)
    cleanup_exp_dirs = [d for d in potential_exp_dirs if 'cleanup' in os.path.basename(d).lower()]

    if not cleanup_exp_dirs:
        print("No 'cleanup' experiment directories found matching the criteria.")
        return

    print(f"Found {len(cleanup_exp_dirs)} potential 'cleanup' experiment directories.")

    if group_by_parent:
        # Group experiments by their base name (removing seed/trial suffixes)
        grouped_experiments = {}
        for exp_dir in cleanup_exp_dirs:
            base_name = os.path.basename(exp_dir)
            # Try to identify a base name (e.g., remove trailing _seedX, _trialY)
            # This is heuristic - adjust the split logic if your naming is different
            group_key = base_name.split("_seed")[0].split("_trial")[0].split("_")[0:-1] # Try removing last part if it looks like ID
            group_key = "_".join(group_key) if group_key else base_name # Fallback to full name

            if group_key not in grouped_experiments:
                grouped_experiments[group_key] = []
            grouped_experiments[group_key].append(exp_dir)

        print(f"Grouping experiments into {len(grouped_experiments)} groups.")
        for group_name, exp_dirs in grouped_experiments.items():
            plot_experiment_group(
                experiment_dirs=exp_dirs,
                output_dir=output_dir,
                group_name=group_name,
                metrics_to_plot=METRICS_TO_PLOT,
                x_axis_metric=X_AXIS_METRIC,
                smoothing_window=args.smooth,
                confidence_interval=args.ci,
                plot_individual=args.individual
            )
    else:
        # Plot each experiment individually
        print("Plotting each experiment individually.")
        for exp_dir in cleanup_exp_dirs:
            group_name = os.path.basename(exp_dir) # Use directory name as group name
            plot_experiment_group(
                experiment_dirs=[exp_dir], # List with single directory
                output_dir=output_dir,
                group_name=group_name,
                metrics_to_plot=METRICS_TO_PLOT,
                x_axis_metric=X_AXIS_METRIC,
                smoothing_window=args.smooth,
                confidence_interval=0, # No CI for single trial
                plot_individual=False # No individual plots needed
            )

    print("\n--- Plotting finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training results from Ray/Tune experiments.")
    parser.add_argument(
        "--ray-dir", type=str, default=DEFAULT_RAY_RESULTS_DIR,
        help="Base directory containing Ray/Tune experiment results."
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the generated plots."
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process experiment directories containing this string in their name."
    )
    parser.add_argument(
        "--group", action="store_true",
        help="Group experiments by base name (e.g., for multiple seeds) and plot mean/CI."
    )
    parser.add_argument(
        "--smooth", type=int, default=10,
        help="Window size for moving average smoothing (set to 1 for no smoothing)."
    )
    parser.add_argument(
        "--ci", type=float, default=0.95,
        help="Confidence interval level (e.g., 0.95 for 95%% CI). Set to 0 to disable."
    )
    parser.add_argument(
        "--individual", action="store_true",
        help="If grouping, also plot individual trial lines faintly."
    )

    args = parser.parse_args()
    main(args)