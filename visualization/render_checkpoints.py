# visualization/render_checkpoints.py
"""
Iterates through multiple RLlib checkpoint directories within an experiment
folder and generates rollout videos for each using visualizer_rllib.py.
Focuses on the Cleanup scenario.
"""

import os
import argparse
import subprocess # To call the visualizer script
import sys

from visualization.utility_funcs import get_all_subdirs

# --- Configuration ---
DEFAULT_RAY_RESULTS_DIR = os.path.expanduser("~/ray_results")
VISUALIZER_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizer_rllib.py"))


def find_checkpoint_dirs(experiment_dir: str) -> list[str]:
    """Finds all 'checkpoint_xxxxx' directories within an experiment folder."""
    checkpoint_dirs = []
    if not os.path.isdir(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return []

    for item in os.listdir(experiment_dir):
        full_path = os.path.join(experiment_dir, item)
        if os.path.isdir(full_path) and item.startswith("checkpoint_"):
            # Optionally check for actual checkpoint files inside
            # checkpoint_file = os.path.join(full_path, f"checkpoint-{item.split('_')[-1]}")
            # if os.path.isfile(checkpoint_file): # Less strict, just check dir exists
            checkpoint_dirs.append(full_path)

    # Sort checkpoints numerically
    checkpoint_dirs.sort(key=lambda x: int(x.split("_")[-1]))
    return checkpoint_dirs


def render_single_experiment(
    experiment_dir: str,
    env_name: str,
    num_agents: int,
    video_base_dir: str,
    steps_per_episode: int,
    num_episodes: int,
    fps: int
):
    """Renders videos for all checkpoints within a single experiment directory."""
    print(f"\n--- Processing Experiment: {os.path.basename(experiment_dir)} ---")
    checkpoint_dirs = find_checkpoint_dirs(experiment_dir)

    if not checkpoint_dirs:
        print(f"No valid checkpoint directories found in {experiment_dir}. Skipping.")
        return

    print(f"Found {len(checkpoint_dirs)} checkpoint directories.")

    output_video_dir = os.path.join(video_base_dir, os.path.basename(experiment_dir))
    os.makedirs(output_video_dir, exist_ok=True)
    print(f"Video output directory: {output_video_dir}")

    for i, chkpt_dir in enumerate(checkpoint_dirs):
        chkpt_num = chkpt_dir.split("_")[-1]
        print(f"\nRendering checkpoint {i+1}/{len(checkpoint_dirs)} (chkpt_{chkpt_num})...")
        print(f"Checkpoint path: {chkpt_dir}")

        # Construct the command to call visualizer_rllib.py
        # Ensure paths with spaces are handled if necessary (though unlikely for checkpoints)
        command = [
            sys.executable, # Use the same python interpreter
            VISUALIZER_SCRIPT_PATH,
            chkpt_dir, # Pass the checkpoint directory path
            "--env", env_name,
            "--agents", str(num_agents),
            "--steps", str(steps_per_episode),
            "--episodes", str(num_episodes),
            "--video-dir", output_video_dir,
            "--video-filename", f"rollout_chkpt_{chkpt_num}", # Unique name per checkpoint
            "--fps", str(fps)
        ]

        print(f"Executing command: {' '.join(command)}")
        try:
            # Run the visualizer script as a subprocess
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Visualizer Output:\n", process.stdout)
            if process.stderr:
                 print("Visualizer Error Output:\n", process.stderr)
            print(f"Checkpoint {chkpt_num} rendering complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error rendering checkpoint {chkpt_num}:")
            print(e.stderr)
            print(f"Visualizer stdout (if any):\n{e.stdout}")
            # Decide whether to continue or stop
            # continue
        except FileNotFoundError:
             print(f"Error: Could not find python executable '{sys.executable}' or visualizer script '{VISUALIZER_SCRIPT_PATH}'")
             return # Stop if scripts can't be found


def render_all_experiments(
    base_ray_dir: str,
    env_name: str,
    num_agents: int,
    video_base_dir: str,
    steps_per_episode: int,
    num_episodes: int,
    fps: int,
    experiment_filter: Optional[str] = None
):
    """Finds experiment folders and renders checkpoints for each."""
    print(f"Searching for experiment directories in: {base_ray_dir}")
    all_dirs = get_all_subdirs(base_ray_dir)

    experiment_dirs = []
    for d in all_dirs:
        # Basic check: does it contain checkpoint folders?
        has_checkpoints = any(name.startswith("checkpoint_") for name in os.listdir(d))
        # Filter by name if provided
        filter_match = (experiment_filter is None or experiment_filter in os.path.basename(d))

        if has_checkpoints and filter_match:
             # Heuristic: Check if it looks like a Tune experiment directory
             # (e.g., contains params.json or result.json)
             if os.path.exists(os.path.join(d, "params.json")) or os.path.exists(os.path.join(d, "result.json")):
                experiment_dirs.append(d)


    if not experiment_dirs:
        print("No valid experiment directories found matching criteria.")
        return

    print(f"Found {len(experiment_dirs)} experiment directories to process.")
    for exp_dir in experiment_dirs:
        render_single_experiment(
            exp_dir, env_name, num_agents, video_base_dir,
            steps_per_episode, num_episodes, fps
        )

    print("\n--- All Rendering Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render rollout videos from RLlib checkpoints.")
    parser.add_argument(
        "--ray-dir", type=str, default=DEFAULT_RAY_RESULTS_DIR,
        help="Base directory containing Ray/Tune experiment results."
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process experiment directories containing this string in their name."
    )
    parser.add_argument(
        "--env", type=str, required=True, choices=["cleanup"], # Focus on cleanup
        help="Environment name used for training (must match)."
    )
    parser.add_argument(
        "--agents", type=int, required=True,
        help="Number of agents used during training (must match)."
    )
    parser.add_argument(
        "--video-dir", type=str, default="rendered_checkpoints",
        help="Base directory to store the output videos."
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Maximum steps (cycles) per rollout episode video."
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to render per checkpoint."
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="FPS for the output videos."
    )

    args = parser.parse_args()

    # Only process cleanup experiments
    if args.env != 'cleanup':
        print("This script is configured to only process 'cleanup' experiments.")
        sys.exit(1)

    # Construct a filter that includes the environment name
    env_filter = f"{args.env}_" # Basic filter assuming env name is in exp name
    combined_filter = env_filter
    if args.filter:
        combined_filter = f"{args.filter}_{env_filter}" # Combine if user provided filter

    render_all_experiments(
        base_ray_dir=args.ray_dir,
        env_name=args.env,
        num_agents=args.agents,
        video_base_dir=args.video_dir,
        steps_per_episode=args.steps,
        num_episodes=args.episodes,
        fps=args.fps,
        experiment_filter=combined_filter # Use combined filter
    )