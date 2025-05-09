# visualization/visualizer_rllib.py
"""
Loads a trained RLlib agent checkpoint and performs rollouts,
rendering the environment steps to generate a video.
Focuses on the Cleanup environment and modern RLlib APIs.
"""

import argparse
import os
import shutil
import sys
import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

# --- Adapt imports for your refactored structure ---
from envs.cleanup_env import env as cleanup_aec_env # AEC env factory
from models.baseline_model import BaselineModel   # Import your custom model(s)
# from models.moa_model import MOAModel
# from models.scm_model import SocialCuriosityModule
from visualization.utility_funcs import make_video_from_rgb_imgs

# --- Environment Registration ---
# Ensure the environment used during training is registered here as well.
ENV_NAME_REGISTERED = "cleanup_v1_refactored_viz" # Use a distinct name if needed
register_env(ENV_NAME_REGISTERED, lambda config: PettingZooEnv(cleanup_aec_env(**config)))

# --- Model Registration (if using custom models) ---
# Make sure your custom models are registered if they were used during training
# This might be redundant if registered elsewhere, but safe to include.
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("baseline_model_refactored", BaselineModel)
# ModelCatalog.register_custom_model("moa_model_refactored", MOAModel)
# ModelCatalog.register_custom_model("scm_model_refactored", SocialCuriosityModule)


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a trained RLlib agent from a checkpoint and generate video.",
        epilog="""
Example Usage:
    python visualizer_rllib.py /path/to/your/checkpoint/dir/ --env cleanup --agents 5
""",
    )

    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the checkpoint directory (e.g., .../checkpoint_000100)."
    )
    # --run argument is often implicit now, deduced from checkpoint's algorithm
    # parser.add_argument(
    #     "--run", type=str, required=True, help="The algorithm class name (e.g., PPO)."
    # )
    parser.add_argument(
        "--env", type=str, default="cleanup", choices=["cleanup"],
        help="Environment name matching the one used for training."
    )
    parser.add_argument(
        "--agents", type=int, default=5, help="Number of agents matching the training config."
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Maximum number of steps (cycles) per rollout episode."
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of complete episodes to roll out."
    )
    parser.add_argument(
        "--video-dir", type=str, default="rollouts/rllib_agent",
        help="Directory to store the output video(s)."
    )
    parser.add_argument(
        "--video-filename", type=str, default=None,
        help="Specific filename for the video (without extension). If None, defaults are used."
    )
    parser.add_argument(
         "--fps", type=int, default=10, help="FPS for the output video."
    )
    # RLlib config overrides (less common for visualization, but possible)
    # parser.add_argument(
    #     "--config", type=json.loads, default="{}", help="Optional JSON config overrides."
    # )

    return parser

def run_rollout(args):
    """Loads checkpoint, runs rollouts, and saves video."""

    print(f"Loading checkpoint from: {args.checkpoint_path}")

    # Ensure Ray is initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Load the Algorithm from the checkpoint
    try:
        # For RLlib 2.x, Algorithm.from_checkpoint is preferred
        algo = Algorithm.from_checkpoint(args.checkpoint_path)
        print(f"Algorithm loaded successfully: {type(algo).__name__}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Attempting legacy restore (might not work for newer checkpoints)...")
        # Fallback for older RLlib versions or different checkpoint structures
        try:
             from ray.tune.analysis import ExperimentAnalysis
             analysis = ExperimentAnalysis(os.path.dirname(os.path.dirname(args.checkpoint_path))) # Go up two levels
             algo_class_str = analysis.best_config['algo_class_name'] # Hacky way to guess algo
             print(f"Guessed Algorithm: {algo_class_str}")
             from ray.rllib.algorithms.registry import get_algorithm_class
             AlgoClass = get_algorithm_class(algo_class_str)
             algo = AlgoClass(config=analysis.best_config)
             algo.restore(args.checkpoint_path)
             print("Legacy restore successful.")
        except Exception as e2:
            print(f"Legacy restore also failed: {e2}")
            print("Please ensure the checkpoint path is correct and Ray/RLlib versions match training.")
            sys.exit(1)


    # Get the environment creator config from the algorithm
    # The actual environment instance is created inside the rollout worker/evaluation worker
    # env_config = algo.config.get("env_config", {})
    env_config = {
        "num_agents": args.agents,
        # Add other necessary env args from your config if they weren't saved
        # or if you need to override them for visualization
        "render_mode": "rgb_array"
    }

    # Create a local environment instance for rendering metadata (optional but helpful)
    # Use the registered name and the deduced/provided config
    local_env = PettingZooEnv(cleanup_aec_env(**env_config))

    print("Starting rollout...")
    episode_count = 0
    total_steps = 0

    while episode_count < args.episodes:
        episode_reward = 0
        episode_steps = 0
        rendered_frames = []

        # Use the registered env_creator lambda for the PettingZooEnv wrapper
        # The actual environment instance lives inside the algo's workers,
        # but we need a local one to step through manually for visualization
        # Let's reset the local env to mirror the start of an episode
        obs, info = local_env.reset(seed=episode_count) # Seed each episode differently
        agent_states = {} # Store LSTM states if model is recurrent
        if algo.config.model.get("use_lstm"):
             for agent_id in local_env.agents:
                 policy = algo.get_policy(local_env.agent_to_policy(agent_id))
                 if hasattr(policy, 'get_initial_state'):
                     agent_states[agent_id] = policy.get_initial_state()
                 else:
                     # Handle non-RNN policies within potential RNN setup
                     agent_states[agent_id] = []

        terminated = {agent_id: False for agent_id in local_env.agents}
        truncated = {agent_id: False for agent_id in local_env.agents}
        active_agents = list(local_env.agents) # Track active agents

        while active_agents and episode_steps < args.steps:
            actions = {}
            current_step_agents = list(active_agents) # Agents acting in this step

            # Get actions for all currently active agents
            for agent_id in current_step_agents:
                if agent_id not in obs: # Agent might have terminated last step
                    continue

                policy_id = local_env.agent_to_policy(agent_id)
                policy = algo.get_policy(policy_id) # Get policy for the agent

                if policy is None:
                    print(f"Warning: Could not get policy for agent {agent_id} (policy_id: {policy_id}). Skipping action.")
                    actions[agent_id] = local_env.action_space(agent_id).sample() # Fallback: random action
                    continue

                state_in = agent_states.get(agent_id, [])
                action_tuple = algo.compute_single_action(
                    observation=obs[agent_id],
                    state=state_in,
                    policy_id=policy_id,
                    # Explore=False likely desired for visualization
                    explore=False
                )

                # Unpack action and new state
                action = action_tuple[0]
                if algo.config.model.get("use_lstm"):
                     agent_states[agent_id] = action_tuple[2] # LSTM state is usually 3rd element

                actions[agent_id] = action


            # Step the *local* environment with the computed actions
            new_obs, rewards, term_step, trunc_step, infos = local_env.step(actions)

            # Update global states
            obs = new_obs
            terminated.update(term_step)
            truncated.update(trunc_step)

            # Accumulate rewards and steps
            step_reward = sum(rewards.values())
            episode_reward += step_reward
            episode_steps += 1
            total_steps += 1

            # Render the current state *after* the step
            frame = local_env.render()
            if frame is not None:
                rendered_frames.append(frame)

            # Update active agents list
            active_agents = [agent_id for agent_id in local_env.agents if not terminated[agent_id] and not truncated[agent_id]]

            # # Optional: Print step info
            # print(f"Episode {episode_count+1}, Step {episode_steps}: Reward={step_reward:.2f}, Actions={actions}")

            # Check for episode end condition (all agents done or step limit)
            if not active_agents:
                print(f"Episode {episode_count+1} finished naturally after {episode_steps} steps.")
                break
            if episode_steps >= args.steps:
                 print(f"Episode {episode_count+1} reached step limit ({args.steps}).")
                 break # Ensure loop terminates

        episode_count += 1
        print(f"Episode {episode_count} finished. Total Reward: {episode_reward:.2f}, Steps: {episode_steps}")

        # Save video for this episode
        if rendered_frames:
            video_subdir = os.path.join(args.video_dir, f"episode_{episode_count}")
            video_name = args.video_filename if args.video_filename else f"{args.env}_checkpoint_{os.path.basename(args.checkpoint_path)}_ep{episode_count}"
            try:
                video_full_path = make_video_from_rgb_imgs(
                    rendered_frames, video_subdir, video_name=video_name, fps=args.fps
                )
                print(f"Video for episode {episode_count} saved to {video_full_path}")
            except Exception as e:
                print(f"Error saving video for episode {episode_count}: {e}")
        else:
            print(f"No frames rendered for episode {episode_count}, skipping video save.")

    print(f"Finished {episode_count} episodes and {total_steps} total steps.")
    local_env.close()
    # Consider stopping the algorithm/workers if they were started just for this
    # algo.stop() # Maybe not needed if loaded read-only


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_path):
         # Check if it's a file path (like checkpoint-100) and try parent dir
         if os.path.isfile(args.checkpoint_path):
             args.checkpoint_path = os.path.dirname(args.checkpoint_path)
             print(f"Checkpoint path adjusted to directory: {args.checkpoint_path}")
         else:
             print(f"Error: Checkpoint path '{args.checkpoint_path}' is not a valid directory.")
             sys.exit(1)

    run_rollout(args)
    ray.shutdown()