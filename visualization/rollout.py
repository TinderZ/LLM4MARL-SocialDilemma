# visualization/rollout.py
"""
Runs a rollout (episode) in the specified environment with random actions
or a simple heuristic, rendering the steps to generate a video.
Focuses on the Cleanup environment using the PettingZoo API.
"""

import argparse
import os
import shutil
import sys
import time
from typing import Optional, Tuple, List, Dict # For potential pauses during rendering

import numpy as np
from pettingzoo.utils.env import ActionType, AECEnv # Import AECEnv type hint

# --- Adapt imports for your refactored structure ---
# Assuming cleanup_env provides an AEC interface via env() factory
from envs.cleanup_env import env as cleanup_aec_env
from visualization.utility_funcs import make_video_from_rgb_imgs

# Define action mapping if needed (or get from env.action_space)
# Example assuming standard actions 0-8 for Cleanup
STAY_ACTION = 4 # Example, adjust if your action mapping differs

class Controller:
    """Handles environment rollout and rendering."""
    def __init__(self, env_name: str = "cleanup", num_agents: int = 5, max_cycles: int = 1000):
        self.env_name = env_name
        self.num_agents = num_agents

        if self.env_name == "cleanup":
            print("Initializing Cleanup environment (AEC)...")
            # Use the AEC env factory
            self.env: AECEnv = cleanup_aec_env(num_agents=num_agents, render_mode="rgb_array", max_cycles=max_cycles)
        # Add elif for other envs like "harvest" if needed later
        else:
            raise ValueError(f"Environment '{env_name}' not supported by this rollout script.")

        print(f"Environment initialized with {self.num_agents} agents.")
        print(f"Observation Space Sample ({self.env.possible_agents[0]}): {self.env.observation_space(self.env.possible_agents[0])}")
        print(f"Action Space Sample ({self.env.possible_agents[0]}): {self.env.action_space(self.env.possible_agents[0])}")


    def rollout(self, horizon: int, save_dir: Optional[str] = None) -> Tuple[List[Dict[str, float]], List[Dict[str, np.ndarray]], List[np.ndarray]]:
        """
        Performs a rollout for a given number of steps (horizon).

        Args:
            horizon: The maximum number of environment steps (cycles) for the rollout.
            save_dir: If provided, saves rendered frames as PNG images in this directory.

        Returns:
            A tuple containing:
            - rewards_history: List of reward dicts per step.
            - observations_history: List of observation dicts per step.
            - rendered_frames: List of RGB frames (numpy arrays) rendered at each step.
        """
        rewards_history: List[Dict[str, float]] = []
        observations_history: List[Dict[str, np.ndarray]] = []
        rendered_frames: List[np.ndarray] = []

        print(f"Starting rollout for {horizon} steps...")
        self.env.reset(seed=42) # Seed for reproducibility

        step = 0
        agents_in_episode = set(self.env.agents)

        for agent_id in self.env.agent_iter():
            if step >= horizon:
                 print(f"Rollout horizon ({horizon} steps) reached.")
                 break

            observation, reward, terminated, truncated, info = self.env.last()
            observations_history.append({agent_id: observation}) # Store individual obs
            rewards_history.append({agent_id: reward})           # Store individual reward

            # Add rendered frame for this agent's turn (represents state *before* action)
            frame = self.env.render()
            if frame is not None:
                 rendered_frames.append(frame)
                 if save_dir:
                     frame_path = os.path.join(save_dir, f"frame_{step:06d}_{agent_id}.png")
                     # utility_funcs.save_img expects RGB
                     cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


            action: ActionType = None
            if terminated or truncated:
                # Agent is done, step with None action
                action = None
                agents_in_episode.discard(agent_id)
                print(f"Agent {agent_id} finished (Terminated: {terminated}, Truncated: {truncated}) at step {step}")
            else:
                # --- Action Selection (Random Agent) ---
                action_space = self.env.action_space(agent_id)
                if isinstance(action_space, gym.spaces.Discrete):
                    # Masking is not directly supported here, sample random valid action
                    action = action_space.sample()
                else:
                    # Handle other action space types if necessary
                    print(f"Warning: Unsupported action space {type(action_space)} for agent {agent_id}. Staying.")
                    action = STAY_ACTION # Default to STAY if space is complex/unknown

            self.env.step(action)
            step += 1 # Increment step count after *each agent* takes a turn

            # Check if all agents for the current step are done
            if not self.env.agents: # If the agent list becomes empty, the episode is over
                 print(f"Episode ended naturally at step {step} (all agents done).")
                 break


        print(f"Rollout finished after {step} steps.")
        return rewards_history, observations_history, rendered_frames


    def render_rollout(self, horizon: int = 50, path: str = "rollouts", fps: int = 10):
        """
        Performs a rollout and renders it as a video.

        Args:
            horizon: The number of environment steps (cycles) for the rollout.
            path: Directory path to save the video and temporary frames.
            fps: Frames per second for the output video.
        """
        video_dir = os.path.abspath(path)
        frames_dir = os.path.join(video_dir, "temp_frames")

        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir) # Clear old frames
        os.makedirs(frames_dir, exist_ok=True)

        print(f"Rendering rollout video to {video_dir}")
        print(f"Temporary frames will be saved in {frames_dir}")

        # Run rollout, saving frames
        _, _, rendered_frames = self.rollout(horizon=horizon, save_dir=None) # Don't save individual frames now

        # Create video from collected frames
        if rendered_frames:
            video_name = f"{self.env_name}_rollout_{horizon}steps"
            video_full_path = make_video_from_rgb_imgs(rendered_frames, video_dir, video_name=video_name, fps=fps)
            print(f"Video saved to {video_full_path}")
        else:
            print("No frames were rendered, video not created.")

        # Clean up temporary frames directory (optional)
        # print(f"Cleaning up temporary frames directory: {frames_dir}")
        # shutil.rmtree(frames_dir)
        print("Rendering complete.")

        self.env.close() # Close the environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a rollout simulation for the Cleanup environment.")
    parser.add_argument("--env", type=str, default="cleanup", choices=["cleanup"], help="Environment name.")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents.")
    parser.add_argument("--horizon", type=int, default=200, help="Number of steps (cycles) for the rollout.")
    parser.add_argument("--path", type=str, default="rollouts/random_agent", help="Directory to save the output video.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the video.")

    args = parser.parse_args()

    controller = Controller(env_name=args.env, num_agents=args.agents, max_cycles=args.horizon + 10) # Ensure max_cycles > horizon
    controller.render_rollout(horizon=args.horizon, path=args.path, fps=args.fps)