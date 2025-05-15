# run_scripts/train_rllib_centralized.py
import argparse
import os
import sys
from datetime import datetime
import pytz
import pandas as pd # 新增导入
# import functools # Ensure functools is not used for this

import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig # Using PPO directly
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks # 新增导入

# Add near the top with other imports
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time # Optional for smoother plotting in some backends
from ray.tune.callback import Callback as TuneCallback # 重命名以避免与rllib callbacks冲突
from ray.tune.experiment.trial import Trial  # Import Trial for type hinting
# import threading # To handle plotting in a separate thread potentially
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting

# --- Import your refactored environment and model ---
# Assuming PettingZoo AEC interface for the environment creator
from envs.cleanup_env import env as cleanup_env_creator
# Import other env creators if needed (e.g., harvest)
# Import your refactored model(s)
from models.baseline_model import BaselineModel
# from models.moa_model import MOAModel  # Example if extending
# from models.scm_model import SocialCuriosityModule # Example if extending


class PlottingCallback(TuneCallback): # 继承自 ray.tune.callback.Callback
    """
    A Tune Callback that plots agent metrics during training and saves the plot periodically.
    Now includes episode-level custom metrics: total/per-agent apples and pollution,
    with combined plots for apple-related metrics and pollution-related metrics.
    """
    def __init__(self, num_agents, policy_mode, plot_freq=5):
        super().__init__()
        self.num_agents = num_agents
        self.policy_mode = policy_mode
        self.plot_freq = plot_freq # Plot every N iterations
        self._iter = 0
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        # Data storage for standard metrics
        self.policy_loss = defaultdict(list)
        self.mean_reward = defaultdict(list)
        self.reward_variance = defaultdict(list) # Only one overall variance line
        self.timesteps = []

        # Data storage for custom episode metrics
        self.ep_total_apples_hist = []
        self.ep_total_pollution_hist = []
        self.ep_agent_apples_hist = defaultdict(list)
        self.ep_agent_pollution_hist = defaultdict(list)

        self._setup_plot()

    def _setup_plot(self):
        self.fig, self.axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True) # 更改为5个子图
        self.fig.suptitle("Agent Training Metrics", fontsize=16)

        self.axes[0].set_ylabel("Mean Policy Loss")
        self.axes[1].set_ylabel("Mean Episode Reward")
        self.axes[2].set_ylabel("Episode Reward Variance (Overall)")
        self.axes[3].set_ylabel("Apples Collected (Episode)") # Combined Apple Metrics
        self.axes[4].set_ylabel("Pollution Cleaned (Episode)") # Combined Pollution Metrics
        self.axes[4].set_xlabel("Training Timesteps") # X-axis label on the last subplot

        # Initial empty plot lines for standard metrics
        self.lines_loss = {}
        self.lines_reward = {}
        self.line_variance, = self.axes[2].plot([], [], label="Overall Reward Variance", color='red', linestyle='--')

        # Lines for combined custom metrics plots
        # Axes[3] for Apples
        self.line_total_apples, = self.axes[3].plot([], [], label="Total Apples", color='darkgreen', linewidth=2)
        self.lines_agent_apples = {}

        # Axes[4] for Pollution
        self.line_total_pollution, = self.axes[4].plot([], [], label="Total Pollution", color='saddlebrown', linewidth=2)
        self.lines_agent_pollution = {}

        colors = plt.cm.viridis(np.linspace(0, 1, self.num_agents))

        for i, agent_id in enumerate(self.agent_ids):
            # Standard metrics
            self.lines_loss[agent_id], = self.axes[0].plot([], [], label=f"{agent_id} Loss", color=colors[i], alpha=0.7)
            self.lines_reward[agent_id], = self.axes[1].plot([], [], label=f"{agent_id} Reward", color=colors[i], alpha=0.7)
            
            # Per-agent custom metrics (on combined plots)
            self.lines_agent_apples[agent_id], = self.axes[3].plot([], [], label=f"{agent_id} Apples", color=colors[i], alpha=0.7, linestyle=':')
            self.lines_agent_pollution[agent_id], = self.axes[4].plot([], [], label=f"{agent_id} Pollution", color=colors[i], alpha=0.7, linestyle=':')

        for i in range(5): # Iterate through all 5 axes
            self.axes[i].legend(loc='best', fontsize='small')
            self.axes[i].grid(True, linestyle='--', alpha=0.5) # Add grid for better readability
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.96])


    def on_trial_result(self, iteration: int, trials: list, trial: Trial, result: dict, **info):
        self._iter += 1
        if self._iter % self.plot_freq != 0:
            return

        if not trial.logdir:
             print(f"PlottingCallback: Warning - trial.logdir not available in iteration {self._iter}. Skipping plot save.")
             return
        save_path = os.path.join(trial.logdir, "training_metrics_detailed.png")

        current_timesteps = result.get("timesteps_total", self._iter)
        if not self.timesteps or self.timesteps[-1] < current_timesteps:
            self.timesteps.append(current_timesteps)
        
        # --- Standard Metrics Extraction (Loss, Reward, Variance) ---
        policy_losses_found = False
        if "info" in result and "learner" in result["info"]:
            for agent_id_full, policy_info in result["info"]["learner"].items():
                target_agent_ids_for_loss = []
                if agent_id_full in self.agent_ids:
                    target_agent_ids_for_loss.append(agent_id_full)
                # Assuming 'shared_policy' or 'default_policy' means it applies to all agents for plotting purposes.
                # This might depend on whether your 'policy_mode' is 'centralized'.
                elif "shared_policy" in agent_id_full or "default_policy" in agent_id_full or self.policy_mode == "centralized":
                    target_agent_ids_for_loss.extend(self.agent_ids)

                if "learner_stats" in policy_info:
                    loss = policy_info["learner_stats"].get("policy_loss")
                    if loss is not None:
                        for ag_id in target_agent_ids_for_loss:
                            self.policy_loss[ag_id].append(loss)
                        policy_losses_found = True
                    else: 
                        for ag_id in target_agent_ids_for_loss:
                            self.policy_loss[ag_id].append(self.policy_loss[ag_id][-1] if self.policy_loss[ag_id] else np.nan)
        
        if not policy_losses_found and "policy_loss" in result: 
             agg_loss = result["policy_loss"]
             for agent_id in self.agent_ids:
                self.policy_loss[agent_id].append(agg_loss)
        
        for agent_id in self.agent_ids:
            while len(self.policy_loss[agent_id]) < len(self.timesteps):
                self.policy_loss[agent_id].insert(0, np.nan) if self.timesteps and self.policy_loss[agent_id] else self.policy_loss[agent_id].append(np.nan)


        rewards_found = False
        if "policy_reward_mean" in result and isinstance(result["policy_reward_mean"], dict):
            processed_agents_reward = set()
            for policy_id_key, reward_val in result["policy_reward_mean"].items():
                # In centralized, key might be 'shared_policy'. In decentralized, 'agent_0', 'agent_1'.
                if policy_id_key in self.agent_ids: # Decentralized
                    self.mean_reward[policy_id_key].append(reward_val)
                    processed_agents_reward.add(policy_id_key)
                    rewards_found = True
                # If centralized, apply this single reward to all agents for plotting
                elif self.policy_mode == "centralized": # Check against actual policy_mode
                    for agent_id_iter in self.agent_ids:
                        self.mean_reward[agent_id_iter].append(reward_val)
                        processed_agents_reward.add(agent_id_iter)
                    rewards_found = True 
                    break # Shared policy reward found and applied to all

            for agent_id_iter in self.agent_ids:
                if agent_id_iter not in processed_agents_reward:
                    self.mean_reward[agent_id_iter].append(self.mean_reward[agent_id_iter][-1] if self.mean_reward[agent_id_iter] else np.nan)

        if not rewards_found and "episode_reward_mean" in result: 
            agg_reward = result["episode_reward_mean"]
            for agent_id_iter in self.agent_ids:
                self.mean_reward[agent_id_iter].append(agg_reward)

        for agent_id_iter in self.agent_ids:
             while len(self.mean_reward[agent_id_iter]) < len(self.timesteps):
                self.mean_reward[agent_id_iter].insert(0, np.nan) if self.timesteps and self.mean_reward[agent_id_iter] else self.mean_reward[agent_id_iter].append(np.nan)

        variance = np.nan
        if "hist_stats" in result and "episode_reward" in result["hist_stats"]:
            episode_rewards_list = result["hist_stats"]["episode_reward"]
            if len(episode_rewards_list) > 1: variance = np.var(episode_rewards_list)
        self.reward_variance["overall"].append(variance)
        while len(self.reward_variance["overall"]) < len(self.timesteps):
            self.reward_variance["overall"].insert(0, np.nan) if self.timesteps and self.reward_variance["overall"] else self.reward_variance["overall"].append(np.nan)


        # --- Custom Metrics Extraction ---
        custom_metrics_data = result.get("custom_metrics", {})
        self.ep_total_apples_hist.append(custom_metrics_data.get("ep_total_apples_collected_mean", np.nan))
        self.ep_total_pollution_hist.append(custom_metrics_data.get("ep_total_pollution_cleaned_mean", np.nan))

        for agent_id in self.agent_ids:
            self.ep_agent_apples_hist[agent_id].append(custom_metrics_data.get(f"{agent_id}_ep_apples_mean", np.nan))
            self.ep_agent_pollution_hist[agent_id].append(custom_metrics_data.get(f"{agent_id}_ep_pollution_mean", np.nan))
        
        # Pad custom metrics history
        current_len = len(self.timesteps)
        while len(self.ep_total_apples_hist) < current_len:
            self.ep_total_apples_hist.insert(0, np.nan)
        while len(self.ep_total_pollution_hist) < current_len:
            self.ep_total_pollution_hist.insert(0, np.nan)
        for agent_id in self.agent_ids:
            while len(self.ep_agent_apples_hist[agent_id]) < current_len:
                self.ep_agent_apples_hist[agent_id].insert(0, np.nan)
            while len(self.ep_agent_pollution_hist[agent_id]) < current_len:
                self.ep_agent_pollution_hist[agent_id].insert(0, np.nan)

        # --- Update Plot ---
        min_len = len(self.timesteps) # Should be same as current_len now
        if min_len == 0: return # Nothing to plot if no timesteps

        for agent_id in self.agent_ids:
            if agent_id in self.lines_loss:
                self.lines_loss[agent_id].set_data(self.timesteps, self.policy_loss[agent_id][-min_len:])
            if agent_id in self.lines_reward:
                self.lines_reward[agent_id].set_data(self.timesteps, self.mean_reward[agent_id][-min_len:])
            # Update custom per-agent metrics plots (on combined axes)
            if agent_id in self.lines_agent_apples:
                self.lines_agent_apples[agent_id].set_data(self.timesteps, self.ep_agent_apples_hist[agent_id][-min_len:])
            if agent_id in self.lines_agent_pollution:
                self.lines_agent_pollution[agent_id].set_data(self.timesteps, self.ep_agent_pollution_hist[agent_id][-min_len:])

        if self.line_variance:
            self.line_variance.set_data(self.timesteps, self.reward_variance["overall"][-min_len:])
        
        # Update custom total metrics plots (on combined axes)
        self.line_total_apples.set_data(self.timesteps, self.ep_total_apples_hist[-min_len:])
        self.line_total_pollution.set_data(self.timesteps, self.ep_total_pollution_hist[-min_len:])

        for ax in self.axes:
            ax.relim()
            ax.autoscale_view(True, True, True)

        try:
            self.fig.canvas.draw_idle()
            self.fig.savefig(save_path)
            # print(f"PlottingCallback: Saved plot to {save_path}") 
        except Exception as e:
             print(f"PlottingCallback: Failed to save plot to {save_path}: {e}")

    def close_plot(self):
        if hasattr(self, 'fig') and self.fig:
            plt.close(self.fig)
            self.fig = None
            self.axes = None


class EpisodeMetricsCallback(DefaultCallbacks):
    """
    RLlib Callback to record detailed episode metrics:
    - Total apples collected by all agents in the episode.
    - Total pollution cleaned by all agents in the episode.
    - Per-agent apples collected in the episode.
    - Per-agent pollution cleaned in the episode.
    These are stored in episode.custom_metrics.
    """
    def __init__(self, num_agents: int = None): # num_agents can be passed if known, or inferred
        super().__init__()
        self.num_agents_param = num_agents 
        self.agent_ids = [] # Will be populated in on_episode_start if num_agents is known

    def _initialize_agent_ids(self, episode):
        if not self.agent_ids: # Initialize only once or if num_agents_param is set
            if self.num_agents_param is not None:
                 self.agent_ids = [f"agent_{i}" for i in range(self.num_agents_param)]
            elif episode.policy_mapping_fn: # Try to infer from episode if possible (more robust)
                 # This is a bit indirect; simpler to require num_agents if not using _agent_ids
                 # For now, rely on num_agents_param or ensure agent_ids are present in infos
                 pass


    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        self._initialize_agent_ids(episode)
        # Initialize accumulators for the episode
        episode.user_data["ep_agent_apples"] = defaultdict(int)
        episode.user_data["ep_agent_pollution"] = defaultdict(int)

    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        #RLlib 2.x : The info dict is now available on the episode via episode.last_info_for(agent_id)
        #or iterate episode.get_infos() which returns a dict {agent_id: info}
        infos = episode.last_info_for() # Gets all agent infos from the last step
        if not infos: # Might be empty if called before first step infos are processed
            active_agents_in_step = list(episode._agent_to_last_info.keys()) # Fallback, less ideal
            infos = episode._agent_to_last_info
        else:
            active_agents_in_step = list(infos.keys())

        # If self.agent_ids wasn't set by num_agents_param, try to populate from observed agents
        if not self.agent_ids and active_agents_in_step:
            # Filter for "agent_X" pattern, sort them for consistency
            potential_ids = sorted([aid for aid in active_agents_in_step if aid.startswith("agent_")])
            if potential_ids:
                self.agent_ids = potential_ids

        for agent_id in self.agent_ids: # Iterate over expected agent_ids
            info = infos.get(agent_id, {}) # Get info for the specific agent
            if info:
                episode.user_data["ep_agent_apples"][agent_id] += info.get("apples_collected_step", 0)
                episode.user_data["ep_agent_pollution"][agent_id] += info.get("pollution_cleaned_step", 0)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        ep_agent_apples = episode.user_data["ep_agent_apples"]
        ep_agent_pollution = episode.user_data["ep_agent_pollution"]

        total_apples = sum(ep_agent_apples.values())
        total_pollution = sum(ep_agent_pollution.values())

        episode.custom_metrics["ep_total_apples_collected"] = total_apples
        episode.custom_metrics["ep_total_pollution_cleaned"] = total_pollution

        for agent_id in self.agent_ids: # Use the initialized list of agent_ids
            episode.custom_metrics[f"{agent_id}_ep_apples"] = ep_agent_apples.get(agent_id, 0)
            episode.custom_metrics[f"{agent_id}_ep_pollution"] = ep_agent_pollution.get(agent_id, 0)


class StepIntervalMetricsCallback(DefaultCallbacks):
    """
    RLlib Callback to record metrics at fixed step intervals (e.g., every 50 steps)
    within an episode. Logs data to a CSV file in the trial's log directory.
    Metrics:
    - Apples collected by each agent in the interval.
    - Pollution cleaned by each agent in the interval.
    - LLM commands received by each agent in the interval.
    """
    def __init__(self, num_agents: int, interval: int = 50):
        super().__init__()
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        self.interval = interval
        self.interval_data_accumulator = [] # Stores dicts for CSV rows
        self.trial_logdir = None # Will be set in on_train_result

    def _reset_current_interval_stats(self, episode):
        episode.user_data["interval_step_count"] = 0
        episode.user_data["interval_agent_apples"] = defaultdict(int)
        episode.user_data["interval_agent_pollution"] = defaultdict(int)
        episode.user_data["interval_agent_llm_commands"] = defaultdict(lambda: defaultdict(int))

    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["interval_step_count"] += 1
        
        infos = episode.last_info_for()
        if not infos: infos = episode._agent_to_last_info # Fallback


        for agent_id in self.agent_ids:
            info = infos.get(agent_id, {})
            if info:
                episode.user_data["interval_agent_apples"][agent_id] += info.get("apples_collected_step", 0)
                episode.user_data["interval_agent_pollution"][agent_id] += info.get("pollution_cleaned_step", 0)
                llm_cmd = info.get("llm_command")
                if llm_cmd:
                    episode.user_data["interval_agent_llm_commands"][agent_id][llm_cmd] += 1
        
        current_ep_steps = episode.user_data["interval_step_count"]
        if current_ep_steps > 0 and current_ep_steps % self.interval == 0:
            interval_end_step = episode.length # Episode length is total steps so far in episode
            interval_start_step = interval_end_step - self.interval + 1

            # Ensure trial_logdir is set (might be set first time in on_train_result if on_train_result wasn't called or available)
            if not self.trial_logdir and hasattr(episode, 'trial_id'): # Check if part of a Tune trial
                 # This is a guess; robust way is through on_train_result's 'trial' object
                 # For now, we will ensure saving happens in on_train_result where 'trial.logdir' is reliable.
                 pass

            record = {
                "episode_id": str(episode.episode_id), # Convert UUID to string for pd compatibility if needed
                "training_iteration": episode.custom_metrics.get("training_iteration", result.get("training_iteration", np.nan) if 'result' in locals() else np.nan), # Try to get training iteration
                "timesteps_total_interval_end": episode.total_reward_これまで + interval_end_step, # Attempt global step context
                "episode_steps_interval": f"{interval_start_step}-{interval_end_step}"
            }

            # Retrieve stats for the completed interval
            # These were accumulated over the last 'self.interval' steps.
            current_interval_apples = episode.user_data["interval_agent_apples"]
            current_interval_pollution = episode.user_data["interval_agent_pollution"]
            current_interval_llm = episode.user_data["interval_agent_llm_commands"]

            for agent_id in self.agent_ids:
                record[f"{agent_id}_apples_interval"] = current_interval_apples.get(agent_id,0)
                record[f"{agent_id}_pollution_interval"] = current_interval_pollution.get(agent_id,0)
                record[f"{agent_id}_llm_collect_apples_interval"] = current_interval_llm.get(agent_id, {}).get("collect apples", 0)
                record[f"{agent_id}_llm_clean_up_interval"] = current_interval_llm.get(agent_id, {}).get("clean up", 0)
            
            self.interval_data_accumulator.append(record)
            
            # Reset stats for the next interval within the same episode
            # Reset !!!
            episode.user_data["interval_agent_apples"] = defaultdict(int)
            episode.user_data["interval_agent_pollution"] = defaultdict(int)
            episode.user_data["interval_agent_llm_commands"] = defaultdict(lambda: defaultdict(int))
            # episode.user_data["interval_step_count"] is continuous within episode, no reset here.

    def on_train_result(self, *, algorithm, result: dict, **kwargs): # For RLlib context
        """Save accumulated interval data to CSV periodically using algorithm's logdir."""
        if not self.trial_logdir: # Ensure logdir is set
            self.trial_logdir = algorithm.logdir

        if self.interval_data_accumulator and self.trial_logdir:
            df = pd.DataFrame(self.interval_data_accumulator)
            output_path = os.path.join(self.trial_logdir, "step_interval_metrics.csv")
            
            # Add training iteration to records if available and not already there
            if "training_iteration" not in df.columns and "training_iteration" in result:
                 df["training_iteration"] = result["training_iteration"]
            if "timesteps_total" not in df.columns and "timesteps_total" in result: # Overall timesteps at this trial result
                 df["timesteps_total_at_result"] = result["timesteps_total"]

            try:
                if os.path.exists(output_path):
                    df.to_csv(output_path, mode='a', header=False, index=False)
                else:
                    df.to_csv(output_path, mode='w', header=True, index=False)
                # print(f"StepIntervalMetricsCallback: Appended/Saved {len(self.interval_data_accumulator)} rows to {output_path}")
            except Exception as e:
                print(f"StepIntervalMetricsCallback: Error saving to {output_path}: {e}")
            
            self.interval_data_accumulator.clear()

# --- Environment Registration ---
# It's common to register environments here before Tune runs.

# +++ START NEW CODE +++
class CombinedRLlibCallbacks(DefaultCallbacks):
    def __init__(self, num_agents: int, interval: int):
        super().__init__()
        self.episode_cb = EpisodeMetricsCallback(num_agents=num_agents)
        self.step_interval_cb = StepIntervalMetricsCallback(num_agents=num_agents, interval=interval)

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        self.episode_cb.on_episode_start(
            worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs
        )
        self.step_interval_cb.on_episode_start(
            worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs
        )

    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        self.episode_cb.on_episode_step(
            worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs
        )
        self.step_interval_cb.on_episode_step(
            worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs
        )

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        self.episode_cb.on_episode_end(
            worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs
        )
        # StepIntervalMetricsCallback does not have on_episode_end, but if it did, it would be called here.

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # EpisodeMetricsCallback does not have on_train_result
        self.step_interval_cb.on_train_result(algorithm=algorithm, result=result, **kwargs)
# +++ END NEW CODE +++

def env_creator(env_config):
    env_name = env_config.get("env_name", "cleanup")
    num_agents = env_config.get("num_agents", 5)
    # ... other env args ...

    if env_name == "cleanup":
        # Create the PettingZoo AEC environment first
        aec_env = cleanup_env_creator(num_agents=num_agents) # , use_llm=use_llm, etc.
        # Wrap it with RLlib's wrapper
        rllib_multi_agent_env = PettingZooEnv(aec_env)
        return rllib_multi_agent_env
    # elif env_name == "harvest":
        # aec_env = harvest_aec_creator(num_agents=num_agents)
        # return PettingZooEnv(aec_env)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
# Register the environment creator function under a unique name
ENV_NAME_REGISTERED = "ssd_cleanup" # Or make this dynamic based on args.env
register_env(ENV_NAME_REGISTERED, env_creator)


# --- Model Registration ---
# Register custom models with RLlib
# You can choose unique names or use the class directly in config
ModelCatalog.register_custom_model("baseline_model&llm", BaselineModel)
# ModelCatalog.register_custom_model("moa_model_refactored", MOAModel) # Example
# ModelCatalog.register_custom_model("scm_model_refactored", SocialCuriosityModule) # Example


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment Identification
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Experiment name prefix."
    )
    parser.add_argument(
        "--env", type=str, default="cleanup", choices=["cleanup", "harvest"], # Add others if refactored
        help="Environment name."
    )
    parser.add_argument(
        "--algorithm", type=str, default="PPO", choices=["PPO"], # Extend if needed
        help="RLlib algorithm."
    )
    parser.add_argument(
        "--model", type=str, default="baseline", choices=["baseline", "moa", "scm"], # Add others if refactored
        help="Model architecture."
    )
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="centralized",
        choices=["centralized", "decentralized", "two_policies"],
        help="Defines the multi-agent policy configuration. "
             "'centralized': Single policy for all agents. "
             "'decentralized': One policy per agent. "
             "'two_policies': Agents 0,1 use policy_A, rest policy_B. "
    )
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of trials to run.")
    parser.add_argument( "--seed", type=int, default=None, help="Set seed for reproducibility.")


    # Ray and Tune Control
    parser.add_argument("--local_mode", action="store_true", help="Run Ray in local mode for debugging.")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if found.")
    parser.add_argument("--restore", type=str, default=None, help="Explicit path to checkpoint to restore from.")
    parser.add_argument("--use_s3", action="store_true", help="Upload results to S3.")
    parser.add_argument("--s3_bucket_prefix", type=str, default="s3://your-bucket-name/ssd-results", help="S3 bucket prefix for uploads.") # CHANGE BUCKET NAME
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="Save checkpoint every N iterations.")
    parser.add_argument("--stop_timesteps", type=int, default=int(500e6), help="Stop after N total env steps.")
    parser.add_argument("--stop_reward", type=float, default=None, help="Stop if avg reward reaches this value.")
    parser.add_argument("--stop_iters", type=int, default=None, help="Stop after N training iterations.")


    # Resource Allocation (match run script)
    parser.add_argument("--num_workers", type=int, default=6, help="Number of rollout workers.")
    parser.add_argument("--num_envs_per_worker", type=int, default=16, help="Number of envs per worker.")
    parser.add_argument("--cpus_per_worker", type=float, default=1, help="CPUs per worker.")
    parser.add_argument("--gpus_per_worker", type=float, default=0, help="GPUs per worker.")
    parser.add_argument("--cpus_for_driver", type=int, default=1, help="CPUs for the driver (trainer).")
    parser.add_argument("--gpus_for_driver", type=float, default=1, help="GPUs for the driver (trainer).")

    # Core PPO Hyperparameters (match run script)
    parser.add_argument("--rollout_fragment_length", type=int, default=1000, help="RLlib rollout fragment length.")
    
    parser.add_argument("--horizon", type=int, default=None, help="Episode horizon (max steps per episode). If None, PPO's default or env max steps will be used.")
    parser.add_argument("--soft_horizon", action="store_true", help="Enable soft horizon. Episodes are truncated at horizon but env is not reset if True.")
    parser.add_argument("--no_done_at_end", action="store_true", help="Set no_done_at_end. If True, the done=True signal will not be set when an episode ends solely due to reaching the horizon. Useful for RNNs.")
    
    parser.add_argument("--train_batch_size", type=int, default=None, help="RLlib train batch size (if None, calculated).")
    parser.add_argument("--sgd_minibatch_size", type=int, default=None, help="RLlib SGD minibatch size (if None, calculated).")
    parser.add_argument("--num_sgd_iter", type=int, default=10, help="Number of SGD iterations per train batch.") # Default PPO is 10-30
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides schedule if set).")
    parser.add_argument("--lr_schedule_steps", nargs="+", type=int, default=[0, 20000000], help="Timesteps for LR schedule points.")
    parser.add_argument("--lr_schedule_weights", nargs="+", type=float, default=[0.00126, 0.000012], help="LR values for schedule points.")
    parser.add_argument("--entropy_coeff", type=float, default=0.00176, help="Entropy coefficient.")
    parser.add_argument("--vf_loss_coeff", type=float, default=0.5, help="Value function loss coefficient (PPO default).") # Common PPO default
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter (PPO default).") # Common PPO default
    parser.add_argument("--grad_clip", type=float, default=40.0, help="Gradient clipping.")

    # Model Hyperparameters (Specific to your refactored models)
    parser.add_argument("--lstm_hidden_size", type=int, default=128, help="LSTM hidden state size.")
    # Add other model-specific args if they differ from defaults in BaselineModel etc.
    # e.g., parser.add_argument("--fcnet_hiddens", nargs='+', type=int, default=[32, 32])

    # Environment Specific Args (if needed)
    parser.add_argument("--use_collective_reward", action="store_true", help="Use collective reward.")
    # Add other env args if they influence the env_creator

    # LLM Args (if applicable)
    parser.add_argument("--use_llm", action="store_true", help="Enable LLM features in the environment.")
    parser.add_argument("--llm_f_step", type=int, default=50, help="LLM update frequency in steps.")

    args = parser.parse_args()

    # Calculate default batch sizes if not provided
    if args.train_batch_size is None:
        args.train_batch_size = args.num_workers * args.num_envs_per_worker * args.rollout_fragment_length
        print(f"Calculated train_batch_size: {args.train_batch_size}")
    if args.sgd_minibatch_size is None:
        # PPO often uses smaller minibatches than the full train batch
        # A common default is 128 or 256, or derived from train_batch_size
        args.sgd_minibatch_size = max(128, args.train_batch_size // 16) # Example derivation
        print(f"Calculated sgd_minibatch_size: {args.sgd_minibatch_size}")

    return args


# --- Main Execution ---
def main(args):
    # Initialize Ray
    if args.local_mode:
        ray.init(num_cpus=args.cpus_for_driver + args.num_workers * args.cpus_per_worker,
                 local_mode=True)
    else:
        # Connect to existing cluster or start new one
        ray.init(address=os.environ.get("RAY_ADDRESS", None)) # Assumes RAY_ADDRESS is set for clusters


    temp_env_config = {
        "env_name": args.env,
        "num_agents": args.num_agents,
        "use_llm": args.use_llm,
        "llm_f_step": args.llm_f_step,
    }
    temp_env = env_creator(temp_env_config)
    obs_space = temp_env.observation_space["agent_0"]
    act_space = temp_env.action_space["agent_0"]
    temp_env.close()

    # Dynamically define policies and mapping function ---
    policies_dict = {}
    actual_policy_mapping_fn = None
    print(f"Using policy mode: {args.policy_mode}")

    if args.policy_mode == "centralized":
        policies_dict = {
            "shared_policy": PolicySpec(
                # policy_class is None to use the default for the algorithm (e.g., PPO TorchPolicy)
                observation_space=obs_space,
                action_space=act_space,
                # config can be added here to override model or other policy-specific settings
            )
        }
        actual_policy_mapping_fn = lambda agent_id, *a, **kw: "shared_policy"
        print("Policy setup: All agents use 'shared_policy'.")

    elif args.policy_mode == "decentralized":
        policies_dict = {
            f"agent_{i}": PolicySpec(
                observation_space=obs_space,
                action_space=act_space,
            )
            for i in range(args.num_agents)
        }
        # Each agent_id (e.g., "agent_0") maps to a policy_id with the same name.
        actual_policy_mapping_fn = lambda agent_id, *a, **kw: agent_id
        print(f"Policy setup: Each of {args.num_agents} agents uses its own policy (agent_0, agent_1, ...).")

    elif args.policy_mode == "two_policies":
        policies_dict = {
            "policy_A": PolicySpec(observation_space=obs_space, action_space=act_space),
            "policy_B": PolicySpec(observation_space=obs_space, action_space=act_space),
        }
        def mapping_fn_two_policies(agent_id, episode, worker, **kwargs):
            agent_num = int(agent_id.split('_')[-1])
            # Example: First 2 agents use policy_A, the rest use policy_B
            # Adjust this condition based on how you want to split them
            if agent_num < 2: # Agents "agent_0" and "agent_1"
                return "policy_A"
            else:
                return "policy_B"
        actual_policy_mapping_fn = mapping_fn_two_policies
        print("Policy setup: Agents 0 & 1 use 'policy_A', others use 'policy_B'.")

    else:
        raise ValueError(f"Unknown --policy_mode: {args.policy_mode}")

    
    # --- Configure Algorithm ---
    if args.algorithm == "PPO":
        config = PPOConfig()
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    # Configure RLlib-specific callbacks
    # Pass the raw class to .callbacks()
    config = config.callbacks(CombinedRLlibCallbacks)
    # Assign the arguments dictionary to the .callbacks_config attribute
    config.callbacks_config = {
        "num_agents": args.num_agents,
        "interval": 50  # Default interval for StepIntervalMetricsCallback, can be from args if needed
    }

    # Select correct registered model name
    if args.model == "baseline":
        model_name_registered = "baseline_model&llm"
    # elif args.model == "moa":
    #     model_name_registered = "moa_model_refactored"
    # elif args.model == "scm":
    #     model_name_registered = "scm_model_refactored"
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Learning Rate Schedule
    if args.lr is not None:
        lr_schedule = None # Override schedule if fixed LR is given
        lr_value = args.lr
    elif args.lr_schedule_steps and args.lr_schedule_weights:
        lr_schedule = list(zip(args.lr_schedule_steps, args.lr_schedule_weights))
        lr_value = args.lr_schedule_weights[0] # Initial LR
    else:
        lr_schedule = None
        lr_value = 0.0001 # Default if nothing specified

    # Environment Config (passed to env_creator)
    env_config = {
        "env_name": args.env,
        "num_agents": args.num_agents,
        "use_llm": args.use_llm,
        "llm_f_step": args.llm_f_step,
        # Add other env-specific args here if needed by creator
    }

    config = (
        config
        .environment(
            env=ENV_NAME_REGISTERED,
            env_config=env_config,
            disable_env_checking=True # Recommended for multi-agent/complex envs
        )
        .framework("torch") # Or "tf2"
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=args.num_envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
            # for short episode
            # horizon=args.horizon,
            # soft_horizon=args.soft_horizon,
            # no_done_at_end=args.no_done_at_end
        )
        .training(
            gamma=0.99,
            lr=lr_value,
            lr_schedule=lr_schedule,
            lambda_=0.95, # GAE lambda (PPO default)
            kl_coeff=0.2, # PPO default
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            train_batch_size=args.train_batch_size,
            vf_loss_coeff=args.vf_loss_coeff,
            entropy_coeff=args.entropy_coeff,
            clip_param=args.clip_param,
            grad_clip=args.grad_clip,
            model={
                "custom_model": model_name_registered,
                "custom_model_config": {
                    # Pass model-specific args from command line or defaults
                    # Ensure these match your refactored model's __init__
                    "conv_filters": [[6, [3, 3], 1]], # Example default
                    "fcnet_hiddens": [32, 32], # Example default
                    "lstm_hidden_size": args.lstm_hidden_size,
                },
                 "use_lstm": False#True, # Let RLlib handle if model is nn.Module? See BaselineModel notes.
                #"lstm_use_prev_action": True
            },
        )

        .multi_agent(
            policies=policies_dict,
            policy_mapping_fn=actual_policy_mapping_fn,
            # Optional: If you only want to train a subset of policies explicitly
            # policies_to_train=["list_of_policy_ids_to_train_if_needed"]
        )
        
        .resources(
            num_gpus=args.gpus_for_driver,
            num_cpus_per_worker=args.cpus_per_worker,
            num_gpus_per_worker=args.gpus_per_worker,
            num_cpus_for_local_worker = args.cpus_for_driver, # Renamed from driver
        )
        
        # .evaluation(
        #     evaluation_interval=2,  # 每 1 次训练迭代运行一次评估
        #     evaluation_duration=1,  # 每次评估运行 1 个 episode
        #     evaluation_duration_unit="episodes",
        #     evaluation_num_workers=1, # 使用 1 个 worker 进行评估
        #     evaluation_config={
        #         "render_env": True,  # <--- 启用环境渲染
        #         # 可选：如果评估需要特定环境配置，可以在这里覆盖
        #         # "env_config": { ... }
        #         # 可选：分配给评估 worker 的资源
        #          "explore": False # 通常在评估时不进行探索
        #     }
        # )
        # Add evaluation config if needed
        #.evaluation(evaluation_interval=10, evaluation_num_workers=1)
        .debugging(seed=args.seed) # Set seed if provided, None otherwise # Set seed if provided
    )


    # --- Stopping Criteria ---
    stop_criteria = {}
    if args.stop_timesteps:
        stop_criteria["timesteps_total"] = args.stop_timesteps
    if args.stop_reward:
        stop_criteria["episode_reward_mean"] = args.stop_reward
    if args.stop_iters:
        stop_criteria["training_iteration"] = args.stop_iters
    if not stop_criteria:
        stop_criteria["training_iteration"] = 100 # Default stop after 100 iters if nothing else set


    # --- Experiment Naming and Storage ---
    experiment_base_name = args.exp_name if args.exp_name else f"{args.env}_{args.model}_{args.algorithm}"
    # Add date/time for uniqueness?
    # timestamp = datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d_%H-%M-%S")
    # experiment_full_name = f"{experiment_base_name}_{timestamp}"
    experiment_full_name = experiment_base_name # Keep it simple for now

    storage_path = os.path.expanduser("~/ray_results")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    if args.use_s3:
        # Ensure path ends with / for S3 uploads
        s3_prefix = args.s3_bucket_prefix
        if not s3_prefix.endswith('/'):
            s3_prefix += '/'
        storage_path = s3_prefix

    plot_callback = PlottingCallback(
        num_agents=args.num_agents,
        policy_mode=args.policy_mode,
        plot_freq=5  # Update plot every 5 training iterations (adjust as needed)
        # save_path=os.path.join(os.path.expanduser("ray_results"), # Save in default results dir
        #                         f"{experiment_full_name}_metrics.png") # Filename based on experiment
    )

    # New Callbacks
    # episode_metrics_cb = EpisodeMetricsCallback(num_agents=args.num_agents) # Will be handled by CombinedRLlibCallbacks
    # step_interval_cb = StepIntervalMetricsCallback(num_agents=args.num_agents, interval=50) # Will be handled by CombinedRLlibCallbacks


    # --- Setup Tune ---
    tuner = tune.Tuner(
        args.algorithm, 
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            name=experiment_full_name,
            stop=stop_criteria,
            storage_path=storage_path, 
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=True,
                num_to_keep=3 
            ),
            callbacks=[plot_callback] # Only Tune-specific callbacks here
            # Add failure config if needed
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples, 
            metric="episode_reward_mean", 
            mode="max", 
        ),
    )

    # --- Restore and Run ---
    if args.resume:
         print(f"Attempting to resume experiment: {experiment_full_name} from {storage_path}")
         # Note: Tuner automatically handles resuming if the experiment name/path exists
         # tuner = tune.Tuner.restore(os.path.join(storage_path, experiment_full_name), trainable=args.algorithm)
         # The above might be needed for specific resume cases, but Tuner(..., run_config=...) often handles it.
         pass # Tuner handles resume based on name/path
    elif args.restore:
         print(f"Restoring experiment from checkpoint: {args.restore}")
         # Restore requires the specific trainable and path to checkpoint *directory*
         tuner = tune.Tuner.restore(path=args.restore, trainable=args.algorithm)
         # Need to potentially re-apply some config/stop criteria if not in checkpoint?
         # tuner.update_config(...) # Less common, usually restore loads most things

    # Run the experiment(s)
    results = tuner.fit()

    print("Training finished.")
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    # --- 增加检查 ---
    if best_result:
        print("Best trial config: {}".format(best_result.config))
        if best_result.metrics and "episode_reward_mean" in best_result.metrics:
            print("Best trial final reward: {}".format(best_result.metrics["episode_reward_mean"]))
        else:
            print("Best trial found, but 'episode_reward_mean' metric is missing.")
            print(f"All metrics for best trial: {best_result.metrics}")
    else:
        print("No best trial found (likely due to errors or no completed trials).")
    

    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    # Handle potential debug mode setting local_mode
    if sys.gettrace() is not None:
         print("Debug mode detected, forcing local_mode=True")
         args.local_mode = True
         if args.exp_name is None:
             args.exp_name = "debug_experiment" # Override name for debug runs

    main(args)