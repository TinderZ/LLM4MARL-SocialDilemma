# cleanup_env.py
import functools
import random
from copy import deepcopy
from typing import Dict, Tuple, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


from envs.cleanup_agent import CleanupAgent
from envs.llm_module import LLMModule

from envs.constants import IMMOBILIZE_DURATION_HIT, IMMOBILIZE_DURATION_FIRE, STAY_ACTION_INDEX
from envs.constants import (ACTION_MEANING, APPLE, APPLE_REWARD, APPLE_RESPAWN_PROBABILITY,
                     APPLE_SPAWN, AGENT_CHARS, CLEANUP_MAP, CLEANUP_VIEW_SIZE, CLEAN_BEAM_LENGTH,
                     CLEAN_BEAM_WIDTH, CLEAN_REWARD, CLEANABLE_TILES, CLEAN_BLOCKING_CELLS,
                     CLEANED_TILE_RESULT, CLEAN_BEAM, DEFAULT_COLOURS, EMPTY, FIRE_BEAM_LENGTH,
                     FIRE_BEAM_WIDTH, FIRE_BLOCKING_CELLS, MOVE_ACTIONS, NON_WALKABLE, NUM_ACTIONS,
                     ORIENTATIONS, ORIENTATION_VECTORS, PENALTY_BEAM, PENALTY_FIRE, PENALTY_HIT,
                     RIVER, ROTATION_MAP, SPECIAL_ACTIONS, STREAM, THRESHOLD_DEPLETION,
                     THRESHOLD_RESTORATION, TURN_ACTIONS, VIEW_PADDING, WALL, WASTE,
                     WASTE_INIT, WASTE_SPAWN_PROBABILITY, AGENT_START)
from envs.constants import (CLEAN_BEAM_LENGTH_VALID)

class CleanupEnv(ParallelEnv):
    """
    Cleanup Social Dilemma Environment based on the original implementation,
    refactored for the PettingZoo Parallel API.

    In this environment, agents are incentivized to collect apples (reward=1).
    Apples grow based on the cleanliness of a nearby river. Cleaning the river
    requires agents to use a cleaning beam. However, the river gets polluted by
    waste generated in a separate spawning area. Agents can use a "penalty" beam
    to punish other agents (cost=-1 for firing, reward=-50 for being hit).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "cleanup_v1"}

    def __init__(
        self,
        num_agents: int = 5,
        render_mode: str | None = None,
        max_cycles: int = 1000,
        use_collective_reward: bool = False,  # Not implemented in detail from original, but kept as option
        inequity_averse_reward: bool = False, # Not implemented in detail from original, but kept as option
        alpha: float = 0.0,                   # Parameter for IAR
        beta: float = 0.0,                    # Parameter for IAR

        use_llm: bool = False,                # Enable/disable LLM features
        llm_f_step: int = 50,                 # Frequency of LLM calling
        llm_type: str = "rule-based"          # Type of LLM to use
    ):
        """
        Initializes the Cleanup environment.

        Args:
            num_agents: The number of agents in the environment.
            render_mode: The mode for rendering ('human' or 'rgb_array').
            max_cycles: The maximum number of steps per episode.
            use_collective_reward: Whether to use a collective reward signal.
            inequity_averse_reward: Whether to use inequity aversion rewards.
            alpha: Inequity aversion parameter (penalty for others having more).
            beta: Inequity aversion parameter (penalty for having less than others).

            use_llm: Whether to enable LLM-based observation masking.
            llm_f_step: How often (in steps) the LLM processes game info.
            llm_type: The type of LLM to use (e.g., "rule-based", "chat-gpt").
        """
        super().__init__()

        if num_agents <= 0:
            raise ValueError("Number of agents must be positive.")
        if num_agents > len(AGENT_CHARS):
             raise ValueError(f"Maximum number of agents is {len(AGENT_CHARS)}")

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]

        self.agent_id_map = {i: f"agent_{i}" for i in range(num_agents)}
        self.render_mode = render_mode
        self.max_cycles = max_cycles

        # Reward structure (advanced options not fully implemented from original)
        self.use_collective_reward = use_collective_reward
        self.inequity_averse_reward = inequity_averse_reward
        self.alpha = alpha
        self.beta = beta

        # LLM-related attributes
        self.use_llm = use_llm 
        self.llm_f_step = llm_f_step
        self.llm_type = llm_type
        self.llm_modules: dict[str, LLMModule] = {}
        self.llm_commands: dict[str, str | None] = {} # Store current command per agent
        if self.use_llm:
            print(f"LLM integration enabled. Update frequency: {self.llm_f_step} steps.")
            for agent_id in self.possible_agents:
                 self.llm_modules[agent_id] = LLMModule(agent_id)
                 self.llm_commands[agent_id] = None # Initialize with no command

        # Load map and initialize state variables
        self.base_map = self._ascii_to_numpy(CLEANUP_MAP)
        self.world_map = deepcopy(self.base_map)
        self.map_height, self.map_width = self.base_map.shape

        # Find initial points
        self.spawn_points = self._find_points(AGENT_START)
        self.apple_spawn_points = self._find_points(APPLE_SPAWN)
        self.waste_init_points = self._find_points(WASTE_INIT)
        self.river_points = self._find_points(RIVER)
        self.stream_points = self._find_points(STREAM) # Stream tiles 'S'
        self.wall_points = self._find_points(WALL)
        

        # Waste dynamics related points
        self.potential_waste_area = len(self.waste_init_points) + len(self.river_points)
        self.waste_spawn_points = self.waste_init_points + self.river_points # All points where waste can exist

        # Initialize agents
        self._agents: dict[str, CleanupAgent] = {} # Use dict for agent management
        # Will be populated in reset()

        # State variables updated during steps
        self.current_waste_density = 0.0
        self.current_apple_spawn_prob = APPLE_RESPAWN_PROBABILITY
        self.current_waste_spawn_prob = WASTE_SPAWN_PROBABILITY
        self._compute_probabilities() # Initial calculation (will also set current_waste_density)

        self.num_cycles = 0
        self.beam_pos = [] # Positions currently occupied by beams for rendering

        # Setup rendering if needed
        self.fig = None
        self.ax = None
        self.render_im = None

        # --- PettingZoo API Properties ---
        # Define observation space: RGB image view for each agent
        self.observation_spaces = {
            agent_id: spaces.Box(
                low=0, high=255,
                shape=(2 * CLEANUP_VIEW_SIZE + 1, 2 * CLEANUP_VIEW_SIZE + 1, 3),
                dtype=np.uint8
            ) for agent_id in self.possible_agents
        }

        # Define action space: Discrete actions for each agent
        self.action_spaces = {
            agent_id: spaces.Discrete(NUM_ACTIONS) for agent_id in self.possible_agents
        }

        # Add this line to store per-step stats for each agent
        self._agent_step_stats: dict[str, dict[str, int]] = {}

    # --- PettingZoo API Methods ---

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """Returns the observation space for a single agent."""
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Returns the action space for a single agent."""
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None) -> dict[str, np.ndarray]:
        """Resets the environment to an initial state."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.agents = self.possible_agents[:] # Active agents list
        self._agents = {} # Clear agent objects

        # Reset map to base state (walls, empty spaces, initial waste/river)
        self.world_map = deepcopy(self.base_map)
        # Custom reset: place initial waste and river tiles correctly
        self._reset_map_features()

        # Spawn agents at unique starting locations
        available_spawn_points = deepcopy(self.spawn_points)
        if len(available_spawn_points) < self.num_agents:
            raise ValueError("Not enough spawn points for the number of agents.")
        random.shuffle(available_spawn_points)

        for i, agent_id in enumerate(self.agents):
            spawn_pos = np.array(available_spawn_points[i])
            # Random initial orientation
            orientation = random.choice(list(ORIENTATIONS.keys()))
            #self._agents[agent_id].reset(spawn_pos, orientation) # Reset agent state
            self._agents[agent_id] = CleanupAgent(
                agent_id_num=i,
                start_pos=spawn_pos,
                start_orientation=orientation,
                view_len=CLEANUP_VIEW_SIZE,
            )
            # Remove agent start 'P' from world map
            self.world_map[spawn_pos[0], spawn_pos[1]] = EMPTY

        # Reset dynamics and counters
        self._compute_probabilities()
        self.num_cycles = 0
        self.beam_pos = []

        # Reset LLM commands
        if self.use_llm:
            for agent_id in self.possible_agents:
                self.llm_commands[agent_id] = None

        # Get initial observations
        observations = {agent_id: self._get_observation(agent_id) for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents} # Create empty info dict
        #print(f"Initial observations: {observations}")
        #print(f"Initial infos: {infos}")

        # In reset method, after agents are initialized
        self._agent_step_stats = {
            agent_id: {"apples_collected_step": 0, "pollution_cleaned_step": 0}
            for agent_id in self.agents
        }

        if self.render_mode == "human":
            self.render()

        return observations, infos # Return both observations and infos



    def _process_agent_movements(self, actions: Dict[str, int]) -> Tuple[Dict[str, str], Dict[str, np.ndarray], Dict[str, str]]:
        """
        Decodes actions, handles turns immediately, and determines intended positions.
        Handles immobilization overrides.
        """
        agent_action_map = {}
        agent_new_positions = {}
        agent_new_orientations = {}
        active_agents = self.agents[:] # Agents active at the start of this sub-step

        for agent_id in active_agents:
            agent = self._agents.get(agent_id)
            if not agent: continue

            action_code = actions.get(agent_id)
            if action_code is None: continue # Agent might not have provided an action

            # --- Handle Immobilization ---
            if agent.is_immobilized():
                action_code = STAY_ACTION_INDEX # Override action
                agent.decrement_immobilization()

            action_str = ACTION_MEANING.get(action_code)
            agent_action_map[agent_id] = action_str

            if action_str in TURN_ACTIONS:
                new_orientation = ROTATION_MAP.get((agent.get_orientation(), action_str))
                agent.set_orientation(new_orientation)
                agent_new_orientations[agent_id] = new_orientation
            elif action_str in MOVE_ACTIONS:
                move_vec = MOVE_ACTIONS[action_str]
                rotated_move = self._rotate_vector(move_vec, agent.get_orientation())
                intended_pos = agent.get_pos() + rotated_move
                agent_new_positions[agent_id] = intended_pos
            # Special actions (FIRE, CLEAN) are handled later

        return agent_action_map, agent_new_positions, agent_new_orientations


    def _resolve_conflicts_and_update_positions(self, agent_new_positions: Dict[str, np.ndarray]):
        """Resolves movement conflicts and updates agent positions."""
        # This uses your existing _resolve_movement_conflicts logic
        final_positions = self._resolve_movement_conflicts(agent_new_positions)
        for agent_id, final_pos in final_positions.items():
            if agent_id in self._agents:
                self._agents[agent_id].set_pos(final_pos)

    def _handle_consumption_and_special_actions(self, agent_action_map: Dict[str, str]):
        """Handles apple consumption and beam firing in a randomized order."""
        beam_updates = [] # Store tile changes from beams
        
        # --- Shuffle agent order for fairness in special actions ---
        # Use the list of agents active at the start of the *main* step
        active_agents_for_step = self.agents[:] 
        # Create a list of agent IDs present in the action map to shuffle
        agents_with_actions = [agent_id for agent_id in active_agents_for_step if agent_id in agent_action_map]
        random.shuffle(agents_with_actions) # Shuffle the order
        
        # --- Handle Consumption (Apples) - Can happen before or after beams, affects reward calculation timing ---
        # Let's do it first based on the *final* positions after conflict resolution.
        agents_active_after_move = self.agents[:] # Re-check active agents
        for agent_id in agents_active_after_move:
            agent = self._agents.get(agent_id)
            command = self.llm_commands.get(agent_id) if self.use_llm else None
            if not agent: continue
            pos = agent.get_pos()
            tile = self.world_map[pos[0], pos[1]]
            if tile == APPLE and command != "clean up":
                agent.add_reward(APPLE_REWARD)
                self._update_map_tile(pos[0], pos[1], APPLE_SPAWN)
                # Update step stats for apple collection
                if agent_id in self._agent_step_stats:
                    self._agent_step_stats[agent_id]["apples_collected_step"] += 1

        # Handle Special Actions (FIRE, CLEAN) in shuffled order 
        for agent_id in agents_with_actions: # Iterate using the shuffled list
            agent = self._agents.get(agent_id)
            if not agent: continue # Agent might have terminated/truncated already

            action_str = agent_action_map.get(agent_id)

            if action_str == "FIRE":
                agent.immobilize(IMMOBILIZE_DURATION_FIRE)
                agent.add_reward(-PENALTY_FIRE) # Cost for firing
                # _fire_beam now returns updates and cleaned_waste_count
                fire_tile_updates, _ = self._fire_beam( # We don't expect waste cleaned by FIRE
                    agent.get_pos(), agent.get_orientation(), FIRE_BEAM_LENGTH,
                    PENALTY_BEAM, [], [], FIRE_BLOCKING_CELLS, FIRE_BEAM_WIDTH
                )
                # beam_updates.extend(fire_tile_updates) # Typically empty for FIRE

            elif action_str == "CLEAN":
                agent.add_reward(CLEAN_REWARD) # Cost/reward for cleaning
                # _fire_beam now returns updates and cleaned_waste_count
                clean_tile_updates, num_waste_cleaned = self._fire_beam(
                    agent.get_pos(), agent.get_orientation(), CLEAN_BEAM_LENGTH,
                    CLEAN_BEAM, CLEANABLE_TILES, CLEANED_TILE_RESULT, CLEAN_BLOCKING_CELLS, CLEAN_BEAM_WIDTH
                )
                beam_updates.extend(clean_tile_updates) # Store tile changes
                # Update step stats for pollution cleaned
                if agent_id in self._agent_step_stats:
                    self._agent_step_stats[agent_id]["pollution_cleaned_step"] += num_waste_cleaned

        # Apply beam updates to the map *after* all beams resolved for the step
        for r, c, char in beam_updates:
            self._update_map_tile(r, c, char)


    def _update_environment_state(self):
        """Handles waste/apple spawning and updates environment probabilities."""
        self._compute_probabilities() # Update probs based on current waste
        spawn_updates = self._spawn_apples_and_waste()
        for r, c, char in spawn_updates:
            self._update_map_tile(r, c, char)

    def _get_llm_commands(self, agents_active_at_step_start: List[str]):
        """Processes game info and gets commands from LLM modules if enabled."""
        if not self.use_llm or (self.num_cycles % self.llm_f_step != 0):
            return # Only run LLM logic at specified frequency

        game_info = ""
        if self.current_waste_density >= THRESHOLD_DEPLETION:
            game_info = "River severely polluted, apples cannot grow."
        elif self.current_waste_density <= THRESHOLD_RESTORATION:
            game_info = "River is clean, apples can grow well."
        else:
            game_info = f"River pollution level moderate (density: {self.current_waste_density:.2f})."

        # Process Info and Get Commands for each agent active at step start
        agents_cumulative_rewards = {}
        for agent_id in agents_active_at_step_start:
            agent = self._agents.get(agent_id)
            if agent:
                agents_cumulative_rewards[agent_id] = agent.get_cumulative_reward()

        for agent_id in agents_active_at_step_start:
            if agent_id in self.llm_modules:
                command = self.llm_modules[agent_id].process_game_info(
                    game_info, self.llm_type, agents_cumulative_rewards
                )
                self.llm_commands[agent_id] = command
            else:
                self.llm_commands[agent_id] = None # Default to None if no LLM module



    def step(self, actions: dict[str, int]) -> tuple[dict, dict, dict, dict, dict]:
        """Advances the environment by one step based on agent actions."""
        self.num_cycles += 1
        self.beam_pos = [] # Clear beams from previous step

        # 记录步骤开始时的活动智能体, 目的是确保后续的奖励、终止/截断状态和观测都基于这个列表计算
        agents_at_step_start = self.agents[:]
        # print(f"Active agents: {agents_at_step_start}")

        # At the beginning of the step method, after self.num_cycles += 1
        for agent_id in agents_at_step_start: # agents_at_step_start is crucial here
            # Ensure entry exists, especially if agents can be added/removed dynamically (though not typical for reset)
            if agent_id not in self._agent_step_stats:
                self._agent_step_stats[agent_id] = {"apples_collected_step": 0, "pollution_cleaned_step": 0}
            self._agent_step_stats[agent_id]["apples_collected_step"] = 0
            self._agent_step_stats[agent_id]["pollution_cleaned_step"] = 0

        # 1. Process Actions (Movement intents, Turns, Immobilization)
        agent_action_map, agent_new_positions, _ = self._process_agent_movements(actions)

        # 2. Resolve Movement Conflicts and Update Positions
        self._resolve_conflicts_and_update_positions(agent_new_positions)

        # 3. Handle Consumption and Special Actions (Apples, Firing/Cleaning Beams)
        #    Uses the shuffled order internally for beam actions.
        self._handle_consumption_and_special_actions(agent_action_map)

        # 4. Update Environment State (Waste/Apple Spawning)
        self._update_environment_state()

        # 5. Calculate Rewards and Termination/Truncation
        # 基于 agents_at_step_start 初始化状态字典 
        rewards = {agent_id: 0.0 for agent_id in agents_at_step_start}
        terminations = {agent_id: False for agent_id in agents_at_step_start}
        truncations = {agent_id: False for agent_id in agents_at_step_start}
        infos = {agent_id: {} for agent_id in agents_at_step_start}
        
        

        # Apply collective/IAR rewards if enabled (Simplified - full IAR needs careful implementation)
        # collective reeward, inequity penalty
        if self.use_collective_reward:
             total_reward = sum(rewards.values())
             rewards = {agent_id: total_reward for agent_id in agents_at_step_start}
        elif self.inequity_averse_reward and self.num_agents > 1:
             current_rewards = rewards.copy()
             for agent_id_i in agents_at_step_start:
                 inequity_penalty = 0
                 for agent_id_j in agents_at_step_start:
                     if agent_id_i == agent_id_j: continue
                     diff = current_rewards[agent_id_j] - current_rewards[agent_id_i]
                     if diff > 0: # Disadvantageous inequity
                         inequity_penalty += self.alpha * diff
                     elif diff < 0: # Advantageous inequity
                         inequity_penalty += self.beta * abs(diff) # beta is typically negative, so abs() or adjust sign
                 rewards[agent_id_i] -= inequity_penalty / (self.num_agents - 1)


        # Check truncation (max cycles)
        is_truncated = self.num_cycles >= self.max_cycles
        if is_truncated:
            truncations = {agent_id: True for agent_id in agents_at_step_start}
            # Don't clear self.agents here yet

        # 6. Get LLM commands (if applicable for this step)
        self._get_llm_commands(agents_at_step_start)


        # 7. Generate observations for all agents active at the start of the step
        #    They need an observation even if they terminate/truncate *in this step*.
        observations = {}
        for agent_id in agents_at_step_start:
            try:
                observations[agent_id] = self._get_observation(agent_id)
            except KeyError:
                # Agent object might not exist if something went wrong, handle gracefully
                print(f"Warning: Agent {agent_id} not found when getting observation, returning default.")
                # Return a default observation (e.g., zeros) matching the space
                obs_space = self.observation_space(agent_id)
                observations[agent_id] = np.zeros(obs_space.shape, dtype=obs_space.dtype)

             # Populate infos for each agent
            # Ensure agent_id from agents_at_step_start has an entry in infos
            current_agent_stats = self._agent_step_stats.get(agent_id, {"apples_collected_step": 0, "pollution_cleaned_step": 0})
            infos[agent_id]["apples_collected_step"] = current_agent_stats["apples_collected_step"]
            infos[agent_id]["pollution_cleaned_step"] = current_agent_stats["pollution_cleaned_step"]
            infos[agent_id]["llm_command"] = self.llm_commands.get(agent_id)



         # Get rewards accumulated by agents
        for agent_id in agents_at_step_start:
            # if agent_id in self._agents: # Ensure agent is still active
            agent = self._agents.get(agent_id)
            if agent:  # Check if agent exists before consuming reward
                rewards[agent_id] += agent.consume_reward() # Add rewards from hits/consumption
                agent.add_cumulative_reward(rewards[agent_id]) # Update cumulative apple/hit reward
                
            else:
                print(f"Warning: Agent {agent_id} not found in self._agents.")


        # Update self.agents list based on term/trunc flags ---
        # Determine the agents who will be active in the *next* step
        next_agents = []
        for agent_id in agents_at_step_start:
            if not terminations[agent_id] and not truncations[agent_id]:
                next_agents.append(agent_id)
        self.agents = next_agents


        if self.render_mode == "human":
            self.render()

        # PettingZoo expects dicts for all return values, keyed by agent ID
        # Ensure all returned dicts have keys from agents_at_step_start
        # (Observations dict is already handled. Rewards/Terms/Truncs/Infos were initialized based on it)
        return observations, rewards, terminations, truncations, infos


    def render(self) -> np.ndarray | None:
        """Renders the environment."""
        rgb_map = self._map_to_colors()

        if self.render_mode == "human":
            if self.fig is None:
                plt.ion() # Interactive mode on
                self.fig, self.ax = plt.subplots(1, 1)
                self.render_im = self.ax.imshow(rgb_map, interpolation='nearest')
                plt.title("Cleanup Social Dilemma")
            else:
                self.render_im.set_data(rgb_map)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events() # Update display
            return None
        elif self.render_mode == "rgb_array":
            return rgb_map
        return None


    def close(self):
        """Closes the rendering window."""
        if self.fig is not None:
            plt.ioff() # Interactive mode off
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.render_im = None


    # --- Helper Methods ---

    def _reset_map_features(self):
        """Places initial waste, river, and stream tiles."""
        for r, c in self.waste_init_points:
             self.world_map[r, c] = WASTE
        for r, c in self.river_points:
             self.world_map[r, c] = RIVER
        for r, c in self.stream_points:
             self.world_map[r, c] = STREAM # Place stream tiles


    def _ascii_to_numpy(self, ascii_map):
        """Converts the ASCII map list to a numpy byte array."""
        return np.array([[c.encode('ascii') for c in row] for row in ascii_map])

    def _find_points(self, char_to_find):
        """Finds all coordinates (row, col) of a given character in the base map."""
        return np.argwhere(self.base_map == char_to_find).tolist()

    def _get_map_with_agents(self):
        """Creates a temporary map view with agents and beams placed."""
        map_view = np.copy(self.world_map)
        # Place agents
        for agent in self._agents.values():
            pos = agent.get_pos()
            # Check bounds just in case
            if 0 <= pos[0] < self.map_height and 0 <= pos[1] < self.map_width:
                 # Check if beam is already there, beams have render priority
                 if map_view[pos[0], pos[1]] not in [PENALTY_BEAM, CLEAN_BEAM]:
                     map_view[pos[0], pos[1]] = agent.get_agent_char()

        # Place beams (render priority over agents)
        for r, c, beam_char in self.beam_pos:
             if 0 <= r < self.map_height and 0 <= c < self.map_width:
                 map_view[r, c] = beam_char
        return map_view

    def _map_to_colors(self) -> np.ndarray:
        """Converts the current world map state (with agents) to an RGB array."""
        map_with_agents = self._get_map_with_agents()
        rgb_map = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)

        for r in range(self.map_height):
            for c in range(self.map_width):
                char = map_with_agents[r, c]
                rgb_map[r, c, :] = DEFAULT_COLOURS.get(char, DEFAULT_COLOURS[b' ']) # Default to black if char not found
        return rgb_map


    def _map_to_colors_mask_apple(self) -> np.ndarray:
        """Generates RGB map masking Apples ('A') as Grass ('B')."""
        map_with_agents = self._get_map_with_agents()
        rgb_map = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        grass_color = DEFAULT_COLOURS[APPLE_SPAWN] # Color of 'B'

        for r in range(self.map_height):
            for c in range(self.map_width):
                char = map_with_agents[r, c]
                if char == APPLE:
                    rgb_map[r, c, :] = grass_color
                else:
                    rgb_map[r, c, :] = DEFAULT_COLOURS.get(char, DEFAULT_COLOURS[b' '])
        return rgb_map

    def _map_to_colors_mask_waste(self) -> np.ndarray:
        """Generates RGB map masking Waste ('H') as River ('R')."""
        map_with_agents = self._get_map_with_agents()
        rgb_map = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        river_color = DEFAULT_COLOURS[RIVER] # Color of 'R'

        for r in range(self.map_height):
            for c in range(self.map_width):
                char = map_with_agents[r, c]
                if char == WASTE:
                    rgb_map[r, c, :] = river_color
                else:
                    rgb_map[r, c, :] = DEFAULT_COLOURS.get(char, DEFAULT_COLOURS[b' '])
        return rgb_map


    def _get_agent_view(self, agent: CleanupAgent, full_rgb_map: np.ndarray) -> np.ndarray:
        """Extracts the agent's egocentric view from the full RGB map."""
        pos = agent.get_pos()
        view_size = CLEANUP_VIEW_SIZE
        padded_map = np.pad(full_rgb_map, ((VIEW_PADDING, VIEW_PADDING), (VIEW_PADDING, VIEW_PADDING), (0, 0)), mode='constant', constant_values=0)

        # Agent's position in the padded map
        padded_r, padded_c = pos[0] + VIEW_PADDING, pos[1] + VIEW_PADDING

        # Extract the square view centered on the agent
        view = padded_map[
            padded_r - view_size : padded_r + view_size + 1,
            padded_c - view_size : padded_c + view_size + 1,
            :
        ]

        # Rotate the view based on agent's orientation
        orientation = agent.get_orientation()
        if orientation == "UP":
            rotated_view = view
        elif orientation == "RIGHT": # 90 deg clockwise
            rotated_view = np.rot90(view, k=3)
        elif orientation == "DOWN": # 180 deg clockwise
            rotated_view = np.rot90(view, k=2)
        elif orientation == "LEFT": # 270 deg clockwise (or 90 counter-clockwise)
            rotated_view = np.rot90(view, k=1)
        else:
            rotated_view = view # Should not happen

        return rotated_view


    def _get_observation(self, agent_id: str) -> np.ndarray:
        """
        Generates the observation for a specific agent, potentially masked by LLM command.
        """
        agent = self._agents[agent_id]
        command = self.llm_commands.get(agent_id) if self.use_llm else None

        # Determine which map rendering function to use
        if self.use_llm and command == "clean up":
            # Mask apples (show as grass 'B')
            map_rgb_for_view = self._map_to_colors_mask_apple()
        elif self.use_llm and command == "collect apples":
            # Mask waste (show as river 'R')
            map_rgb_for_view = self._map_to_colors_mask_waste()
        else:
            # Default: no masking or LLM not used
            map_rgb_for_view = self._map_to_colors()

        # Get the egocentric view from the chosen map
        agent_view_rgb = self._get_agent_view(agent, map_rgb_for_view)
        return agent_view_rgb


    def _rotate_vector(self, vector: np.ndarray, orientation: str) -> np.ndarray:
        """Rotates a move vector based on the agent's orientation."""
        # Check if the input vector is the 'STAY' action first
        if np.array_equal(vector, MOVE_ACTIONS["STAY"]):
            return np.array([0, 0])

        # If orientation is UP, no rotation is needed
        if orientation == "UP":
            return vector

        # For other orientations (RIGHT, DOWN, LEFT), calculate rotation
        orientation_vec = ORIENTATION_VECTORS[orientation]

        # Determine the relative move type
        if np.array_equal(vector, MOVE_ACTIONS["MOVE_UP"]): # Relative Forward
            return orientation_vec
        elif np.array_equal(vector, MOVE_ACTIONS["MOVE_DOWN"]): # Relative Backward
            return -orientation_vec
        elif np.array_equal(vector, MOVE_ACTIONS["MOVE_LEFT"]): # Relative Strafe Left
            return np.array([-orientation_vec[1], orientation_vec[0]]) # 
        elif np.array_equal(vector, MOVE_ACTIONS["MOVE_RIGHT"]): # Relative Strafe Right
            return np.array([orientation_vec[1], -orientation_vec[0]])
        else:
            # Should not happen if vector is a valid move action from MOVE_ACTIONS
            print(f"Warning: Unexpected move vector {vector} in _rotate_vector")
            return np.array([0, 0]) # Return STAY as a safe default


    def _is_position_valid(self, pos: np.ndarray) -> bool:
        """Checks if a position is within map bounds."""
        return 0 <= pos[0] < self.map_height and 0 <= pos[1] < self.map_width

    def _is_tile_walkable(self, pos: np.ndarray) -> bool:
        """Checks if the tile at the given position is walkable."""
        if not self._is_position_valid(pos):
            return False
        # Check against non-walkable tile types
        tile = self.world_map[pos[0], pos[1]]
        if tile in NON_WALKABLE:
            return False
        # Check if another agent is already there (this is handled in conflict resolution)
        return True

    def _resolve_movement_conflicts(self, intended_positions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Resolves conflicts where multiple agents intend to move to the same cell."""
        final_positions = {}
        agent_current_positions = {agent_id: agent.get_pos() for agent_id, agent in self._agents.items()}
        
        # Agents who didn't request a move stay put
        for agent_id, current_pos in agent_current_positions.items():
            if agent_id not in intended_positions:
                final_positions[agent_id] = current_pos

        # Identify conflicting moves
        target_cells = {} # { (r, c): [agent_id1, agent_id2, ...], ... }
        for agent_id, target_pos_tuple in intended_positions.items():
             target_pos = tuple(target_pos_tuple)
             # Check basic validity (bounds and non-walkable static tiles)
             if not self._is_position_valid(target_pos_tuple) or \
                self.world_map[target_pos[0], target_pos[1]] in NON_WALKABLE:
                 # Invalid move, agent stays put
                 final_positions[agent_id] = agent_current_positions[agent_id]
                 continue # Skip this agent

             if target_pos not in target_cells:
                 target_cells[target_pos] = []
             target_cells[target_pos].append(agent_id)

        processed_agents = set(final_positions.keys()) # Track agents whose moves are decided

        # Resolve conflicts for cells targeted by multiple agents
        for target_pos, agents_targeting in target_cells.items():
            if len(agents_targeting) > 1:
                # Conflict! Randomly choose one agent to succeed
                winner = random.choice(agents_targeting)
                final_positions[winner] = np.array(target_pos)
                processed_agents.add(winner)
                # Losers stay in their original positions
                for loser in agents_targeting:
                    if loser != winner:
                        final_positions[loser] = agent_current_positions[loser]
                        processed_agents.add(loser)

        # Process non-conflicting moves (agents targeting unique, valid cells)
        for target_pos, agents_targeting in target_cells.items():
            if len(agents_targeting) == 1:
                agent_id = agents_targeting[0]
                if agent_id not in processed_agents: # Ensure not already processed as loser
                     # Check for swap conflicts (A->B, B->A) - Simplified: just allow if cell is targeted by only one
                     # Check if target cell is occupied by another agent *that is also moving*
                     is_occupied_by_moving_agent = False
                     occupying_agent_id = None
                     for other_agent_id, other_current_pos in agent_current_positions.items():
                         if other_agent_id != agent_id and tuple(other_current_pos) == target_pos:
                             # Check if the occupying agent has an intended move
                             if other_agent_id in intended_positions:
                                  is_occupied_by_moving_agent = True
                             occupying_agent_id = other_agent_id
                             break

                     if not is_occupied_by_moving_agent:
                          final_positions[agent_id] = np.array(target_pos)
                     else:
                          # Occupied by an agent that is also trying to move. Prevent move.
                          # More complex resolution (like the original's loop) could be added here.
                          # For simplicity now, the mover stays put if target is occupied by *any* other agent.
                          is_occupied = False
                          for other_id, other_pos in agent_current_positions.items():
                              if other_id != agent_id and tuple(other_pos) == target_pos:
                                   is_occupied = True
                                   break
                          if not is_occupied:
                               final_positions[agent_id] = np.array(target_pos)
                          else:
                               final_positions[agent_id] = agent_current_positions[agent_id]

                     processed_agents.add(agent_id)


        # Ensure all agents have a final position assigned
        for agent_id in self.agents:
            if agent_id not in final_positions:
                 # This agent must have intended an invalid move or stayed put implicitly
                 final_positions[agent_id] = agent_current_positions[agent_id]

        return final_positions


    def _update_map_tile(self, row: int, col: int, char: bytes):
        """Updates a single tile on the world map."""
        if 0 <= row < self.map_height and 0 <= col < self.map_width:
            self.world_map[row, col] = char


    def _fire_beam(self, start_pos: np.ndarray, orientation: str, length: int,
                   beam_char: bytes, cell_types: list[bytes], update_char: list[bytes],
                   blocking_cells: list[bytes], beam_width: int) -> tuple[list[tuple[int, int, bytes]], int]:
        """
        Fires a beam, potentially hitting agents or changing tiles.
        'length' is the range of the beam (FIRE_BEAM_LENGTH or CLEAN_BEAM_LENGTH).
        'affect_num' logic:
            - PENALTY_BEAM: affects 1 agent per line.
            - CLEAN_BEAM: affects CLEAN_BEAM_LENGTH_VALID waste tiles per line.
        Returns:
            - list of (row, col, new_char) for map updates
            - count of waste tiles cleaned by this beam action (across all lines)
        """
        firing_direction = ORIENTATION_VECTORS[orientation]
        updates = []
        cleaned_waste_count = 0 # Total waste tiles cleaned by this entire beam fire action

        # Determine initial relative offsets for beam lines based on width
        if beam_width == 3:
            if orientation == "UP" or orientation == "DOWN":
                # Beam lines originate from agent's column, and columns +/- 1
                init_relative_offsets = [np.array([0, 0]), np.array([0, 1]), np.array([0, -1])]
            elif orientation == "RIGHT" or orientation == "LEFT":
                # Beam lines originate from agent's row, and rows +/- 1
                init_relative_offsets = [np.array([0, 0]), np.array([1, 0]), np.array([-1, 0])]
            else: # Should not happen
                init_relative_offsets = [np.array([0,0])]
        elif beam_width == 1:
            init_relative_offsets = [np.array([0,0])] # Single beam line from agent's center
        else: # Default or unsupported width
            init_relative_offsets = [np.array([0,0])]

        agent_positions = {tuple(agent.get_pos()): agent_id for agent_id, agent in self._agents.items()}

        # Determine max number of targets this beam type can affect per line
        max_affect_on_line = 0
        if beam_char == PENALTY_BEAM:
            max_affect_on_line = 1  # Hits 1 agent
        elif beam_char == CLEAN_BEAM:
            max_affect_on_line = CLEAN_BEAM_LENGTH_VALID # Cleans up to N waste tiles

        for offset in init_relative_offsets:
            # current_pos_beam_origin is the starting point of this specific beam line segment, relative to agent's start_pos
            current_pos_beam_origin = start_pos + offset
            # temp_beam_path_pos will track the tip of the beam as it travels
            temp_beam_path_pos = np.copy(current_pos_beam_origin) # Use copy to avoid modifying origin
            
            num_affected_this_line = 0

            for _ in range(length): # Iterate for the range of the beam
                # Move beam one step forward from its previous position
                temp_beam_path_pos = temp_beam_path_pos + firing_direction

                if not self._is_position_valid(temp_beam_path_pos):
                    break # Beam line goes out of map bounds

                row, col = int(temp_beam_path_pos[0]), int(temp_beam_path_pos[1])
                tile_char_on_map = self.world_map[row, col]

                # Add to beam_pos for rendering.
                # To avoid duplicate rendering if multiple lines hit same cell, could use a set then convert.
                # For now, simple append. Last beam char wins if overlap.
                self.beam_pos.append((row, col, beam_char))

                # 1. Check for agent hit
                current_pos_tuple = tuple(temp_beam_path_pos)
                if current_pos_tuple in agent_positions:
                    hit_agent_id = agent_positions[current_pos_tuple]
                    #firing_agent = None

                    if beam_char == PENALTY_BEAM:
                        if num_affected_this_line < max_affect_on_line:
                            self._agents[hit_agent_id].add_reward(-PENALTY_HIT)
                            self._agents[hit_agent_id].immobilize(IMMOBILIZE_DURATION_HIT)
                            num_affected_this_line += 1
                        break # PENALTY_BEAM stops at the first agent it hits on this line
                    elif beam_char == CLEAN_BEAM:
                        # CLEAN_BEAM don't stop when it hits an agent on this line
                        continue 
                
                # 2. Check if the tile itself blocks the beam (e.g., a wall)
                if tile_char_on_map in blocking_cells:
                    break # Beam line is blocked

                # 3. Check if the tile can be affected by this beam type
                if tile_char_on_map in cell_types: # cell_types for CLEAN are [WASTE]
                    if beam_char == CLEAN_BEAM:
                        # Only process if we haven't reached the max affectable targets for this line
                        if num_affected_this_line < max_affect_on_line:
                            try:
                                type_index = cell_types.index(tile_char_on_map)
                                new_char_to_place = update_char[type_index] # e.g., WASTE -> RIVER   # 实际上就一对替换的，从waste替换为river，不用index(list)

                                # Append update, will be applied later
                                updates.append((row, col, new_char_to_place))
                                
                                # If it was a waste tile that got cleaned
                                if tile_char_on_map == WASTE:
                                    cleaned_waste_count += 1
                                
                                num_affected_this_line += 1
                                # Beam continues along its path even after cleaning,
                                # unless max_affect_on_line for cleaning is now reached,
                                # or it hits max range, wall, or agent.
                            except (ValueError, IndexError):
                                # Should not happen if cell_types and update_char are well-defined
                                pass
                        # If num_affected_this_line >= max_affect_on_line, this beam line can't clean more.
                        # It still continues its path until range end, wall, or agent.
                    # Add logic for other beam types and cell interactions if needed
            # End of this beam line's travel
        # End of all beam lines for this fire action

        # Remove duplicate tile updates by converting to a set of tuples and back to list
        # This ensures a tile is not queued for update multiple times if hit by overlapping beam lines
        if updates:
            updates = list(set(updates))

        return updates, cleaned_waste_count

    def _compute_probabilities(self):
        """Computes the apple and waste spawn probabilities based on waste density."""
        current_waste = np.count_nonzero(self.world_map == WASTE)
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = current_waste / self.potential_waste_area
        self.current_waste_density = waste_density

        if self.current_waste_density >= THRESHOLD_DEPLETION:  # 0.45
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = WASTE_SPAWN_PROBABILITY
        
        if self.current_waste_density >= THRESHOLD_DEPLETION:  # 0.45
            self.current_apple_spawn_prob = 0
        else: # waste_density < THRESHOLD_DEPLETION
    
            if self.current_waste_density <= THRESHOLD_RESTORATION:
                self.current_apple_spawn_prob = APPLE_RESPAWN_PROBABILITY
            else:
                # Linear interpolation
                prob = (1.0 - (waste_density - THRESHOLD_RESTORATION) / (THRESHOLD_DEPLETION - THRESHOLD_RESTORATION)) * APPLE_RESPAWN_PROBABILITY
                self.current_apple_spawn_prob = max(0, prob) # Ensure non-negative

    def _spawn_apples_and_waste(self) -> list[tuple[int, int, bytes]]:
        """Attempts to spawn apples and waste based on current probabilities."""
        spawn_updates = []
        agent_pos_list = []
        for agent_id in self.agents:
            agent = self._agents.get(agent_id)
            if agent:
                agent_pos_list.append(tuple(agent.get_pos()))

        # Try to spawn apples
        for r, c in self.apple_spawn_points:
            if (self.world_map[r, c] == APPLE_SPAWN or self.world_map[r, c] == EMPTY) and \
            tuple([r, c]) not in agent_pos_list:
                if random.random() < self.current_apple_spawn_prob:
                    spawn_updates.append((r, c, APPLE))

        # Try to spawn waste 
        for r, c in self.waste_spawn_points: 
            if (self.world_map[r, c] == EMPTY or self.world_map[r, c] == RIVER) and \
            tuple([r, c]) not in agent_pos_list:
                    if random.random() < self.current_waste_spawn_prob:
                        spawn_updates.append((r, c, WASTE))

        return spawn_updates

# --- PettingZoo AEC Wrapper --- (Optional but common)
# def env(**kwargs):
#     """Creates a PettingZoo AEC environment."""
#     parallel_env = CleanupEnv(**kwargs)
#     aec_env = parallel_to_aec(parallel_env)
#     #aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env) # Good for debugging
#     #aec_env = wrappers.OrderEnforcingWrapper(aec_env)   # Ensures order
#     return aec_env
# --- 新的定义 ---
def env(render_mode=None, **kwargs):
    """
    Creates a PettingZoo AEC environment.

    Args:
        render_mode: The rendering mode ('human', 'rgb_array', or None).
        **kwargs: Other arguments to pass to the CleanupEnv constructor
                  (e.g., num_agents, max_cycles).
    """
    # 将 render_mode 和其他 kwargs 传递给 CleanupEnv
    parallel_env = CleanupEnv(render_mode=render_mode, **kwargs)
    aec_env = parallel_to_aec(parallel_env)
    #aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env) # Good for debugging
    #aec_env = wrappers.OrderEnforcingWrapper(aec_env)   # Ensures order
    return aec_env
# --- 修改结束 ---


# # # --- Example Usage ---
# if __name__ == "__main__":
#     num_agents = 2
#     render_mode = "human" # "rgb_array" or None
#     env = CleanupEnv(num_agents=num_agents, render_mode=render_mode, max_cycles=500)
#     observations = env.reset()

#     for _ in range(env.max_cycles):
#         actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
#         observations, rewards, terminations, truncations, infos = env.step(actions)
#         #print(f"Step: {env.num_cycles}, Agents: {env.agents}, Rewards: {rewards}")

#         if not env.agents: # Episode ended (all terminated or truncated)
#             print("Episode Finished!")
#             observations = env.reset()

#     env.close()
#     print("Cleanup Env Example Done.")