"""
Baseline RL agent: tabular Q-learning with no motivational layer.
Reward-driven task-seeker with compact state features.
"""

import numpy as np
import random


class BaselineAgent:
    """
    Tabular Q-learning agent.

    State is encoded as compact relative task and local hazard features.
    No internal motivational state - purely reward-driven.

    The environment randomizes lava every episode, so this agent uses a compact
    tabular representation that can generalize across positions:
    relative mineral direction/distance, immediate lava mask, boundary mask,
    and a lava-distance bin. Action selection remains epsilon-greedy over
    learned Q-values; there is no shortest-path planner or motivational layer.
    """

    ACTIONS = 4  
    ACTION_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(
        self,
        grid_size:    int   = 10,
        alpha:        float = 0.3,   
        gamma:        float = 0.95, 
        epsilon:      float = 1.0,   
        epsilon_min:  float = 0.05,
        epsilon_decay:float = 0.97,
        seed:         int   = 42,
        max_distance_bin: int = 4,
        safe_exploration_probability: float = 0.7,
        progress_shaping_weight: float = 8.0,
        lava_shaping_penalty: float = 80.0,
        boundary_shaping_penalty: float = 5.0,
        danger_shaping_penalty: float = 2.0,
        mask_lava_on_exploit: bool = False,
    ):
        self.grid_size     = grid_size
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = random.Random(seed)
        self.max_distance_bin = max_distance_bin
        self.safe_exploration_probability = safe_exploration_probability
        self.progress_shaping_weight = progress_shaping_weight
        self.lava_shaping_penalty = lava_shaping_penalty
        self.boundary_shaping_penalty = boundary_shaping_penalty
        self.danger_shaping_penalty = danger_shaping_penalty
        self.mask_lava_on_exploit = mask_lava_on_exploit

        # Sparse Q-table: encoded_state -> [action values]
        self.q_table: dict[tuple, np.ndarray] = {}
        self.visit_counts: dict[tuple, np.ndarray] = {}

    
    def _encode(self, state: dict) -> tuple:
        """ 
        Encode the environment state into a compact tabular representation.

        The encoding captures the relative mineral direction and distance,
        nearby lava and boundary information, and proximity to lava,
        allowing the Q-table to generalize across different grid layouts.
       """
        ar, ac = state["pos"]
        mr, mc = state["mineral_pos"]
        lava_cells = set(state.get("lava_cells", ()))

        dy = mr - ar
        dx = mc - ac
        local_lava_mask = 0
        boundary_mask = 0

        for action in range(self.ACTIONS):
            next_pos, hit_boundary = self._next_position((ar, ac), action)
            if hit_boundary:
                boundary_mask |= 1 << action
            if next_pos in lava_cells:
                local_lava_mask |= 1 << action

        lava_distance = int(state.get("lava_distance", self.grid_size * 2))
        lava_distance_bin = min(lava_distance, self.max_distance_bin + 1)

        return (
            self._sign(dy),
            self._sign(dx),
            min(abs(dy), self.max_distance_bin),
            min(abs(dx), self.max_distance_bin),
            local_lava_mask,
            boundary_mask,
            lava_distance_bin,
            int(state.get("in_lava", False)),
        )

    def select_action(self, state: dict) -> int:
        """
       Select an action using an epsilon-greedy policy.

       During exploration, a safety-aware random action may be selected.
       During exploitation, the action with the highest learned Q-value
       is chosen.
       """
        if self.rng.random() < self.epsilon:
            return self._select_exploratory_action(state)

        key = self._encode(state)
        q_values = self._q_values(key)
        actions = self._valid_actions(
            state,
            avoid_lava=self.mask_lava_on_exploit,
        )
        return self._argmax_with_random_tie(q_values, actions)

    def update(self, state: dict, action: int, reward: float,
               next_state: dict, done: bool):
        """
        Update the Q-table using the Q-learning temporal-difference rule.

       The environment reward is first shaped to encourage progress toward
       minerals while discouraging dangerous or inefficient behaviour.
       """
        s = self._encode(state)
        ns = self._encode(next_state)
        q_values = self._q_values(s)
        next_q_values = self._q_values(ns)
        shaped_reward = self._shape_reward(state, reward, next_state)

        if done:
            next_value = 0.0
        else:
            next_actions = self._valid_actions(next_state)
            next_value = max(next_q_values[next_action] for next_action in next_actions)

        td_target = shaped_reward + self.gamma * next_value
        td_error = td_target - q_values[action]
        q_values[action] += self.alpha * td_error
        self.visit_counts[s][action] += 1.0

    def decay_epsilon(self):
        """Decay the exploration rate while respecting the minimum value."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_episode(self):
        """
        Reset any episode-specific agent state.
        Reserved for future episode-local state. 
        """
        pass

    def _q_values(self, key: tuple) -> np.ndarray:
        """
        Retrieve or initialize the Q-values associated with an encoded state.
        """
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.ACTIONS)
            self.visit_counts[key] = np.zeros(self.ACTIONS)
        return self.q_table[key]

    def _select_exploratory_action(self, state: dict) -> int:
        """
        Sample a random exploratory action, optionally avoiding nearby lava.
        """

        avoid_lava = self.rng.random() < self.safe_exploration_probability
        return self.rng.choice(self._valid_actions(state, avoid_lava=avoid_lava))

    def _shape_reward(self, state: dict, reward: float, next_state: dict) -> float:
        """
        Apply reward shaping to improve learning.

        Rewards progress toward minerals while penalizing lava exposure,
        boundary collisions, and movement through dangerous regions.
        """
        shaped_reward = reward
        old_mineral_pos = state["mineral_pos"]
        collected_mineral = next_state["pos"] == old_mineral_pos

        if not collected_mineral:
            before_dist = self._manhattan_distance(state["pos"], old_mineral_pos)
            after_dist = self._manhattan_distance(next_state["pos"], old_mineral_pos)
            shaped_reward += self.progress_shaping_weight * (before_dist - after_dist)

        if next_state.get("in_lava", False):
            shaped_reward -= self.lava_shaping_penalty

        if next_state["pos"] == state["pos"] and not collected_mineral:
            shaped_reward -= self.boundary_shaping_penalty

        lava_distance = int(next_state.get("lava_distance", self.max_distance_bin + 1))
        danger_depth = max(0, 3 - lava_distance)
        shaped_reward -= self.danger_shaping_penalty * danger_depth

        return shaped_reward

    def _valid_actions(self, state: dict, avoid_lava: bool = False) -> list[int]:
        """
        Return all valid movement actions.

        Optionally filters actions that would immediately move into lava.
        """
        pos = state["pos"]
        lava_cells = set(state.get("lava_cells", ()))
        actions = set()
        for action in range(self.ACTIONS):
            next_pos, hit_boundary = self._next_position(pos, action)
            if hit_boundary:
                continue
            if avoid_lava and next_pos in lava_cells:
                continue
            actions.add(action)

        if actions:
            return list(actions)
        return [action for action in range(self.ACTIONS)]

    def _next_position(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], bool]:
        """
        Compute the next position resulting from an action.

        Returns the new position together with a flag indicating whether
        the move would cross the environment boundary.
        """
        dr, dc = self.ACTION_DELTAS[action]
        next_pos = (pos[0] + dr, pos[1] + dc)
        if self._in_bounds(next_pos):
            return next_pos, False
        return pos, True

    def _argmax_with_random_tie(self, q_values: np.ndarray, actions: list[int]) -> int:
        """
        Select the highest-valued action, breaking ties uniformly at random.
        """
        max_score = max(q_values[action] for action in actions)
        best_actions = [
            action for action in actions
            if np.isclose(q_values[action], max_score)
        ]
        return self.rng.choice(best_actions)

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        """Return True if the given grid position lies within the environment."""
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    @staticmethod
    def _manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        """Compute the Manhattan distance between two grid positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _sign(value: int) -> int:
        """Return the sign of an integer as -1, 0, or 1."""
        return (value > 0) - (value < 0)
