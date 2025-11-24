"""
Gymnasium environment for DQN pathfinding training.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from apathion.game.map import Map
from apathion.game.tower import Tower, TowerType
from apathion.game.enemy import Enemy, EnemyType
from apathion.pathfinding.dqn_reward_profiles import get_reward_weights, RewardProfile


class PathfindingEnv(gym.Env):
    """
    Gymnasium environment for training DQN agents to navigate tower defense maps.
    
    The agent controls a single enemy unit trying to reach a goal while avoiding
    tower damage. The environment uses a feature vector state representation.
    
    Attributes:
        map_type: Type of map to use ("simple", "branching", "open_arena")
        max_steps: Maximum steps per episode
        render_mode: Optional rendering mode
        state_size: Size of the observation vector
        action_space: Discrete(8) for 8 directional movements
        observation_space: Box for feature vector
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    # Action mapping: 8 directional movements
    ACTION_DIRECTIONS = [
        (0, -1),   # 0: North
        (1, -1),   # 1: Northeast
        (1, 0),    # 2: East
        (1, 1),    # 3: Southeast
        (0, 1),    # 4: South
        (-1, 1),   # 5: Southwest
        (-1, 0),   # 6: West
        (-1, -1),  # 7: Northwest
    ]
    
    def __init__(
        self,
        map_type: str = "simple",
        max_steps: int = 500,
        num_towers: int = 3,
        random_towers: bool = True,
        state_size: int = 42,  # Increased to 42 for danger-aware features
        render_mode: Optional[str] = None,
        reward_profile: str = "survival",  # "speed", "balanced", or "survival"
    ):
        """
        Initialize the pathfinding environment.
        
        Args:
            map_type: Type of map ("simple", "branching", "open_arena")
            max_steps: Maximum steps per episode
            num_towers: Number of towers to place
            random_towers: Whether to randomize tower positions each episode
            state_size: Size of observation vector
            render_mode: Optional rendering mode
            reward_profile: Reward optimization ("speed", "balanced", "survival")
        """
        super().__init__()
        
        self.map_type = map_type
        self.max_steps = max_steps
        self.num_towers = num_towers
        self.random_towers = random_towers
        self.state_size = state_size
        self.render_mode = render_mode
        
        # Set reward weights based on profile
        self._set_reward_weights(reward_profile)
        
        # Create map
        self.game_map = self._create_map(map_type)
        
        # Get spawn and goal from map
        self.spawn_position = self.game_map.spawn_points[0]
        self.goal_position = self.game_map.goal_positions[0]
        
        # Define action and observation space
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32
        )
        
        # Environment state
        self.agent_position: Tuple[int, int] = self.spawn_position
        self.agent_health: float = 100.0
        self.agent_max_health: float = 100.0
        self.towers: List[Tower] = []
        self.steps: int = 0
        self.last_distance_to_goal: float = 0.0
        self.last_action: int = 0
        self.cumulative_damage: float = 0.0
        
        # Initialize towers
        self._place_towers()
    
    def _set_reward_weights(self, profile_name: str) -> None:
        """Set reward weights based on the profile."""
        try:
            profile = RewardProfile(profile_name)
            weights = get_reward_weights(profile)
            
            self.goal_reward = weights["goal_reward"]
            self.health_bonus_multiplier = weights["health_bonus_multiplier"]
            self.death_penalty = weights["death_penalty"]
            self.step_penalty = weights["step_penalty"]
            self.damage_penalty_multiplier = weights["damage_penalty_multiplier"]
            self.progress_reward_multiplier = weights["progress_reward_multiplier"]
            self.spawn_distance_bonus_multiplier = weights["spawn_distance_bonus_multiplier"]
            
        except ValueError:
            raise ValueError(f"Unknown reward profile: {profile_name}. Use 'speed', 'balanced', or 'survival'")
    
    def _create_map(self, map_type: str) -> Map:
        """Create a map based on the map type."""
        if map_type == "simple":
            return Map.create_simple_map()
        elif map_type == "branching":
            return Map.create_branching_map()
        elif map_type == "open_arena":
            return Map.create_open_arena()
        else:
            return Map.create_simple_map()
    
    def _place_towers(self) -> None:
        """Place towers on the map."""
        self.towers = []
        
        if not self.random_towers:
            # Fixed tower positions for consistent training (not recommended)
            if self.map_type == "simple":
                tower_positions = [(10, 10), (15, 10), (20, 10)]
            elif self.map_type == "branching":
                tower_positions = [(8, 8), (12, 12), (16, 8)]
            else:  # open_arena
                tower_positions = [(10, 10), (20, 10), (15, 15)]
        else:
            # Random tower positions - VARY NUMBER AND TYPE for generalization!
            # Randomize number of towers (50% variation from num_towers)
            min_towers = max(1, int(self.num_towers * 0.5))
            max_towers = int(self.num_towers * 1.5)
            actual_num_towers = np.random.randint(min_towers, max_towers + 1)
            
            tower_positions = []
            attempts = 0
            while len(tower_positions) < actual_num_towers and attempts < 200:
                x = np.random.randint(5, self.game_map.width - 5)
                y = np.random.randint(5, self.game_map.height - 5)
                
                # Check if position is valid (walkable and not too close to spawn/goal)
                if self.game_map.is_walkable(x, y):
                    spawn_dist = abs(x - self.spawn_position[0]) + abs(y - self.spawn_position[1])
                    goal_dist = abs(x - self.goal_position[0]) + abs(y - self.goal_position[1])
                    
                    # Must be at least 3 cells from spawn, and not blocking the goal
                    # Allow closer to goal to create challenging scenarios
                    if spawn_dist > 3 and goal_dist > 2:
                        # Check not too close to existing towers (prevent clustering)
                        too_close = False
                        for tx, ty in tower_positions:
                            if abs(x - tx) + abs(y - ty) < 3:
                                too_close = True
                                break
                        
                        if not too_close:
                            tower_positions.append((x, y))
                
                attempts += 1
        
        # Create towers with RANDOM VARIED types for better generalization
        tower_type_pool = [TowerType.BASIC, TowerType.SNIPER, TowerType.RAPID]
        
        for i, pos in enumerate(tower_positions):
            # Randomly select tower type (more variation)
            tower_type = np.random.choice(tower_type_pool)
            tower_id = f"tower_{i}"
            
            if tower_type == TowerType.BASIC:
                tower = Tower.create_basic(tower_id, pos)
            elif tower_type == TowerType.SNIPER:
                tower = Tower.create_sniper(tower_id, pos)
            else:
                tower = Tower.create_rapid(tower_id, pos)
            
            self.towers.append(tower)
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    
    def _get_observation(self) -> np.ndarray:
        """
        Encode the current state into a feature vector.
        
        Returns:
            Feature vector of size state_size
        """
        obs = []
        
        # 1. Normalized relative position to goal (2 features)
        rel_x = (self.goal_position[0] - self.agent_position[0]) / self.game_map.width
        rel_y = (self.goal_position[1] - self.agent_position[1]) / self.game_map.height
        obs.extend([rel_x, rel_y])
        
        # 1b. NEW: Normalized absolute position (2 features)
        # Allows agent to know "where I am" to handle fixed obstacles (walls)
        abs_x = self.agent_position[0] / self.game_map.width
        abs_y = self.agent_position[1] / self.game_map.height
        obs.extend([abs_x, abs_y])
        
        # 2. Normalized distance to goal (1 feature)
        max_distance = np.sqrt(self.game_map.width ** 2 + self.game_map.height ** 2)
        distance = self._calculate_distance(self.agent_position, self.goal_position)
        obs.append(distance / max_distance)
        
        # 3. Current health ratio (1 feature)
        obs.append(self.agent_health / self.agent_max_health)
        
        # 4. Directional walkability (8 features - one per action direction)
        # This is CRITICAL - agent needs to know which directions are blocked!
        for dx, dy in self.ACTION_DIRECTIONS:
            x = self.agent_position[0] + dx
            y = self.agent_position[1] + dy
            is_walkable = 1.0 if self.game_map.is_walkable(x, y) else 0.0
            obs.append(is_walkable)
        
        # 5. Tower threat features (up to 5 nearest towers, 4 features each = 20 features)
        # Features per tower: rel_x, rel_y, is_in_range, damage_rate
        towers_sorted = sorted(
            self.towers,
            key=lambda t: self._calculate_distance(self.agent_position, t.position)
        )
        
        max_towers_encoded = 5
        for i in range(max_towers_encoded):
            if i < len(towers_sorted):
                tower = towers_sorted[i]
                t_rel_x = (tower.position[0] - self.agent_position[0]) / self.game_map.width
                t_rel_y = (tower.position[1] - self.agent_position[1]) / self.game_map.height
                t_distance = self._calculate_distance(self.agent_position, tower.position)
                t_in_range = 1.0 if t_distance <= tower.range else 0.0
                t_damage_rate = (tower.damage * tower.attack_rate) / 50.0  # Normalize
                
                obs.extend([t_rel_x, t_rel_y, t_in_range, t_damage_rate])
            else:
                # Padding for missing towers
                obs.extend([0.0, 0.0, 0.0, 0.0])
        
        # 6. Danger-aware directional features (8 features - danger level per direction)
        # This helps agent understand which directions are safer
        for dx, dy in self.ACTION_DIRECTIONS:
            x, y = self.agent_position[0] + dx, self.agent_position[1] + dy
            danger_level = 0.0
            
            if self.game_map.is_walkable(x, y):
                # Calculate danger at this position
                danger_level = self._get_damage_at_position((x, y))
                # Normalize danger (typical max DPS ~50)
                danger_level = min(1.0, danger_level / 50.0)
            else:
                # Obstacle = maximum danger
                danger_level = 1.0
            
            obs.append(danger_level)
        
        # Total features: 2 + 2 + 1 + 1 + 8 + 20 + 8 = 42
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(
        self,
        reached_goal: bool,
        is_dead: bool,
        damage_taken: float,
        distance_improved: float,
        invalid_move: bool = False
    ) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            reached_goal: Whether agent reached the goal
            is_dead: Whether agent died
            damage_taken: Damage taken this step
            distance_improved: Change in distance to goal (positive = closer)
            invalid_move: Whether agent tried to move into an obstacle
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Terminal rewards
        if reached_goal:
            # Goal reward + health bonus (weights depend on profile)
            health_bonus = (self.agent_health / self.agent_max_health) * self.health_bonus_multiplier
            reward += self.goal_reward + health_bonus
        elif is_dead:
            # Death penalty (much larger for survival profile)
            reward -= self.death_penalty
        
        # Invalid move penalty (constant across profiles)
        if invalid_move:
            reward -= 0.1
        
        # Step penalty (smaller for survival profile - length matters less)
        reward -= self.step_penalty
        
        # Damage penalty (VERY punishing for survival profile)
        if damage_taken > 0:
            # Damage penalty scales with damage taken
            # With survival profile (multiplier=50), this is EXTREMELY punishing
            damage_ratio = damage_taken / self.agent_max_health
            reward -= damage_ratio * self.damage_penalty_multiplier
        
        # Safe progress reward - reward progress that avoids danger
        # This creates incentive to find safe efficient paths, not just short ones
        if not invalid_move and distance_improved > 0:
            # Calculate current danger level
            current_danger = self._get_damage_at_position(self.agent_position)
            
            # Safety factor: 1.0 = perfectly safe, 0.0 = maximum danger
            # Typical max DPS is ~50 for multiple towers
            safety_factor = max(0.0, 1.0 - (current_danger / 50.0))
            
            # Reward progress MORE if agent is in safer areas
            # If safe (factor=1.0): full progress reward
            # If dangerous (factor=0.0): minimal progress reward
            safe_progress_multiplier = 0.5 + 0.5 * safety_factor  # Range: 0.5 to 1.0
            
            reward += distance_improved * self.progress_reward_multiplier * safe_progress_multiplier
            
        elif not invalid_move and distance_improved < 0:
            # Penalty for moving away from goal
            # BUT: Allow retreating if current position is dangerous
            current_danger = self._get_damage_at_position(self.agent_position)
            
            # Reduce penalty if retreating from danger
            retreat_penalty_multiplier = 0.2 if current_danger < 10.0 else 0.05
            reward -= abs(distance_improved) * self.progress_reward_multiplier * retreat_penalty_multiplier
        
        # Spawn distance bonus (encourages leaving start area)
        distance_from_spawn = abs(self.agent_position[0] - self.spawn_position[0]) + \
                             abs(self.agent_position[1] - self.spawn_position[1])
        
        if distance_from_spawn < 10:
            reward += distance_from_spawn * self.spawn_distance_bonus_multiplier
        
        # Tower proximity penalty (NEW: proactive avoidance, not just damage reaction)
        # This gives IMMEDIATE negative feedback for being near towers, even before taking damage
        if not reached_goal and not is_dead:
            for tower in self.towers:
                dist = self._calculate_distance(self.agent_position, tower.position)
                # Penalty increases as agent gets closer to tower
                if dist < tower.range:
                    # Inside range: IMMENSE penalty to force avoidance
                    # Non-linear scaling (squared) makes center terrifying
                    proximity_ratio = 1.0 - (dist / tower.range)
                    
                    # Base penalty for just being in range (prevents "skimming")
                    base_penalty = 5.0
                    
                    # Scaled penalty
                    proximity_penalty = base_penalty + (proximity_ratio ** 2) * self.damage_penalty_multiplier
                    reward -= proximity_penalty
                    
                # elif dist < tower.range * 1.5:
                #     # Near tower: Stronger warning gradient
                #     # Increase multiplier from 0.01 to 0.2 to make it "felt"
                #     proximity_ratio = 1.0 - ((dist - tower.range) / (tower.range * 0.5))
                #     proximity_penalty = proximity_ratio * 0.2 * self.damage_penalty_multiplier
                #     reward -= proximity_penalty
        
        return reward
    
    def _get_damage_at_position(self, position: Tuple[int, int]) -> float:
        """Calculate damage per second at a position."""
        total_dps = 0.0
        for tower in self.towers:
            distance = self._calculate_distance(position, tower.position)
            if distance <= tower.range:
                total_dps += tower.damage * tower.attack_rate
        return total_dps
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset agent state
        self.agent_position = self.spawn_position
        self.agent_health = self.agent_max_health
        self.steps = 0
        self.cumulative_damage = 0.0
        self.last_action = 0
        
        # Recalculate initial distance
        self.last_distance_to_goal = self._calculate_distance(
            self.agent_position,
            self.goal_position
        )
        
        # Optionally randomize towers
        if self.random_towers:
            self._place_towers()
        
        observation = self._get_observation()
        info = {
            "position": self.agent_position,
            "health": self.agent_health,
            "distance_to_goal": self.last_distance_to_goal,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action index (0-7 for 8 directions)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.steps += 1
        self.last_action = action
        
        # Get action direction
        dx, dy = self.ACTION_DIRECTIONS[action]
        new_x = self.agent_position[0] + dx
        new_y = self.agent_position[1] + dy
        
        # Check if new position is valid
        invalid_move = False
        if self.game_map.is_walkable(new_x, new_y):
            self.agent_position = (new_x, new_y)
        else:
            # Invalid move - agent tried to walk into obstacle
            invalid_move = True
        
        # Calculate damage at new position
        damage_per_second = self._get_damage_at_position(self.agent_position)
        # Each step represents ~0.1 seconds (10 steps per second typical movement)
        damage_this_step = damage_per_second * 0.1
        self.agent_health -= damage_this_step
        self.cumulative_damage += damage_this_step
        
        # Calculate distance improvement
        current_distance = self._calculate_distance(self.agent_position, self.goal_position)
        distance_improved = self.last_distance_to_goal - current_distance
        self.last_distance_to_goal = current_distance
        
        # Check terminal conditions
        reached_goal = current_distance < 1.5  # Close enough to goal
        is_dead = self.agent_health <= 0
        terminated = reached_goal or is_dead
        truncated = self.steps >= self.max_steps
        
        # Calculate reward
        reward = self._calculate_reward(
            reached_goal,
            is_dead,
            damage_this_step,
            distance_improved,
            invalid_move
        )
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            "position": self.agent_position,
            "health": self.agent_health,
            "distance_to_goal": current_distance,
            "reached_goal": reached_goal,
            "is_dead": is_dead,
            "cumulative_damage": self.cumulative_damage,
            "steps": self.steps,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            # Could implement pygame rendering here if needed
            pass
    
    def close(self):
        """Clean up resources."""
        pass

