"""
Deep Q-Network (DQN) pathfinding algorithm implementation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import warnings

from apathion.pathfinding.base import BasePathfinder
from apathion.game.map import Map
from apathion.game.tower import Tower

# Import stable-baselines3 if available
try:
    from stable_baselines3 import DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn(
        "stable-baselines3 not installed. DQN pathfinding will use fallback behavior. "
        "Install with: pip install stable-baselines3"
    )


class DQNPathfinder(BasePathfinder):
    """
    Deep Q-Network reinforcement learning pathfinder.
    
    This implementation uses a neural network to learn movement policies
    through experience. The agent learns to balance survival, path efficiency,
    and group success.
    
    Attributes:
        state_size: Size of the state vector
        action_size: Number of possible actions (typically 8 directions)
        model: Stable-baselines3 DQN model
        use_cache: Whether to cache decisions
        cache_duration: Number of frames to reuse cached decisions
    """
    
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
        name: str = "DQN",
        state_size: int = 42,  # Updated to 42 for danger-aware features
        action_size: int = 8,
        use_cache: bool = True,
        cache_duration: int = 5,
        model_path: Optional[str] = None,
        plan_full_path: bool = True,  # Changed: False to match step-by-step training
    ):
        """
        Initialize DQN pathfinder.
        
        Args:
            name: Name identifier
            state_size: Size of state representation
            action_size: Number of possible actions
            use_cache: Whether to cache decisions for performance
            cache_duration: Frames to reuse cached decisions
            model_path: Optional path to trained model
            plan_full_path: If True, generate complete paths; if False, step-by-step
        """
        super().__init__(name)
        self.state_size = state_size
        self.action_size = action_size
        self.use_cache = use_cache
        self.cache_duration = cache_duration
        self.plan_full_path = plan_full_path
        self.model = None
        self.decision_cache: Dict[str, Tuple[List[Tuple[int, int]], int]] = {}
        self._last_tower_positions: set = set()  # Track tower positions for cache invalidation
        
        # Load model if path provided
        if model_path is not None:
            self.load_model(model_path)
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs
    ) -> List[Tuple[int, int]]:
        """
        Find path using DQN policy.
        
        Args:
            start: Starting position
            goal: Goal position
            **kwargs: Optional parameters (enemy_id for caching, health)
            
        Returns:
            List of positions from start to goal (single next step)
        """
        if self.game_map is None:
            return [start, goal]
        
        # Check cache if enabled
        enemy_id = kwargs.get("enemy_id", "default")
        if self.use_cache and enemy_id in self.decision_cache:
            cached_path, frame_count = self.decision_cache[enemy_id]
            if frame_count < self.cache_duration:
                self.decision_cache[enemy_id] = (cached_path, frame_count + 1)
                return cached_path
            else:
                del self.decision_cache[enemy_id]
        
        # Use DQN model if available, otherwise fallback
        if self.model is not None and SB3_AVAILABLE:
            # Pass plan_full_path setting to DQN decision
            kwargs['plan_full_path'] = self.plan_full_path
            path = self._dqn_decision(start, goal, **kwargs)
        else:
            path = self._simple_path_placeholder(start, goal)
        
        # Cache the decision
        if self.use_cache:
            self.decision_cache[enemy_id] = (path, 0)
        
        return path
    
    def _dqn_decision(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs
    ) -> List[Tuple[int, int]]:
        """
        Make a pathfinding decision using the DQN model.
        
        Args:
            start: Current position
            goal: Goal position
            **kwargs: Additional state info (health, etc.)
            
        Returns:
            Next position to move to (or complete path if plan_full_path=True)
        """
        # Check if we should plan a complete path
        plan_full_path = kwargs.get("plan_full_path", False)
        
        if plan_full_path:
            return self._plan_complete_path(start, goal, **kwargs)
        
        # Otherwise, just return next step (original behavior)
        # Encode current state
        state = self.encode_state(start, goal, **kwargs)
        
        # Get action from model (greedy, no exploration during gameplay)
        action, _ = self.model.predict(state, deterministic=True)
        action = int(action)
        
        # Convert action to next position
        dx, dy = self.ACTION_DIRECTIONS[action]
        next_x = start[0] + dx
        next_y = start[1] + dy
        
        # Validate the move
        if self.game_map.is_walkable(next_x, next_y):
            return [start, (next_x, next_y)]
        else:
            # If move is invalid, find a valid alternative
            # Try to find any walkable direction, preferring toward goal
            best_move = start
            best_score = float('inf')
            
            for alt_dx, alt_dy in self.ACTION_DIRECTIONS:
                alt_x = start[0] + alt_dx
                alt_y = start[1] + alt_dy
                
                if self.game_map.is_walkable(alt_x, alt_y):
                    # Score based on distance to goal (lower is better)
                    dist = abs(alt_x - goal[0]) + abs(alt_y - goal[1])
                    if dist < best_score:
                        best_score = dist
                        best_move = (alt_x, alt_y)
            
            # Return best valid move (or stay in place if no valid moves)
            return [start, best_move]
    
    def _plan_complete_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs
    ) -> List[Tuple[int, int]]:
        """
        Plan a complete path from start to goal using DQN model.
        
        This simulates the DQN model forward step-by-step to generate
        a complete path, similar to how the agent would navigate during training.
        
        Args:
            start: Starting position
            goal: Goal position
            **kwargs: Additional state info (health, etc.)
            
        Returns:
            Complete path from start to goal
        """
        if self.game_map is None:
            return [start, goal]
        
        path = [start]
        current_pos = start
        health = kwargs.get("health", 100.0)
        max_health = kwargs.get("max_health", 100.0)
        
        # Calculate reasonable max steps based on map size and current distance
        straight_line_dist = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        max_steps = kwargs.get("max_path_steps", max(200, straight_line_dist * 5))
        
        visited = set()
        visited.add(current_pos)
        stuck_counter = 0  # Track if agent is stuck
        last_distance = straight_line_dist
        
        for step in range(max_steps):
            # Check if reached goal
            distance_to_goal = abs(current_pos[0] - goal[0]) + abs(current_pos[1] - goal[1])
            
            # More lenient goal detection
            if distance_to_goal == 0:
                # Exactly at goal
                break
            elif distance_to_goal == 1:
                # One step away - add goal and finish
                if goal not in path:
                    path.append(goal)
                break
            
            # Encode current state
            state = self.encode_state(
                current_pos, 
                goal, 
                health=health, 
                max_health=max_health
            )
            
            # Get action from model
            action, _ = self.model.predict(state, deterministic=True)
            action = int(action)
            
            # Convert action to next position
            dx, dy = self.ACTION_DIRECTIONS[action]
            next_x = current_pos[0] + dx
            next_y = current_pos[1] + dy
            next_pos = (next_x, next_y)
            
            # Check if move is valid
            if not self.game_map.is_walkable(next_x, next_y):
                # Invalid move - try to find a valid alternative
                found_valid = False
                for alt_dx, alt_dy in self.ACTION_DIRECTIONS:
                    alt_x = current_pos[0] + alt_dx
                    alt_y = current_pos[1] + alt_dy
                    alt_pos = (alt_x, alt_y)
                    
                    if self.game_map.is_walkable(alt_x, alt_y) and alt_pos not in visited:
                        # Use this position
                        next_pos = alt_pos
                        found_valid = True
                        break
                
                if not found_valid:
                    # No valid moves - return path so far
                    break
            
            # Check for loops (visiting same position)
            if next_pos in visited:
                # Try to find unvisited position
                found_unvisited = False
                for alt_dx, alt_dy in self.ACTION_DIRECTIONS:
                    alt_x = current_pos[0] + alt_dx
                    alt_y = current_pos[1] + alt_dy
                    alt_pos = (alt_x, alt_y)
                    
                    if self.game_map.is_walkable(alt_x, alt_y) and alt_pos not in visited:
                        next_pos = alt_pos
                        found_unvisited = True
                        break
                
                if not found_unvisited:
                    # Stuck - return path so far
                    break
            
            # Move to next position
            path.append(next_pos)
            visited.add(next_pos)
            current_pos = next_pos
            
            # Check if making progress
            if distance_to_goal >= last_distance:
                stuck_counter += 1
                if stuck_counter > 10:
                    # Agent is stuck or moving in circles - force toward goal
                    break
            else:
                stuck_counter = 0
            
            last_distance = distance_to_goal
            
            # Simulate damage (reduce health for next prediction)
            damage = self._estimate_damage_at_position(current_pos)
            health = max(0, health - damage * 0.1)
            
            # If health too low, agent might give up - break and return partial path
            if health <= 10:
                break
        
        # If didn't reach goal, forcefully complete the path
        if len(path) > 0 and path[-1] != goal:
            last_pos = path[-1]
            dist = abs(last_pos[0] - goal[0]) + abs(last_pos[1] - goal[1])
            
            if dist > 0:
                # Use greedy pathfinding to complete the path
                # This ensures we ALWAYS reach the goal
                attempt_pos = last_pos
                completion_steps = 0
                max_completion_steps = dist * 3  # Allow some detours
                
                while attempt_pos != goal and completion_steps < max_completion_steps:
                    # Move toward goal greedily
                    dx = int(np.sign(goal[0] - attempt_pos[0]))
                    dy = int(np.sign(goal[1] - attempt_pos[1]))
                    
                    # Try moving in x direction first
                    if dx != 0:
                        next_pos = (attempt_pos[0] + dx, attempt_pos[1])
                        if self.game_map.is_walkable(next_pos[0], next_pos[1]):
                            if next_pos not in path:
                                path.append(next_pos)
                            attempt_pos = next_pos
                            completion_steps += 1
                            continue
                    
                    # Try moving in y direction
                    if dy != 0:
                        next_pos = (attempt_pos[0], attempt_pos[1] + dy)
                        if self.game_map.is_walkable(next_pos[0], next_pos[1]):
                            if next_pos not in path:
                                path.append(next_pos)
                            attempt_pos = next_pos
                            completion_steps += 1
                            continue
                    
                    # Try diagonal or any valid move
                    moved = False
                    for alt_dx, alt_dy in self.ACTION_DIRECTIONS:
                        next_pos = (attempt_pos[0] + alt_dx, attempt_pos[1] + alt_dy)
                        next_dist = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                        
                        if (self.game_map.is_walkable(next_pos[0], next_pos[1]) and 
                            next_pos not in path and next_dist < dist):
                            path.append(next_pos)
                            attempt_pos = next_pos
                            dist = next_dist
                            completion_steps += 1
                            moved = True
                            break
                    
                    if not moved:
                        # Completely stuck
                        break
        
        return path
    
    def _estimate_damage_at_position(self, position: Tuple[int, int]) -> float:
        """Estimate damage per step at a position (for path planning)."""
        total_dps = 0.0
        for tower in self.towers:
            distance = np.sqrt(
                (position[0] - tower.position[0]) ** 2 + 
                (position[1] - tower.position[1]) ** 2
            )
            if distance <= tower.range:
                total_dps += tower.damage * tower.attack_rate
        return total_dps
    
    def _simple_path_placeholder(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Fallback pathfinding when no DQN model is available.
        
        Returns a single step toward the goal, avoiding obstacles.
        """
        if self.game_map is None:
            return [start, goal]
        
        # Find best walkable move toward goal
        best_move = start
        best_distance = float('inf')
        
        for dx, dy in self.ACTION_DIRECTIONS:
            next_x = start[0] + dx
            next_y = start[1] + dy
            
            if self.game_map.is_walkable(next_x, next_y):
                # Calculate distance to goal
                distance = abs(next_x - goal[0]) + abs(next_y - goal[1])
                if distance < best_distance:
                    best_distance = distance
                    best_move = (next_x, next_y)
        
        return [start, best_move]
    
    def update_state(self, game_map: Map, towers: List[Tower]) -> None:
        """
        Update map and tower information.
        
        Args:
            game_map: Current game map
            towers: List of active towers
        """
        # Get current tower positions as a set (for comparison)
        current_tower_positions = {t.position for t in towers}
        
        # Clear cache if tower configuration changed
        if current_tower_positions != self._last_tower_positions:
            self.decision_cache.clear()
            # print(f"Cache cleared: towers changed from {len(self._last_tower_positions)} to {len(current_tower_positions)}")
        
        # Update state
        self.game_map = game_map
        self.towers = towers
        self._last_tower_positions = current_tower_positions
    
    def encode_state(
        self,
        position: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs
    ) -> np.ndarray:
        """
        Encode game state into a feature vector for the neural network.
        
        This encoding matches the PathfindingEnv observation space.
        
        Args:
            position: Current position
            goal: Goal position
            **kwargs: Additional state info (health, etc.)
            
        Returns:
            State vector as numpy array (shape: (state_size,))
        """
        if self.game_map is None:
            return np.zeros(self.state_size, dtype=np.float32)
        
        obs = []
        
        # 1. Normalized relative position to goal (2 features)
        rel_x = (goal[0] - position[0]) / self.game_map.width
        rel_y = (goal[1] - position[1]) / self.game_map.height
        obs.extend([rel_x, rel_y])
        
        # 1b. NEW: Normalized absolute position (2 features)
        abs_x = position[0] / self.game_map.width
        abs_y = position[1] / self.game_map.height
        obs.extend([abs_x, abs_y])
        
        # 2. Normalized distance to goal (1 feature)
        max_distance = np.sqrt(self.game_map.width ** 2 + self.game_map.height ** 2)
        distance = np.sqrt((goal[0] - position[0]) ** 2 + (goal[1] - position[1]) ** 2)
        obs.append(distance / max_distance)
        
        # 3. Current health ratio (1 feature)
        health = kwargs.get("health", 100.0)
        max_health = kwargs.get("max_health", 100.0)
        obs.append(health / max_health)
        
        # 4. Directional walkability (8 features - one per action direction)
        # CRITICAL: Tells agent which directions are blocked by obstacles
        for dx, dy in self.ACTION_DIRECTIONS:
            x, y = position[0] + dx, position[1] + dy
            is_walkable = 1.0 if self.game_map.is_walkable(x, y) else 0.0
            obs.append(is_walkable)
        
        # 5. Tower threat features (up to 5 nearest towers, 4 features each = 20 features)
        towers_sorted = sorted(
            self.towers,
            key=lambda t: np.sqrt((position[0] - t.position[0]) ** 2 + (position[1] - t.position[1]) ** 2)
        )
        
        max_towers_encoded = 5
        for i in range(max_towers_encoded):
            if i < len(towers_sorted):
                tower = towers_sorted[i]
                t_rel_x = (tower.position[0] - position[0]) / self.game_map.width
                t_rel_y = (tower.position[1] - position[1]) / self.game_map.height
                t_distance = np.sqrt(
                    (position[0] - tower.position[0]) ** 2 + 
                    (position[1] - tower.position[1]) ** 2
                )
                t_in_range = 1.0 if t_distance <= tower.range else 0.0
                t_damage_rate = (tower.damage * tower.attack_rate) / 50.0  # Normalize
                
                obs.extend([t_rel_x, t_rel_y, t_in_range, t_damage_rate])
            else:
                # Padding for missing towers
                obs.extend([0.0, 0.0, 0.0, 0.0])
        
        # 6. Danger-aware directional features (8 features - danger level per direction)
        for dx, dy in self.ACTION_DIRECTIONS:
            x, y = position[0] + dx, position[1] + dy
            danger_level = 0.0
            
            if self.game_map.is_walkable(x, y):
                # Calculate danger at this position
                danger_level = self._estimate_damage_at_position((x, y))
                # Normalize danger (typical max DPS ~50)
                danger_level = min(1.0, danger_level / 50.0)
            else:
                # Obstacle = maximum danger
                danger_level = 1.0
            
            obs.append(danger_level)
        
        # Total features: 2 + 2 + 1 + 1 + 8 + 20 + 8 = 42
        return np.array(obs, dtype=np.float32)
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state: Encoded state vector
            epsilon: Exploration rate (0 = greedy, 1 = random)
            
        Returns:
            Action index
        """
        # PLACEHOLDER: This would use the neural network
        
        if np.random.random() < epsilon:
            # Exploration: random action
            return np.random.randint(0, self.action_size)
        else:
            # Exploitation: use model (placeholder returns random for now)
            # TODO: q_values = self.model.predict(state)
            # return np.argmax(q_values)
            return 0  # Placeholder
    
    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> float:
        """
        Perform one training step (placeholder).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
            
        Returns:
            Loss value
        """
        # PLACEHOLDER: This would implement DQN training logic
        # - Store experience in replay buffer
        # - Sample batch
        # - Calculate TD target
        # - Backpropagate and update weights
        
        return 0.0
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to saved model (without .zip extension)
            
        Returns:
            True if loaded successfully
        """
        if not SB3_AVAILABLE:
            print("Warning: stable-baselines3 not available. Cannot load model.")
            return False
        
        try:
            # Load model on CPU for inference (works regardless of training device)
            self.model = DQN.load(model_path, device="cpu")
            print(f"Successfully loaded DQN model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """
        Save the current model to disk.
        
        Args:
            model_path: Path to save model (without .zip extension)
            
        Returns:
            True if saved successfully
        """
        if not SB3_AVAILABLE or self.model is None:
            print("Warning: No model to save or stable-baselines3 not available.")
            return False
        
        try:
            self.model.save(model_path)
            print(f"Successfully saved DQN model to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model to {model_path}: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = super().to_dict()
        data.update({
            "state_size": self.state_size,
            "action_size": self.action_size,
            "use_cache": self.use_cache,
            "cache_duration": self.cache_duration,
            "cached_decisions": len(self.decision_cache),
            "model_loaded": self.model is not None,
        })
        return data

