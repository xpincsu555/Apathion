"""
Ant Colony Optimization (ACO) pathfinding algorithm implementation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from collections import deque

from apathion.pathfinding.base import BasePathfinder
from apathion.game.map import Map
from apathion.game.tower import Tower


class ACOPathfinder(BasePathfinder):
    """
    Ant Colony Optimization pathfinding with pheromone trails and damage awareness.
    
    This implementation enables swarm intelligence where enemy groups
    deposit virtual pheromones on paths, influencing future routing decisions.
    Includes tower damage avoidance to prevent enemies from exploring near towers.
    
    Attributes:
        num_ants: Number of ants to simulate per path search
        pheromone_grid: 2D grid storing pheromone levels
        evaporation_rate: Rate at which pheromones evaporate (0-1)
        deposit_strength: Amount of pheromone deposited by successful paths
        alpha: Pheromone importance weight
        beta: Heuristic importance weight
        gamma: Damage avoidance weight (higher = avoid towers more strongly)
    """
    
    def __init__(
        self,
        name: str = "ACO",
        num_ants: int = 5,
        evaporation_rate: float = 0.05,
        deposit_strength: float = 2.0,
        alpha: float = 1.0,
        beta: float = 2.5,
        gamma: float = 5.0,
        epsilon: float = 0.15,
        use_path_cache: bool = False,
        cache_duration: int = 3,
    ):
        """
        Initialize ACO pathfinder.
        
        Args:
            name: Name identifier
            num_ants: Number of ants per iteration (reduced for efficiency)
            evaporation_rate: Pheromone evaporation rate (0-1)
            deposit_strength: Pheromone deposit amount
            alpha: Pheromone importance (higher = follow trails more)
            beta: Heuristic importance (higher = prefer goal-directed paths)
            gamma: Damage avoidance weight (higher = avoid towers more)
            epsilon: Exploration rate (0-1, higher = more random exploration)
            use_path_cache: Enable path caching for similar positions
            cache_duration: Number of frames to cache paths (short to allow diversity)
        """
        super().__init__(name)
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.deposit_strength = deposit_strength
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_path_cache = use_path_cache
        self.cache_duration = cache_duration
        self.pheromone_grid: Optional[np.ndarray] = None
        self.anti_pheromone_grid: Optional[np.ndarray] = None
        self.path_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[List[Tuple[int, int]], int]] = {}
        self.frame_count = 0
        self.enemy_count = 0  # Track number of enemies for exploration diversity
        self._astar_fallback_pathfinder = None  # Lazy initialization of A* for fallback
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs
    ) -> List[Tuple[int, int]]:
        """
        Find path using ACO algorithm.
        
        Args:
            start: Starting position
            goal: Goal position
            **kwargs: Optional parameters:
                - num_ants: Override number of ants (int)
                - max_iterations: Maximum iterations per ant (int, default 500)
            
        Returns:
            List of positions from start to goal
        """
        if self.game_map is None or self.pheromone_grid is None:
            return [start, goal]
        
        # Track enemy for exploration diversity
        self.enemy_count += 1
        
        # Check cache with probabilistic use (to maintain diversity)
        # Only use cache sometimes to ensure exploration continues
        use_cache_this_time = (self.use_path_cache and 
                              np.random.random() > self.epsilon and
                              self.enemy_count % 3 != 0)  # Skip cache every 3rd enemy
        
        if use_cache_this_time:
            cache_key = (start, goal)
            if cache_key in self.path_cache:
                cached_path, cached_frame = self.path_cache[cache_key]
                if self.frame_count - cached_frame < self.cache_duration:
                    # Validate cached path is still valid
                    if self._is_path_valid(cached_path):
                        return cached_path
                else:
                    # Cache expired
                    del self.path_cache[cache_key]
        
        # Get parameters
        num_ants = kwargs.get('num_ants', self.num_ants)
        max_iterations = kwargs.get('max_iterations', 500)
        
        # Track best path found
        best_path: List[Tuple[int, int]] = []
        best_cost = float('inf')
        all_failed_paths: List[List[Tuple[int, int]]] = []
        
        # Run multiple ants to explore paths
        for ant_idx in range(num_ants):
            path = self._construct_ant_path(start, goal, max_iterations)
            
            if path and path[-1] == goal:
                # Successful path - calculate cost including damage
                cost = self._calculate_path_cost_with_damage(path)
                
                # Update best path
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
                
                # Deposit pheromones on this path
                # Better paths (lower cost) get more pheromone
                quality = 100.0 / (cost + 1.0)
                self._deposit_pheromones_with_quality(path, quality)
            else:
                # Failed path - track for anti-pheromone
                if path:
                    all_failed_paths.append(path)
        
        # Deposit anti-pheromone on failed paths to discourage future ants
        for failed_path in all_failed_paths:
            self._deposit_anti_pheromones(failed_path)
        
        # If no valid path found, use A* as fallback
        if not best_path:
            # Try with more iterations first
            for _ in range(num_ants):
                path = self._construct_ant_path(start, goal, max_iterations * 2)
                if path and path[-1] == goal:
                    best_path = path
                    break
            
            # If still no path, use A* fallback
            if not best_path:
                best_path = self._get_astar_fallback_path(start, goal)
                
            # If A* also fails, return just start position (don't move)
            if not best_path or best_path[-1] != goal:
                return [start]
        
        # Smooth the path
        best_path = self._smooth_path(best_path)
        
        # Cache the path
        if self.use_path_cache:
            self.path_cache[(start, goal)] = (best_path, self.frame_count)
        
        return best_path
    
    def _construct_ant_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_iterations: int
    ) -> List[Tuple[int, int]]:
        """
        Construct a path for a single ant using probabilistic selection.
        
        Args:
            start: Starting position
            goal: Goal position
            max_iterations: Maximum steps before giving up
            
        Returns:
            Path constructed by the ant (may not reach goal)
        """
        path = [start]
        current = start
        visited = {start}
        iterations = 0
        stuck_count = 0
        max_stuck = 5
        
        while current != goal and iterations < max_iterations:
            # Early termination if very close to goal
            dist_to_goal = self._manhattan_distance(current, goal)
            if dist_to_goal <= 1:
                # Close enough, add goal directly
                if self.game_map and self.game_map.is_walkable(goal[0], goal[1]):
                    path.append(goal)
                    current = goal
                    break
            
            # Get valid neighbors
            neighbors = self._get_valid_neighbors(current)
            
            # Filter out visited neighbors
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            # If stuck, allow limited revisiting
            if not unvisited_neighbors:
                stuck_count += 1
                if stuck_count >= max_stuck:
                    break
                # Allow revisiting but prefer less-visited nodes
                unvisited_neighbors = neighbors
            else:
                stuck_count = 0
            
            if not unvisited_neighbors:
                break
            
            # Select next position probabilistically
            next_pos = self._select_next_position(current, unvisited_neighbors, goal)
            
            if next_pos is None:
                break
            
            path.append(next_pos)
            visited.add(next_pos)
            current = next_pos
            iterations += 1
        
        return path
    
    def _get_valid_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid walkable neighbor positions.
        
        Args:
            pos: Current position
            
        Returns:
            List of valid neighbor positions
        """
        if self.game_map is None:
            return []
        
        x, y = pos
        neighbors = []
        
        # 8-directional movement (cardinal + diagonal)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < self.game_map.width and 0 <= ny < self.game_map.height:
                # Check walkability
                if self.game_map.is_walkable(nx, ny):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _select_next_position(
        self,
        current: Tuple[int, int],
        neighbors: List[Tuple[int, int]],
        goal: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Probabilistically select next position based on pheromones and heuristics.
        Uses epsilon-greedy exploration to maintain path diversity.
        
        Args:
            current: Current position
            neighbors: List of candidate neighbors
            goal: Goal position
            
        Returns:
            Selected neighbor or None if no valid selection
        """
        if not neighbors:
            return None
        
        # Epsilon-greedy exploration: sometimes choose randomly
        if np.random.random() < self.epsilon:
            # Random exploration - but prefer positions closer to goal
            distances = [self._euclidean_distance(n, goal) for n in neighbors]
            min_dist = min(distances)
            # Filter to neighbors that are reasonably close to best
            good_neighbors = [n for n, d in zip(neighbors, distances) if d <= min_dist * 1.5]
            if good_neighbors:
                return good_neighbors[np.random.randint(len(good_neighbors))]
            return neighbors[np.random.randint(len(neighbors))]
        
        # Calculate probabilities for each neighbor
        probabilities = []
        for neighbor in neighbors:
            prob = self.calculate_transition_probability(current, neighbor, goal)
            probabilities.append(prob)
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            # If all probabilities are zero, choose randomly
            return neighbors[np.random.randint(len(neighbors))]
        
        probabilities = np.array([p / total for p in probabilities])
        
        # Add small amount of noise to prevent complete convergence
        noise = np.random.random(len(probabilities)) * 0.01
        probabilities = probabilities + noise
        
        # Ensure probabilities sum to exactly 1.0 (handle floating point errors)
        probabilities = probabilities / probabilities.sum()
        
        # Select based on probability distribution
        selected_idx = np.random.choice(len(neighbors), p=probabilities)
        return neighbors[selected_idx]
    
    def update_state(self, game_map: Map, towers: List[Tower]) -> None:
        """
        Update map, towers, and pheromone grid.
        
        Args:
            game_map: Current game map
            towers: List of active towers
        """
        self.game_map = game_map
        self.towers = towers
        self.frame_count += 1
        
        # Initialize pheromone grid if needed
        if (self.pheromone_grid is None or
            self.pheromone_grid.shape != (game_map.height, game_map.width)):
            self._initialize_pheromone_grid()
        
        # Initialize anti-pheromone grid if needed
        if (self.anti_pheromone_grid is None or
            self.anti_pheromone_grid.shape != (game_map.height, game_map.width)):
            self._initialize_anti_pheromone_grid()
        
        # Apply evaporation
        self._evaporate_pheromones()
        self._evaporate_anti_pheromones()
    
    def _initialize_pheromone_grid(self) -> None:
        """
        Initialize pheromone grid with uniform values.
        
        PLACEHOLDER: Could use smarter initialization strategies.
        """
        if self.game_map is None:
            return
        
        # Initialize with small uniform value
        initial_value = 0.1
        self.pheromone_grid = np.full(
            (self.game_map.height, self.game_map.width),
            initial_value,
            dtype=np.float32
        )
    
    def _evaporate_pheromones(self) -> None:
        """
        Apply pheromone evaporation to all cells.
        
        Pheromones decay over time to allow adaptation to changing conditions.
        """
        if self.pheromone_grid is not None:
            self.pheromone_grid *= (1.0 - self.evaporation_rate)
            # Maintain minimum pheromone level
            self.pheromone_grid = np.maximum(self.pheromone_grid, 0.01)
    
    def _deposit_pheromones(self, path: List[Tuple[int, int]]) -> None:
        """
        Deposit pheromones along a path.
        
        Args:
            path: List of positions where pheromones should be deposited
        """
        self._deposit_pheromones_with_quality(path, quality=1.0)
    
    def _deposit_pheromones_with_quality(
        self,
        path: List[Tuple[int, int]],
        quality: float
    ) -> None:
        """
        Deposit pheromones along a path with quality weighting.
        
        Better paths (higher quality) receive more pheromone deposit.
        
        Args:
            path: List of positions where pheromones should be deposited
            quality: Quality multiplier for pheromone deposit (0-1, higher is better)
        """
        if self.pheromone_grid is None or not path:
            return
        
        # Calculate deposit amount based on path length and quality
        path_length = len(path)
        if path_length == 0:
            return
        
        # Shorter paths get more pheromone per cell
        # Quality factor further amplifies good paths
        deposit_per_cell = (self.deposit_strength / path_length) * quality
        
        # Deposit on each cell in path
        for x, y in path:
            if 0 <= y < self.pheromone_grid.shape[0] and 0 <= x < self.pheromone_grid.shape[1]:
                self.pheromone_grid[y, x] += deposit_per_cell
    
    def calculate_transition_probability(
        self,
        current: Tuple[int, int],
        neighbor: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> float:
        """
        Calculate probability of moving from current to neighbor.
        
        Combines pheromone trails, distance heuristic, and damage avoidance
        to guide path selection away from dangerous tower zones.
        
        Args:
            current: Current position
            neighbor: Neighboring position to evaluate
            goal: Goal position
            
        Returns:
            Transition probability
        """
        if self.pheromone_grid is None or self.anti_pheromone_grid is None:
            return 0.0
        
        # Get pheromone level
        nx, ny = neighbor
        if not (0 <= ny < self.pheromone_grid.shape[0] and 
                0 <= nx < self.pheromone_grid.shape[1]):
            return 0.0
        
        pheromone = self.pheromone_grid[ny, nx]
        anti_pheromone = self.anti_pheromone_grid[ny, nx]
        
        # Net pheromone: positive minus negative
        net_pheromone = max(0.01, pheromone - anti_pheromone * 0.5)
        
        # Calculate heuristic (inverse of distance to goal)
        # Use Euclidean distance for better accuracy
        dx = abs(neighbor[0] - goal[0])
        dy = abs(neighbor[1] - goal[1])
        distance = (dx ** 2 + dy ** 2) ** 0.5
        heuristic = 1.0 / (distance + 0.1)
        
        # Calculate damage avoidance factor (STRENGTHENED)
        damage_avoidance = 1.0
        if self.gamma > 0 and self.towers:
            damage = self.estimate_damage_at_position(neighbor)
            # Much stronger penalty for damage
            # Use exponential decay with configurable strength
            if damage > 0:
                damage_avoidance = np.exp(-damage * 0.05)
            else:
                damage_avoidance = 1.0
        
        # Additional bonus for moving closer to goal (reduced to allow more exploration)
        current_dist = self._euclidean_distance(current, goal)
        neighbor_dist = self._euclidean_distance(neighbor, goal)
        progress_bonus = 1.0
        if neighbor_dist < current_dist:
            progress_bonus = 1.3  # 30% bonus for getting closer to goal (reduced from 50%)
        elif neighbor_dist > current_dist:
            progress_bonus = 0.8  # Milder penalty for moving away (was 0.7)
        
        # Combine all factors
        probability = (
            (net_pheromone ** self.alpha) * 
            (heuristic ** self.beta) * 
            (damage_avoidance ** self.gamma) *
            progress_bonus
        )
        
        return max(0.0, probability)
    
    def get_pheromone_at(self, position: Tuple[int, int]) -> float:
        """
        Get pheromone level at a position.
        
        Args:
            position: (x, y) grid position
            
        Returns:
            Pheromone level
        """
        if self.pheromone_grid is None:
            return 0.0
        
        x, y = position
        if 0 <= y < self.pheromone_grid.shape[0] and 0 <= x < self.pheromone_grid.shape[1]:
            return float(self.pheromone_grid[y, x])
        return 0.0
    
    def _initialize_anti_pheromone_grid(self) -> None:
        """
        Initialize anti-pheromone grid (tracks failed paths).
        """
        if self.game_map is None:
            return
        
        self.anti_pheromone_grid = np.zeros(
            (self.game_map.height, self.game_map.width),
            dtype=np.float32
        )
    
    def _evaporate_anti_pheromones(self) -> None:
        """
        Apply anti-pheromone evaporation (faster than regular pheromones).
        """
        if self.anti_pheromone_grid is not None:
            # Anti-pheromones evaporate faster to allow recovery
            self.anti_pheromone_grid *= (1.0 - self.evaporation_rate * 2.0)
            self.anti_pheromone_grid = np.maximum(self.anti_pheromone_grid, 0.0)
    
    def _deposit_anti_pheromones(self, path: List[Tuple[int, int]]) -> None:
        """
        Deposit anti-pheromones on a failed path to discourage future exploration.
        
        Args:
            path: Failed path where ant got stuck or killed
        """
        if self.anti_pheromone_grid is None or not path:
            return
        
        # Deposit less anti-pheromone than regular pheromone
        deposit_per_cell = self.deposit_strength * 0.3
        
        for x, y in path:
            if 0 <= y < self.anti_pheromone_grid.shape[0] and 0 <= x < self.anti_pheromone_grid.shape[1]:
                self.anti_pheromone_grid[y, x] += deposit_per_cell
    
    def _calculate_path_cost_with_damage(self, path: List[Tuple[int, int]]) -> float:
        """
        Calculate path cost including both distance and damage exposure.
        
        Args:
            path: Path to evaluate
            
        Returns:
            Total cost (distance + weighted damage)
        """
        if not path or len(path) < 2:
            return 0.0
        
        # Distance cost
        distance_cost = self.calculate_path_cost(path)
        
        # Damage cost
        damage_cost = 0.0
        for pos in path:
            damage = self.estimate_damage_at_position(pos)
            damage_cost += damage
        
        # Combine costs (damage is weighted more heavily)
        total_cost = distance_cost + damage_cost * 2.0
        
        return total_cost
    
    def _is_path_valid(self, path: List[Tuple[int, int]]) -> bool:
        """
        Check if a cached path is still valid (all positions walkable).
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is valid, False otherwise
        """
        if not path or self.game_map is None:
            return False
        
        for x, y in path:
            if not (0 <= x < self.game_map.width and 0 <= y < self.game_map.height):
                return False
            if not self.game_map.is_walkable(x, y):
                return False
        
        return True
    
    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Smooth path by removing unnecessary waypoints using line-of-sight.
        
        Args:
            path: Original path
            
        Returns:
            Smoothed path with fewer waypoints
        """
        if not path or len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to find furthest visible point
            furthest_idx = current_idx + 1
            
            for test_idx in range(len(path) - 1, current_idx, -1):
                if self._has_line_of_sight(path[current_idx], path[test_idx]):
                    furthest_idx = test_idx
                    break
            
            smoothed.append(path[furthest_idx])
            current_idx = furthest_idx
        
        return smoothed
    
    def _has_line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        Check if there's a clear line of sight between two points.
        
        Args:
            start: Starting position
            end: Ending position
            
        Returns:
            True if line of sight exists, False otherwise
        """
        if self.game_map is None:
            return False
        
        x0, y0 = start
        x1, y1 = end
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # Check if current position is walkable
            if not self.game_map.is_walkable(x, y):
                return False
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Euclidean distance
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx ** 2 + dy ** 2) ** 0.5
    
    def _get_astar_fallback_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Use A* pathfinding as fallback when ACO fails.
        
        This delegates to the AStarPathfinder class to reuse existing
        tested implementation instead of duplicating code.
        
        Args:
            start: Starting position
            goal: Goal position
            
        Returns:
            Path from start to goal, or empty list if no path exists
        """
        # Lazy import to avoid circular dependency
        from apathion.pathfinding.astar import AStarPathfinder
        
        # Lazy initialization of A* pathfinder
        if self._astar_fallback_pathfinder is None:
            self._astar_fallback_pathfinder = AStarPathfinder(
                name="ACO-Fallback-AStar",
                use_enhanced=False,  # Use basic A* for fallback (no damage consideration)
                diagonal_movement=True
            )
        
        # Update A* with current state
        if self.game_map is not None:
            self._astar_fallback_pathfinder.update_state(self.game_map, self.towers)
        
        # Get path from A*
        path = self._astar_fallback_pathfinder.find_path(start, goal)
        
        # A* returns [start, goal] on failure, we want empty list
        if path == [start, goal]:
            return []
        
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = super().to_dict()
        data.update({
            "num_ants": self.num_ants,
            "evaporation_rate": self.evaporation_rate,
            "deposit_strength": self.deposit_strength,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "use_path_cache": self.use_path_cache,
            "cache_duration": self.cache_duration,
            "pheromone_initialized": self.pheromone_grid is not None,
            "anti_pheromone_initialized": self.anti_pheromone_grid is not None,
            "cached_paths": len(self.path_cache),
            "frame_count": self.frame_count,
            "enemy_count": self.enemy_count,
        })
        return data

