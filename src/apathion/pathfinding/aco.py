"""
Ant Colony Optimization (ACO) pathfinding algorithm implementation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

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
        num_ants: int = 10,
        evaporation_rate: float = 0.01,
        deposit_strength: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.5,
        gamma: float = 3.5,
    ):
        """
        Initialize ACO pathfinder.
        
        Args:
            name: Name identifier
            num_ants: Number of ants per iteration
            evaporation_rate: Pheromone evaporation rate (0-1)
            deposit_strength: Pheromone deposit amount
            alpha: Pheromone importance (higher = follow trails more)
            beta: Heuristic importance (higher = prefer shorter paths)
            gamma: Damage avoidance weight (higher = avoid towers more)
        """
        super().__init__(name)
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.deposit_strength = deposit_strength
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pheromone_grid: Optional[np.ndarray] = None
    
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
                - max_iterations: Maximum iterations per ant (int, default 1000)
            
        Returns:
            List of positions from start to goal
        """
        if self.game_map is None or self.pheromone_grid is None:
            return [start, goal]
        
        # Get parameters
        num_ants = kwargs.get('num_ants', self.num_ants)
        max_iterations = kwargs.get('max_iterations', 1000)
        
        # Track best path found
        best_path: List[Tuple[int, int]] = []
        best_cost = float('inf')
        
        # Run multiple ants to explore paths
        for _ in range(num_ants):
            path = self._construct_ant_path(start, goal, max_iterations)
            
            if path and path[-1] == goal:
                cost = self.calculate_path_cost(path)
                
                # Update best path
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
                
                # Deposit pheromones on this path
                # Better paths (shorter) get more pheromone
                quality = 1.0 / (cost + 1.0)
                self._deposit_pheromones_with_quality(path, quality)
        
        # If no valid path found, return fallback
        if not best_path:
            return [start, goal]
        
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
        
        while current != goal and iterations < max_iterations:
            # Get valid neighbors
            neighbors = self._get_valid_neighbors(current)
            
            # Filter out visited neighbors
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            # If stuck, allow revisiting (with penalty)
            if not unvisited_neighbors:
                unvisited_neighbors = neighbors
            
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
        
        Args:
            current: Current position
            neighbors: List of candidate neighbors
            goal: Goal position
            
        Returns:
            Selected neighbor or None if no valid selection
        """
        if not neighbors:
            return None
        
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
        
        # Initialize pheromone grid if needed
        if (self.pheromone_grid is None or
            self.pheromone_grid.shape != (game_map.height, game_map.width)):
            self._initialize_pheromone_grid()
        
        # Apply evaporation
        self._evaporate_pheromones()
    
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
        if self.pheromone_grid is None:
            return 0.0
        
        # Get pheromone level
        nx, ny = neighbor
        if not (0 <= ny < self.pheromone_grid.shape[0] and 
                0 <= nx < self.pheromone_grid.shape[1]):
            return 0.0
        
        pheromone = self.pheromone_grid[ny, nx]
        
        # Calculate heuristic (inverse of distance to goal)
        dx = abs(neighbor[0] - goal[0])
        dy = abs(neighbor[1] - goal[1])
        distance = (dx ** 2 + dy ** 2) ** 0.5
        heuristic = 1.0 / (distance + 1.0)
        
        # Calculate damage avoidance factor
        damage_avoidance = 1.0
        if self.gamma > 0 and self.towers:
            damage = self.estimate_damage_at_position(neighbor)
            # Higher damage = lower avoidance factor
            # Use exponential decay to strongly penalize high-damage areas
            damage_avoidance = 1.0 / (1.0 + damage * 0.1)
        
        # Combine pheromone, heuristic, and damage avoidance
        probability = (
            (pheromone ** self.alpha) * 
            (heuristic ** self.beta) * 
            (damage_avoidance ** self.gamma)
        )
        
        return probability
    
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
            "pheromone_initialized": self.pheromone_grid is not None,
        })
        return data

