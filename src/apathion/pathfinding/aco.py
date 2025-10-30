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
    Ant Colony Optimization pathfinding with pheromone trails.
    
    This implementation enables swarm intelligence where enemy groups
    deposit virtual pheromones on paths, influencing future routing decisions.
    
    Attributes:
        num_ants: Number of ants to simulate per path search
        pheromone_grid: 2D grid storing pheromone levels
        evaporation_rate: Rate at which pheromones evaporate (0-1)
        deposit_strength: Amount of pheromone deposited by successful paths
        alpha: Pheromone importance weight
        beta: Heuristic importance weight
    """
    
    def __init__(
        self,
        name: str = "ACO",
        num_ants: int = 10,
        evaporation_rate: float = 0.1,
        deposit_strength: float = 1.0,
        alpha: float = 1.0,
        beta: float = 2.0,
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
        """
        super().__init__(name)
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.deposit_strength = deposit_strength
        self.alpha = alpha
        self.beta = beta
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
            **kwargs: Optional parameters
            
        Returns:
            List of positions from start to goal
        """
        # PLACEHOLDER: Actual ACO implementation would go here
        # For now, return a simple path
        
        if self.game_map is None or self.pheromone_grid is None:
            return [start, goal]
        
        # TODO: Implement full ACO search with:
        # - Multiple ant simulations
        # - Probabilistic path selection based on pheromones
        # - Path evaluation and pheromone deposit
        # - Evaporation step
        # - Return best path found
        
        path = self._simple_path_placeholder(start, goal)
        self._deposit_pheromones(path)
        
        return path
    
    def _simple_path_placeholder(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Placeholder for simple path generation."""
        return [start, goal]
    
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
        if self.pheromone_grid is None or not path:
            return
        
        # Calculate deposit amount (could be based on path quality)
        path_length = len(path)
        deposit_per_cell = self.deposit_strength / path_length if path_length > 0 else 0
        
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
        
        # Combine pheromone and heuristic
        probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
        
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
            "pheromone_initialized": self.pheromone_grid is not None,
        })
        return data

