"""
Base pathfinding module defining the abstract interface for all pathfinding algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

from apathion.game.map import Map
from apathion.game.tower import Tower


class BasePathfinder(ABC):
    """
    Abstract base class for pathfinding algorithms.
    
    All pathfinding algorithms (A*, ACO, DQN) must inherit from this class
    and implement the required methods.
    
    Attributes:
        name: Name of the pathfinding algorithm
        game_map: Reference to the game map
        towers: List of active towers (for damage zone awareness)
    """
    
    def __init__(self, name: str):
        """
        Initialize the pathfinder.
        
        Args:
            name: Name identifier for this pathfinder
        """
        self.name = name
        self.game_map: Optional[Map] = None
        self.towers: List[Tower] = []
    
    @abstractmethod
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs
    ) -> List[Tuple[int, int]]:
        """
        Find a path from start to goal.
        
        Args:
            start: Starting (x, y) grid position
            goal: Goal (x, y) grid position
            **kwargs: Algorithm-specific parameters
            
        Returns:
            List of (x, y) positions forming a path from start to goal
        """
        pass
    
    @abstractmethod
    def update_state(self, game_map: Map, towers: List[Tower]) -> None:
        """
        Update the pathfinder with current game state.
        
        This is called when the game state changes (e.g., tower placement).
        
        Args:
            game_map: Current game map
            towers: List of active towers
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of this pathfinding algorithm.
        
        Returns:
            Algorithm name
        """
        return self.name
    
    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """
        Calculate the total cost of a path.
        
        Args:
            path: List of (x, y) positions
            
        Returns:
            Total path cost (distance-based by default)
        """
        if len(path) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            cost += (dx ** 2 + dy ** 2) ** 0.5
        
        return cost
    
    def get_damage_zones(self) -> List[Dict[str, Any]]:
        """
        Get damage zone information from all towers.
        
        Returns:
            List of damage zone dictionaries
        """
        return [tower.get_damage_zone() for tower in self.towers]
    
    def estimate_damage_at_position(self, position: Tuple[int, int]) -> float:
        """
        Estimate the damage exposure at a given position.
        
        Args:
            position: (x, y) grid position
            
        Returns:
            Estimated damage per second at this position
        """
        total_dps = 0.0
        
        for tower in self.towers:
            tx, ty = tower.position
            px, py = position
            distance = ((tx - px) ** 2 + (ty - py) ** 2) ** 0.5
            
            if distance <= tower.range:
                total_dps += tower.damage * tower.attack_rate
        
        return total_dps
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pathfinder state to dictionary for logging.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "towers_tracked": len(self.towers),
        }

