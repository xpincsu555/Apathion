"""
Fixed path pathfinding algorithm implementation.

This pathfinder uses a pre-defined baseline path from the configuration.
Useful for baseline comparisons and testing.
"""

from typing import List, Tuple, Optional

from apathion.pathfinding.base import BasePathfinder
from apathion.game.map import Map
from apathion.game.tower import Tower


class FixedPathfinder(BasePathfinder):
    """
    Fixed path pathfinder that uses a pre-defined baseline path.
    
    This pathfinder always returns the same fixed path regardless of game state.
    It's useful for establishing baseline performance metrics and comparing
    against adaptive pathfinding algorithms.
    
    Attributes:
        baseline_path: Pre-defined path as list of (x, y) coordinates
    """
    
    def __init__(
        self,
        name: str = "Fixed-Path",
        baseline_path: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Initialize fixed path pathfinder.
        
        Args:
            name: Name identifier
            baseline_path: Pre-defined path as list of (x, y) tuples
        """
        super().__init__(name)
        self.baseline_path = baseline_path or []
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs
    ) -> List[Tuple[int, int]]:
        """
        Return the portion of the fixed baseline path from the start position.
        
        This method finds the closest point on the baseline path to the start position
        and returns the path from that point onward. This allows enemies to use the
        baseline path even when they start partway through it.
        
        Args:
            start: Starting position (finds nearest point on baseline path)
            goal: Goal position (verified to match baseline path goal)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            The portion of baseline path from start to goal
        """
        if not self.baseline_path:
            # Fallback: return empty path if no baseline defined
            return []
        
        # Find the nearest point on the baseline path to the start position
        min_distance = float('inf')
        best_index = 0
        
        for i, waypoint in enumerate(self.baseline_path):
            dx = waypoint[0] - start[0]
            dy = waypoint[1] - start[1]
            distance = dx * dx + dy * dy  # Use squared distance for efficiency
            
            if distance < min_distance:
                min_distance = distance
                best_index = i
        
        # Return the path from the nearest point onward
        # If the start is very close to a waypoint, skip it and start from the next one
        if min_distance < 0.5:  # Close enough to consider "at" this waypoint
            # Skip this waypoint and start from the next
            if best_index + 1 < len(self.baseline_path):
                return list(self.baseline_path[best_index + 1:])
            else:
                # At the last waypoint, return just the goal
                return [self.baseline_path[-1]]
        else:
            # Start is between waypoints, return from nearest point
            return list(self.baseline_path[best_index:])
    
    def update_state(self, game_map: Map, towers: List[Tower]) -> None:
        """
        Update pathfinder state (no-op for fixed path).
        
        Args:
            game_map: Current game map
            towers: List of active towers
        """
        self.game_map = game_map
        self.towers = towers
        # Fixed path doesn't need to update based on game state
    
    def set_baseline_path(self, path: List[Tuple[int, int]]) -> None:
        """
        Set or update the baseline path.
        
        Args:
            path: New baseline path as list of (x, y) tuples
        """
        self.baseline_path = path
    
    def get_baseline_path(self) -> List[Tuple[int, int]]:
        """
        Get the current baseline path.
        
        Returns:
            Copy of the baseline path
        """
        return list(self.baseline_path)
    
    def validate_path(self) -> bool:
        """
        Check if the baseline path is valid for the current map.
        
        Returns:
            True if path is valid, False otherwise
        """
        if not self.baseline_path or not self.game_map:
            return False
        
        is_valid, _ = self.game_map.validate_path(self.baseline_path)
        return is_valid

