"""
Map module for grid-based game environment representation.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class Map:
    """
    Grid-based map representation for the tower defense game.
    
    The map defines walkable areas, obstacles, spawn points, and goal positions
    for enemies to navigate.
    
    Attributes:
        width: Width of the grid in cells
        height: Height of the grid in cells
        grid: 2D numpy array representing the map (0=walkable, 1=obstacle)
        spawn_points: List of (x, y) coordinates where enemies spawn
        goal_positions: List of (x, y) coordinates enemies try to reach
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        spawn_points: Optional[List[Tuple[int, int]]] = None,
        goal_positions: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Initialize a new map.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            obstacles: List of (x, y) coordinates marked as obstacles
            spawn_points: List of (x, y) coordinates where enemies spawn
            goal_positions: List of (x, y) coordinates for goals
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        
        # Set obstacles
        if obstacles:
            for x, y in obstacles:
                if 0 <= x < width and 0 <= y < height:
                    self.grid[y, x] = 1
        
        # Default spawn and goal if not provided
        self.spawn_points = spawn_points or [(0, height // 2)]
        self.goal_positions = goal_positions or [(width - 1, height // 2)]
    
    def is_walkable(self, x: int, y: int) -> bool:
        """
        Check if a grid position is walkable.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if the position is within bounds and not an obstacle
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self.grid[y, x] == 0
    
    def get_neighbors(self, x: int, y: int, diagonal: bool = True) -> List[Tuple[int, int]]:
        """
        Get walkable neighboring cells of a position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            diagonal: Whether to include diagonal neighbors
            
        Returns:
            List of (x, y) coordinates of walkable neighbors
        """
        neighbors = []
        
        # Cardinal directions
        directions = [
            (0, -1),  # North
            (1, 0),   # East
            (0, 1),   # South
            (-1, 0),  # West
        ]
        
        # Add diagonal directions if requested
        if diagonal:
            directions.extend([
                (1, -1),  # Northeast
                (1, 1),   # Southeast
                (-1, 1),  # Southwest
                (-1, -1), # Northwest
            ])
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def add_obstacle(self, x: int, y: int) -> bool:
        """
        Add an obstacle at the specified position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if obstacle was added successfully
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1
            return True
        return False
    
    def remove_obstacle(self, x: int, y: int) -> bool:
        """
        Remove an obstacle at the specified position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if obstacle was removed successfully
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 0
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert map to a dictionary for logging.
        
        Returns:
            Dictionary representation of the map
        """
        return {
            "width": self.width,
            "height": self.height,
            "obstacles": np.argwhere(self.grid == 1).tolist(),
            "spawn_points": self.spawn_points,
            "goal_positions": self.goal_positions,
        }
    
    @classmethod
    def create_simple_map(cls, width: int = 30, height: int = 20) -> "Map":
        """
        Create a simple open map with minimal obstacles.
        
        Args:
            width: Width of the map
            height: Height of the map
            
        Returns:
            New Map instance
        """
        return cls(
            width=width,
            height=height,
            spawn_points=[(0, height // 2)],
            goal_positions=[(width - 1, height // 2)],
        )
    
    @classmethod
    def create_branching_map(cls) -> "Map":
        """
        Create a map with branching paths (2-3 route choices).
        
        Returns:
            New Map instance with branching layout
        """
        width, height = 30, 20
        obstacles = []
        
        # Create walls to force branching
        # Placeholder for actual branching logic
        for y in range(5, 15):
            obstacles.append((width // 2, y))
        
        return cls(
            width=width,
            height=height,
            obstacles=obstacles,
            spawn_points=[(0, height // 2)],
            goal_positions=[(width - 1, height // 2)],
        )
    
    @classmethod
    def create_open_arena(cls) -> "Map":
        """
        Create an open arena with maximum routing freedom.
        
        Returns:
            New Map instance with minimal obstacles
        """
        width, height = 40, 30
        return cls(
            width=width,
            height=height,
            spawn_points=[(0, height // 2)],
            goal_positions=[(width - 1, height // 2)],
        )

