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
    
    def validate_path(self, path: List[Tuple[int, int]]) -> Tuple[bool, str]:
        """
        Validate that a path is continuous, avoids obstacles, and goes from start to goal.
        
        Args:
            path: List of (x, y) coordinates representing a path
            
        Returns:
            Tuple of (is_valid, error_message). error_message is empty string if valid.
        """
        if not path or len(path) < 2:
            return False, "Path must have at least 2 points"
        
        # Check if path starts at a spawn point
        start = path[0]
        if start not in self.spawn_points:
            return False, f"Path start {start} is not a spawn point. Spawn points: {self.spawn_points}"
        
        # Check if path ends at a goal position
        end = path[-1]
        if end not in self.goal_positions:
            return False, f"Path end {end} is not a goal position. Goal positions: {self.goal_positions}"
        
        # Check each point in the path
        for i, (x, y) in enumerate(path):
            # Check if point is within bounds
            if not (0 <= x < self.width and 0 <= y < self.height):
                return False, f"Point {i} at ({x}, {y}) is out of bounds"
            
            # Check if point is walkable (not an obstacle)
            if not self.is_walkable(x, y):
                return False, f"Point {i} at ({x}, {y}) is an obstacle"
        
        # Check if path is continuous (each point is adjacent to the next)
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # Allow cardinal and diagonal moves (max distance of 1 in each direction)
            if dx > 1 or dy > 1:
                return False, f"Path is not continuous: ({x1}, {y1}) to ({x2}, {y2}) is too far"
        
        return True, ""
    
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
    def create_branching_map(cls, config: Optional[Any] = None) -> "Map":
        """
        Create a map with branching paths (2-3 route choices).
        
        Creates a map with diagonal obstacle stripes (shadow regions) similar to the reference image.
        The obstacles form diagonal bands from upper-left to lower-right, creating multiple
        distinct path options (upper, middle, lower routes).
        
        Args:
            config: Optional MapConfig object with obstacle_regions
        
        Returns:
            New Map instance with branching layout
        """
        width, height = 30, 20
        obstacles = []
        
        # Default obstacle regions (can be overridden by config)
        default_obstacle_regions = [
            (0, 0, 5, 10),
            (5, 0, 17, 8),
            (17, 0, 22, 5),
            (22, 0, 30, 4),
            (0, 12, 15, 20),
            (15, 18, 28, 20),
            (28, 7, 30, 20),
            (17, 11, 26, 16),
            (22, 8, 26, 11),
            (24, 6, 26, 8),
        ]
        
        # Use obstacle regions from config if provided, otherwise use defaults
        if config and hasattr(config, 'obstacle_regions') and config.obstacle_regions:
            obstacle_regions = [tuple(region) for region in config.obstacle_regions]
        else:
            obstacle_regions = default_obstacle_regions
        
        # Generate obstacles for each rectangular region
        for x1, y1, x2, y2 in obstacle_regions:
            for x in range(x1, x2):
                for y in range(y1, y2):
                    if 0 <= x < width and 0 <= y < height:
                        obstacles.append((x, y))
        
        return cls(
            width=width,
            height=height,
            obstacles=obstacles,
            spawn_points=[(0, 11)],
            goal_positions=[(29, 5)],
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

