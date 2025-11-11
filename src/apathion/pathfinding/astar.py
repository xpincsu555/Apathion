"""
A* pathfinding algorithm implementation with enhanced cost functions.
"""

from typing import List, Tuple, Dict, Any, Optional, Set
import heapq
import math

from apathion.pathfinding.base import BasePathfinder
from apathion.game.map import Map
from apathion.game.tower import Tower


class AStarPathfinder(BasePathfinder):
    """
    A* pathfinding with basic and enhanced variants.
    
    This implementation provides two modes:
    - Basic A*: Uses only g(n) and h(n) costs
    - Enhanced A*: Incorporates damage and congestion costs
    
    Attributes:
        alpha: Weight for damage cost component
        beta: Weight for congestion cost component
        diagonal_movement: Whether diagonal movement is allowed
        use_enhanced: Whether to use enhanced cost function
    """
    
    def __init__(
        self,
        name: str = "A*-Enhanced",
        alpha: float = 0.5,
        beta: float = 0.3,
        diagonal_movement: bool = True,
        use_enhanced: bool = True,
    ):
        """
        Initialize A* pathfinder.
        
        Args:
            name: Name identifier
            alpha: Weight for damage cost (0.0 = ignore damage)
            beta: Weight for congestion cost (0.0 = ignore congestion)
            diagonal_movement: Allow diagonal movement
            use_enhanced: Use enhanced cost function (damage + congestion)
        """
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta
        self.diagonal_movement = diagonal_movement
        self.use_enhanced = use_enhanced
        self.congestion_map: Dict[Tuple[int, int], float] = {}
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs
    ) -> List[Tuple[int, int]]:
        """
        Find path using A* algorithm.
        
        Args:
            start: Starting position
            goal: Goal position
            **kwargs: Optional parameters:
                - use_enhanced: Override default enhanced mode (bool)
                - alpha: Override damage weight (float)
                - beta: Override congestion weight (float)
            
        Returns:
            List of positions from start to goal
        """
        if self.game_map is None:
            return [start, goal]
        
        # Allow runtime override of enhanced mode
        use_enhanced = kwargs.get('use_enhanced', self.use_enhanced)
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)
        
        # Initialize data structures
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0  # Tiebreaker for heap
        heapq.heappush(open_set, (0.0, counter, start))
        counter += 1
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            # Goal reached
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Skip if not walkable
                if not self._is_walkable(neighbor):
                    continue
                
                # Calculate movement cost
                movement_cost = self._calculate_movement_cost(current, neighbor)
                
                # Add enhanced costs if enabled
                if use_enhanced:
                    if alpha > 0:
                        damage = self.estimate_damage_at_position(neighbor)
                        movement_cost += alpha * damage
                    if beta > 0:
                        congestion = self.congestion_map.get(neighbor, 0.0)
                        movement_cost += beta * congestion
                
                tentative_g_score = g_score[current] + movement_cost
                
                # Update if better path found
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.calculate_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1
        
        # No path found, return straight line
        return [start, goal]
    
    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct path from start to goal.
        
        Args:
            came_from: Parent mapping
            current: Current (goal) position
            
        Returns:
            List of positions from start to goal
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighbor positions.
        
        Args:
            pos: Current position
            
        Returns:
            List of neighbor positions
        """
        x, y = pos
        neighbors = []
        
        # Cardinal directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbors.append((x + dx, y + dy))
        
        # Diagonal directions
        if self.diagonal_movement:
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbors.append((x + dx, y + dy))
        
        return neighbors
    
    def _is_walkable(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is walkable.
        
        Args:
            pos: Position to check
            
        Returns:
            True if walkable, False otherwise
        """
        if self.game_map is None:
            return True
        
        x, y = pos
        
        # Check bounds
        if x < 0 or y < 0 or x >= self.game_map.width or y >= self.game_map.height:
            return False
        
        # Check walkability
        return self.game_map.is_walkable(x, y)
    
    def _calculate_movement_cost(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int]
    ) -> float:
        """
        Calculate base movement cost between positions.
        
        Args:
            from_pos: Starting position
            to_pos: Destination position
            
        Returns:
            Movement cost (Euclidean distance)
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def update_state(self, game_map: Map, towers: List[Tower]) -> None:
        """
        Update map and tower information.
        
        Args:
            game_map: Current game map
            towers: List of active towers
        """
        self.game_map = game_map
        self.towers = towers
        self._update_congestion_map()
    
    def _update_congestion_map(self) -> None:
        """
        Update congestion metrics based on current game state.
        
        PLACEHOLDER: This would track enemy density in different areas.
        """
        # TODO: Implement congestion tracking
        # - Track number of enemies per grid cell
        # - Apply smoothing/decay over time
        # - Use for pathfinding cost calculation
        self.congestion_map.clear()
    
    def calculate_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate heuristic distance to goal using Euclidean distance.
        
        Args:
            pos: Current position
            goal: Goal position
            
        Returns:
            Estimated distance to goal (Euclidean)
        """
        dx = pos[0] - goal[0]
        dy = pos[1] - goal[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def calculate_composite_cost(
        self,
        position: Tuple[int, int],
        base_cost: float
    ) -> float:
        """
        Calculate composite cost incorporating damage and congestion.
        
        Args:
            position: Grid position
            base_cost: Base movement cost
            
        Returns:
            Total cost including all factors
        """
        # Base geometric cost
        total_cost = base_cost
        
        # Add damage cost
        if self.alpha > 0:
            damage = self.estimate_damage_at_position(position)
            total_cost += self.alpha * damage
        
        # Add congestion cost
        if self.beta > 0:
            congestion = self.congestion_map.get(position, 0.0)
            total_cost += self.beta * congestion
        
        return total_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = super().to_dict()
        data.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "diagonal_movement": self.diagonal_movement,
            "use_enhanced": self.use_enhanced,
            "congestion_cells": len(self.congestion_map),
        })
        return data

