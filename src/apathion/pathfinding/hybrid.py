"""
Hybrid pathfinding system combining DQN leaders with A* followers.

This implementation reduces computational cost by having only a subset of enemies
(leaders) use DQN inference, while the majority (followers) use simpler A* logic
to follow their nearest leader.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from apathion.pathfinding.base import BasePathfinder
from apathion.pathfinding.dqn import DQNPathfinder
from apathion.pathfinding.astar import AStarPathfinder
from apathion.game.map import Map
from apathion.game.tower import Tower
from apathion.game.enemy import Enemy, EnemyType


class HybridPathfinder(BasePathfinder):
    """
    Hybrid pathfinder that delegates to DQN for leaders and A* for followers.
    
    This system optimizes performance by:
    - Using DQN for 5-10 leader enemies per wave (strategic pathfinding)
    - Using A* for remaining followers (tactical following behavior)
    - Followers navigate toward their nearest leader's position
    
    Attributes:
        dqn_pathfinder: DQN pathfinder for leaders
        astar_pathfinder: A* pathfinder for followers
        leader_positions: Dict tracking current leader positions
        leaders_per_wave: Number of leaders to designate per wave
    """
    
    def __init__(
        self,
        name: str = "Hybrid-DQN-Leader",
        model_path: Optional[str] = None,
        leaders_per_wave: int = 5,
        dqn_cache_duration: int = 5,
        use_enhanced_astar: bool = True,
    ):
        """
        Initialize hybrid pathfinder.
        
        Args:
            name: Name identifier
            model_path: Path to trained DQN model for leaders
            leaders_per_wave: Number of leaders (5-10 recommended)
            dqn_cache_duration: Frames to cache DQN decisions
            use_enhanced_astar: Whether followers use enhanced A* with damage costs
        """
        super().__init__(name)
        
        # Create sub-pathfinders
        self.dqn_pathfinder = DQNPathfinder(
            name="DQN-Leader",
            model_path=model_path,
            cache_duration=dqn_cache_duration,
        )
        
        self.astar_pathfinder = AStarPathfinder(
            name="A*-Follower",
            use_enhanced=use_enhanced_astar,
        )
        
        self.leaders_per_wave = leaders_per_wave
        self.leader_positions: Dict[str, Tuple[int, int]] = {}
        
        # Track enemy designations
        self.designated_leaders: set = set()
        self.wave_enemy_count: Dict[int, int] = {}
        self.current_wave: int = 0
    
    def update_state(self, game_map: Map, towers: List[Tower]) -> None:
        """
        Update both sub-pathfinders with current game state.
        
        Args:
            game_map: Current game map
            towers: List of active towers
        """
        self.game_map = game_map
        self.towers = towers
        
        # Update both pathfinders
        self.dqn_pathfinder.update_state(game_map, towers)
        self.astar_pathfinder.update_state(game_map, towers)
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        **kwargs
    ) -> List[Tuple[int, int]]:
        """
        Find path using hybrid strategy.
        
        For leaders: Use DQN for strategic pathfinding
        For followers: Use A* to follow nearest leader
        
        Args:
            start: Starting position
            goal: Goal position
            **kwargs: Must include 'enemy_id' and optionally 'enemy_type', 'wave'
            
        Returns:
            List of positions forming the path
        """
        enemy_id = kwargs.get("enemy_id", "default")
        enemy_type = kwargs.get("enemy_type", EnemyType.NORMAL)
        wave = kwargs.get("wave", self.current_wave)
        
        # Update wave tracking
        if wave != self.current_wave:
            self.current_wave = wave
            self.wave_enemy_count[wave] = 0
        
        # Determine if this enemy should be a leader
        is_leader = self._should_be_leader(enemy_id, enemy_type, wave)
        
        if is_leader:
            # Leaders use DQN
            path = self.dqn_pathfinder.find_path(start, goal, **kwargs)
            
            # Track leader position for followers
            if len(path) > 0:
                self.leader_positions[enemy_id] = path[-1] if len(path) > 1 else start
        else:
            # Followers use A* toward nearest leader or goal
            target = self._get_follower_target(start, goal)
            path = self.astar_pathfinder.find_path(start, target, **kwargs)
        
        return path
    
    def _should_be_leader(
        self,
        enemy_id: str,
        enemy_type: EnemyType,
        wave: int
    ) -> bool:
        """
        Determine if an enemy should be designated as a leader.
        
        Args:
            enemy_id: Unique enemy identifier
            enemy_type: Type of enemy
            wave: Current wave number
            
        Returns:
            True if enemy should be a leader
        """
        # Already designated as leader
        if enemy_id in self.designated_leaders:
            return True
        
        # Explicit LEADER type
        if enemy_type == EnemyType.LEADER:
            self.designated_leaders.add(enemy_id)
            return True
        
        # Check if wave still needs leaders
        if wave not in self.wave_enemy_count:
            self.wave_enemy_count[wave] = 0
        
        wave_leaders = sum(
            1 for eid in self.designated_leaders 
            if eid.startswith(f"wave_{wave}_")
        )
        
        if wave_leaders < self.leaders_per_wave:
            # Designate this enemy as a leader
            self.designated_leaders.add(enemy_id)
            return True
        
        return False
    
    def _get_follower_target(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Get target position for a follower enemy.
        
        Followers navigate toward their nearest leader. If no leaders exist
        or all leaders are behind the follower, navigate directly to goal.
        
        Args:
            start: Follower's current position
            goal: Final goal position
            
        Returns:
            Target position (leader position or goal)
        """
        if not self.leader_positions:
            return goal
        
        # Find nearest leader ahead of follower
        min_distance = float('inf')
        nearest_leader = None
        
        for leader_id, leader_pos in self.leader_positions.items():
            # Check if leader is ahead (closer to goal)
            leader_to_goal = self._manhattan_distance(leader_pos, goal)
            follower_to_goal = self._manhattan_distance(start, goal)
            
            if leader_to_goal < follower_to_goal:
                distance = self._manhattan_distance(start, leader_pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_leader = leader_pos
        
        # If found a leader ahead, follow them; otherwise go to goal
        return nearest_leader if nearest_leader else goal
    
    def _manhattan_distance(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def update_leader_position(self, enemy_id: str, position: Tuple[int, int]) -> None:
        """
        Update a leader's position for follower tracking.
        
        Args:
            enemy_id: Leader enemy identifier
            position: Current position
        """
        if enemy_id in self.designated_leaders:
            self.leader_positions[enemy_id] = position
    
    def remove_leader(self, enemy_id: str) -> None:
        """
        Remove a leader from tracking (e.g., when defeated or reached goal).
        
        Args:
            enemy_id: Leader enemy identifier
        """
        self.designated_leaders.discard(enemy_id)
        self.leader_positions.pop(enemy_id, None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = super().to_dict()
        data.update({
            "leaders_per_wave": self.leaders_per_wave,
            "active_leaders": len(self.leader_positions),
            "total_designated_leaders": len(self.designated_leaders),
            "current_wave": self.current_wave,
        })
        return data

