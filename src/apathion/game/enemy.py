"""
Enemy module for enemy entity management and behavior.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class EnemyType(Enum):
    """Enum for different enemy types."""
    NORMAL = "normal"
    FAST = "fast"
    TANK = "tank"
    LEADER = "leader"


@dataclass
class Enemy:
    """
    Enemy entity with position, health, and movement capabilities.
    
    Attributes:
        id: Unique identifier for the enemy
        position: Current (x, y) grid position
        enemy_type: Type of enemy (affects stats)
        health: Current health points
        max_health: Maximum health points
        speed: Movement speed (cells per second)
        damage_taken: Total damage taken so far
        path_history: List of positions the enemy has visited
        current_path: Planned path as list of positions
        is_alive: Whether the enemy is still active
        reached_goal: Whether the enemy reached the goal
    """
    
    id: str
    position: Tuple[float, float]
    enemy_type: EnemyType = EnemyType.NORMAL
    health: float = 100.0
    max_health: float = 100.0
    speed: float = 1.0
    damage_taken: float = 0.0
    path_history: List[Tuple[float, float]] = field(default_factory=list)
    current_path: List[Tuple[int, int]] = field(default_factory=list)
    is_alive: bool = True
    reached_goal: bool = False
    
    def __post_init__(self):
        """Initialize path history with starting position."""
        if not self.path_history:
            self.path_history.append(self.position)
    
    def move(self, target_position: Tuple[float, float], delta_time: float) -> bool:
        """
        Move the enemy toward a target position.
        
        Args:
            target_position: Target (x, y) position to move toward
            delta_time: Time elapsed since last update (seconds)
            
        Returns:
            True if the enemy reached the target position
        """
        if not self.is_alive:
            return False
        
        # Calculate direction and distance
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        
        if distance < 0.01:  # Already at target
            return True
        
        # Calculate movement amount
        move_distance = self.speed * delta_time
        
        if move_distance >= distance:
            # Reached target
            self.position = (float(target_position[0]), float(target_position[1]))
            self.path_history.append(self.position)
            return True
        else:
            # Move toward target
            ratio = move_distance / distance
            new_x = self.position[0] + dx * ratio
            new_y = self.position[1] + dy * ratio
            self.position = (new_x, new_y)
            return False
    
    def take_damage(self, damage: float) -> bool:
        """
        Apply damage to the enemy.
        
        Args:
            damage: Amount of damage to apply
            
        Returns:
            True if the enemy is still alive after damage
        """
        if not self.is_alive:
            return False
        
        self.health -= damage
        self.damage_taken += damage
        
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            return False
        
        return True
    
    def set_path(self, path: List[Tuple[int, int]]) -> None:
        """
        Set a new path for the enemy to follow.
        
        Args:
            path: List of (x, y) grid positions to follow
        """
        self.current_path = path.copy()
    
    def get_next_waypoint(self) -> Optional[Tuple[int, int]]:
        """
        Get the next waypoint in the current path.
        
        Returns:
            Next (x, y) position in the path, or None if no path
        """
        if self.current_path:
            return self.current_path[0]
        return None
    
    def advance_waypoint(self) -> None:
        """Remove the first waypoint from the current path."""
        if self.current_path:
            self.current_path.pop(0)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the enemy for AI input.
        
        Returns:
            Dictionary containing enemy state information
        """
        return {
            "id": self.id,
            "position": self.position,
            "type": self.enemy_type.value,
            "health": self.health,
            "max_health": self.max_health,
            "health_ratio": self.health / self.max_health if self.max_health > 0 else 0,
            "speed": self.speed,
            "damage_taken": self.damage_taken,
            "is_alive": self.is_alive,
            "reached_goal": self.reached_goal,
            "path_length": len(self.current_path),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert enemy to a dictionary for logging.
        
        Returns:
            Dictionary representation of the enemy
        """
        return {
            "id": self.id,
            "position": self.position,
            "type": self.enemy_type.value,
            "health": self.health,
            "max_health": self.max_health,
            "speed": self.speed,
            "damage_taken": self.damage_taken,
            "is_alive": self.is_alive,
            "reached_goal": self.reached_goal,
            "path_history_length": len(self.path_history),
        }
    
    @classmethod
    def create_normal(cls, enemy_id: str, position: Tuple[float, float]) -> "Enemy":
        """Create a normal enemy."""
        return cls(
            id=enemy_id,
            position=position,
            enemy_type=EnemyType.NORMAL,
            health=100.0,
            max_health=100.0,
            speed=1.0,
        )
    
    @classmethod
    def create_fast(cls, enemy_id: str, position: Tuple[float, float]) -> "Enemy":
        """Create a fast enemy with less health."""
        return cls(
            id=enemy_id,
            position=position,
            enemy_type=EnemyType.FAST,
            health=60.0,
            max_health=60.0,
            speed=2.0,
        )
    
    @classmethod
    def create_tank(cls, enemy_id: str, position: Tuple[float, float]) -> "Enemy":
        """Create a tank enemy with high health."""
        return cls(
            id=enemy_id,
            position=position,
            enemy_type=EnemyType.TANK,
            health=200.0,
            max_health=200.0,
            speed=0.5,
        )

