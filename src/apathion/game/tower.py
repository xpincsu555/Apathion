"""
Tower module for tower entity management and attack logic.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time


class TowerType(Enum):
    """Enum for different tower types."""
    BASIC = "basic"
    SNIPER = "sniper"
    RAPID = "rapid"
    AREA = "area"


@dataclass
class Tower:
    """
    Tower entity that attacks enemies within range.
    
    Attributes:
        id: Unique identifier for the tower
        position: (x, y) grid position
        tower_type: Type of tower (affects stats)
        damage: Damage dealt per attack
        range: Attack range in grid cells
        attack_rate: Attacks per second
        last_attack_time: Timestamp of last attack
        total_damage_dealt: Total damage dealt by this tower
        kills: Number of enemies eliminated
        cost: Gold cost to purchase this tower
    """
    
    id: str
    position: Tuple[int, int]
    tower_type: TowerType = TowerType.BASIC
    damage: float = 20.0
    range: float = 3.0
    attack_rate: float = 1.0  # attacks per second
    last_attack_time: float = 0.0
    total_damage_dealt: float = 0.0
    kills: int = 0
    cost: int = 2
    
    def can_attack(self, target_position: Tuple[float, float], current_time: float) -> bool:
        """
        Check if the tower can attack a target at the given position.
        
        Args:
            target_position: (x, y) position of potential target
            current_time: Current game time in seconds
            
        Returns:
            True if target is in range and attack cooldown is ready
        """
        # Check if cooldown is ready
        time_since_last_attack = current_time - self.last_attack_time
        if time_since_last_attack < (1.0 / self.attack_rate):
            return False
        
        # Check if target is in range
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        
        return distance <= self.range
    
    def attack(self, current_time: float) -> float:
        """
        Perform an attack.
        
        Args:
            current_time: Current game time in seconds
            
        Returns:
            Damage dealt by this attack
        """
        self.last_attack_time = current_time
        self.total_damage_dealt += self.damage
        return self.damage
    
    def record_kill(self) -> None:
        """Record that this tower eliminated an enemy."""
        self.kills += 1
    
    def get_damage_zone(self) -> Dict[str, Any]:
        """
        Get the damage zone information for pathfinding algorithms.
        
        Returns:
            Dictionary with position, range, and damage information
        """
        return {
            "position": self.position,
            "range": self.range,
            "damage": self.damage,
            "attack_rate": self.attack_rate,
            "dps": self.damage * self.attack_rate,  # Damage per second
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tower to a dictionary for logging.
        
        Returns:
            Dictionary representation of the tower
        """
        return {
            "id": self.id,
            "position": self.position,
            "type": self.tower_type.value,
            "damage": self.damage,
            "range": self.range,
            "attack_rate": self.attack_rate,
            "total_damage_dealt": self.total_damage_dealt,
            "kills": self.kills,
        }
    
    @classmethod
    def create_basic(cls, tower_id: str, position: Tuple[int, int]) -> "Tower":
        """Create a basic balanced tower."""
        return cls(
            id=tower_id,
            position=position,
            tower_type=TowerType.BASIC,
            damage=20.0,
            range=3.0,
            attack_rate=1.0,
            cost=8,
        )
    
    @classmethod
    def create_sniper(cls, tower_id: str, position: Tuple[int, int]) -> "Tower":
        """Create a sniper tower with high damage and range."""
        return cls(
            id=tower_id,
            position=position,
            tower_type=TowerType.SNIPER,
            damage=50.0,
            range=6.0,
            attack_rate=0.5,
            cost=25,
        )
    
    @classmethod
    def create_rapid(cls, tower_id: str, position: Tuple[int, int]) -> "Tower":
        """Create a rapid-fire tower."""
        return cls(
            id=tower_id,
            position=position,
            tower_type=TowerType.RAPID,
            damage=10.0,
            range=2.5,
            attack_rate=3.0,
            cost=11,
        )
    
    @classmethod
    def create_area(cls, tower_id: str, position: Tuple[int, int]) -> "Tower":
        """Create an area-effect tower."""
        return cls(
            id=tower_id,
            position=position,
            tower_type=TowerType.AREA,
            damage=15.0,
            range=4.0,
            attack_rate=0.75,
            cost=14,
        )

