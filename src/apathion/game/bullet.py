"""
Bullet module for projectile-based tower attacks.
"""

from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Bullet:
    """
    Bullet projectile fired by towers.
    
    Attributes:
        position: Current (x, y) grid position (float coordinates)
        target_enemy_id: ID of the target enemy
        damage: Damage to deal on impact
        speed: Movement speed in grid cells per second
        sprite: Optional sprite surface for rendering
    """
    
    position: Tuple[float, float]
    target_enemy_id: str
    damage: float
    speed: float = 8.0  # grid cells per second
    sprite: Optional[object] = None  # pygame.Surface, but avoid circular import
    
    def update(self, target_position: Tuple[float, float], delta_time: float) -> bool:
        """
        Move the bullet toward its target position.
        
        Args:
            target_position: Current target (x, y) grid position
            delta_time: Time elapsed since last update (seconds)
            
        Returns:
            True if the bullet reached the target
        """
        # Calculate direction and distance
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        
        if distance < 0.2:  # Close enough to hit (0.2 grid cells)
            return True
        
        # Calculate movement amount
        move_distance = self.speed * delta_time
        
        if move_distance >= distance:
            # Reached target
            self.position = target_position
            return True
        else:
            # Move toward target
            ratio = move_distance / distance
            new_x = self.position[0] + dx * ratio
            new_y = self.position[1] + dy * ratio
            self.position = (new_x, new_y)
            return False


@dataclass
class HitEffect:
    """
    Visual effect displayed when a bullet hits an enemy.
    
    Attributes:
        position: (x, y) grid position of the effect
        sprite: Sprite surface for the effect
        duration: How long the effect should be displayed (seconds)
        elapsed: Time elapsed since effect started (seconds)
    """
    
    position: Tuple[float, float]
    sprite: object  # pygame.Surface
    duration: float = 0.3  # seconds
    elapsed: float = 0.0
    
    def update(self, delta_time: float) -> bool:
        """
        Update the effect animation.
        
        Args:
            delta_time: Time elapsed since last update (seconds)
            
        Returns:
            True if the effect is still active, False if it should be removed
        """
        self.elapsed += delta_time
        return self.elapsed < self.duration
    
    def get_alpha(self) -> int:
        """
        Get the current alpha value for fade-out effect.
        
        Returns:
            Alpha value (0-255)
        """
        # Fade out over the duration
        progress = self.elapsed / self.duration
        return int(255 * (1.0 - progress))

