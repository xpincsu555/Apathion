"""
Particle system for visual effects like gold drops, sparkles, and floating text.
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass, field
import random
import pygame


@dataclass
class FloatingText:
    """
    Floating text that moves upward and fades out.
    
    Used for displaying gold gain notifications.
    
    Attributes:
        text: Text content to display
        position: Starting (x, y) position in screen coordinates
        velocity: (vx, vy) velocity vector
        lifetime: Total duration in seconds
        elapsed_time: Time elapsed since creation
        color: RGB color tuple
        font_size: Font size for rendering
    """
    text: str
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, -50.0)  # Move upward
    lifetime: float = 1.5
    elapsed_time: float = 0.0
    color: Tuple[int, int, int] = (255, 215, 0)  # Gold color
    font_size: int = 24
    
    def update(self, delta_time: float) -> bool:
        """
        Update floating text animation.
        
        Args:
            delta_time: Time elapsed since last update
            
        Returns:
            True if still active, False if expired
        """
        self.elapsed_time += delta_time
        
        if self.elapsed_time >= self.lifetime:
            return False
        
        # Update position
        self.position = (
            self.position[0] + self.velocity[0] * delta_time,
            self.position[1] + self.velocity[1] * delta_time
        )
        
        return True
    
    def get_alpha(self) -> int:
        """Get current alpha value based on elapsed time."""
        progress = self.elapsed_time / self.lifetime
        # Fade out in the last 50% of lifetime
        if progress > 0.5:
            fade_progress = (progress - 0.5) / 0.5
            return int(255 * (1.0 - fade_progress))
        return 255


@dataclass
class CoinDrop:
    """
    Coin drop animation at enemy death position.
    
    Attributes:
        position: Current (x, y) position in grid coordinates
        velocity: (vx, vy) velocity vector in grid units per second
        lifetime: Total duration in seconds
        elapsed_time: Time elapsed since creation
        sprite: Pygame surface for the coin image
        jitter_x: Random horizontal jitter amount
    """
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, -2.0)  # Move upward in grid space
    lifetime: float = 0.6
    elapsed_time: float = 0.0
    sprite: Optional[pygame.Surface] = None
    jitter_x: float = 0.0
    
    def __post_init__(self):
        """Initialize with random jitter."""
        # Add random left/right jitter
        self.jitter_x = random.uniform(-0.3, 0.3)
        self.velocity = (self.jitter_x, self.velocity[1])
    
    def update(self, delta_time: float) -> bool:
        """
        Update coin drop animation.
        
        Args:
            delta_time: Time elapsed since last update
            
        Returns:
            True if still active, False if expired
        """
        self.elapsed_time += delta_time
        
        if self.elapsed_time >= self.lifetime:
            return False
        
        # Update position
        self.position = (
            self.position[0] + self.velocity[0] * delta_time,
            self.position[1] + self.velocity[1] * delta_time
        )
        
        return True
    
    def get_alpha(self) -> int:
        """Get current alpha value based on elapsed time."""
        progress = self.elapsed_time / self.lifetime
        # Start fading after 40% of lifetime
        if progress > 0.4:
            fade_progress = (progress - 0.4) / 0.6
            return int(255 * (1.0 - fade_progress))
        return 255


@dataclass
class SparkleParticle:
    """
    Sparkle particle effect around coin drops.
    
    Attributes:
        position: Current (x, y) position in grid coordinates
        velocity: (vx, vy) velocity vector in grid units per second
        lifetime: Total duration in seconds
        elapsed_time: Time elapsed since creation
        size: Particle size in pixels
        color: RGB color tuple
    """
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    lifetime: float = 0.4
    elapsed_time: float = 0.0
    size: float = 3.0
    color: Tuple[int, int, int] = (255, 255, 200)  # Bright yellow-white
    
    def __post_init__(self):
        """Initialize with random velocity."""
        # Random outward velocity
        angle = random.uniform(0, 2 * 3.14159)
        speed = random.uniform(0.5, 1.5)
        self.velocity = (
            speed * random.uniform(-1, 1),
            speed * random.uniform(-1, 1)
        )
    
    def update(self, delta_time: float) -> bool:
        """
        Update sparkle particle animation.
        
        Args:
            delta_time: Time elapsed since last update
            
        Returns:
            True if still active, False if expired
        """
        self.elapsed_time += delta_time
        
        if self.elapsed_time >= self.lifetime:
            return False
        
        # Update position
        self.position = (
            self.position[0] + self.velocity[0] * delta_time,
            self.position[1] + self.velocity[1] * delta_time
        )
        
        return True
    
    def get_alpha(self) -> int:
        """Get current alpha value based on elapsed time."""
        progress = self.elapsed_time / self.lifetime
        return int(255 * (1.0 - progress))
    
    def get_size(self) -> float:
        """Get current size based on elapsed time."""
        progress = self.elapsed_time / self.lifetime
        # Shrink over time
        return self.size * (1.0 - progress * 0.5)


@dataclass
class ParticleSystem:
    """
    Manager for all particle effects in the game.
    
    Attributes:
        floating_texts: List of active floating text effects
        coin_drops: List of active coin drop animations
        sparkles: List of active sparkle particles
    """
    floating_texts: List[FloatingText] = field(default_factory=list)
    coin_drops: List[CoinDrop] = field(default_factory=list)
    sparkles: List[SparkleParticle] = field(default_factory=list)
    
    def update(self, delta_time: float) -> None:
        """
        Update all particles.
        
        Args:
            delta_time: Time elapsed since last update
        """
        # Update floating texts
        self.floating_texts = [
            text for text in self.floating_texts
            if text.update(delta_time)
        ]
        
        # Update coin drops
        self.coin_drops = [
            coin for coin in self.coin_drops
            if coin.update(delta_time)
        ]
        
        # Update sparkles
        self.sparkles = [
            sparkle for sparkle in self.sparkles
            if sparkle.update(delta_time)
        ]
    
    def add_gold_gain(
        self,
        amount: int,
        screen_position: Tuple[float, float]
    ) -> None:
        """
        Add a gold gain floating text effect.
        
        Args:
            amount: Amount of gold gained
            screen_position: Screen (x, y) position to display text
        """
        text = FloatingText(
            text=f"+{amount}",
            position=screen_position,
            color=(255, 215, 0),  # Gold color
            font_size=28
        )
        self.floating_texts.append(text)
    
    def add_coin_drop(
        self,
        grid_position: Tuple[float, float],
        sprite: Optional[pygame.Surface] = None
    ) -> None:
        """
        Add a coin drop animation at enemy death position.
        
        Args:
            grid_position: Grid (x, y) position where enemy died
            sprite: Coin sprite image
        """
        # Position coin slightly above enemy position
        coin_pos = (grid_position[0], grid_position[1] - 0.3)
        
        coin = CoinDrop(
            position=coin_pos,
            sprite=sprite
        )
        self.coin_drops.append(coin)
        
        # Add sparkles around the coin
        self._add_sparkles_at_position(coin_pos)
    
    def _add_sparkles_at_position(self, grid_position: Tuple[float, float]) -> None:
        """
        Add sparkle particles around a position.
        
        Args:
            grid_position: Grid (x, y) position for sparkles
        """
        # Create 5-8 sparkle particles
        num_sparkles = random.randint(5, 8)
        
        for _ in range(num_sparkles):
            # Random offset from center
            offset_x = random.uniform(-0.2, 0.2)
            offset_y = random.uniform(-0.2, 0.2)
            
            sparkle = SparkleParticle(
                position=(
                    grid_position[0] + offset_x,
                    grid_position[1] + offset_y
                ),
                velocity=(0.0, 0.0),  # Will be set in __post_init__
                color=random.choice([
                    (255, 255, 200),  # Yellow-white
                    (255, 230, 150),  # Light gold
                    (255, 255, 255),  # White
                ])
            )
            self.sparkles.append(sparkle)
    
    def clear(self) -> None:
        """Clear all particles."""
        self.floating_texts.clear()
        self.coin_drops.clear()
        self.sparkles.clear()

