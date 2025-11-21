"""
Pygame renderer for tower defense game visualization.
"""

from typing import Tuple, Optional, List, Dict
from enum import Enum
import pygame
import numpy as np
import os

from apathion.game.game import GameState
from apathion.game.enemy import Enemy, EnemyType
from apathion.game.tower import Tower, TowerType
from apathion.pathfinding.base import BasePathfinder
from apathion.pathfinding.aco import ACOPathfinder
from apathion.config import VisualizationConfig


class VisualizationMode(Enum):
    """Enum for different visualization modes."""
    MINIMAL = "minimal"
    NORMAL = "normal"
    DEBUG = "debug"


class GameRenderer:
    """
    Pygame renderer for the tower defense game.
    
    Handles all drawing operations including map, entities, UI, and debug info.
    
    Attributes:
        screen: Pygame display surface
        config: Visualization configuration
        mode: Current visualization mode
        cell_size: Size of each grid cell in pixels
        offset_x: X offset for centering the grid
        offset_y: Y offset for centering the grid
    """
    
    def __init__(
        self,
        screen: pygame.Surface,
        config: VisualizationConfig,
        map_width: int,
        map_height: int,
    ):
        """
        Initialize the renderer.
        
        Args:
            screen: Pygame display surface
            config: Visualization configuration
            map_width: Width of the game map in cells
            map_height: Height of the game map in cells
        """
        self.screen = screen
        self.config = config
        self.mode = VisualizationMode.NORMAL
        
        # Calculate cell size and offsets for centering
        available_width = config.window_width - 40  # Margins
        available_height = config.window_height - 100  # Margins + UI space
        
        self.cell_size = min(
            available_width // map_width,
            available_height // map_height
        )
        
        grid_width = map_width * self.cell_size
        grid_height = map_height * self.cell_size
        
        self.offset_x = (config.window_width - grid_width) // 2
        self.offset_y = (config.window_height - grid_height) // 2 + 40  # Leave space for stats
        
        # Colors
        self.COLOR_BACKGROUND = (30, 30, 40)
        self.COLOR_GRID = (60, 60, 70)
        self.COLOR_WALKABLE = (50, 50, 60)
        self.COLOR_OBSTACLE = (20, 20, 25)
        self.COLOR_SPAWN = (0, 200, 0)
        self.COLOR_GOAL = (200, 0, 0)
        self.COLOR_TOWER = (50, 100, 255)
        self.COLOR_TOWER_RANGE = (50, 100, 255, 30)
        self.COLOR_ENEMY_HEALTHY = (255, 150, 0)
        self.COLOR_ENEMY_DAMAGED = (255, 50, 50)
        self.COLOR_HEALTH_BG = (100, 100, 100)
        self.COLOR_HEALTH_FG = (0, 255, 0)
        self.COLOR_PATH = (200, 200, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)
        
        # Font
        pygame.font.init()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)
        
        # Load enemy sprites
        self.enemy_sprites: Dict[EnemyType, pygame.Surface] = {}
        self._load_enemy_sprites()
        
        # Load tower sprites
        self.tower_sprites: Dict[TowerType, pygame.Surface] = {}
        self._load_tower_sprites()
        
        # Load hit effect sprite
        self.blood_sprite: Optional[pygame.Surface] = None
        self._load_hit_effect_sprite()
        
        # Load palm tree decoration sprite
        self.palmtree_sprite: Optional[pygame.Surface] = None
        self._load_palmtree_sprite()
    
    def toggle_mode(self) -> None:
        """Toggle between visualization modes."""
        modes = list(VisualizationMode)
        current_idx = modes.index(self.mode)
        next_idx = (current_idx + 1) % len(modes)
        self.mode = modes[next_idx]
    
    def _auto_crop_sprite(self, sprite: pygame.Surface) -> pygame.Surface:
        """
        Auto-crop a sprite to remove transparent borders and background padding.
        
        This function detects the character by finding pixels that are significantly
        different from the background/border colors.
        
        Args:
            sprite: Pygame surface with alpha channel
            
        Returns:
            Cropped sprite with only visible character pixels
        """
        width, height = sprite.get_size()
        
        # Sample border colors to identify background
        # Check corners and edges to find the typical background color
        sprite.lock()
        
        border_samples = []
        edge_width = min(50, width // 10)  # Sample edge region
        
        # Sample top and bottom edges
        for x in range(0, width, 10):
            for y in [0, height - 1]:
                pixel = sprite.get_at((x, y))
                if pixel.a > 200:  # Only consider opaque pixels
                    border_samples.append((pixel.r, pixel.g, pixel.b))
        
        # Sample left and right edges
        for y in range(0, height, 10):
            for x in [0, width - 1]:
                pixel = sprite.get_at((x, y))
                if pixel.a > 200:  # Only consider opaque pixels
                    border_samples.append((pixel.r, pixel.g, pixel.b))
        
        # Calculate average border color
        if border_samples:
            avg_r = sum(c[0] for c in border_samples) / len(border_samples)
            avg_g = sum(c[1] for c in border_samples) / len(border_samples)
            avg_b = sum(c[2] for c in border_samples) / len(border_samples)
            background_color = (avg_r, avg_g, avg_b)
        else:
            background_color = None
        
        # Find the bounds of character pixels (pixels significantly different from background)
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        
        color_threshold = 75  # How different from background to be considered "character"
        
        for y in range(height):
            for x in range(width):
                pixel = sprite.get_at((x, y))
                
                # First check: pixel must not be transparent
                if pixel.a <= 10:
                    continue
                
                # Second check: if we have a background color, pixel must differ from it
                is_character_pixel = False
                
                if background_color:
                    # Calculate color distance from background
                    color_diff = abs(pixel.r - background_color[0]) + \
                                 abs(pixel.g - background_color[1]) + \
                                 abs(pixel.b - background_color[2])
                    
                    # If significantly different from background, it's character
                    if color_diff > color_threshold:
                        is_character_pixel = True
                else:
                    # No background detected, use alpha only
                    if pixel.a > 10:
                        is_character_pixel = True
                
                if is_character_pixel:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        
        sprite.unlock()
        
        # If no visible pixels found, return original
        if min_x >= max_x or min_y >= max_y:
            return sprite
        
        # Add a small padding to avoid cutting too tight
        padding = 2
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(width - 1, max_x + padding)
        max_y = min(height - 1, max_y + padding)
        
        # Calculate crop rectangle
        crop_width = max_x - min_x + 1
        crop_height = max_y - min_y + 1
        
        # Create cropped surface
        cropped = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
        cropped.blit(sprite, (0, 0), (min_x, min_y, crop_width, crop_height))
        
        return cropped
    
    def _load_enemy_sprites(self) -> None:
        """Load and scale enemy sprite images."""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, "..", "assets", "enemies")
        
        # Map enemy types to sprite filenames
        sprite_files = {
            EnemyType.NORMAL: "blue_transparent.png",
            EnemyType.FAST: "red_transparent.png",
            EnemyType.TANK: "green_transparent.png",
            EnemyType.LEADER: "red_transparent.png",  # Use red for leader as well
        }
        
        # Calculate desired sprite size (slightly smaller than cell to leave room for health bars)
        sprite_size = int(self.cell_size * 0.8)
        
        # Load and scale each sprite
        for enemy_type, filename in sprite_files.items():
            filepath = os.path.join(assets_dir, filename)
            
            try:
                # Load the sprite image
                sprite = pygame.image.load(filepath).convert_alpha()
                
                # Auto-crop to remove transparent borders
                sprite = self._auto_crop_sprite(sprite)
                
                # Scale to fit cell size while maintaining aspect ratio
                sprite = pygame.transform.scale(sprite, (sprite_size, sprite_size))
                
                # Store the scaled sprite
                self.enemy_sprites[enemy_type] = sprite
                
            except (pygame.error, FileNotFoundError) as e:
                # If sprite loading fails, create a fallback colored surface
                print(f"Warning: Could not load sprite {filepath}: {e}")
                print(f"Using fallback rendering for {enemy_type.value} enemies")
                
                # Create a simple colored rectangle as fallback
                fallback_surface = pygame.Surface((sprite_size, sprite_size), pygame.SRCALPHA)
                fallback_color = {
                    EnemyType.NORMAL: (100, 150, 255),     # Blue
                    EnemyType.FAST: (255, 100, 100),       # Red
                    EnemyType.TANK: (100, 255, 100),       # Green
                    EnemyType.LEADER: (255, 150, 100),     # Orange/Red
                }.get(enemy_type, (200, 200, 200))
                
                pygame.draw.circle(
                    fallback_surface,
                    fallback_color,
                    (sprite_size // 2, sprite_size // 2),
                    sprite_size // 2
                )
                
                self.enemy_sprites[enemy_type] = fallback_surface
    
    def _load_tower_sprites(self) -> None:
        """Load and scale tower sprite images."""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, "..", "assets", "towers")
        
        # Map tower types to sprite filenames
        sprite_files = {
            TowerType.BASIC: "buyucu1.png",
            TowerType.SNIPER: "buyucu2.png",
            TowerType.RAPID: "kale1.png",
            TowerType.AREA: "kuled.png",
        }
        
        # Calculate desired sprite dimensions (1.5 tiles wide, 2 tiles tall)
        sprite_width = int(self.cell_size * 1.5)
        sprite_height = int(self.cell_size * 2)
        
        # Load and scale each sprite
        for tower_type, filename in sprite_files.items():
            filepath = os.path.join(assets_dir, filename)
            
            try:
                # Load the sprite image
                sprite = pygame.image.load(filepath).convert_alpha()
                
                # Auto-crop to remove transparent borders
                sprite = self._auto_crop_sprite(sprite)
                
                # Scale to desired dimensions (1.5 tiles wide, 2 tiles tall)
                sprite = pygame.transform.scale(sprite, (sprite_width, sprite_height))
                
                # Store the scaled sprite
                self.tower_sprites[tower_type] = sprite
                
            except (pygame.error, FileNotFoundError) as e:
                # If sprite loading fails, create a fallback colored surface
                print(f"Warning: Could not load tower sprite {filepath}: {e}")
                print(f"Using fallback rendering for {tower_type.value} towers")
                
                # Create a simple colored rectangle as fallback
                fallback_surface = pygame.Surface((sprite_width, sprite_height), pygame.SRCALPHA)
                fallback_color = {
                    TowerType.BASIC: (80, 150, 255),       # Blue
                    TowerType.SNIPER: (255, 100, 255),     # Purple
                    TowerType.RAPID: (255, 200, 100),      # Orange
                    TowerType.AREA: (100, 255, 150),       # Green
                }.get(tower_type, (150, 150, 150))
                
                pygame.draw.rect(
                    fallback_surface,
                    fallback_color,
                    pygame.Rect(0, 0, sprite_width, sprite_height)
                )
                
                self.tower_sprites[tower_type] = fallback_surface
    
    def _load_hit_effect_sprite(self) -> None:
        """Load the blood splash hit effect sprite."""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, "..", "assets", "towers")
        filepath = os.path.join(assets_dir, "bloodtrail_1.png")
        
        try:
            # Load the blood sprite
            sprite = pygame.image.load(filepath).convert_alpha()
            
            # Scale to a reasonable size (about half a cell)
            effect_size = int(self.cell_size * 0.6)
            sprite = pygame.transform.scale(sprite, (effect_size, effect_size))
            
            self.blood_sprite = sprite
            
        except (pygame.error, FileNotFoundError) as e:
            print(f"Warning: Could not load blood effect sprite {filepath}: {e}")
            print(f"Using fallback red circle for hit effects")
            
            # Create a simple red circle as fallback
            effect_size = int(self.cell_size * 0.6)
            fallback_surface = pygame.Surface((effect_size, effect_size), pygame.SRCALPHA)
            pygame.draw.circle(
                fallback_surface,
                (255, 0, 0, 200),
                (effect_size // 2, effect_size // 2),
                effect_size // 2
            )
            self.blood_sprite = fallback_surface
    
    def _load_palmtree_sprite(self) -> None:
        """Load the palm tree decoration sprite."""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, "..", "assets", "map")
        filepath = os.path.join(assets_dir, "palmtree.png")
        
        try:
            # Load the palm tree sprite
            sprite = pygame.image.load(filepath).convert_alpha()
            
            # Auto-crop to remove transparent borders
            sprite = self._auto_crop_sprite(sprite)
            
            # Scale to fit cell size (slightly larger for visual appeal)
            tree_size = int(self.cell_size * 1.2)
            sprite = pygame.transform.scale(sprite, (tree_size, tree_size))
            
            self.palmtree_sprite = sprite
            
        except (pygame.error, FileNotFoundError) as e:
            print(f"Warning: Could not load palm tree sprite {filepath}: {e}")
            print(f"Map will be displayed without palm tree decorations")
            self.palmtree_sprite = None
    
    def _is_path_tile(self, x: int, y: int, game_state: GameState) -> bool:
        """
        Check if a tile is part of an enemy path.
        
        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            game_state: Current game state
            
        Returns:
            True if tile is part of any enemy's path
        """
        # Check if any enemy has this tile in their current path
        for enemy in game_state.enemies:
            if enemy.current_path and (x, y) in enemy.current_path:
                return True
        
        # Check if any enemy is currently on this tile
        for enemy in game_state.enemies:
            enemy_grid_x = int(enemy.position[0])
            enemy_grid_y = int(enemy.position[1])
            if (enemy_grid_x, enemy_grid_y) == (x, y):
                return True
        
        # Check if tile is spawn or goal
        if (x, y) in game_state.map.spawn_points or (x, y) in game_state.map.goal_positions:
            return True
        
        return False
    
    def _is_tower_at(self, x: int, y: int, game_state: GameState) -> bool:
        """
        Check if a tower exists at this tile.
        
        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            game_state: Current game state
            
        Returns:
            True if a tower exists at this position
        """
        for tower in game_state.towers:
            if tower.position == (x, y):
                return True
        return False
    
    def get_grid_position(self, mouse_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Convert screen position to grid coordinates.
        
        Args:
            mouse_pos: (x, y) screen position
            
        Returns:
            (x, y) grid position or None if outside grid
        """
        mx, my = mouse_pos
        
        # Convert to grid coordinates
        grid_x = (mx - self.offset_x) // self.cell_size
        grid_y = (my - self.offset_y) // self.cell_size
        
        return (grid_x, grid_y)
    
    def render(
        self,
        game_state: GameState,
        pathfinder: Optional[BasePathfinder] = None,
        algorithm_name: str = "Unknown",
        fps: float = 0.0,
    ) -> None:
        """
        Render the complete game state.
        
        Args:
            game_state: Current game state
            pathfinder: Pathfinder instance (for pheromone visualization)
            algorithm_name: Name of the pathfinding algorithm
            fps: Current frames per second
        """
        # Clear screen
        self.screen.fill(self.COLOR_BACKGROUND)
        
        # Draw game elements
        self._draw_map(game_state)
        
        # Draw palm tree decorations (before other elements)
        self._draw_palmtrees(game_state)
        
        # Draw pheromones in debug mode for ACO
        if self.mode == VisualizationMode.DEBUG and isinstance(pathfinder, ACOPathfinder):
            if self.config.show_pheromones:
                self._draw_pheromones(pathfinder)
        
        # Draw tower ranges in normal/debug mode
        if self.mode != VisualizationMode.MINIMAL and self.config.show_tower_ranges:
            self._draw_tower_ranges(game_state)
        
        # Draw entities
        self._draw_towers(game_state)
        self._draw_enemies(game_state)
        self._draw_bullets(game_state)
        self._draw_hit_effects(game_state)
        
        # Draw paths in debug mode
        if self.mode == VisualizationMode.DEBUG and self.config.show_paths:
            self._draw_enemy_paths(game_state)
        
        # Draw UI
        if self.mode != VisualizationMode.MINIMAL:
            self._draw_stats(game_state, algorithm_name, fps)
        
        # Draw mode indicator
        self._draw_mode_indicator()
    
    def _draw_map(self, game_state: GameState) -> None:
        """Draw the game map with grid, obstacles, spawn, and goal."""
        game_map = game_state.map
        
        # Create a set of tower positions for quick lookup
        tower_positions = {tower.position for tower in game_state.towers}
        
        # Draw grid cells
        for y in range(game_map.height):
            for x in range(game_map.width):
                rect = pygame.Rect(
                    self.offset_x + x * self.cell_size,
                    self.offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Determine cell color
                if (x, y) in tower_positions:
                    # Tower cells get a much lighter, distinct background for visibility
                    color = (70, 80, 100)
                elif game_map.grid[y, x] == 1:  # Obstacle (but not a tower)
                    color = self.COLOR_OBSTACLE
                else:  # Walkable
                    color = self.COLOR_WALKABLE
                
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw grid lines if enabled
                if self.config.show_grid:
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        # Draw spawn points
        for spawn in game_map.spawn_points:
            self._draw_special_cell(spawn, self.COLOR_SPAWN, "S")
        
        # Draw goal positions
        for goal in game_map.goal_positions:
            self._draw_special_cell(goal, self.COLOR_GOAL, "G")
    
    def _draw_special_cell(
        self,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        label: str
    ) -> None:
        """Draw a special cell (spawn/goal) with label."""
        x, y = position
        rect = pygame.Rect(
            self.offset_x + x * self.cell_size,
            self.offset_y + y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        
        # Draw colored border
        pygame.draw.rect(self.screen, color, rect, 3)
        
        # Draw label
        text = self.font_small.render(label, True, color)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)
    
    def _draw_palmtrees(self, game_state: GameState) -> None:
        """Draw palm tree decorations on obstacle tiles (not on paths or towers)."""
        if self.palmtree_sprite is None:
            return
        
        game_map = game_state.map
        
        # Draw palm tree on obstacle tiles only (not on walkable paths)
        for y in range(game_map.height):
            for x in range(game_map.width):
                # Only draw on obstacle tiles
                if game_map.grid[y, x] != 1:
                    continue
                
                # Skip if tile has a tower (towers can be placed on obstacles with force=True)
                if self._is_tower_at(x, y, game_state):
                    continue
                
                # Draw palm tree centered on the tile
                screen_x = self.offset_x + x * self.cell_size + self.cell_size // 2
                screen_y = self.offset_y + y * self.cell_size + self.cell_size // 2
                
                tree_rect = self.palmtree_sprite.get_rect()
                tree_rect.center = (screen_x, screen_y)
                self.screen.blit(self.palmtree_sprite, tree_rect)
    
    def _draw_pheromones(self, pathfinder: ACOPathfinder) -> None:
        """Draw pheromone trails for ACO algorithm."""
        if pathfinder.pheromone_grid is None:
            return
        
        pheromone_surface = pygame.Surface(
            (self.config.window_width, self.config.window_height),
            pygame.SRCALPHA
        )
        
        # Find min/max for normalization
        pheromone_grid = pathfinder.pheromone_grid
        min_pheromone = np.min(pheromone_grid)
        max_pheromone = np.max(pheromone_grid)
        pheromone_range = max_pheromone - min_pheromone
        
        if pheromone_range < 0.001:
            return
        
        # Draw pheromone overlay
        for y in range(pheromone_grid.shape[0]):
            for x in range(pheromone_grid.shape[1]):
                pheromone = pheromone_grid[y, x]
                normalized = (pheromone - min_pheromone) / pheromone_range
                
                # Color gradient from yellow (low) to red (high)
                if normalized > 0.1:  # Only draw visible pheromones
                    red = int(255 * normalized)
                    green = int(255 * (1.0 - normalized * 0.7))
                    alpha = int(100 * normalized)
                    
                    rect = pygame.Rect(
                        self.offset_x + x * self.cell_size,
                        self.offset_y + y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    
                    pygame.draw.rect(
                        pheromone_surface,
                        (red, green, 0, alpha),
                        rect
                    )
        
        self.screen.blit(pheromone_surface, (0, 0))
    
    def _draw_tower_ranges(self, game_state: GameState) -> None:
        """Draw tower attack ranges."""
        range_surface = pygame.Surface(
            (self.config.window_width, self.config.window_height),
            pygame.SRCALPHA
        )
        
        for tower in game_state.towers:
            center_x = self.offset_x + int(tower.position[0] * self.cell_size + self.cell_size / 2)
            center_y = self.offset_y + int(tower.position[1] * self.cell_size + self.cell_size / 2)
            radius = int(tower.range * self.cell_size)
            
            pygame.draw.circle(
                range_surface,
                self.COLOR_TOWER_RANGE,
                (center_x, center_y),
                radius
            )
        
        self.screen.blit(range_surface, (0, 0))
    
    def _draw_towers(self, game_state: GameState) -> None:
        """Draw all towers using PNG sprites."""
        for tower in game_state.towers:
            # Get the sprite for this tower type
            sprite = self.tower_sprites.get(tower.tower_type)
            if sprite is None:
                continue
            
            # Calculate position - towers are taller (1.5 tiles), so position them
            # centered horizontally and with bottom aligned to tile bottom
            x = self.offset_x + int(tower.position[0] * self.cell_size)
            y = self.offset_y + int(tower.position[1] * self.cell_size)
            
            # Position sprite: center horizontally, align bottom with tile bottom
            sprite_rect = sprite.get_rect()
            sprite_rect.centerx = x + self.cell_size // 2
            sprite_rect.bottom = y + self.cell_size
            
            # Draw the sprite
            self.screen.blit(sprite, sprite_rect)
    
    def _draw_enemies(self, game_state: GameState) -> None:
        """Draw all enemies using PNG sprites."""
        for enemy in game_state.enemies:
            if not enemy.is_alive:
                continue
            
            # Get the sprite for this enemy type
            sprite = self.enemy_sprites.get(enemy.enemy_type)
            if sprite is None:
                continue
            
            # Calculate position - center the sprite on the enemy's position
            x = self.offset_x + int(enemy.position[0] * self.cell_size)
            y = self.offset_y + int(enemy.position[1] * self.cell_size)
            
            # Center the sprite
            sprite_rect = sprite.get_rect()
            sprite_rect.center = (x + self.cell_size // 2, y + self.cell_size // 2)
            
            # Apply color tint based on health (for visual feedback)
            health_ratio = enemy.health / enemy.max_health
            
            # Create a copy of the sprite to apply color tinting without modifying the original
            tinted_sprite = sprite.copy()
            
            # Apply red tint if enemy is damaged (health below 50%)
            # if health_ratio < 0.5:
            #     # Create a red overlay
            #     red_overlay = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
            #     red_intensity = int(100 * (1.0 - health_ratio * 2))  # More red as health decreases
            #     red_overlay.fill((255, 0, 0, red_intensity))
            #     tinted_sprite.blit(red_overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            
            # Draw the sprite
            self.screen.blit(tinted_sprite, sprite_rect)
            
            # Draw health bar in normal/debug mode
            if self.mode != VisualizationMode.MINIMAL and self.config.show_health_bars:
                self._draw_health_bar(enemy, x, y)
    
    def _draw_health_bar(self, enemy: Enemy, x: int, y: int) -> None:
        """Draw health bar above an enemy."""
        bar_width = self.cell_size - 4
        bar_height = 4
        bar_x = x + 2
        bar_y = y - 6
        
        health_ratio = enemy.health / enemy.max_health
        
        # Background
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
        
        # Foreground
        fg_width = int(bar_width * health_ratio)
        if fg_width > 0:
            fg_rect = pygame.Rect(bar_x, bar_y, fg_width, bar_height)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, fg_rect)
    
    def _draw_bullets(self, game_state: GameState) -> None:
        """Draw all active bullets."""
        for bullet in game_state.bullets:
            # Convert bullet grid position to screen position
            screen_x = self.offset_x + int(bullet.position[0] * self.cell_size)
            screen_y = self.offset_y + int(bullet.position[1] * self.cell_size)
            
            # Draw bullet as a small yellow/orange circle
            bullet_radius = 3
            pygame.draw.circle(
                self.screen,
                (255, 200, 50),  # Yellow-orange color
                (screen_x, screen_y),
                bullet_radius
            )
            
            # Add a darker outline for visibility
            pygame.draw.circle(
                self.screen,
                (200, 150, 0),
                (screen_x, screen_y),
                bullet_radius,
                1
            )
    
    def _draw_hit_effects(self, game_state: GameState) -> None:
        """Draw all active hit effects."""
        if self.blood_sprite is None:
            return
        
        for effect in game_state.hit_effects:
            # Set sprite if not already set
            if effect.sprite is None:
                effect.sprite = self.blood_sprite
            
            # Convert effect grid position to screen position
            screen_x = self.offset_x + int(effect.position[0] * self.cell_size)
            screen_y = self.offset_y + int(effect.position[1] * self.cell_size)
            
            # Create a copy of the sprite for fading
            effect_sprite = effect.sprite.copy()
            
            # Apply fade-out effect based on elapsed time
            alpha = effect.get_alpha()
            effect_sprite.set_alpha(alpha)
            
            # Draw the effect centered on the position
            effect_rect = effect_sprite.get_rect()
            effect_rect.center = (screen_x, screen_y)
            self.screen.blit(effect_sprite, effect_rect)
    
    def _draw_enemy_paths(self, game_state: GameState) -> None:
        """Draw planned paths for all enemies."""
        for enemy in game_state.enemies:
            if not enemy.is_alive or not enemy.current_path:
                continue
            
            # Draw path as line segments
            points = [enemy.position]
            points.extend(enemy.current_path)
            
            # Convert to screen coordinates
            screen_points = []
            for px, py in points:
                sx = self.offset_x + int(px * self.cell_size + self.cell_size / 2)
                sy = self.offset_y + int(py * self.cell_size + self.cell_size / 2)
                screen_points.append((sx, sy))
            
            if len(screen_points) >= 2:
                pygame.draw.lines(
                    self.screen,
                    self.COLOR_PATH,
                    False,
                    screen_points,
                    2
                )
    
    def _draw_stats(
        self,
        game_state: GameState,
        algorithm_name: str,
        fps: float
    ) -> None:
        """Draw game statistics overlay."""
        stats = game_state.get_statistics()
        
        # Prepare stats text
        lines = [
            f"Algorithm: {algorithm_name}",
            f"Wave: {stats['wave']}",
            f"Time: {stats['game_time']:.1f}s",
            f"Enemies: {stats['enemies_active']} active",
            f"Defeated: {stats['enemies_defeated']}",
            f"Escaped: {stats['enemies_escaped']}",
            f"Towers: {stats['towers']}",
        ]
        
        # Add FPS in debug mode
        if self.mode == VisualizationMode.DEBUG:
            lines.append(f"FPS: {fps:.1f}")
        
        # Draw background box
        padding = 10
        line_height = 24
        box_width = 250
        box_height = len(lines) * line_height + padding * 2
        
        box_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        box_surface.fill((0, 0, 0, 180))
        self.screen.blit(box_surface, (10, 10))
        
        # Draw text lines
        y_offset = 10 + padding
        for line in lines:
            text = self.font_small.render(line, True, self.COLOR_TEXT)
            self.screen.blit(text, (20, y_offset))
            y_offset += line_height
    
    def _draw_mode_indicator(self) -> None:
        """Draw current visualization mode indicator."""
        mode_text = f"Mode: {self.mode.value.upper()} (Tab to toggle)"
        text = self.font_small.render(mode_text, True, self.COLOR_TEXT)
        
        text_x = self.config.window_width - text.get_width() - 10
        text_y = 10
        
        # Background
        bg_rect = pygame.Rect(
            text_x - 5,
            text_y - 5,
            text.get_width() + 10,
            text.get_height() + 10
        )
        bg_surface = pygame.Surface(
            (bg_rect.width, bg_rect.height),
            pygame.SRCALPHA
        )
        bg_surface.fill((0, 0, 0, 180))
        self.screen.blit(bg_surface, (bg_rect.x, bg_rect.y))
        
        # Text
        self.screen.blit(text, (text_x, text_y))
    
    def _interpolate_color(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        ratio: float
    ) -> Tuple[int, int, int]:
        """
        Interpolate between two colors.
        
        Args:
            color1: First color (RGB)
            color2: Second color (RGB)
            ratio: Interpolation ratio (0.0 to 1.0)
            
        Returns:
            Interpolated color (RGB)
        """
        r = int(color1[0] + (color2[0] - color1[0]) * ratio)
        g = int(color1[1] + (color2[1] - color1[1]) * ratio)
        b = int(color1[2] + (color2[2] - color1[2]) * ratio)
        return (r, g, b)
    
    def draw_game_over(self, victory: bool) -> None:
        """
        Draw game over screen.
        
        Args:
            victory: True if player won, False if lost
        """
        # Semi-transparent overlay
        overlay = pygame.Surface(
            (self.config.window_width, self.config.window_height),
            pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        if victory:
            main_text = "VICTORY!"
            color = (0, 255, 0)
        else:
            main_text = "GAME OVER"
            color = (255, 0, 0)
        
        text = self.font_large.render(main_text, True, color)
        text_rect = text.get_rect(
            center=(self.config.window_width // 2, self.config.window_height // 2)
        )
        self.screen.blit(text, text_rect)
        
        # Instructions
        inst_text = "Press ESC to quit"
        inst = self.font_medium.render(inst_text, True, self.COLOR_TEXT)
        inst_rect = inst.get_rect(
            center=(self.config.window_width // 2, self.config.window_height // 2 + 50)
        )
        self.screen.blit(inst, inst_rect)

