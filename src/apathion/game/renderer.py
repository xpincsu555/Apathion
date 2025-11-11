"""
Pygame renderer for tower defense game visualization.
"""

from typing import Tuple, Optional, List
from enum import Enum
import pygame
import numpy as np

from apathion.game.game import GameState
from apathion.game.enemy import Enemy
from apathion.game.tower import Tower
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
    
    def toggle_mode(self) -> None:
        """Toggle between visualization modes."""
        modes = list(VisualizationMode)
        current_idx = modes.index(self.mode)
        next_idx = (current_idx + 1) % len(modes)
        self.mode = modes[next_idx]
    
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
        """Draw all towers."""
        for tower in game_state.towers:
            x = self.offset_x + int(tower.position[0] * self.cell_size)
            y = self.offset_y + int(tower.position[1] * self.cell_size)
            
            # Draw tower with high visibility - much brighter and larger
            padding = max(1, self.cell_size // 8)
            tower_rect = pygame.Rect(
                x + padding,
                y + padding,
                self.cell_size - 2 * padding,
                self.cell_size - 2 * padding
            )
            
            # Draw filled tower with bright blue color
            pygame.draw.rect(self.screen, (80, 150, 255), tower_rect)
            
            # Draw bright border for extra visibility
            border_width = max(2, self.cell_size // 15)
            pygame.draw.rect(self.screen, (150, 200, 255), tower_rect, border_width)
            
            # Draw tower type indicator in normal mode too for visibility
            if self.mode != VisualizationMode.MINIMAL:
                type_char = tower.tower_type.value[0].upper()
                text = self.font_small.render(type_char, True, (255, 255, 255))
                text_rect = text.get_rect(center=tower_rect.center)
                
                # Draw text shadow for better readability
                shadow_offset = 1
                shadow_text = self.font_small.render(type_char, True, (0, 0, 0))
                shadow_rect = text_rect.move(shadow_offset, shadow_offset)
                self.screen.blit(shadow_text, shadow_rect)
                self.screen.blit(text, text_rect)
    
    def _draw_enemies(self, game_state: GameState) -> None:
        """Draw all enemies with different shapes based on type."""
        from apathion.game.enemy import EnemyType
        
        for enemy in game_state.enemies:
            if not enemy.is_alive:
                continue
            
            # Position
            x = self.offset_x + int(enemy.position[0] * self.cell_size)
            y = self.offset_y + int(enemy.position[1] * self.cell_size)
            
            # Color based on health
            health_ratio = enemy.health / enemy.max_health
            color = self._interpolate_color(
                self.COLOR_ENEMY_DAMAGED,
                self.COLOR_ENEMY_HEALTHY,
                health_ratio
            )
            
            # Center position
            center_x = x + self.cell_size // 2
            center_y = y + self.cell_size // 2
            
            # Draw enemy with different shapes based on type
            if enemy.enemy_type == EnemyType.NORMAL:
                # Circle for normal enemies
                radius = self.cell_size // 3
                pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
                pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), radius, 1)
            
            elif enemy.enemy_type == EnemyType.FAST:
                # Triangle for fast enemies (pointing down/forward)
                size = self.cell_size // 2
                points = [
                    (center_x, center_y - size // 2),  # Top
                    (center_x - size // 2, center_y + size // 2),  # Bottom left
                    (center_x + size // 2, center_y + size // 2),  # Bottom right
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)
            
            elif enemy.enemy_type == EnemyType.TANK:
                # Square for tank enemies
                size = self.cell_size // 2
                rect = pygame.Rect(
                    center_x - size // 2,
                    center_y - size // 2,
                    size,
                    size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)
            
            elif enemy.enemy_type == EnemyType.LEADER:
                # Diamond for leader enemies
                size = self.cell_size // 2
                points = [
                    (center_x, center_y - size // 2),  # Top
                    (center_x + size // 2, center_y),  # Right
                    (center_x, center_y + size // 2),  # Bottom
                    (center_x - size // 2, center_y),  # Left
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)
            
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

