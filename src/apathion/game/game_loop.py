"""
Main game loop for pygame-based tower defense simulation.
"""

from typing import Optional, List
import pygame
import sys

from apathion.game.game import GameState
from apathion.game.map import Map
from apathion.game.enemy import EnemyType
from apathion.game.renderer import GameRenderer
from apathion.pathfinding.base import BasePathfinder
from apathion.config import GameConfig


class GameLoop:
    """
    Main game loop handler for the tower defense game.
    
    Manages pygame initialization, input handling, game state updates,
    and rendering.
    
    Attributes:
        config: Game configuration
        game_state: Current game state
        pathfinder: Pathfinding algorithm instance
        renderer: Pygame renderer
        screen: Pygame display surface
        clock: Pygame clock for frame timing
        running: Whether the game loop is active
        paused: Whether the game is paused
    """
    
    def __init__(
        self,
        config: GameConfig,
        game_state: GameState,
        pathfinder: BasePathfinder,
    ):
        """
        Initialize the game loop.
        
        Args:
            config: Game configuration
            game_state: Game state instance
            pathfinder: Pathfinding algorithm instance
        """
        self.config = config
        self.game_state = game_state
        self.pathfinder = pathfinder
        
        # Initialize pygame
        pygame.init()
        
        # Create window
        self.screen = pygame.display.set_mode(
            (config.visualization.window_width, config.visualization.window_height)
        )
        pygame.display.set_caption(
            f"Apathion Tower Defense - {pathfinder.get_name()}"
        )
        
        # Create renderer
        self.renderer = GameRenderer(
            self.screen,
            config.visualization,
            game_state.map.width,
            game_state.map.height
        )
        
        # Game loop state
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # Wave management
        self.current_wave = 0
        self.max_waves = config.enemies.wave_count
        self.enemies_per_wave = config.enemies.enemies_per_wave
        self.spawn_delay = config.enemies.spawn_delay
        
        # Parse enemy types from config
        self.enemy_types = self._parse_enemy_types(config.enemies.enemy_types)
        
        # FPS tracking
        self.fps_history: List[float] = []
        self.current_fps = 0.0
        
        # Selected tower type for placement
        self.selected_tower_type = "basic"
        self.tower_type_index = 0
        self.available_tower_types = config.towers.tower_types
    
    def _parse_enemy_types(self, type_names: List[str]) -> List[EnemyType]:
        """
        Parse enemy type names to EnemyType enum values.
        
        Args:
            type_names: List of enemy type name strings
            
        Returns:
            List of EnemyType enum values
        """
        types = []
        for name in type_names:
            name_lower = name.lower()
            if name_lower == "fast":
                types.append(EnemyType.FAST)
            elif name_lower == "tank":
                types.append(EnemyType.TANK)
            elif name_lower == "leader":
                types.append(EnemyType.LEADER)
            else:
                types.append(EnemyType.NORMAL)
        return types
    
    def run(self) -> None:
        """Run the main game loop."""
        # Update pathfinder with initial state
        self.pathfinder.update_state(self.game_state.map, self.game_state.towers)
        
        # Start the game
        self.game_state.start()
        
        # Spawn first wave
        self._spawn_next_wave()
        
        # Main loop
        while self.running:
            # Calculate delta time
            delta_time = self.clock.tick(self.config.target_fps) / 1000.0
            delta_time *= self.config.simulation_speed
            
            # Track FPS
            self._update_fps()
            
            # Handle input
            self._handle_input()
            
            # Update game state
            if not self.paused:
                self._update_game(delta_time)
            
            # Render
            self._render()
            
            # Flip display
            pygame.display.flip()
        
        # Cleanup
        pygame.quit()
    
    def _handle_input(self) -> None:
        """Handle keyboard and mouse input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                # ESC to quit
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                # Space to pause/resume
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                
                # Tab to toggle visualization mode
                elif event.key == pygame.K_TAB:
                    self.renderer.toggle_mode()
                
                # Number keys to select tower type
                elif event.key == pygame.K_1:
                    self.selected_tower_type = "basic"
                elif event.key == pygame.K_2:
                    self.selected_tower_type = "sniper"
                elif event.key == pygame.K_3:
                    self.selected_tower_type = "rapid"
                elif event.key == pygame.K_4:
                    self.selected_tower_type = "area"
                
                # T to cycle tower types
                elif event.key == pygame.K_t:
                    self._cycle_tower_type()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_tower_placement(event.pos)
    
    def _cycle_tower_type(self) -> None:
        """Cycle through available tower types."""
        if self.available_tower_types:
            self.tower_type_index = (self.tower_type_index + 1) % len(self.available_tower_types)
            self.selected_tower_type = self.available_tower_types[self.tower_type_index]
    
    def _handle_tower_placement(self, mouse_pos: tuple) -> None:
        """
        Handle tower placement at mouse position.
        
        Args:
            mouse_pos: (x, y) screen position of mouse click
        """
        if not self.config.towers.allow_dynamic_placement:
            return
        
        # Convert to grid position
        grid_pos = self.renderer.get_grid_position(mouse_pos)
        if grid_pos is None:
            return
        
        # Check bounds
        if not (0 <= grid_pos[0] < self.game_state.map.width and
                0 <= grid_pos[1] < self.game_state.map.height):
            return
        
        # Check if player can afford this tower type
        tower_cost = self.game_state.get_tower_cost(self.selected_tower_type)
        if not self.game_state.can_afford_tower(self.selected_tower_type):
            print(f"Cannot afford {self.selected_tower_type} tower (cost: {tower_cost}, gold: {self.game_state.gold})")
            return
        
        # Try to place tower (force=False enforces placement rules, check_gold=True enforces cost)
        tower = self.game_state.place_tower(grid_pos, self.selected_tower_type, force=False, check_gold=True)
        
        if tower:
            print(f"Placed {self.selected_tower_type} tower at {grid_pos} for {tower_cost} gold (remaining: {self.game_state.gold})")
            # Update pathfinder with new tower
            self.pathfinder.update_state(self.game_state.map, self.game_state.towers)
            
            # Recalculate paths for all active enemies
            goal = self.game_state.map.goal_positions[0]
            self.game_state.update_enemy_paths(self.pathfinder, goal)
        else:
            # Get error message from game state
            error_msg = self.game_state.get_last_placement_error()
            if error_msg:
                print(f"Failed to place tower at {grid_pos}: {error_msg}")
            else:
                print(f"Failed to place tower at {grid_pos}")
    
    def _update_game(self, delta_time: float) -> None:
        """
        Update game state for one frame.
        
        Args:
            delta_time: Time elapsed since last frame (seconds)
        """
        # Update game state
        self.game_state.update(delta_time)
        
        # Update wave spawning
        newly_spawned = self.game_state.update_wave_spawning(delta_time)
        
        # Assign paths to newly spawned enemies
        if newly_spawned:
            goal = self.game_state.map.goal_positions[0]
            for enemy in newly_spawned:
                start = (int(enemy.position[0]), int(enemy.position[1]))
                path = self.pathfinder.find_path(
                    start, goal,
                    enemy_id=enemy.id,
                    health=enemy.health,
                    max_health=enemy.max_health,
                    wave=self.current_wave,
                    enemy_type=enemy.enemy_type
                )
                enemy.set_path(path)
        
        # Check if wave is complete and spawn next
        if self.game_state.is_wave_complete() and self.current_wave < self.max_waves:
            self._spawn_next_wave()
        
        # Check game over condition
        if self.game_state.is_game_over():
            self._handle_game_over(victory=False)
        
        # Check victory condition
        if self.current_wave >= self.max_waves and self.game_state.is_wave_complete():
            self._handle_game_over(victory=True)
    
    def _spawn_next_wave(self) -> None:
        """Spawn the next wave of enemies."""
        if self.current_wave >= self.max_waves:
            return
        
        # Determine enemy types for this wave
        wave_enemy_types = []
        for i in range(self.enemies_per_wave):
            enemy_type = self.enemy_types[i % len(self.enemy_types)]
            wave_enemy_types.append(enemy_type)
        
        # Prepare wave with delayed spawning
        self.game_state.prepare_wave_with_delay(
            num_enemies=self.enemies_per_wave,
            enemy_types=wave_enemy_types,
            spawn_delay=self.spawn_delay
        )
        
        self.current_wave += 1
    
    def _render(self) -> None:
        """Render the current game state."""
        self.renderer.render(
            self.game_state,
            self.pathfinder,
            self.pathfinder.get_name(),
            self.current_fps
        )
        
        # Draw pause indicator
        if self.paused:
            self._draw_pause_indicator()
        
        # Draw controls help
        self._draw_controls_help()
    
    def _draw_pause_indicator(self) -> None:
        """Draw pause indicator overlay."""
        font = pygame.font.Font(None, 72)
        text = font.render("PAUSED", True, (255, 255, 255))
        text_rect = text.get_rect(
            center=(
                self.config.visualization.window_width // 2,
                self.config.visualization.window_height // 2
            )
        )
        
        # Semi-transparent background
        bg_rect = text_rect.inflate(40, 20)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 180))
        self.screen.blit(bg_surface, bg_rect)
        
        # Text
        self.screen.blit(text, text_rect)
    
    def _draw_controls_help(self) -> None:
        """Draw controls help at the bottom of the screen."""
        # Get tower costs
        tower_costs = {
            "basic": self.game_state.get_tower_cost("basic"),
            "sniper": self.game_state.get_tower_cost("sniper"),
            "rapid": self.game_state.get_tower_cost("rapid"),
            "area": self.game_state.get_tower_cost("area"),
        }
        
        controls = [
            "Controls:",
            "Click: Place tower",
            f"T: Change tower ({self.selected_tower_type.upper()}, cost: {tower_costs[self.selected_tower_type]})",
            "1: Basic (8g)  2: Sniper (25g)",
            "3: Rapid (11g)  4: Area (14g)",
            "Space: Pause  Tab: View  ESC: Quit"
        ]
        
        font = pygame.font.Font(None, 18)
        y_offset = self.config.visualization.window_height - 15 * len(controls) - 10
        
        # Draw semi-transparent background
        bg_height = 15 * len(controls) + 15
        bg_surface = pygame.Surface((280, bg_height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 180))
        self.screen.blit(bg_surface, (5, y_offset - 5))
        
        for i, line in enumerate(controls):
            # Highlight current tower selection
            if i == 2:  # Tower type line
                color = (100, 255, 100) if self.game_state.can_afford_tower(self.selected_tower_type) else (255, 100, 100)
            else:
                color = (200, 200, 200)
            
            text = font.render(line, True, color)
            self.screen.blit(text, (10, y_offset))
            y_offset += 15
    
    def _update_fps(self) -> None:
        """Update FPS tracking."""
        fps = self.clock.get_fps()
        self.fps_history.append(fps)
        
        # Keep only last 30 frames
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        # Calculate average
        if self.fps_history:
            self.current_fps = sum(self.fps_history) / len(self.fps_history)
    
    def _handle_game_over(self, victory: bool) -> None:
        """
        Handle game over state.
        
        Args:
            victory: True if player won, False if lost
        """
        # Stop game
        self.game_state.pause()
        
        # Render game over screen
        self.renderer.draw_game_over(victory)
        pygame.display.flip()
        
        # Wait for user to quit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False
                        self.running = False
            
            self.clock.tick(10)  # Low FPS while waiting


def run_game_loop(
    config: GameConfig,
    game_map: Map,
    pathfinder: BasePathfinder,
    initial_towers: Optional[List[tuple]] = None,
) -> None:
    """
    Run the main game loop with pygame visualization.
    
    Args:
        config: Game configuration
        game_map: Map instance
        pathfinder: Pathfinding algorithm instance
        initial_towers: Optional list of (position, type) tuples for initial towers
    """
    # Create game state
    game_state = GameState(game_map)
    
    # Place initial towers if provided
    if initial_towers:
        for position, tower_type in initial_towers:
            game_state.place_tower(position, tower_type, force=True, check_gold=False)
    elif config.towers.initial_tower_placements:
        # Use custom tower placements from config (force placement even on obstacles)
        for placement in config.towers.initial_tower_placements:
            position = tuple(placement["position"])
            tower_type = placement.get("type", "basic")
            result = game_state.place_tower(position, tower_type, force=True, check_gold=False)
            if result is None:
                print(f"Warning: Failed to place {tower_type} tower at {position}")
    elif config.towers.initial_towers > 0:
        # Place some default towers
        # Find suitable positions (simple strategy: evenly spaced)
        tower_spacing = game_map.width // (config.towers.initial_towers + 1)
        for i in range(config.towers.initial_towers):
            x = (i + 1) * tower_spacing
            y = game_map.height // 2
            tower_type = config.towers.tower_types[i % len(config.towers.tower_types)]
            game_state.place_tower((x, y), tower_type, check_gold=False)
    
    # Create and run game loop
    game_loop = GameLoop(config, game_state, pathfinder)
    game_loop.run()

