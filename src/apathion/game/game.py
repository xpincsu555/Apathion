"""
Game module for main game state and loop management.
"""

from typing import List, Dict, Any, Optional, Tuple
import time

from apathion.game.map import Map
from apathion.game.enemy import Enemy, EnemyType
from apathion.game.tower import Tower


class GameState:
    """
    Main game state managing enemies, towers, and game flow.
    
    Attributes:
        map: Game map instance
        enemies: List of active enemies
        towers: List of placed towers
        wave_number: Current wave number
        game_time: Total elapsed game time in seconds
        enemies_spawned: Total number of enemies spawned
        enemies_defeated: Number of enemies defeated
        enemies_escaped: Number of enemies that reached the goal
        is_running: Whether the game is active
    """
    
    def __init__(self, game_map: Optional[Map] = None):
        """
        Initialize game state.
        
        Args:
            game_map: Map instance, or None to create default map
        """
        self.map = game_map or Map.create_simple_map()
        self.enemies: List[Enemy] = []
        self.towers: List[Tower] = []
        self.wave_number = 0
        self.game_time = 0.0
        self.enemies_spawned = 0
        self.enemies_defeated = 0
        self.enemies_escaped = 0
        self.is_running = False
        self._last_update_time = time.time()
    
    def update(self, delta_time: float) -> None:
        """
        Update game state for one frame.
        
        Args:
            delta_time: Time elapsed since last update (seconds)
        """
        if not self.is_running:
            return
        
        self.game_time += delta_time
        
        # Update enemies
        self._update_enemies(delta_time)
        
        # Process tower attacks
        self._process_tower_attacks()
        
        # Remove dead enemies
        self._cleanup_dead_enemies()
    
    def _update_enemies(self, delta_time: float) -> None:
        """
        Update all enemy positions and states.
        
        Args:
            delta_time: Time elapsed since last update (seconds)
        """
        for enemy in self.enemies:
            if not enemy.is_alive or enemy.reached_goal:
                continue
            
            # Get next waypoint
            waypoint = enemy.get_next_waypoint()
            if waypoint is None:
                # No path - check if at goal
                goal_positions = self.map.goal_positions
                for goal in goal_positions:
                    dx = enemy.position[0] - goal[0]
                    dy = enemy.position[1] - goal[1]
                    if (dx ** 2 + dy ** 2) ** 0.5 < 0.5:
                        enemy.reached_goal = True
                        self.enemies_escaped += 1
                        break
                continue
            
            # Move toward waypoint
            reached = enemy.move(waypoint, delta_time)
            if reached:
                enemy.advance_waypoint()
    
    def _process_tower_attacks(self) -> None:
        """Process attacks from all towers."""
        current_time = self.game_time
        
        for tower in self.towers:
            # Find enemies in range
            for enemy in self.enemies:
                if not enemy.is_alive or enemy.reached_goal:
                    continue
                
                if tower.can_attack(enemy.position, current_time):
                    damage = tower.attack(current_time)
                    is_alive = enemy.take_damage(damage)
                    
                    if not is_alive:
                        tower.record_kill()
                        self.enemies_defeated += 1
                    
                    break  # Tower attacks one enemy per attack cycle
    
    def _cleanup_dead_enemies(self) -> None:
        """Remove dead or escaped enemies from active list."""
        self.enemies = [
            e for e in self.enemies
            if e.is_alive and not e.reached_goal
        ]
    
    def spawn_wave(
        self,
        num_enemies: int = 10,
        enemy_types: Optional[List[EnemyType]] = None,
    ) -> List[Enemy]:
        """
        Spawn a wave of enemies.
        
        Args:
            num_enemies: Number of enemies to spawn
            enemy_types: List of enemy types to spawn (random if None)
            
        Returns:
            List of spawned enemies
        """
        if enemy_types is None:
            enemy_types = [EnemyType.NORMAL] * num_enemies
        
        spawned = []
        spawn_point = self.map.spawn_points[0]  # Use first spawn point
        
        for i in range(num_enemies):
            enemy_id = f"E{self.wave_number:03d}_{self.enemies_spawned:04d}"
            enemy_type = enemy_types[i] if i < len(enemy_types) else EnemyType.NORMAL
            
            if enemy_type == EnemyType.FAST:
                enemy = Enemy.create_fast(enemy_id, spawn_point)
            elif enemy_type == EnemyType.TANK:
                enemy = Enemy.create_tank(enemy_id, spawn_point)
            else:
                enemy = Enemy.create_normal(enemy_id, spawn_point)
            
            self.enemies.append(enemy)
            spawned.append(enemy)
            self.enemies_spawned += 1
        
        self.wave_number += 1
        return spawned
    
    def place_tower(
        self,
        position: Tuple[int, int],
        tower_type: str = "basic",
    ) -> Optional[Tower]:
        """
        Place a tower at the specified position.
        
        Args:
            position: (x, y) grid position
            tower_type: Type of tower to place
            
        Returns:
            Placed tower instance, or None if placement failed
        """
        # Check if position is valid
        if not self.map.is_walkable(position[0], position[1]):
            return None
        
        # Check if a tower already exists at this position
        for tower in self.towers:
            if tower.position == position:
                return None
        
        # Create tower
        tower_id = f"T{len(self.towers):03d}"
        
        if tower_type == "sniper":
            tower = Tower.create_sniper(tower_id, position)
        elif tower_type == "rapid":
            tower = Tower.create_rapid(tower_id, position)
        elif tower_type == "area":
            tower = Tower.create_area(tower_id, position)
        else:
            tower = Tower.create_basic(tower_id, position)
        
        self.towers.append(tower)
        
        # Mark position as obstacle for pathfinding
        self.map.add_obstacle(position[0], position[1])
        
        return tower
    
    def remove_tower(self, tower_id: str) -> bool:
        """
        Remove a tower by ID.
        
        Args:
            tower_id: ID of tower to remove
            
        Returns:
            True if tower was removed
        """
        for i, tower in enumerate(self.towers):
            if tower.id == tower_id:
                # Remove obstacle from map
                self.map.remove_obstacle(tower.position[0], tower.position[1])
                self.towers.pop(i)
                return True
        return False
    
    def start(self) -> None:
        """Start the game."""
        self.is_running = True
        self._last_update_time = time.time()
    
    def pause(self) -> None:
        """Pause the game."""
        self.is_running = False
    
    def reset(self) -> None:
        """Reset game state."""
        self.enemies.clear()
        self.towers.clear()
        self.wave_number = 0
        self.game_time = 0.0
        self.enemies_spawned = 0
        self.enemies_defeated = 0
        self.enemies_escaped = 0
        self.is_running = False
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            True if game over conditions are met (placeholder logic)
        """
        # Placeholder: game over if too many enemies escaped
        return self.enemies_escaped >= 20
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current game statistics.
        
        Returns:
            Dictionary of game statistics
        """
        total_enemies = self.enemies_spawned
        active_enemies = len(self.enemies)
        
        return {
            "wave": self.wave_number,
            "game_time": self.game_time,
            "enemies_spawned": self.enemies_spawned,
            "enemies_active": active_enemies,
            "enemies_defeated": self.enemies_defeated,
            "enemies_escaped": self.enemies_escaped,
            "towers": len(self.towers),
            "survival_rate": (
                self.enemies_escaped / total_enemies * 100
                if total_enemies > 0 else 0
            ),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert game state to a dictionary for logging.
        
        Returns:
            Dictionary representation of game state
        """
        return {
            "wave": self.wave_number,
            "game_time": self.game_time,
            "enemies": [e.to_dict() for e in self.enemies],
            "towers": [t.to_dict() for t in self.towers],
            "statistics": self.get_statistics(),
        }

