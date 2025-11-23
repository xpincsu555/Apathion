"""
Game module for main game state and loop management.
"""

from typing import List, Dict, Any, Optional, Tuple
import time

from apathion.game.map import Map
from apathion.game.enemy import Enemy, EnemyType
from apathion.game.tower import Tower
from apathion.game.bullet import Bullet, HitEffect
from apathion.game.particles import ParticleSystem


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
        gold: Current player gold amount
        particles: Particle system for visual effects
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
        self.bullets: List[Bullet] = []
        self.hit_effects: List[HitEffect] = []
        self.wave_number = 0
        self.game_time = 0.0
        self.enemies_spawned = 0
        self.enemies_defeated = 0
        self.enemies_escaped = 0
        self.is_running = False
        self._last_update_time = time.time()
        
        # Gold system
        self.gold = 18  # Starting gold
        self.total_gold_earned = 0
        self._pending_gold_drops: List[Dict[str, Any]] = []
        
        # Particle system for visual effects
        self.particles = ParticleSystem()
        
        # Wave spawning state
        self._pending_spawns: List[Dict[str, Any]] = []
        self._next_spawn_time = 0.0
        self._current_wave_size = 0
        self._current_wave_spawned = 0
    
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
        
        # Process tower attacks (spawns bullets)
        self._process_tower_attacks()
        
        # Update bullets
        self._update_bullets(delta_time)
        
        # Update hit effects
        self._update_hit_effects(delta_time)
        
        # Update particle system
        self.particles.update(delta_time)
        
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
            
            # Move toward waypoint (convert to float to ensure type consistency)
            waypoint_float = (float(waypoint[0]), float(waypoint[1]))
            reached = enemy.move(waypoint_float, delta_time)
            if reached:
                enemy.advance_waypoint()
    
    def _process_tower_attacks(self) -> None:
        """Process attacks from all towers by spawning bullets."""
        current_time = self.game_time
        
        for tower in self.towers:
            # Find enemies in range
            for enemy in self.enemies:
                if not enemy.is_alive or enemy.reached_goal:
                    continue
                
                if tower.can_attack(enemy.position, current_time):
                    # Record attack on tower
                    damage = tower.attack(current_time)
                    
                    # Calculate tower center position in grid coordinates (with 0.5 offset for center)
                    tower_grid_x = tower.position[0] + 0.5
                    tower_grid_y = tower.position[1] + 0.5
                    
                    # Create bullet (positions in grid coordinates, speed in grid cells per second)
                    bullet = Bullet(
                        position=(tower_grid_x, tower_grid_y),
                        target_enemy_id=enemy.id,
                        damage=damage,
                        speed=8.0  # grid cells per second (fast enough to look good)
                    )
                    self.bullets.append(bullet)
                    
                    break  # Tower attacks one enemy per attack cycle
    
    def _update_bullets(self, delta_time: float) -> None:
        """Update all bullets and handle collisions."""
        bullets_to_remove = []
        
        for bullet in self.bullets:
            # Find the target enemy
            target_enemy = None
            for enemy in self.enemies:
                if enemy.id == bullet.target_enemy_id and enemy.is_alive:
                    target_enemy = enemy
                    break
            
            # If target is dead or missing, remove bullet
            if target_enemy is None:
                bullets_to_remove.append(bullet)
                continue
            
            # Target enemy position (already in grid coordinates)
            target_pos = target_enemy.position
            
            # Update bullet position (moves in grid coordinate space)
            reached = bullet.update(target_pos, delta_time)
            
            # If bullet reached target, deal damage and create hit effect
            if reached:
                is_alive = target_enemy.take_damage(bullet.damage)
                
                if not is_alive:
                    # Find the tower that shot this bullet and record kill
                    for tower in self.towers:
                        if tower.last_attack_time == self.game_time or \
                           abs(tower.last_attack_time - self.game_time) < 0.1:
                            tower.record_kill()
                            break
                    self.enemies_defeated += 1
                    
                    # Award gold for killing the enemy
                    gold_earned = target_enemy.gold_value
                    self.gold += gold_earned
                    self.total_gold_earned += gold_earned
                    
                    # Store enemy position and gold earned for renderer to create effects
                    # The renderer will access this through a method on game state
                    if not hasattr(self, '_pending_gold_drops'):
                        self._pending_gold_drops = []
                    self._pending_gold_drops.append({
                        'position': target_enemy.position,
                        'amount': gold_earned
                    })
                    
                    # Trigger gold drop particles at enemy position
                    self.particles.add_coin_drop(target_enemy.position)
                
                # Create hit effect at enemy position (in grid coordinates)
                # Note: hit effect sprite will be loaded by renderer
                self.hit_effects.append(HitEffect(
                    position=bullet.position,
                    sprite=None,  # Will be set by renderer
                    duration=0.3
                ))
                
                bullets_to_remove.append(bullet)
        
        # Remove bullets that hit or lost their target
        for bullet in bullets_to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
    
    def _update_hit_effects(self, delta_time: float) -> None:
        """Update all hit effects and remove expired ones."""
        effects_to_remove = []
        
        for effect in self.hit_effects:
            still_active = effect.update(delta_time)
            if not still_active:
                effects_to_remove.append(effect)
        
        for effect in effects_to_remove:
            if effect in self.hit_effects:
                self.hit_effects.remove(effect)
    
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
            elif enemy_type == EnemyType.LEADER:
                enemy = Enemy.create_leader(enemy_id, spawn_point)
            else:
                enemy = Enemy.create_normal(enemy_id, spawn_point)
            
            self.enemies.append(enemy)
            spawned.append(enemy)
            self.enemies_spawned += 1
        
        self.wave_number += 1
        return spawned
    
    def prepare_wave_with_delay(
        self,
        num_enemies: int = 10,
        enemy_types: Optional[List[EnemyType]] = None,
        spawn_delay: float = 1.0,
    ) -> None:
        """
        Prepare a wave of enemies to spawn with delays.
        
        Args:
            num_enemies: Number of enemies to spawn
            enemy_types: List of enemy types to spawn (random if None)
            spawn_delay: Delay in seconds between each enemy spawn
        """
        if enemy_types is None:
            enemy_types = [EnemyType.NORMAL] * num_enemies
        
        self._pending_spawns.clear()
        self._current_wave_size = num_enemies
        self._current_wave_spawned = 0
        
        spawn_point = self.map.spawn_points[0]
        
        for i in range(num_enemies):
            enemy_id = f"E{self.wave_number:03d}_{self.enemies_spawned + i:04d}"
            enemy_type = enemy_types[i] if i < len(enemy_types) else EnemyType.NORMAL
            
            self._pending_spawns.append({
                "id": enemy_id,
                "type": enemy_type,
                "position": spawn_point,
                "spawn_time": self.game_time + i * spawn_delay,
            })
        
        self._next_spawn_time = self.game_time
        self.wave_number += 1
    
    def update_wave_spawning(self, delta_time: float) -> List[Enemy]:
        """
        Update wave spawning and spawn pending enemies.
        
        Args:
            delta_time: Time elapsed since last update
            
        Returns:
            List of newly spawned enemies this frame
        """
        spawned = []
        
        while self._pending_spawns and self._pending_spawns[0]["spawn_time"] <= self.game_time:
            spawn_info = self._pending_spawns.pop(0)
            
            enemy_type = spawn_info["type"]
            if enemy_type == EnemyType.FAST:
                enemy = Enemy.create_fast(spawn_info["id"], spawn_info["position"])
            elif enemy_type == EnemyType.TANK:
                enemy = Enemy.create_tank(spawn_info["id"], spawn_info["position"])
            elif enemy_type == EnemyType.LEADER:
                enemy = Enemy.create_leader(spawn_info["id"], spawn_info["position"])
            else:
                enemy = Enemy.create_normal(spawn_info["id"], spawn_info["position"])
            
            self.enemies.append(enemy)
            spawned.append(enemy)
            self.enemies_spawned += 1
            self._current_wave_spawned += 1
        
        return spawned
    
    def is_wave_active(self) -> bool:
        """
        Check if a wave is currently active (enemies present or pending).
        
        Returns:
            True if wave is active
        """
        return len(self.enemies) > 0 or len(self._pending_spawns) > 0
    
    def is_wave_complete(self) -> bool:
        """
        Check if current wave is complete (all enemies spawned and cleared).
        
        Returns:
            True if wave is complete
        """
        return len(self._pending_spawns) == 0 and len(self.enemies) == 0
    
    def _filter_path_from_position(
        self,
        path: List[Tuple[int, int]],
        current_position: Tuple[float, float]
    ) -> List[Tuple[int, int]]:
        """
        Filter a path to remove waypoints that the enemy is currently on or has already passed.
        
        This prevents enemies from backtracking when replanning. When a tower is placed and
        paths need to be recalculated, enemies should continue from their current position,
        not go back to their starting grid cell.
        
        Args:
            path: Full path from pathfinder (includes start position)
            current_position: Enemy's actual (x, y) position (can have fractional coordinates)
            
        Returns:
            Filtered path starting from the next waypoint ahead of current position
        """
        if not path:
            return path
        
        # Get the enemy's current grid cell
        enemy_grid_x = int(current_position[0])
        enemy_grid_y = int(current_position[1])
        
        # Find the first waypoint that's not in the enemy's current grid cell
        # or in a cell the enemy has already passed through
        for i, waypoint in enumerate(path):
            # Skip waypoints that are in the same grid cell as the enemy
            if waypoint == (enemy_grid_x, enemy_grid_y):
                continue
            
            # Calculate distance to this waypoint
            dx = waypoint[0] - current_position[0]
            dy = waypoint[1] - current_position[1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            
            # Skip waypoints that are too close (enemy is effectively already there)
            # Use a threshold slightly larger than diagonal distance within a cell (sqrt(2) ≈ 1.41)
            if distance < 0.5:
                continue
            
            # This waypoint is far enough ahead, start from here
            return path[i:]
        
        # If all waypoints are too close or in current cell, return the last waypoint
        # (the goal) so the enemy has something to move toward
        if path:
            return [path[-1]]
        return path
    
    def update_enemy_paths(self, pathfinder, goal: Tuple[int, int]) -> None:
        """
        Recalculate paths for all active enemies from their current position.
        
        When an enemy needs to replan (e.g., due to tower placement), we calculate
        a new path from their current position. However, we need to skip waypoints
        that are at or behind the enemy's current position to prevent backtracking.
        
        Args:
            pathfinder: Pathfinder instance to use
            goal: Goal position
        """
        for enemy in self.enemies:
            if enemy.is_alive and not enemy.reached_goal:
                start = (int(enemy.position[0]), int(enemy.position[1]))
                path = pathfinder.find_path(
                    start, goal,
                    enemy_id=enemy.id,
                    health=enemy.health,
                    max_health=enemy.max_health,
                    enemy_type=enemy.enemy_type
                )
                
                # Skip waypoints that the enemy has already passed
                # The pathfinder returns a path starting from 'start', but the enemy
                # may have already moved past that grid position
                filtered_path = self._filter_path_from_position(path, enemy.position)
                enemy.set_path(filtered_path)
    
    def get_tower_cost(self, tower_type: str) -> int:
        """
        Get the gold cost for a tower type.
        
        Args:
            tower_type: Type of tower
            
        Returns:
            Gold cost for the tower
        """
        costs = {
            "basic": 8,
            "sniper": 25,
            "rapid": 11,
            "area": 14,
        }
        return costs.get(tower_type, 8)
    
    def can_afford_tower(self, tower_type: str) -> bool:
        """
        Check if player can afford a tower type.
        
        Args:
            tower_type: Type of tower
            
        Returns:
            True if player has enough gold
        """
        return self.gold >= self.get_tower_cost(tower_type)
    
    def place_tower(
        self,
        position: Tuple[int, int],
        tower_type: str = "basic",
        force: bool = False,
        check_gold: bool = True,
    ) -> Optional[Tower]:
        """
        Place a tower at the specified position.
        
        Args:
            position: (x, y) grid position
            tower_type: Type of tower to place
            force: If True, allows placement on obstacles (for initial config)
            check_gold: If True, checks gold cost (set False for initial towers)
            
        Returns:
            Placed tower instance, or None if placement failed
        """
        # Check if position is within bounds
        if not (0 <= position[0] < self.map.width and 0 <= position[1] < self.map.height):
            return None
        
        # Check if position is valid (unless forcing)
        if not force and not self.map.is_walkable(position[0], position[1]):
            return None
        
        # Check if a tower already exists at this position
        for tower in self.towers:
            if tower.position == position:
                return None
        
        # Check gold cost
        tower_cost = self.get_tower_cost(tower_type)
        if check_gold and self.gold < tower_cost:
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
        
        # Deduct gold cost
        if check_gold:
            self.gold -= tower_cost
        
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
        self.bullets.clear()
        self.hit_effects.clear()
        self.wave_number = 0
        self.game_time = 0.0
        self.enemies_spawned = 0
        self.enemies_defeated = 0
        self.enemies_escaped = 0
        self.is_running = False
        self.gold = 18
        self.total_gold_earned = 0
        self._pending_gold_drops.clear()
        self.particles.clear()
        self._pending_spawns.clear()
        self._next_spawn_time = 0.0
        self._current_wave_size = 0
        self._current_wave_spawned = 0
    
    def consume_pending_gold_drops(self) -> List[Dict[str, Any]]:
        """
        Get and clear pending gold drop notifications.
        
        Returns:
            List of gold drop events with position and amount
        """
        drops = self._pending_gold_drops.copy()
        self._pending_gold_drops.clear()
        return drops
    
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

