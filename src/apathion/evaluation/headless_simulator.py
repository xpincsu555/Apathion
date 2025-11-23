"""
Headless simulation runner for experiments without pygame display.

This module provides a simulation runner that executes the full game logic
without requiring a pygame window, enabling automated batch experiments.
"""

from typing import List, Tuple, Dict, Any, Optional
import time

from apathion.game.game import GameState
from apathion.game.map import Map
from apathion.game.enemy import EnemyType
from apathion.pathfinding.base import BasePathfinder
from apathion.config import GameConfig
from apathion.evaluation.logger import GameLogger


class HeadlessSimulator:
    """
    Headless game simulator for batch experiments.
    
    Runs the full game simulation without visualization, collecting
    performance data and metrics for analysis.
    
    Attributes:
        config: Game configuration
        logger: Data logger for experiment results
        verbose: Whether to print progress messages
    """
    
    def __init__(
        self,
        config: GameConfig,
        logger: Optional[GameLogger] = None,
        verbose: bool = True
    ):
        """
        Initialize headless simulator.
        
        Args:
            config: Game configuration
            logger: Optional logger instance
            verbose: Print progress messages
        """
        self.config = config
        self.logger = logger or GameLogger()
        self.verbose = verbose
    
    def run_simulation(
        self,
        game_map: Map,
        pathfinder: BasePathfinder,
        num_waves: int,
        enemies_per_wave: int,
        initial_towers: Optional[List[Tuple[Tuple[int, int], str]]] = None,
        enemy_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete simulation.
        
        Args:
            game_map: Map to use
            pathfinder: Pathfinding algorithm
            num_waves: Number of waves to simulate
            enemies_per_wave: Enemies per wave
            initial_towers: Optional list of ((x, y), type) tuples
            enemy_types: Optional list of enemy type names
            
        Returns:
            Dictionary with simulation results and metrics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting simulation: {pathfinder.get_name()}")
            print(f"Waves: {num_waves}, Enemies/wave: {enemies_per_wave}")
            print(f"{'='*60}")
        
        # Initialize game state
        game_state = GameState(game_map)
        game_state.start()
        
        # Place initial towers (without gold checks for experiments)
        if initial_towers:
            for position, tower_type in initial_towers:
                game_state.place_tower(position, tower_type, force=True, check_gold=False)
        elif self.config.towers.initial_tower_placements:
            for placement in self.config.towers.initial_tower_placements:
                position = tuple(placement["position"])
                tower_type = placement.get("type", "basic")
                game_state.place_tower(position, tower_type, force=True, check_gold=False)
        
        # Update pathfinder with initial state
        pathfinder.update_state(game_state.map, game_state.towers)
        
        # Parse enemy types
        parsed_enemy_types = self._parse_enemy_types(enemy_types or ["normal"])
        
        # Simulation tracking
        sim_start_time = time.time()
        all_decision_times = []
        all_paths = []  # Track all paths for diversity calculation
        wave_results = []
        
        # Track previous counts to calculate deltas
        prev_defeated = 0
        prev_escaped = 0
        
        # Run waves
        goal = game_state.map.goal_positions[0]
        
        for wave_num in range(num_waves):
            if self.verbose:
                print(f"\nWave {wave_num + 1}/{num_waves}")
            
            wave_start_time = time.time()
            
            # Determine enemy types for this wave
            wave_enemy_types = []
            for i in range(enemies_per_wave):
                enemy_type = parsed_enemy_types[i % len(parsed_enemy_types)]
                wave_enemy_types.append(enemy_type)
            
            # Spawn wave (instant, no delay for batch experiments)
            enemies = game_state.spawn_wave(
                num_enemies=enemies_per_wave,
                enemy_types=wave_enemy_types
            )
            
            # Assign initial paths
            for enemy in enemies:
                start = (int(enemy.position[0]), int(enemy.position[1]))
                
                # Time the pathfinding
                path_start = time.time()
                path = pathfinder.find_path(start, goal, enemy_id=enemy.id)
                path_time = (time.time() - path_start) * 1000  # ms
                
                enemy.set_path(path)
                all_decision_times.append(path_time)
                all_paths.append(path)  # Collect path for diversity analysis
                
                # Log the decision
                self.logger.log_decision(
                    timestamp=game_state.game_time,
                    algorithm=pathfinder.get_name(),
                    enemy_id=enemy.id,
                    chosen_path=path,
                    alternative_paths=0,
                    decision_time_ms=path_time,
                )
            
            # Simulate wave until all enemies are cleared
            wave_frame_count = 0
            max_wave_time = 300.0  # Maximum 5 minutes per wave
            target_dt = 1.0 / 60.0  # Simulate at 60 FPS
            
            while not game_state.is_wave_complete():
                game_state.update(target_dt)
                wave_frame_count += 1
                
                # Safety check to prevent infinite loops
                if game_state.game_time - wave_start_time > max_wave_time:
                    if self.verbose:
                        print(f"  Warning: Wave timeout after {max_wave_time}s")
                    break
            
            wave_elapsed = time.time() - wave_start_time
            
            # Calculate deltas (change since last wave)
            defeated_this_wave = game_state.enemies_defeated - prev_defeated
            escaped_this_wave = game_state.enemies_escaped - prev_escaped
            
            # Update previous counts for next wave
            prev_defeated = game_state.enemies_defeated
            prev_escaped = game_state.enemies_escaped
            
            # Collect wave statistics (use deltas, not cumulative)
            wave_stats = {
                "wave_number": wave_num + 1,
                "enemies_spawned": enemies_per_wave,
                "enemies_defeated": defeated_this_wave,
                "enemies_escaped": escaped_this_wave,
                "wave_time_seconds": wave_elapsed,
                "frames_simulated": wave_frame_count,
            }
            wave_results.append(wave_stats)
            
            # Log wave results
            self.logger.log_wave_results(
                wave=wave_num + 1,
                algorithm=pathfinder.get_name(),
                enemies_spawned=enemies_per_wave,
                enemies_survived=wave_stats["enemies_escaped"],
                avg_damage=0.0,  # Will be calculated from enemy data
                path_diversity=0.0,  # Will be calculated from paths
                total_cpu_time=wave_elapsed,
            )
            
            if self.verbose:
                print(f"  Defeated: {wave_stats['enemies_defeated']}, "
                      f"Escaped: {wave_stats['enemies_escaped']}, "
                      f"Time: {wave_elapsed:.2f}s")
        
        sim_elapsed = time.time() - sim_start_time
        
        # Calculate aggregate metrics
        total_spawned = sum(w["enemies_spawned"] for w in wave_results)
        total_defeated = sum(w["enemies_defeated"] for w in wave_results)
        total_escaped = sum(w["enemies_escaped"] for w in wave_results)
        
        # Calculate path diversity
        from apathion.evaluation.metrics import path_diversity as calc_path_diversity
        path_diversity_metrics = calc_path_diversity(all_paths)
        
        results = {
            "algorithm": pathfinder.get_name(),
            "algorithm_config": pathfinder.to_dict(),
            "num_waves": num_waves,
            "enemies_per_wave": enemies_per_wave,
            "total_enemies": total_spawned,
            "total_defeated": total_defeated,
            "total_escaped": total_escaped,
            "survival_rate_percent": (total_escaped / total_spawned * 100) if total_spawned > 0 else 0,
            "defeat_rate_percent": (total_defeated / total_spawned * 100) if total_spawned > 0 else 0,
            "avg_decision_time_ms": sum(all_decision_times) / len(all_decision_times) if all_decision_times else 0,
            "max_decision_time_ms": max(all_decision_times) if all_decision_times else 0,
            "min_decision_time_ms": min(all_decision_times) if all_decision_times else 0,
            "total_simulation_time_seconds": sim_elapsed,
            "path_diversity": path_diversity_metrics,  # Add path diversity metrics
            "wave_results": wave_results,
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Simulation Complete")
            print(f"Total Time: {sim_elapsed:.2f}s")
            print(f"Survival Rate: {results['survival_rate_percent']:.1f}%")
            print(f"Avg Decision Time: {results['avg_decision_time_ms']:.3f}ms")
            print(f"Path Diversity: {path_diversity_metrics['diversity_index']:.3f} "
                  f"({path_diversity_metrics['unique_paths']} unique paths)")
            print(f"{'='*60}")
        
        return results
    
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

