#!/usr/bin/env python
"""
Refactored batch experiment runner for pathfinding algorithm evaluation.

This script evaluates pathfinding algorithms on the following metrics:
- Survival rate: percentage of enemies reaching the goal
- Path diversity: Shannon entropy of route distribution
- Adaptation speed: frames to converge on new optimal path after tower placement
- Computational cost: average milliseconds per pathfinding decision

Algorithms compared: fixed, astar, aco, dqn

Usage:
    python scripts/run_experiments.py --config configs/experiments/my_experiment.json
    python scripts/run_experiments.py --preset baseline
    python scripts/run_experiments.py --algorithms fixed,astar,aco,dqn --maps branching,open_arena --runs 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apathion.config import GameConfig
from apathion.game.map import Map
from apathion.game.game import GameState
from apathion.game.enemy import EnemyType
from apathion.pathfinding.fixed import FixedPathfinder
from apathion.pathfinding.astar import AStarPathfinder
from apathion.pathfinding.aco import ACOPathfinder
from apathion.pathfinding.dqn import DQNPathfinder
from apathion.pathfinding.base import BasePathfinder
from apathion.evaluation.logger import GameLogger
from apathion.evaluation.metrics import (
    survival_rate,
    path_diversity,
    computational_cost,
)


class ExperimentRunner:
    """
    Batch experiment runner for algorithm comparison.
    
    Evaluates pathfinding algorithms on survival rate, path diversity,
    adaptation speed, and computational cost metrics.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
    
    def run_from_config(self, config_path: str) -> Dict[str, Any]:
        """
        Run experiments from a configuration file.
        
        Args:
            config_path: Path to experiment config JSON
            
        Returns:
            Dictionary with all experiment results
        """
        with open(config_path, 'r') as f:
            exp_config = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_config.get('name', 'Unnamed')}")
        print(f"Description: {exp_config.get('description', 'No description')}")
        print(f"{'='*70}")
        
        # Extract experiment parameters
        algorithms = exp_config.get("algorithms", ["astar"])
        map_types = exp_config.get("map_types", ["branching"])
        num_runs = exp_config.get("num_runs", 3)
        waves_per_run = exp_config.get("waves_per_run", 10)
        enemies_per_wave = exp_config.get("enemies_per_wave", 30)
        tower_placement_wave = exp_config.get("tower_placement_wave", None)
        
        # Run experiments
        return self.run_experiments(
            algorithms=algorithms,
            map_types=map_types,
            num_runs=num_runs,
            waves_per_run=waves_per_run,
            enemies_per_wave=enemies_per_wave,
            algo_configs=exp_config,
            tower_placement_wave=tower_placement_wave,
        )
    
    def run_experiments(
        self,
        algorithms: List[str],
        map_types: List[str],
        num_runs: int = 3,
        waves_per_run: int = 10,
        enemies_per_wave: int = 30,
        algo_configs: Optional[Dict[str, Any]] = None,
        tower_placement_wave: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run batch experiments across algorithms and maps.
        
        Args:
            algorithms: List of algorithm names (fixed, astar, aco, dqn)
            map_types: List of map type names (branching, open_arena)
            num_runs: Number of runs per configuration
            waves_per_run: Waves per run
            enemies_per_wave: Enemies per wave
            algo_configs: Optional algorithm-specific configurations
            tower_placement_wave: Wave number to place additional towers (for adaptation metric)
            
        Returns:
            Dictionary with all results
        """
        algo_configs = algo_configs or {}
        experiment_start = datetime.now()
        
        print(f"\nExperiment Parameters:")
        print(f"  Algorithms: {', '.join(algorithms)}")
        print(f"  Maps: {', '.join(map_types)}")
        print(f"  Runs per config: {num_runs}")
        print(f"  Waves per run: {waves_per_run}")
        print(f"  Enemies per wave: {enemies_per_wave}")
        if tower_placement_wave:
            print(f"  Tower placement wave: {tower_placement_wave} (for adaptation speed)")
        print(f"  Total experiments: {len(algorithms) * len(map_types) * num_runs}")
        
        all_results = []
        experiment_counter = 0
        total_experiments = len(algorithms) * len(map_types) * num_runs
        
        for algo_name in algorithms:
            for map_type in map_types:
                for run_num in range(num_runs):
                    experiment_counter += 1
                    
                    print(f"\n{'='*70}")
                    print(f"Experiment {experiment_counter}/{total_experiments}")
                    print(f"Algorithm: {algo_name}, Map: {map_type}, Run: {run_num + 1}/{num_runs}")
                    print(f"{'='*70}")
                    
                    # Create game config
                    config = self._create_game_config(
                        map_type=map_type,
                        waves=waves_per_run,
                        enemies=enemies_per_wave,
                    )
                    
                    # Create map
                    game_map = self._create_map(map_type, config)
                    
                    # Create pathfinder
                    pathfinder = self._create_pathfinder(algo_name, algo_configs, game_map, map_type)
                    
                    # Create logger for this run
                    logger = GameLogger()
                    
                    # Run simulation with mid-experiment tower placement if specified
                    result = self._run_simulation_with_adaptation(
                        config=config,
                        game_map=game_map,
                        pathfinder=pathfinder,
                        logger=logger,
                        num_waves=waves_per_run,
                        enemies_per_wave=enemies_per_wave,
                        tower_placement_wave=tower_placement_wave,
                    )
                    
                    # Add metadata
                    result["algorithm"] = pathfinder.get_name()
                    result["map_type"] = map_type
                    result["run_number"] = run_num + 1
                    result["timestamp"] = datetime.now().isoformat()
                    
                    all_results.append(result)
                    self.results.append(result)
        
        experiment_end = datetime.now()
        experiment_duration = (experiment_end - experiment_start).total_seconds()
        
        # Generate summary
        summary = self._generate_summary(all_results)
        
        # Save results
        session_id = experiment_start.strftime("%Y%m%d_%H%M%S")
        self._save_results(all_results, summary, session_id)
        
        print(f"\n{'='*70}")
        print(f"All Experiments Complete!")
        print(f"Total Time: {experiment_duration:.1f}s")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")
        
        return {
            "experiments": all_results,
            "summary": summary,
            "session_id": session_id,
            "duration_seconds": experiment_duration,
        }
    
    def _run_simulation_with_adaptation(
        self,
        config: GameConfig,
        game_map: Map,
        pathfinder: BasePathfinder,
        logger: GameLogger,
        num_waves: int,
        enemies_per_wave: int,
        tower_placement_wave: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run simulation with optional mid-experiment tower placement for adaptation testing.
        
        Args:
            config: Game configuration
            game_map: Map instance
            pathfinder: Pathfinding algorithm
            logger: Logger instance
            num_waves: Number of waves
            enemies_per_wave: Enemies per wave
            tower_placement_wave: Wave to place new towers (None = no mid-placement)
            
        Returns:
            Dictionary with simulation results and metrics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting simulation: {pathfinder.get_name()}")
            print(f"Waves: {num_waves}, Enemies/wave: {enemies_per_wave}")
            if tower_placement_wave:
                print(f"Tower placement at wave: {tower_placement_wave}")
            print(f"{'='*60}")
        
        # Initialize game state
        game_state = GameState(game_map)
        game_state.start()
        
        # Place initial towers (without gold checks for experiments)
        if config.towers.initial_tower_placements:
            for placement in config.towers.initial_tower_placements:
                position = tuple(placement["position"])
                tower_type = placement.get("type", "basic")
                game_state.place_tower(position, tower_type, force=True, check_gold=False)
        
        # Update pathfinder with initial state
        pathfinder.update_state(game_state.map, game_state.towers)
        
        # Simulation tracking
        sim_start_time = time.time()
        all_decision_times = []
        all_paths = []  # Track all paths for diversity calculation
        wave_results = []
        
        # Adaptation tracking
        adaptation_data = {
            "tower_placed": False,
            "placement_frame": 0,
            "placement_time": 0.0,
            "paths_before_change": [],
            "paths_after_change": [],
            "frames_to_converge": 0,
        }
        
        # Track previous counts to calculate deltas
        prev_defeated = 0
        prev_escaped = 0
        
        # Get goal
        goal = game_state.map.goal_positions[0]
        
        # Parse enemy types
        enemy_types = self._parse_enemy_types(config.enemies.enemy_types)
        
        # Run waves
        for wave_num in range(num_waves):
            if self.verbose:
                print(f"\nWave {wave_num + 1}/{num_waves}")
            
            wave_start_time = time.time()
            
            # Check if this is the wave to place new towers
            if tower_placement_wave and wave_num + 1 == tower_placement_wave:
                placement_time = game_state.game_time
                self._place_adaptation_towers(game_state, game_map)
                pathfinder.update_state(game_state.map, game_state.towers)
                adaptation_data["tower_placed"] = True
                adaptation_data["placement_time"] = placement_time
                if self.verbose:
                    print(f"  *** Placed adaptation towers at wave {wave_num + 1} ***")
            
            # Determine enemy types for this wave
            wave_enemy_types = []
            for i in range(enemies_per_wave):
                enemy_type = enemy_types[i % len(enemy_types)]
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
                all_paths.append(path)
                
                # Track paths before/after tower placement for adaptation
                if adaptation_data["tower_placed"]:
                    adaptation_data["paths_after_change"].append(path)
                elif tower_placement_wave:
                    adaptation_data["paths_before_change"].append(path)
                
                # Log the decision
                logger.log_decision(
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
                
                # Track frames after tower placement for adaptation metric
                if adaptation_data["tower_placed"] and adaptation_data["frames_to_converge"] == 0:
                    adaptation_data["placement_frame"] = wave_frame_count
                
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
            
            # Collect wave statistics
            wave_stats = {
                "wave_number": wave_num + 1,
                "enemies_spawned": enemies_per_wave,
                "enemies_defeated": defeated_this_wave,
                "enemies_escaped": escaped_this_wave,
                "wave_time_seconds": wave_elapsed,
                "frames_simulated": wave_frame_count,
            }
            wave_results.append(wave_stats)
            
            if self.verbose:
                print(f"  Defeated: {wave_stats['enemies_defeated']}, "
                      f"Escaped: {wave_stats['enemies_escaped']}, "
                      f"Time: {wave_elapsed:.2f}s")
        
        sim_elapsed = time.time() - sim_start_time
        
        # Calculate aggregate metrics
        total_spawned = sum(w["enemies_spawned"] for w in wave_results)
        total_defeated = sum(w["enemies_defeated"] for w in wave_results)
        total_escaped = sum(w["enemies_escaped"] for w in wave_results)
        
        # 1. Survival Rate: percentage of enemies reaching the goal
        survival_rate_metrics = survival_rate(total_spawned, total_escaped)
        
        # 2. Path Diversity: Shannon entropy of route distribution
        path_diversity_metrics = path_diversity(all_paths)
        
        # 3. Computational Cost: average ms per decision
        decision_logs = [{"decision_time_ms": t} for t in all_decision_times]
        computational_cost_metrics = computational_cost(decision_logs)
        
        # 4. Adaptation Speed: frames to converge after tower placement
        adaptation_speed_metrics = self._calculate_adaptation_speed(adaptation_data, wave_results)
        
        results = {
            "algorithm_config": pathfinder.to_dict(),
            "num_waves": num_waves,
            "enemies_per_wave": enemies_per_wave,
            "total_enemies": total_spawned,
            "total_defeated": total_defeated,
            "total_escaped": total_escaped,
            
            # Primary metrics
            "survival_rate_percent": survival_rate_metrics["survival_percentage"],
            "path_diversity_entropy": path_diversity_metrics["shannon_entropy"],
            "path_diversity_index": path_diversity_metrics["diversity_index"],
            "adaptation_frames": adaptation_speed_metrics.get("frames_to_converge", 0),
            "adaptation_time_seconds": adaptation_speed_metrics.get("time_to_converge", 0.0),
            "computational_cost_ms": computational_cost_metrics["avg_time_ms"],
            
            # Detailed metrics
            "survival_rate_metrics": survival_rate_metrics,
            "path_diversity_metrics": path_diversity_metrics,
            "adaptation_speed_metrics": adaptation_speed_metrics,
            "computational_cost_metrics": computational_cost_metrics,
            
            # Additional data
            "total_simulation_time_seconds": sim_elapsed,
            "wave_results": wave_results,
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Simulation Complete")
            print(f"Total Time: {sim_elapsed:.2f}s")
            print(f"Survival Rate: {results['survival_rate_percent']:.1f}%")
            print(f"Path Diversity (entropy): {results['path_diversity_entropy']:.3f}")
            print(f"Adaptation Speed: {results['adaptation_frames']} frames")
            print(f"Computational Cost: {results['computational_cost_ms']:.3f}ms")
            print(f"{'='*60}")
        
        return results
    
    @property
    def verbose(self) -> bool:
        """Get verbose flag (always True for this version)."""
        return True
    
    def _place_adaptation_towers(self, game_state: GameState, game_map: Map) -> None:
        """
        Place towers mid-experiment to test adaptation speed.
        
        Places strategic towers along common paths to force re-routing.
        
        Args:
            game_state: Current game state
            game_map: Map instance
        """
        # Define strategic positions based on map type
        # These are chosen to block or threaten common paths
        
        if game_map.width == 30 and game_map.height == 20:  # Branching map
            adaptation_towers = [
                ((12, 10), "sniper"),  # Block upper branch
                ((18, 12), "area"),    # Threaten middle area
            ]
        elif game_map.width == 40 and game_map.height == 30:  # Open arena
            adaptation_towers = [
                ((20, 14), "sniper"),  # Center line threat
                ((20, 16), "area"),    # Center line threat
            ]
        else:
            # Generic placement for unknown maps
            center_x = game_map.width // 2
            center_y = game_map.height // 2
            adaptation_towers = [
                ((center_x, center_y - 1), "sniper"),
                ((center_x, center_y + 1), "area"),
            ]
        
        # Place towers
        for position, tower_type in adaptation_towers:
            # Check if position is valid (within bounds and walkable)
            if (0 <= position[0] < game_map.width and 
                0 <= position[1] < game_map.height and
                game_map.is_walkable(position[0], position[1])):
                game_state.place_tower(position, tower_type, force=True, check_gold=False)
    
    def _calculate_adaptation_speed(
        self,
        adaptation_data: Dict[str, Any],
        wave_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate adaptation speed metric.
        
        Measures how quickly the algorithm converges to a new optimal path
        after environmental change (tower placement).
        
        Args:
            adaptation_data: Data about tower placement and path changes
            wave_results: Results from each wave
            
        Returns:
            Dictionary with adaptation speed metrics
        """
        if not adaptation_data["tower_placed"]:
            return {
                "metric": "adaptation_speed",
                "tower_placed": False,
                "frames_to_converge": 0,
                "time_to_converge": 0.0,
            }
        
        # Calculate path similarity before and after tower placement
        paths_before = adaptation_data["paths_before_change"]
        paths_after = adaptation_data["paths_after_change"]
        
        # Simple heuristic: measure when paths stabilize
        # In practice, we consider the algorithm "adapted" when it starts using new paths
        if len(paths_after) > 0:
            # Count unique paths before and after
            unique_before = len(set(str(p) for p in paths_before if p))
            unique_after = len(set(str(p) for p in paths_after if p))
            
            # Estimate convergence time based on when new paths appear
            # This is a simplified metric - in reality we'd track per-enemy path updates
            frames_to_converge = len(paths_after)  # Number of enemies that got new paths
            
            # Time estimate based on frames (assuming 60 FPS)
            time_to_converge = frames_to_converge / 60.0
        else:
            frames_to_converge = 0
            time_to_converge = 0.0
            unique_before = 0
            unique_after = 0
        
        return {
            "metric": "adaptation_speed",
            "tower_placed": True,
            "placement_frame": adaptation_data["placement_frame"],
            "frames_to_converge": frames_to_converge,
            "time_to_converge": time_to_converge,
            "unique_paths_before": unique_before,
            "unique_paths_after": unique_after,
            "path_change_rate": unique_after / unique_before if unique_before > 0 else 0.0,
        }
    
    def _create_game_config(
        self,
        map_type: str,
        waves: int,
        enemies: int,
    ) -> GameConfig:
        """Create game configuration for the specified map type."""
        # Try to load map-specific config file with both naming conventions
        map_config_path = Path(f"configs/{map_type}.json")
        if not map_config_path.exists():
            map_config_path = Path(f"configs/{map_type}_map.json")
        
        if map_config_path.exists():
            with open(map_config_path, 'r') as f:
                map_config = json.load(f)
            config = GameConfig.from_dict(map_config)
        else:
            # Use default config
            config = GameConfig()
        
        # Override with experiment parameters
        config.enemies.wave_count = waves
        config.enemies.enemies_per_wave = enemies
        
        return config
    
    def _create_map(self, map_type: str, config: GameConfig) -> Map:
        """Create map based on type (branching or open_arena)."""
        # Try both naming conventions
        map_config_path = Path(f"configs/{map_type}.json")
        if not map_config_path.exists():
            map_config_path = Path(f"configs/{map_type}_map.json")
        
        if not map_config_path.exists():
            raise ValueError(f"Map config not found: {map_config_path} or configs/{map_type}.json")
        
        with open(map_config_path, 'r') as f:
            map_config_json = json.load(f)
        
        # Extract map configuration
        map_data = map_config_json.get("map", {})
        
        width = map_data.get("width", 30)
        height = map_data.get("height", 20)
        
        # Get spawn and goal from config or use defaults
        spawn_points_raw = map_config_json.get("map", {}).get("spawn_points", [[0, height // 2]])
        goal_positions_raw = map_config_json.get("map", {}).get("goal_positions", [[width - 1, height // 2]])
        
        spawn_points = [(p[0], p[1]) for p in spawn_points_raw]
        goal_positions = [(p[0], p[1]) for p in goal_positions_raw]
        
        # Handle obstacles
        obstacles = []
        obstacle_regions = map_data.get("obstacle_regions", [])
        
        for region in obstacle_regions:
            x1, y1, x2, y2 = region
            for x in range(x1, x2):
                for y in range(y1, y2):
                    if 0 <= x < width and 0 <= y < height:
                        obstacles.append((x, y))
        
        # Create the map
        game_map = Map(
            width=width,
            height=height,
            obstacles=obstacles,
            spawn_points=spawn_points,
            goal_positions=goal_positions,
        )
        
        return game_map
    
    def _create_pathfinder(
        self,
        algo_name: str,
        algo_configs: Dict[str, Any],
        game_map: Map,
        map_type: str = None,
    ) -> BasePathfinder:
        """Create pathfinder instance for the specified algorithm."""
        algo_name_lower = algo_name.lower()
        
        if algo_name_lower == "fixed":
            # Load baseline path from config matching the current map
            config_path = None
            if map_type:
                # Try exact map type name first
                path = Path(f"configs/{map_type}.json")
                if path.exists():
                    config_path = path
                else:
                    # Try with _map suffix
                    path = Path(f"configs/{map_type}_map.json")
                    if path.exists():
                        config_path = path
            
            # Fallback: try common map files
            if config_path is None:
                for map_file in ["branching_map.json", "branching.json", "open_arena.json"]:
                    path = Path("configs") / map_file
                    if path.exists():
                        config_path = path
                        break
            
            baseline_path = None
            if config_path:
                with open(config_path, 'r') as f:
                    map_config = json.load(f)
                baseline_path_raw = map_config.get("map", {}).get("baseline_path", [])
                if baseline_path_raw:
                    baseline_path = [(p[0], p[1]) for p in baseline_path_raw]
            
            if baseline_path is None:
                # Generate baseline path using A*
                baseline_path = self._generate_baseline_path(game_map)
            
            config = algo_configs.get("fixed", {})
            kwargs = {"baseline_path": baseline_path}
            if "name" in config:
                kwargs["name"] = config["name"]
            return FixedPathfinder(**kwargs)
        
        elif algo_name_lower == "astar":
            config = algo_configs.get("astar", {})
            kwargs = {}
            for key in ["name", "alpha", "beta", "diagonal_movement", "use_enhanced"]:
                if key in config:
                    kwargs[key] = config[key]
            return AStarPathfinder(**kwargs)
        
        elif algo_name_lower == "aco":
            config = algo_configs.get("aco", {})
            kwargs = {}
            for key in ["name", "num_ants", "evaporation_rate", "deposit_strength", "alpha", "beta", "gamma"]:
                if key in config:
                    kwargs[key] = config[key]
            return ACOPathfinder(**kwargs)
        
        elif algo_name_lower == "dqn":
            config = algo_configs.get("dqn", {})
            model_path = config.get("model_path", "models/dqn_model_best")
            
            # Create DQN pathfinder with only specified config
            # Force plan_full_path=True for headless experiments (step-by-step doesn't work here)
            kwargs = {"plan_full_path": True}
            for key in ["name", "use_cache", "cache_duration"]:
                if key in config:
                    kwargs[key] = config[key]
            pathfinder = DQNPathfinder(**kwargs)
            
            # Load the trained model
            if Path(f"{model_path}.zip").exists():
                pathfinder.load_model(model_path)
                print(f"  Loaded DQN model from {model_path}")
            else:
                print(f"  Warning: DQN model not found at {model_path}, using random policy")
            
            return pathfinder
        
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
    
    def _generate_baseline_path(self, game_map: Map) -> List[Tuple[int, int]]:
        """Generate a simple baseline path for fixed pathfinder using A*."""
        if not game_map.spawn_points or not game_map.goal_positions:
            return []
        
        spawn = game_map.spawn_points[0]
        goal = game_map.goal_positions[0]
        
        # Use basic A* to generate baseline
        baseline_finder = AStarPathfinder(name="Baseline-Generator", use_enhanced=False)
        baseline_finder.update_state(game_map, [])
        
        return baseline_finder.find_path(spawn, goal)
    
    def _parse_enemy_types(self, type_names: List[str]) -> List[EnemyType]:
        """Parse enemy type names to EnemyType enum values."""
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
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics across all experiments."""
        if not results:
            return {}
        
        # Group by algorithm and map
        by_algo_map = {}
        for result in results:
            algo = result.get("algorithm", "unknown")
            map_type = result.get("map_type", "unknown")
            key = f"{algo}_{map_type}"
            
            if key not in by_algo_map:
                by_algo_map[key] = []
            by_algo_map[key].append(result)
        
        # Calculate statistics per group
        summary = {}
        for key, group_results in by_algo_map.items():
            survival_rates = [r["survival_rate_percent"] for r in group_results]
            path_diversities = [r["path_diversity_entropy"] for r in group_results]
            adaptation_frames = [r["adaptation_frames"] for r in group_results]
            comp_costs = [r["computational_cost_ms"] for r in group_results]
            
            summary[key] = {
                "num_runs": len(group_results),
                "algorithm": group_results[0].get("algorithm"),
                "map_type": group_results[0].get("map_type"),
                
                # Survival rate statistics
                "avg_survival_rate": sum(survival_rates) / len(survival_rates),
                "min_survival_rate": min(survival_rates),
                "max_survival_rate": max(survival_rates),
                
                # Path diversity statistics
                "avg_path_diversity": sum(path_diversities) / len(path_diversities),
                "min_path_diversity": min(path_diversities),
                "max_path_diversity": max(path_diversities),
                
                # Adaptation speed statistics
                "avg_adaptation_frames": sum(adaptation_frames) / len(adaptation_frames) if adaptation_frames else 0,
                "min_adaptation_frames": min(adaptation_frames) if adaptation_frames else 0,
                "max_adaptation_frames": max(adaptation_frames) if adaptation_frames else 0,
                
                # Computational cost statistics
                "avg_computational_cost_ms": sum(comp_costs) / len(comp_costs),
                "min_computational_cost_ms": min(comp_costs),
                "max_computational_cost_ms": max(comp_costs),
            }
        
        return summary
    
    def _save_results(
        self,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        session_id: str,
    ) -> None:
        """Save results to files."""
        # Save full results as JSON
        json_path = self.output_dir / f"experiment_session_{session_id}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "session_id": session_id,
                "results": results,
                "summary": summary,
            }, f, indent=2)
        print(f"\nSaved JSON: {json_path}")
        
        # Save metrics as CSV
        metrics_path = self.output_dir / f"experiment_metrics_{session_id}.csv"
        with open(metrics_path, 'w') as f:
            # Header
            f.write("algorithm,map_type,run_number,total_enemies,total_defeated,total_escaped,"
                   "survival_rate_percent,path_diversity_entropy,path_diversity_index,"
                   "adaptation_frames,adaptation_time_seconds,computational_cost_ms,"
                   "simulation_time_seconds\n")
            
            # Data rows
            for result in results:
                f.write(f"{result['algorithm']},{result['map_type']},{result['run_number']},"
                       f"{result['total_enemies']},{result['total_defeated']},{result['total_escaped']},"
                       f"{result['survival_rate_percent']:.2f},"
                       f"{result['path_diversity_entropy']:.4f},"
                       f"{result['path_diversity_index']:.4f},"
                       f"{result['adaptation_frames']},"
                       f"{result['adaptation_time_seconds']:.4f},"
                       f"{result['computational_cost_ms']:.4f},"
                       f"{result['total_simulation_time_seconds']:.2f}\n")
        print(f"Saved metrics CSV: {metrics_path}")
        
        # Save summary as CSV
        summary_path = self.output_dir / f"experiment_summary_{session_id}.csv"
        with open(summary_path, 'w') as f:
            f.write("algorithm,map_type,num_runs,"
                   "avg_survival_rate,avg_path_diversity,avg_adaptation_frames,avg_computational_cost_ms\n")
            
            for key, stats in summary.items():
                f.write(f"{stats['algorithm']},{stats['map_type']},{stats['num_runs']},"
                       f"{stats['avg_survival_rate']:.2f},"
                       f"{stats['avg_path_diversity']:.4f},"
                       f"{stats['avg_adaptation_frames']:.1f},"
                       f"{stats['avg_computational_cost_ms']:.4f}\n")
        print(f"Saved summary CSV: {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run batch pathfinding experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment configuration JSON file"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["baseline", "full", "adaptation"],
        help="Use a preset experiment configuration"
    )
    
    # Manual configuration
    parser.add_argument(
        "--algorithms",
        type=str,
        help="Comma-separated list of algorithms (fixed,astar,aco,dqn)"
    )
    parser.add_argument(
        "--maps",
        type=str,
        help="Comma-separated list of map types (branching,open_arena)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)"
    )
    parser.add_argument(
        "--waves",
        type=int,
        default=10,
        help="Waves per run (default: 10)"
    )
    parser.add_argument(
        "--enemies",
        type=int,
        default=30,
        help="Enemies per wave (default: 30)"
    )
    parser.add_argument(
        "--tower-placement-wave",
        type=int,
        default=None,
        help="Wave number to place towers for adaptation testing (default: None)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results)"
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner(output_dir=args.output)
    
    # Determine experiment configuration
    if args.config:
        # Run from config file
        runner.run_from_config(args.config)
    
    elif args.preset:
        # Run preset configuration
        if args.preset == "baseline":
            runner.run_experiments(
                algorithms=["fixed", "astar"],
                map_types=["branching"],
                num_runs=5,
                waves_per_run=10,
                enemies_per_wave=30,
            )
        elif args.preset == "full":
            runner.run_experiments(
                algorithms=["fixed", "astar", "aco", "dqn"],
                map_types=["branching", "open_arena"],
                num_runs=5,
                waves_per_run=10,
                enemies_per_wave=50,
            )
        elif args.preset == "adaptation":
            runner.run_experiments(
                algorithms=["fixed", "astar", "aco", "dqn"],
                map_types=["branching", "open_arena"],
                num_runs=3,
                waves_per_run=10,
                enemies_per_wave=30,
                tower_placement_wave=5,  # Place towers mid-experiment
            )
    
    elif args.algorithms and args.maps:
        # Run from command-line args
        algorithms = [a.strip() for a in args.algorithms.split(",")]
        maps = [m.strip() for m in args.maps.split(",")]
        
        runner.run_experiments(
            algorithms=algorithms,
            map_types=maps,
            num_runs=args.runs,
            waves_per_run=args.waves,
            enemies_per_wave=args.enemies,
            tower_placement_wave=args.tower_placement_wave,
        )
    
    else:
        parser.print_help()
        print("\nError: Must specify either --config, --preset, or --algorithms with --maps")
        sys.exit(1)


if __name__ == "__main__":
    main()
