#!/usr/bin/env python
"""
Batch experiment runner for pathfinding algorithm evaluation.

This script runs multiple experiments across different algorithms and maps,
collecting performance data for analysis and reporting.

Usage:
    python scripts/run_experiments.py --config configs/my_experiment.json
    python scripts/run_experiments.py --preset baseline
    python scripts/run_experiments.py --algorithms fixed,astar,aco --maps simple,branching --runs 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apathion.config import GameConfig
from apathion.game.map import Map
from apathion.pathfinding.fixed import FixedPathfinder
from apathion.pathfinding.astar import AStarPathfinder
from apathion.pathfinding.aco import ACOPathfinder
from apathion.evaluation.headless_simulator import HeadlessSimulator
from apathion.evaluation.logger import GameLogger


class ExperimentRunner:
    """
    Batch experiment runner for algorithm comparison.
    
    Manages multiple experiment runs across different configurations,
    collecting and exporting results for analysis.
    """
    
    def __init__(self, output_dir: str = "data/results"):
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
        map_types = exp_config.get("map_types", ["simple"])
        num_runs = exp_config.get("num_runs", 3)
        waves_per_run = exp_config.get("waves_per_run", 10)
        enemies_per_wave = exp_config.get("enemies_per_wave", 30)
        
        # Run experiments
        return self.run_experiments(
            algorithms=algorithms,
            map_types=map_types,
            num_runs=num_runs,
            waves_per_run=waves_per_run,
            enemies_per_wave=enemies_per_wave,
            algo_configs=exp_config,
        )
    
    def run_experiments(
        self,
        algorithms: List[str],
        map_types: List[str],
        num_runs: int = 3,
        waves_per_run: int = 10,
        enemies_per_wave: int = 30,
        algo_configs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run batch experiments across algorithms and maps.
        
        Args:
            algorithms: List of algorithm names
            map_types: List of map type names
            num_runs: Number of runs per configuration
            waves_per_run: Waves per run
            enemies_per_wave: Enemies per wave
            algo_configs: Optional algorithm-specific configurations
            
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
                    pathfinder = self._create_pathfinder(algo_name, algo_configs, game_map)
                    
                    # Create logger for this run
                    logger = GameLogger()
                    
                    # Run simulation
                    simulator = HeadlessSimulator(config, logger, verbose=True)
                    result = simulator.run_simulation(
                        game_map=game_map,
                        pathfinder=pathfinder,
                        num_waves=waves_per_run,
                        enemies_per_wave=enemies_per_wave,
                        initial_towers=None,  # Use config defaults
                    )
                    
                    # Add metadata
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
    
    def _create_game_config(
        self,
        map_type: str,
        waves: int,
        enemies: int,
    ) -> GameConfig:
        """Create game configuration."""
        # Try to load map-specific config file
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
        """Create map based on type."""
        # Always load branching map from config file
        if map_type == "branching":
            map_config_path = Path("configs/branching_map.json")
            if map_config_path.exists():
                with open(map_config_path, 'r') as f:
                    map_config_json = json.load(f)
                
                # Extract map configuration
                map_data = map_config_json.get("map", {})
                
                # Create map using the data
                width = map_data.get("width", 30)
                height = map_data.get("height", 20)
                obstacle_regions = map_data.get("obstacle_regions", [])
                
                # Convert obstacle regions to individual obstacles
                obstacles = []
                for region in obstacle_regions:
                    x1, y1, x2, y2 = region
                    for x in range(x1, x2):
                        for y in range(y1, y2):
                            if 0 <= x < width and 0 <= y < height:
                                obstacles.append((x, y))
                
                # Create the map
                return Map(
                    width=width,
                    height=height,
                    obstacles=obstacles,
                    spawn_points=[(0, 11)],  # From branching_map.json
                    goal_positions=[(29, 5)],  # From branching_map.json
                )
            else:
                print(f"Warning: branching_map.json not found, using default branching map")
                return Map.create_branching_map()
        else:
            print(f"Warning: Only branching map is supported. Using branching map.")
            map_config_path = Path("configs/branching_map.json")
            if map_config_path.exists():
                with open(map_config_path, 'r') as f:
                    map_config_json = json.load(f)
                map_data = map_config_json.get("map", {})
                width = map_data.get("width", 30)
                height = map_data.get("height", 20)
                obstacle_regions = map_data.get("obstacle_regions", [])
                obstacles = []
                for region in obstacle_regions:
                    x1, y1, x2, y2 = region
                    for x in range(x1, x2):
                        for y in range(y1, y2):
                            if 0 <= x < width and 0 <= y < height:
                                obstacles.append((x, y))
                return Map(
                    width=width,
                    height=height,
                    obstacles=obstacles,
                    spawn_points=[(0, 11)],
                    goal_positions=[(29, 5)],
                )
            return Map.create_branching_map()
    
    def _create_pathfinder(
        self,
        algo_name: str,
        algo_configs: Dict[str, Any],
        game_map: Map,
    ) -> Any:
        """Create pathfinder instance."""
        algo_name_lower = algo_name.lower()
        
        if algo_name_lower == "fixed":
            # Load baseline path from branching_map.json
            baseline_path = None
            map_config_path = Path("configs/branching_map.json")
            if map_config_path.exists():
                with open(map_config_path, 'r') as f:
                    map_config = json.load(f)
                baseline_path_raw = map_config.get("map", {}).get("baseline_path", [])
                if baseline_path_raw:
                    baseline_path = [(p[0], p[1]) for p in baseline_path_raw]
                    print(f"  Loaded baseline path with {len(baseline_path)} waypoints from branching_map.json")
            
            if baseline_path is None:
                print("  Warning: No baseline path found, generating one")
                baseline_path = self._generate_baseline_path(game_map)
            
            config = algo_configs.get("fixed", {})
            return FixedPathfinder(
                name=config.get("name", "Fixed-Path"),
                baseline_path=baseline_path,
            )
        
        elif algo_name_lower == "astar":
            config = algo_configs.get("astar", {})
            return AStarPathfinder(
                name=config.get("name", "A*-Enhanced"),
                alpha=config.get("alpha", 0.5),
                beta=config.get("beta", 0.3),
                diagonal_movement=config.get("diagonal_movement", True),
                use_enhanced=config.get("use_enhanced", True),
            )
        
        elif algo_name_lower == "astar_basic":
            # Basic A* with only distance costs (no damage/congestion)
            config = algo_configs.get("astar_basic", {})
            return AStarPathfinder(
                name=config.get("name", "A*-Basic"),
                alpha=config.get("alpha", 0.0),
                beta=config.get("beta", 0.0),
                diagonal_movement=config.get("diagonal_movement", True),
                use_enhanced=False,
            )
        
        elif algo_name_lower == "astar_enhanced":
            # Enhanced A* with damage and congestion costs
            config = algo_configs.get("astar_enhanced", {})
            return AStarPathfinder(
                name=config.get("name", "A*-Enhanced"),
                alpha=config.get("alpha", 0.5),
                beta=config.get("beta", 0.3),
                diagonal_movement=config.get("diagonal_movement", True),
                use_enhanced=True,
            )
        
        elif algo_name_lower == "aco":
            config = algo_configs.get("aco", {})
            return ACOPathfinder(
                name=config.get("name", "ACO"),
                num_ants=config.get("num_ants", 10),
                evaporation_rate=config.get("evaporation_rate", 0.01),
                deposit_strength=config.get("deposit_strength", 1.0),
                alpha=config.get("alpha", 1.0),
                beta=config.get("beta", 1.5),
                gamma=config.get("gamma", 3.5),
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
    
    def _generate_baseline_path(self, game_map: Map) -> List[tuple]:
        """Generate a simple baseline path for fixed pathfinder."""
        if not game_map.spawn_points or not game_map.goal_positions:
            return []
        
        spawn = game_map.spawn_points[0]
        goal = game_map.goal_positions[0]
        
        # Use A* to generate baseline
        from apathion.pathfinding.astar import AStarPathfinder
        baseline_finder = AStarPathfinder(name="Baseline-Generator", use_enhanced=False)
        baseline_finder.update_state(game_map, [])
        
        return baseline_finder.find_path(spawn, goal)
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not results:
            return {}
        
        # Group by algorithm
        by_algorithm = {}
        for result in results:
            algo = result["algorithm"]
            if algo not in by_algorithm:
                by_algorithm[algo] = []
            by_algorithm[algo].append(result)
        
        # Calculate statistics per algorithm
        summary = {}
        for algo, algo_results in by_algorithm.items():
            survival_rates = [r["survival_rate_percent"] for r in algo_results]
            decision_times = [r["avg_decision_time_ms"] for r in algo_results]
            
            summary[algo] = {
                "num_runs": len(algo_results),
                "avg_survival_rate": sum(survival_rates) / len(survival_rates),
                "min_survival_rate": min(survival_rates),
                "max_survival_rate": max(survival_rates),
                "avg_decision_time_ms": sum(decision_times) / len(decision_times),
                "min_decision_time_ms": min(decision_times),
                "max_decision_time_ms": max(decision_times),
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
                   "survival_rate_percent,avg_decision_time_ms,max_decision_time_ms,"
                   "simulation_time_seconds\n")
            
            # Data rows
            for result in results:
                f.write(f"{result['algorithm']},{result['map_type']},{result['run_number']},"
                       f"{result['total_enemies']},{result['total_defeated']},{result['total_escaped']},"
                       f"{result['survival_rate_percent']:.2f},{result['avg_decision_time_ms']:.4f},"
                       f"{result['max_decision_time_ms']:.4f},{result['total_simulation_time_seconds']:.2f}\n")
        print(f"Saved CSV: {metrics_path}")
        
        # Save decision logs as CSV
        decisions_path = self.output_dir / f"experiment_decisions_{session_id}.csv"
        with open(decisions_path, 'w') as f:
            f.write("algorithm,map_type,run_number,wave,avg_decision_time_ms\n")
            
            for result in results:
                for wave_result in result.get("wave_results", []):
                    f.write(f"{result['algorithm']},{result['map_type']},{result['run_number']},"
                           f"{wave_result['wave_number']},{result['avg_decision_time_ms']:.4f}\n")
        print(f"Saved decisions CSV: {decisions_path}")


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
        choices=["baseline", "comparison", "full"],
        help="Use a preset experiment configuration"
    )
    
    # Manual configuration
    parser.add_argument(
        "--algorithms",
        type=str,
        help="Comma-separated list of algorithms (fixed,astar,aco)"
    )
    parser.add_argument(
        "--maps",
        type=str,
        help="Comma-separated list of map types"
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
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="data/results",
        help="Output directory (default: data/results)"
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
                map_types=["simple"],
                num_runs=5,
                waves_per_run=10,
                enemies_per_wave=30,
            )
        elif args.preset == "comparison":
            runner.run_experiments(
                algorithms=["fixed", "astar", "aco"],
                map_types=["simple", "branching"],
                num_runs=5,
                waves_per_run=10,
                enemies_per_wave=50,
            )
        elif args.preset == "full":
            runner.run_experiments(
                algorithms=["fixed", "astar", "aco"],
                map_types=["simple", "branching", "open_arena"],
                num_runs=5,
                waves_per_run=15,
                enemies_per_wave=50,
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
        )
    
    else:
        parser.print_help()
        print("\nError: Must specify either --config, --preset, or --algorithms with --maps")
        sys.exit(1)


if __name__ == "__main__":
    main()

