"""
Command-line interface for Apathion using Fire.

This module provides CLI commands for running experiments, training models,
and analyzing results.
"""

import fire
from typing import Optional, List
from pathlib import Path

from apathion.game.map import Map
from apathion.game.game import GameState
from apathion.pathfinding.astar import AStarPathfinder
from apathion.pathfinding.aco import ACOPathfinder
from apathion.pathfinding.dqn import DQNPathfinder
from apathion.evaluation.logger import GameLogger
from apathion.evaluation.evaluator import Evaluator
from apathion.config import (
    GameConfig,
    ExperimentConfig,
    get_experiment_preset,
    AStarConfig,
    ACOConfig,
    DQNConfig,
)


class ApathionCLI:
    """
    Command-line interface for the Apathion framework.
    
    Commands:
        play: Run an interactive game session
        evaluate: Run evaluation experiments
        train: Train DQN model (placeholder)
        analyze: Analyze logged results
    """
    
    def play(
        self,
        algorithm: str = "astar",
        map_type: str = "simple",
        waves: int = 5,
        enemies: int = 10,
        config_file: Optional[str] = None,
    ):
        """
        Run an interactive game session.
        
        Args:
            algorithm: Pathfinding algorithm to use (astar, aco, dqn, fixed)
            map_type: Type of map (simple, branching, open_arena)
            waves: Number of waves to spawn
            enemies: Enemies per wave
            config_file: Optional path to config JSON file
        
        Example:
            apathion play --algorithm=astar --map_type=branching --waves=10
        """
        print(f"Starting Apathion with {algorithm} on {map_type} map...")
        
        # Load configuration
        if config_file:
            config = GameConfig.from_json(config_file)
        else:
            config = GameConfig()
            config.algorithm = algorithm
            config.enemies.wave_count = waves
            config.enemies.enemies_per_wave = enemies
        
        # Create map
        game_map = self._create_map(map_type)
        print(f"Map created: {game_map.width}x{game_map.height}")
        
        # Create pathfinder
        pathfinder = self._create_pathfinder(algorithm)
        print(f"Pathfinder initialized: {pathfinder.get_name()}")
        
        # Create game state
        game = GameState(game_map)
        
        # Place some initial towers (placeholder positions)
        game.place_tower((10, 10), "basic")
        game.place_tower((15, 10), "basic")
        print(f"Towers placed: {len(game.towers)}")
        
        # Update pathfinder with game state
        pathfinder.update_state(game.map, game.towers)
        
        # Spawn and run waves
        for wave_num in range(waves):
            print(f"\nWave {wave_num + 1}/{waves}")
            enemies_list = game.spawn_wave(num_enemies=enemies)
            print(f"  Spawned {len(enemies_list)} enemies")
            
            # Assign paths
            goal = game.map.goal_positions[0]
            for enemy in enemies_list:
                start = (int(enemy.position[0]), int(enemy.position[1]))
                path = pathfinder.find_path(start, goal, enemy_id=enemy.id)
                enemy.set_path(path)
            
            print(f"  Paths calculated")
            
            # Placeholder: In a real implementation, would run game simulation here
            # For now, just report that wave completed
            print(f"  Wave {wave_num + 1} completed")
        
        # Show final statistics
        stats = game.get_statistics()
        print(f"\n{'-' * 50}")
        print(f"Game Statistics:")
        print(f"  Total enemies spawned: {stats['enemies_spawned']}")
        print(f"  Enemies defeated: {stats['enemies_defeated']}")
        print(f"  Enemies escaped: {stats['enemies_escaped']}")
        print(f"  Survival rate: {stats['survival_rate']:.1f}%")
        print(f"{'-' * 50}")
    
    def evaluate(
        self,
        algorithms: Optional[List[str]] = None,
        maps: Optional[List[str]] = None,
        waves: int = 5,
        enemies: int = 30,
        runs: int = 3,
        preset: Optional[str] = None,
        config_file: Optional[str] = None,
        output: str = "data/results",
    ):
        """
        Run comparative evaluation experiments.
        
        Args:
            algorithms: List of algorithms to compare (default: [astar, aco])
            maps: List of map types to test (default: [simple, branching])
            waves: Number of waves per experiment
            enemies: Enemies per wave
            runs: Number of repetitions per configuration
            preset: Use predefined experiment preset (baseline, full_comparison, performance_test)
            config_file: Optional path to experiment config JSON file
            output: Directory for results
        
        Example:
            apathion evaluate --algorithms=astar,aco --maps=simple,branching --waves=10
            apathion evaluate --preset=full_comparison
        """
        print("Starting evaluation experiments...")
        
        # Load configuration
        if config_file:
            exp_config = ExperimentConfig.from_json(config_file)
        elif preset:
            exp_config = get_experiment_preset(preset)
            print(f"Using preset: {preset}")
        else:
            exp_config = ExperimentConfig()
            if algorithms:
                exp_config.algorithms = algorithms
            if maps:
                exp_config.map_types = maps
            exp_config.waves_per_run = waves
            exp_config.enemies_per_wave = enemies
            exp_config.num_runs = runs
        
        print(f"Configuration:")
        print(f"  Algorithms: {', '.join(exp_config.algorithms)}")
        print(f"  Maps: {', '.join(exp_config.map_types)}")
        print(f"  Runs: {exp_config.num_runs}")
        print(f"  Waves per run: {exp_config.waves_per_run}")
        print(f"  Enemies per wave: {exp_config.enemies_per_wave}")
        
        # Create evaluator
        logger = GameLogger(output_dir=output)
        evaluator = Evaluator(logger=logger)
        
        # Create test maps
        test_maps = [self._create_map(mt) for mt in exp_config.map_types]
        
        # Run experiments for each algorithm
        for run_num in range(exp_config.num_runs):
            print(f"\n{'=' * 60}")
            print(f"Run {run_num + 1}/{exp_config.num_runs}")
            print(f"{'=' * 60}")
            
            for algo_name in exp_config.algorithms:
                print(f"\nTesting algorithm: {algo_name}")
                
                # Create pathfinder
                pathfinder = self._create_pathfinder(algo_name, exp_config)
                
                # Compare across maps
                result = evaluator.compare_algorithms(
                    algorithms=[pathfinder],
                    test_maps=test_maps,
                    map_names=exp_config.map_types,
                    num_waves=exp_config.waves_per_run,
                    enemies_per_wave=exp_config.enemies_per_wave,
                )
                
                print(f"  Completed: {len(result['comparison_results'])} experiments")
        
        # Generate and display report
        print(f"\n{'=' * 60}")
        print("Generating final report...")
        print(f"{'=' * 60}\n")
        
        report = evaluator.generate_report()
        print(report)
        
        # Export results
        exported = evaluator.export_results(prefix=f"{exp_config.name}_")
        print(f"\nResults exported:")
        for key, filepath in exported.items():
            print(f"  {key}: {filepath}")
    
    def train(
        self,
        episodes: int = 1000,
        map_type: str = "simple",
        save_path: str = "models/dqn_model.pth",
        config_file: Optional[str] = None,
    ):
        """
        Train a DQN model (placeholder).
        
        Args:
            episodes: Number of training episodes
            map_type: Type of map to train on
            save_path: Path to save trained model
            config_file: Optional path to training config JSON file
        
        Example:
            apathion train --episodes=5000 --map_type=branching
        """
        print(f"Training DQN model for {episodes} episodes...")
        print(f"Map type: {map_type}")
        print(f"Save path: {save_path}")
        
        # PLACEHOLDER: Actual training implementation
        print("\n⚠️  DQN training not yet implemented.")
        print("This is a placeholder for the training pipeline.")
        print("\nPlanned implementation:")
        print("  1. Initialize DQN agent with neural network")
        print("  2. Create training environment with specified map")
        print("  3. Run episodes with experience replay")
        print("  4. Track training metrics (loss, reward, success rate)")
        print("  5. Save trained model to specified path")
        print("\nTo implement, integrate stable-baselines3 or PyTorch RL framework.")
    
    def analyze(
        self,
        log_dir: str = "data/logs",
        output: Optional[str] = None,
    ):
        """
        Analyze logged results and generate visualizations.
        
        Args:
            log_dir: Directory containing log files
            output: Optional output file for analysis report
        
        Example:
            apathion analyze --log_dir=data/logs --output=analysis_report.txt
        """
        print(f"Analyzing logs from: {log_dir}")
        
        log_path = Path(log_dir)
        if not log_path.exists():
            print(f"Error: Log directory not found: {log_dir}")
            return
        
        # Find CSV files
        csv_files = list(log_path.glob("*.csv"))
        json_files = list(log_path.glob("*.json"))
        
        print(f"\nFound {len(csv_files)} CSV files and {len(json_files)} JSON files")
        
        if csv_files:
            print("\nCSV files:")
            for f in csv_files:
                print(f"  - {f.name}")
        
        if json_files:
            print("\nJSON files:")
            for f in json_files:
                print(f"  - {f.name}")
        
        # PLACEHOLDER: Actual analysis implementation
        print("\n⚠️  Detailed analysis not yet implemented.")
        print("This is a placeholder for the analysis pipeline.")
        print("\nPlanned features:")
        print("  - Load and parse CSV/JSON logs")
        print("  - Calculate aggregate statistics")
        print("  - Generate comparison plots (survival rate, path diversity, etc.)")
        print("  - Statistical significance testing")
        print("  - Export analysis report")
    
    def _create_map(self, map_type: str) -> Map:
        """Create a map of the specified type."""
        if map_type == "simple":
            return Map.create_simple_map()
        elif map_type == "branching":
            return Map.create_branching_map()
        elif map_type == "open_arena":
            return Map.create_open_arena()
        else:
            print(f"Unknown map type '{map_type}', using simple map")
            return Map.create_simple_map()
    
    def _create_pathfinder(
        self,
        algorithm: str,
        exp_config: Optional[ExperimentConfig] = None
    ):
        """Create a pathfinder of the specified type."""
        if algorithm == "astar":
            if exp_config and hasattr(exp_config, 'astar'):
                cfg = exp_config.astar
                return AStarPathfinder(
                    name=cfg.name,
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    diagonal_movement=cfg.diagonal_movement,
                )
            return AStarPathfinder()
        
        elif algorithm == "aco":
            if exp_config and hasattr(exp_config, 'aco'):
                cfg = exp_config.aco
                return ACOPathfinder(
                    name=cfg.name,
                    num_ants=cfg.num_ants,
                    evaporation_rate=cfg.evaporation_rate,
                    deposit_strength=cfg.deposit_strength,
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                )
            return ACOPathfinder()
        
        elif algorithm == "dqn":
            if exp_config and hasattr(exp_config, 'dqn'):
                cfg = exp_config.dqn
                return DQNPathfinder(
                    name=cfg.name,
                    state_size=cfg.state_size,
                    action_size=cfg.action_size,
                    use_cache=cfg.use_cache,
                    cache_duration=cfg.cache_duration,
                )
            return DQNPathfinder()
        
        else:
            print(f"Unknown algorithm '{algorithm}', using A*")
            return AStarPathfinder()


def main():
    """Main entry point for the CLI."""
    fire.Fire(ApathionCLI)


if __name__ == "__main__":
    main()

