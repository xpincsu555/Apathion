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
from apathion.game.game_loop import run_game_loop
from apathion.pathfinding.astar import AStarPathfinder
from apathion.pathfinding.aco import ACOPathfinder
from apathion.pathfinding.dqn import DQNPathfinder
from apathion.pathfinding.fixed import FixedPathfinder
from apathion.evaluation.logger import GameLogger
from apathion.evaluation.evaluator import Evaluator
from apathion.config import (
    GameConfig,
    ExperimentConfig,
    get_experiment_preset,
    AStarConfig,
    ACOConfig,
    DQNConfig,
    FixedConfig,
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
        Run an interactive game session with pygame visualization.
        
        Args:
            algorithm: Pathfinding algorithm to use (astar, astar_basic, astar_enhanced, aco, dqn, fixed)
            map_type: Type of map (simple, branching, open_arena)
            waves: Number of waves to spawn
            enemies: Enemies per wave
            config_file: Optional path to config JSON file
        
        Example:
            apathion play --algorithm=astar_basic --map_type=branching --waves=10
            apathion play --algorithm=astar_enhanced --map_type=branching --waves=10
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
            
            # Update map type
            config.map.map_type = map_type
            config.map.baseline_path = [
                [0, 11], [1, 11], [2, 11], [3, 11], [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [9, 11], [10, 11], [11, 11], [12, 11], [13, 11], [14, 11], [15, 11], [16, 11], [16, 10], [16, 9], [17, 9], [18, 9], [19, 9], [19, 9], [19, 8], [19, 7], [19, 6], [20, 6], [21, 6], [22, 6], [23, 6], [23, 5], [24, 5], [25, 5], [26, 5], [27, 5], [28, 5], [29, 5]
            ]
        
        # Create map
        game_map = self._create_map(config.map)
        print(f"Map created: {game_map.width}x{game_map.height}")
        
        # Create pathfinder (pass config for fixed path support)
        pathfinder = self._create_pathfinder(algorithm, game_config=config)
        print(f"Pathfinder initialized: {pathfinder.get_name()}")
        
        # Update pathfinder with map
        pathfinder.update_state(game_map, [])
        
        # Run pygame game loop
        print(f"\nStarting game with {waves} waves of {enemies} enemies each")
        print(f"Controls:")
        print(f"  Click: Place tower")
        print(f"  T: Change tower type")
        print(f"  Space: Pause/Resume")
        print(f"  Tab: Toggle visualization mode")
        print(f"  ESC: Quit")
        print(f"\nLaunching game window...")
        
        run_game_loop(config, game_map, pathfinder)
    
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
                       Options: astar, astar_basic, astar_enhanced, aco, dqn, fixed
            maps: List of map types to test (default: [simple, branching])
            waves: Number of waves per experiment
            enemies: Enemies per wave
            runs: Number of repetitions per configuration
            preset: Use predefined experiment preset (baseline, full_comparison, performance_test)
            config_file: Optional path to experiment config JSON file
            output: Directory for results
        
        Example:
            apathion evaluate --algorithms=astar_basic,astar_enhanced --maps=simple,branching --waves=10
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
        
        # Create test maps (using map_type strings for backward compatibility)
        test_maps = [self._create_map(mt, validate_path=False) for mt in exp_config.map_types]
        
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
        session_file: Optional[str] = None,
        log_dir: str = "data/results",
        plots: bool = False,
        output_dir: str = "report_figures",
    ):
        """
        Analyze experiment results and generate visualizations.
        
        Args:
            session_file: Path to experiment session JSON file (finds latest if not specified)
            log_dir: Directory containing result files (default: data/results)
            plots: Generate visualization plots (requires matplotlib)
            output_dir: Directory for output plots and tables
        
        Example:
            apathion analyze --session_file=data/results/experiment_session_20251111_120000.json --plots
            apathion analyze --log_dir=data/results --plots --output_dir=my_figures
        """
        import json
        from collections import defaultdict
        
        log_path = Path(log_dir)
        if not log_path.exists():
            print(f"Error: Results directory not found: {log_dir}")
            return
        
        # Find session file
        if session_file is None:
            # Find the most recent session file
            json_files = sorted(log_path.glob("experiment_session_*.json"), reverse=True)
            if not json_files:
                print(f"Error: No experiment session files found in {log_dir}")
                print("Please run experiments first with: python scripts/run_experiments.py")
                return
            session_file = str(json_files[0])
            print(f"Using most recent session: {Path(session_file).name}")
        
        # Load session data
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading session file: {e}")
            return
        
        results = data.get("results", [])
        if not results:
            print("No results found in session file.")
            return
        
        # Print summary
        self._print_analysis_summary(results, data.get("session_id", "Unknown"))
        
        # Generate plots if requested
        if plots:
            self._generate_analysis_plots(results, output_dir)
        
        # Generate CSV table
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._generate_analysis_table(results, output_path / "summary_table.csv")
    
    def _print_analysis_summary(self, results: List[dict], session_id: str):
        """Print analysis summary to console."""
        from collections import defaultdict
        
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS ANALYSIS")
        print("="*80)
        print(f"\nSession ID: {session_id}")
        print(f"Total Experiments: {len(results)}")
        
        # Group by algorithm
        by_algo = defaultdict(list)
        for result in results:
            by_algo[result["algorithm"]].append(result)
        
        print("\n" + "-"*80)
        print("ALGORITHM COMPARISON")
        print("-"*80)
        print(f"\n{'Algorithm':<20} {'Avg Survival':<13} {'Path Div':<12} {'Avg Decision':<15} {'Max Decision':<15}")
        print(f"{'':20} {'Rate (%)':<13} {'Index':<12} {'Time (ms)':<15} {'Time (ms)':<15}")
        print("-"*80)
        
        for algo_name in sorted(by_algo.keys()):
            algo_results = by_algo[algo_name]
            avg_survival = sum(r["survival_rate_percent"] for r in algo_results) / len(algo_results)
            avg_decision = sum(r["avg_decision_time_ms"] for r in algo_results) / len(algo_results)
            max_decision = max(r["max_decision_time_ms"] for r in algo_results)
            
            # Get path diversity if available
            path_div_values = []
            for r in algo_results:
                if "path_diversity" in r and isinstance(r["path_diversity"], dict):
                    path_div_values.append(r["path_diversity"].get("diversity_index", 0.0))
            avg_path_div = sum(path_div_values) / len(path_div_values) if path_div_values else 0.0
            
            print(f"{algo_name:<20} {avg_survival:<13.2f} {avg_path_div:<12.3f} {avg_decision:<15.4f} {max_decision:<15.4f}")
        
        # Evaluate success criteria
        print("\n" + "-"*80)
        print("SUCCESS CRITERIA EVALUATION")
        print("-"*80)
        print("\nTarget: ≥25% higher survival rate than fixed paths")
        print("Target: <5 ms per enemy for real-time performance (60 FPS)\n")
        
        # Get fixed baseline
        fixed_results = by_algo.get("Fixed-Path-Baseline", by_algo.get("Fixed-Path", []))
        if fixed_results:
            baseline_survival = sum(r["survival_rate_percent"] for r in fixed_results) / len(fixed_results)
            target_survival = baseline_survival * 1.25
            print(f"Baseline (Fixed) Survival Rate: {baseline_survival:.2f}%")
            print(f"Target Survival Rate (25% improvement): {target_survival:.2f}%\n")
            
            for algo_name, algo_results in by_algo.items():
                if algo_name in ["Fixed-Path-Baseline", "Fixed-Path"]:
                    continue
                avg_survival = sum(r["survival_rate_percent"] for r in algo_results) / len(algo_results)
                avg_decision = sum(r["avg_decision_time_ms"] for r in algo_results) / len(algo_results)
                
                # Get path diversity
                path_div_values = []
                for r in algo_results:
                    if "path_diversity" in r and isinstance(r["path_diversity"], dict):
                        path_div_values.append(r["path_diversity"].get("diversity_index", 0.0))
                avg_path_div = sum(path_div_values) / len(path_div_values) if path_div_values else 0.0
                
                survival_check = "✓" if avg_survival >= target_survival else "✗"
                performance_check = "✓" if avg_decision < 5.0 else "✗"
                diversity_check = "✓" if avg_path_div >= 0.5 else "○"  # ○ for low diversity (expected for deterministic)
                
                print(f"{algo_name}:")
                print(f"  Survival Rate: {avg_survival:.2f}% {survival_check}")
                print(f"  Path Diversity: {avg_path_div:.3f} {diversity_check}")
                print(f"  Avg Decision Time: {avg_decision:.4f}ms {performance_check}\n")
        else:
            for algo_name, algo_results in by_algo.items():
                avg_decision = sum(r["avg_decision_time_ms"] for r in algo_results) / len(algo_results)
                performance_check = "✓" if avg_decision < 5.0 else "✗"
                print(f"{algo_name}:")
                print(f"  Avg Decision Time: {avg_decision:.4f}ms {performance_check}\n")
    
    def _generate_analysis_plots(self, results: List[dict], output_dir: str):
        """Generate visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("\nWarning: matplotlib not installed. Cannot generate plots.")
            print("Install with: pip install matplotlib")
            return
        
        from collections import defaultdict
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group by algorithm
        by_algo = defaultdict(list)
        for result in results:
            by_algo[result["algorithm"]].append(result)
        
        print(f"\nGenerating plots in: {output_path}")
        
        # Plot 1: Survival rates
        algorithms = sorted(by_algo.keys())
        avg_rates = []
        std_rates = []
        
        for algo in algorithms:
            rates = [r["survival_rate_percent"] for r in by_algo[algo]]
            avg_rates.append(sum(rates) / len(rates))
            std_rates.append(self._std_dev(rates))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(algorithms))
        bars = ax.bar(x, avg_rates, yerr=std_rates, capsize=5, alpha=0.8)
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        for bar, color in zip(bars, colors[:len(bars)]):
            bar.set_color(color)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Survival Rate (%)', fontsize=12)
        ax.set_title('Enemy Survival Rate by Algorithm', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'survival_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ survival_rates.png")
        
        # Plot 2: Decision times
        avg_times = []
        max_times = []
        for algo in algorithms:
            results_list = by_algo[algo]
            avg_times.append(sum(r["avg_decision_time_ms"] for r in results_list) / len(results_list))
            max_times.append(max(r["max_decision_time_ms"] for r in results_list))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(algorithms))
        width = 0.35
        ax.bar(x - width/2, avg_times, width, label='Average', alpha=0.8, color='#4ecdc4')
        ax.bar(x + width/2, max_times, width, label='Maximum', alpha=0.8, color='#ff6b6b')
        ax.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='Real-time Target (5ms)')
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Decision Time (ms)', fontsize=12)
        ax.set_title('Pathfinding Computation Time by Algorithm', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'decision_times.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ decision_times.png")
        
        print(f"\nPlots saved to: {output_path}")
    
    def _generate_analysis_table(self, results: List[dict], output_file: Path):
        """Generate CSV table for report."""
        from collections import defaultdict
        
        by_algo_map = defaultdict(list)
        for result in results:
            key = (result["algorithm"], result.get("map_type", "unknown"))
            by_algo_map[key].append(result)
        
        with open(output_file, 'w') as f:
            f.write("Algorithm,Map Type,Avg Survival Rate (%),Std Dev,Avg Decision Time (ms),"
                   "Max Decision Time (ms),Sample Size\n")
            
            for (algo, map_type), results_list in sorted(by_algo_map.items()):
                survival_rates = [r["survival_rate_percent"] for r in results_list]
                decision_times = [r["avg_decision_time_ms"] for r in results_list]
                avg_survival = sum(survival_rates) / len(survival_rates)
                std_survival = self._std_dev(survival_rates)
                avg_decision = sum(decision_times) / len(decision_times)
                max_decision = max(r["max_decision_time_ms"] for r in results_list)
                
                f.write(f"{algo},{map_type},{avg_survival:.2f},{std_survival:.2f},"
                       f"{avg_decision:.4f},{max_decision:.4f},{len(results_list)}\n")
        
        print(f"\n✓ Generated table: {output_file}")
    
    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _create_map(self, map_config, validate_path: bool = True) -> Map:
        """
        Create a map of the specified type and validate baseline path if provided.
        
        Args:
            map_config: MapConfig or string specifying map type
            validate_path: Whether to validate baseline path
            
        Returns:
            Map instance
        """
        # Handle legacy string input for backward compatibility
        if isinstance(map_config, str):
            map_type = map_config
            baseline_path = None
        else:
            map_type = map_config.map_type
            baseline_path = map_config.baseline_path
        
        # Create the map
        if map_type == "simple":
            game_map = Map.create_simple_map()
        elif map_type == "branching":
            game_map = Map.create_branching_map(config=map_config if not isinstance(map_config, str) else None)
        elif map_type == "open_arena":
            game_map = Map.create_open_arena()
        else:
            print(f"Unknown map type '{map_type}', using simple map")
            game_map = Map.create_simple_map()
        
        # Validate baseline path if provided
        if baseline_path and validate_path:
            # Convert path from list of lists to list of tuples
            path_tuples = [(p[0], p[1]) for p in baseline_path]
            is_valid, error_msg = game_map.validate_path(path_tuples)
            
            if not is_valid:
                print(f"⚠️  Warning: Baseline path validation failed: {error_msg}")
                print(f"   The game will continue, but the baseline path may not be usable.")
            else:
                print(f"✓ Baseline path validated successfully ({len(path_tuples)} points)")
        
        return game_map
    
    def _create_pathfinder(
        self,
        algorithm: str,
        exp_config: Optional[ExperimentConfig] = None,
        game_config: Optional[GameConfig] = None
    ):
        """
        Create a pathfinder of the specified type.
        
        Args:
            algorithm: Algorithm name (astar, astar_basic, astar_enhanced, aco, dqn, fixed)
            exp_config: Optional experiment configuration
            game_config: Optional game configuration (for fixed path baseline)
            
        Returns:
            Pathfinder instance
        """
        if algorithm in ["astar", "astar_enhanced"]:
            # Enhanced A* (with damage and congestion costs)
            if exp_config and hasattr(exp_config, 'astar'):
                cfg = exp_config.astar
                return AStarPathfinder(
                    name=cfg.name,
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    diagonal_movement=cfg.diagonal_movement,
                    use_enhanced=cfg.use_enhanced,
                )
            return AStarPathfinder(
                name="A*-Enhanced",
                use_enhanced=True
            )
        
        elif algorithm == "astar_basic":
            # Basic A* (only g(n) and h(n))
            if exp_config and hasattr(exp_config, 'astar'):
                cfg = exp_config.astar
                return AStarPathfinder(
                    name="A*-Basic",
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    diagonal_movement=cfg.diagonal_movement,
                    use_enhanced=False,
                )
            return AStarPathfinder(
                name="A*-Basic",
                use_enhanced=False
            )
        
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
                    gamma=cfg.gamma,
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
        
        elif algorithm == "fixed":
            # Extract baseline path from game config
            baseline_path = None
            if game_config and game_config.map.baseline_path:
                # Convert from list of lists to list of tuples
                baseline_path = [(p[0], p[1]) for p in game_config.map.baseline_path]
                print(f"  Using baseline path with {len(baseline_path)} points")
            else:
                print("  Warning: No baseline path provided for fixed algorithm")
            
            return FixedPathfinder(baseline_path=baseline_path)
        
        else:
            print(f"Unknown algorithm '{algorithm}', using A*")
            return AStarPathfinder()


def main():
    """Main entry point for the CLI."""
    fire.Fire(ApathionCLI)


if __name__ == "__main__":
    main()

