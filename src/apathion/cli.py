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
from apathion.pathfinding.hybrid import HybridPathfinder
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
        model_path: Optional[str] = None,
    ):
        """
        Run an interactive game session with pygame visualization.
        
        Args:
            algorithm: Pathfinding algorithm to use (astar, astar_basic, astar_enhanced, aco, dqn, fixed)
            map_type: Type of map (simple, branching, open_arena)
            waves: Number of waves to spawn
            enemies: Enemies per wave
            config_file: Optional path to config JSON file
            model_path: Optional path to trained DQN model (for dqn algorithm)
        
        Example:
            apathion play --algorithm=astar_basic --map_type=branching --waves=10
            apathion play --algorithm=astar_enhanced --map_type=branching --waves=10
            apathion play --algorithm=dqn --model_path=models/dqn_model
        """
        print(f"Starting Apathion with {algorithm} on {map_type} map...")
        
        # Load configuration
        if config_file:
            config = GameConfig.from_json(config_file)
            # Override config file settings with command-line arguments
            # This allows: apathion play --algorithm=dqn --config_file=...
            print(f"  Loaded config from: {config_file}")
            print(f"  Config algorithm: {config.algorithm}")
            
            # Command-line algorithm overrides config file
            if algorithm != "astar":  # Default value
                print(f"  Overriding algorithm: {config.algorithm} -> {algorithm}")
                config.algorithm = algorithm
            else:
                # Use config file algorithm
                algorithm = config.algorithm
                
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
        
        # Set model path for DQN if provided (works for both config file and no config)
        if algorithm == "dqn" and model_path:
            print(f"  Setting DQN model path: {model_path}")
            config.dqn.model_path = model_path
        
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
        save_path: str = "models/dqn_model",
        config_file: Optional[str] = None,
        device: str = "auto",
        num_towers: int = 3,
        random_towers: bool = True,  # DEFAULT: True for generalization
        num_envs: int = 1,  # Number of parallel environments (1=no parallelization, 4-8 recommended for GPU)
        learning_rate: float = 0.0003,  # Will be auto-adjusted for reward profile
        buffer_size: int = 100000,
        batch_size: int = 64,  # Increased from 32 for better hardware utilization
        learning_starts: int = 1000,
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.125,  # 12.5% of training for epsilon decay
        exploration_final_eps: float = 0.05,
        save_freq: int = 10000,
        log_interval: int = 100,
        reward_profile: str = "balanced",
    ):
        """
        Train a DQN model for pathfinding.
        
        Args:
            episodes: Number of training episodes (converted to timesteps)
            map_type: Type of map to train on ("simple", "branching", "open_arena")
            save_path: Path to save trained model (without .zip extension)
            config_file: Optional path to training config JSON file
            device: Device for training ("auto", "cpu", "cuda")
            num_towers: Number of towers to place
            random_towers: Whether to randomize tower positions each episode
            num_envs: Number of parallel environments (1=single, 4-8 recommended for GPU)
            learning_rate: Learning rate for optimizer
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            learning_starts: Steps before learning starts
            gamma: Discount factor
            target_update_interval: Steps between target network updates
            exploration_fraction: Fraction of training for epsilon decay
            exploration_final_eps: Final epsilon value
            save_freq: Frequency (in steps) to save checkpoints
            log_interval: Frequency (in episodes) to log progress
            reward_profile: Reward optimization ("speed", "balanced", "survival") [default: balanced]
        
        Example:
            apathion train --episodes=5000 --map_type=simple --device=cuda --num_envs=4
            apathion train --episodes=10000 --map_type=branching --random_towers=True
            apathion train --episodes=3000 --reward_profile=survival  # Max survival
            apathion train --episodes=5000 --reward_profile=speed  # Fastest paths
            apathion train --episodes=5000 --num_envs=8 --device=cuda  # Fast GPU training
        """
        try:
            from stable_baselines3 import DQN
            from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
            from stable_baselines3.common.monitor import Monitor
            from apathion.pathfinding.dqn_env import PathfindingEnv
            import torch.nn as nn
            import os
        except ImportError as e:
            print(f"Error: Required packages not installed: {e}")
            print("\nPlease install dependencies:")
            print("  uv pip install stable-baselines3 gymnasium torch")
            return
        
        print("=" * 70)
        print("DQN Pathfinding Training")
        print("=" * 70)
        
        # Auto-adjust parameters for stability based on reward profile
        if reward_profile == "survival":
            if learning_rate == 0.0003:  # Using default
                learning_rate = 0.0001  # Balanced for large rewards (increased from 0.00005)
                print(f"  ⚙️  Auto-adjusted learning_rate for SURVIVAL profile: 0.0003 → 0.0001")
            if target_update_interval == 1000:  # Using default
                target_update_interval = 5000  # Less frequent updates
                print(f"  ⚙️  Auto-adjusted target_update_interval for SURVIVAL: 1000 → 5000")
            if exploration_fraction == 0.125:  # Using default
                exploration_fraction = 0.3  # Balanced exploration for survival (reduced from 0.8)
                print(f"  ⚙️  Auto-adjusted exploration_fraction for SURVIVAL: 0.125 → 0.3")
        elif reward_profile == "balanced":
            if learning_rate == 0.0003:
                learning_rate = 0.00005  # Lower for large damage penalties
                print(f"  ⚙️  Auto-adjusted learning_rate for BALANCED profile: 0.0003 → 0.00005")
            if target_update_interval == 1000:
                target_update_interval = 3000  # More stable with large rewards
                print(f"  ⚙️  Auto-adjusted target_update_interval for BALANCED: 1000 → 3000")
            if exploration_fraction == 0.125:
                exploration_fraction = 0.15  # Slightly more exploration
                print(f"  ⚙️  Auto-adjusted exploration_fraction for BALANCED: 0.125 → 0.15")
        
        print(f"\nConfiguration:")
        print(f"  Map type: {map_type}")
        print(f"  Episodes: {episodes}")
        print(f"  Device: {device}")
        print(f"  Parallel environments: {num_envs}")
        print(f"  Towers: {num_towers} ({'random' if random_towers else 'fixed'})")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gamma: {gamma}")
        print(f"  Target update interval: {target_update_interval}")
        print(f"  Exploration fraction: {exploration_fraction}")
        print(f"  Save path: {save_path}")
        print(f"  Reward profile: {reward_profile}")
        print()
        
        # Create save directory if needed
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        checkpoint_dir = os.path.join(os.path.dirname(save_path) or ".", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create training environment(s)
        print("Creating training environment...")
        
        def make_env(rank: int):
            """Create a single environment (for vectorization)."""
            def _init():
                env = PathfindingEnv(
                    map_type=map_type,
                    max_steps=500,
                    num_towers=num_towers,
                    random_towers=random_towers,
                    state_size=42,  # Updated for danger-aware features
                    reward_profile=reward_profile,
                )
                # Wrap with Monitor to track episode rewards/lengths in logs
                return Monitor(env)
            return _init
        
        if num_envs == 1:
            # Single environment (no vectorization)
            env = PathfindingEnv(
                map_type=map_type,
                max_steps=500,
                num_towers=num_towers,
                random_towers=random_towers,
                state_size=42,  # Updated for danger-aware features
                reward_profile=reward_profile,
            )
        else:
            # Vectorized environments for parallel training
            # Use SubprocVecEnv for true parallelization (better for CPU-intensive envs)
            # Use DummyVecEnv for sequential execution (better for simple envs)
            use_subprocess = num_envs > 2  # Use subprocess for 3+ envs
            
            if use_subprocess:
                print(f"  Using SubprocVecEnv for {num_envs} parallel environments")
                env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
            else:
                print(f"  Using DummyVecEnv for {num_envs} sequential environments")
                env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print()
        
        # Convert episodes to timesteps (map-specific estimates)
        # Based on empirical observations of average episode lengths
        # Adjust for tower count - more towers = longer episodes during training
        base_steps = {
            "simple": 200,
            "branching": 180,
            "open_arena": 250,
        }
        
        # Scale timesteps based on tower count (more towers = more timesteps needed)
        tower_difficulty_multiplier = 1.0 + (num_towers - 3) * 0.5  # +50% per tower above 3
        steps_estimate = int(base_steps.get(map_type, 200) * tower_difficulty_multiplier)
        total_timesteps = episodes * steps_estimate
        
        print(f"  Estimated steps per episode: {steps_estimate}")
        print(f"  Tower difficulty multiplier: {tower_difficulty_multiplier:.1f}x")
        print(f"  Total timesteps target: {total_timesteps:,} (~{episodes} episodes)")
        
        # Create DQN model with gradient clipping for stability
        print("Initializing DQN model...")
        
        # Configure improved policy with deeper network and better capacity
        # Observation space is now 42 features (includes danger-aware directional info)
        # Use deeper, wider network with skip-like structure for complex spatial reasoning
        network_arch = [512, 256, 256, 128, 64]  # Deeper, wider network
        
        policy_kwargs = {
            "net_arch": network_arch,
            "activation_fn": nn.ReLU,  # ReLU activation function
            "optimizer_kwargs": {
                "eps": 1e-5,  # Adam epsilon for numerical stability
            }
        }
        print(f"  Network architecture: {network_arch}")
        print(f"  State size: 42 features (with danger-aware directional info)")
        
        # Add gradient clipping for survival profile (large rewards)
        max_grad_norm = 10.0  # Default
        if reward_profile == "survival":
            max_grad_norm = 1.0  # Stricter clipping for large rewards
            print(f"  ⚙️  Enabled strict gradient clipping: max_grad_norm={max_grad_norm}")
        elif reward_profile == "balanced":
            max_grad_norm = 5.0
        
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            train_freq=4,
            gradient_steps=2,  # Increased from 1: do 2 gradient updates per training step
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
        )
        print(f"  Model initialized with {device}")
        print()
        
        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix="dqn_checkpoint",
        )
        
        # Train the model
        print(f"\nStarting training for {total_timesteps:,} timesteps (~{episodes} episodes)...")
        print("-" * 70)
        
        try:
            # Try with progress bar, fall back to no progress bar if tqdm not installed
            try:
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=checkpoint_callback,
                    log_interval=log_interval,
                    progress_bar=True,
                )
            except ImportError:
                print("Note: Progress bar disabled (install tqdm and rich for progress bar)")
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=checkpoint_callback,
                    log_interval=log_interval,
                    progress_bar=False,
                )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
        
        # Save final model
        print("\n" + "-" * 70)
        print("Training complete!")
        print(f"\nSaving model to {save_path}...")
        model.save(save_path)
        print(f"✓ Model saved successfully")
        
        # Save training metadata
        metadata = {
            "map_type": map_type,
            "episodes": episodes,
            "timesteps": total_timesteps,
            "num_towers": num_towers,
            "random_towers": random_towers,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "reward_profile": reward_profile,
        }
        
        import json
        metadata_path = f"{save_path}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_path}")
        
        # Test the model
        print("\nTesting trained model...")
        
        # Create a single environment for testing (not vectorized)
        test_env = PathfindingEnv(
            map_type=map_type,
            max_steps=500,
            num_towers=num_towers,
            random_towers=random_towers,
            state_size=34,
            reward_profile=reward_profile,
        )
        
        obs, info = test_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        print(f"  Test episode: {steps} steps, reward: {total_reward:.2f}")
        if info.get("reached_goal"):
            print("  ✓ Agent reached the goal!")
        elif info.get("is_dead"):
            print("  ✗ Agent was eliminated")
        
        test_env.close()
        
        print("\n" + "=" * 70)
        print(f"Training complete! Model saved to: {save_path}.zip")
        print("=" * 70)
        print("\nTo use the trained model:")
        print(f"  apathion play --algorithm=dqn --model_path={save_path}")
        print()
    
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
        
        # Extract spawn_points and goal_positions from config if provided
        spawn_points = None
        goal_positions = None
        if not isinstance(map_config, str):
            if hasattr(map_config, 'spawn_points') and map_config.spawn_points:
                spawn_points = [tuple(sp) for sp in map_config.spawn_points]
            if hasattr(map_config, 'goal_positions') and map_config.goal_positions:
                goal_positions = [tuple(gp) for gp in map_config.goal_positions]
        
        # Create the map
        if map_type == "simple":
            game_map = Map.create_simple_map(
                spawn_points=spawn_points,
                goal_positions=goal_positions
            )
        elif map_type == "branching":
            game_map = Map.create_branching_map(config=map_config if not isinstance(map_config, str) else None)
        elif map_type == "open_arena":
            game_map = Map.create_open_arena(
                spawn_points=spawn_points,
                goal_positions=goal_positions
            )
        else:
            print(f"Unknown map type '{map_type}', using simple map")
            game_map = Map.create_simple_map(
                spawn_points=spawn_points,
                goal_positions=goal_positions
            )
        
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
            cfg = None
            if exp_config and hasattr(exp_config, 'astar'):
                cfg = exp_config.astar
            elif game_config and hasattr(game_config, 'astar'):
                cfg = game_config.astar
            
            if cfg:
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
            cfg = None
            if exp_config and hasattr(exp_config, 'astar'):
                cfg = exp_config.astar
            elif game_config and hasattr(game_config, 'astar'):
                cfg = game_config.astar
            
            if cfg:
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
            # Try exp_config first, then game_config
            cfg = None
            if exp_config and hasattr(exp_config, 'aco'):
                cfg = exp_config.aco
            elif game_config and hasattr(game_config, 'aco'):
                cfg = game_config.aco
            
            if cfg:
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
            # Try exp_config first, then game_config
            cfg = None
            if exp_config and hasattr(exp_config, 'dqn'):
                cfg = exp_config.dqn
            elif game_config and hasattr(game_config, 'dqn'):
                cfg = game_config.dqn
            
            if cfg:
                return DQNPathfinder(
                    name=cfg.name,
                    state_size=cfg.state_size,
                    action_size=cfg.action_size,
                    use_cache=cfg.use_cache,
                    cache_duration=cfg.cache_duration,
                    model_path=cfg.model_path,
                    plan_full_path=cfg.plan_full_path if hasattr(cfg, 'plan_full_path') else False,  # Changed: default to False to match training
                )
            return DQNPathfinder(plan_full_path=False)  # Changed: single-step mode like training
        
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
        
        elif algorithm in ["hybrid", "hybrid_dqn"]:
            # Hybrid DQN leader-follower system
            cfg = None
            if exp_config and hasattr(exp_config, 'dqn'):
                cfg = exp_config.dqn
            elif game_config and hasattr(game_config, 'dqn'):
                cfg = game_config.dqn
            
            model_path = cfg.model_path if cfg else None
            leaders_per_wave = 5  # Default 5 leaders
            
            if model_path:
                print(f"  Using hybrid system with DQN model: {model_path}")
                print(f"  Leaders per wave: {leaders_per_wave}")
            else:
                print("  Warning: No DQN model specified for hybrid algorithm")
            
            return HybridPathfinder(
                model_path=model_path,
                leaders_per_wave=leaders_per_wave,
            )
        
        else:
            print(f"Unknown algorithm '{algorithm}', using A*")
            return AStarPathfinder()


def main():
    """Main entry point for the CLI."""
    fire.Fire(ApathionCLI)


if __name__ == "__main__":
    main()

