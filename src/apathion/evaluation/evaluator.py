"""
Evaluator module for comparative analysis of pathfinding algorithms.
"""

from typing import List, Dict, Any, Optional, Type
import time

from apathion.game.game import GameState
from apathion.game.map import Map
from apathion.pathfinding.base import BasePathfinder
from apathion.evaluation.logger import GameLogger
from apathion.evaluation import metrics


class Evaluator:
    """
    Comparative evaluation framework for pathfinding algorithms.
    
    Runs experiments across different algorithms, maps, and scenarios,
    collecting performance data and generating comparative reports.
    
    Attributes:
        logger: GameLogger instance for data collection
        results: Storage for experiment results
    """
    
    def __init__(self, logger: Optional[GameLogger] = None):
        """
        Initialize evaluator.
        
        Args:
            logger: GameLogger instance (creates new one if None)
        """
        self.logger = logger or GameLogger()
        self.results: List[Dict[str, Any]] = []
    
    def run_experiment(
        self,
        algorithm: BasePathfinder,
        game_map: Map,
        num_waves: int = 5,
        enemies_per_wave: int = 10,
        tower_placements: Optional[List[tuple]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a single experiment with one algorithm and map configuration.
        
        Args:
            algorithm: Pathfinding algorithm to test
            game_map: Map to use for the experiment
            num_waves: Number of waves to simulate
            enemies_per_wave: Enemies per wave
            tower_placements: Optional list of (x, y) tower positions
            **kwargs: Additional experiment parameters
            
        Returns:
            Dictionary with experiment results
        """
        # Initialize game state
        game = GameState(game_map)
        algorithm.update_state(game.map, game.towers)
        
        # Place towers if specified (without gold checks for evaluation)
        if tower_placements:
            for pos in tower_placements:
                game.place_tower(pos, tower_type="basic", check_gold=False)
            algorithm.update_state(game.map, game.towers)
        
        # Track experiment metrics
        experiment_start = time.time()
        all_enemies_data = []
        all_paths = []
        all_decision_times = []
        
        # Run waves
        for wave_num in range(num_waves):
            # Spawn wave
            enemies = game.spawn_wave(num_enemies=enemies_per_wave)
            
            # Assign paths to enemies
            goal = game.map.goal_positions[0]
            for enemy in enemies:
                start_pos = (int(enemy.position[0]), int(enemy.position[1]))
                
                # Time the pathfinding
                path_start = time.time()
                path = algorithm.find_path(start_pos, goal, enemy_id=enemy.id)
                path_time = (time.time() - path_start) * 1000  # Convert to ms
                
                enemy.set_path(path)
                all_paths.append(path)
                all_decision_times.append(path_time)
                
                # Log decision
                self.logger.log_decision(
                    timestamp=game.game_time,
                    algorithm=algorithm.get_name(),
                    enemy_id=enemy.id,
                    chosen_path=path,
                    alternative_paths=0,  # Placeholder
                    decision_time_ms=path_time,
                )
            
            # Simulate wave (placeholder - in real implementation, run game loop)
            # For now, just mark enemies as reaching goal or defeated randomly
            for enemy in enemies:
                enemy_data = enemy.to_dict()
                # PLACEHOLDER: Simulate outcome
                # In real implementation, would run actual game simulation
                enemy_data["reached_goal"] = True  # Simplified
                all_enemies_data.append(enemy_data)
        
        experiment_time = time.time() - experiment_start
        
        # Calculate metrics
        survival = metrics.survival_rate(
            enemies_spawned=len(all_enemies_data),
            enemies_reached_goal=sum(1 for e in all_enemies_data if e.get("reached_goal"))
        )
        
        damage_eff = metrics.damage_efficiency(all_enemies_data)
        
        path_div = metrics.path_diversity(all_paths)
        
        comp_cost = metrics.computational_cost([
            {"decision_time_ms": t} for t in all_decision_times
        ])
        
        # Compile results
        result = {
            "algorithm": algorithm.get_name(),
            "map_type": kwargs.get("map_type", "unknown"),
            "num_waves": num_waves,
            "enemies_per_wave": enemies_per_wave,
            "total_enemies": len(all_enemies_data),
            "experiment_time_seconds": experiment_time,
            "survival_rate": survival,
            "damage_efficiency": damage_eff,
            "path_diversity": path_div,
            "computational_cost": comp_cost,
        }
        
        self.results.append(result)
        
        # Log wave results
        self.logger.log_wave_results(
            wave=num_waves,
            algorithm=algorithm.get_name(),
            enemies_spawned=len(all_enemies_data),
            enemies_survived=sum(1 for e in all_enemies_data if e.get("reached_goal")),
            avg_damage=damage_eff["avg_damage_taken"],
            path_diversity=path_div["diversity_index"],
            total_cpu_time=experiment_time,
        )
        
        return result
    
    def compare_algorithms(
        self,
        algorithms: List[BasePathfinder],
        test_maps: List[Map],
        map_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple algorithms across multiple maps.
        
        Args:
            algorithms: List of pathfinding algorithms to compare
            test_maps: List of maps to test on
            map_names: Optional names for the maps
            **kwargs: Experiment parameters
            
        Returns:
            Dictionary with comparative results
        """
        if map_names is None:
            map_names = [f"map_{i}" for i in range(len(test_maps))]
        
        comparison_results = []
        
        for algorithm in algorithms:
            for game_map, map_name in zip(test_maps, map_names):
                print(f"Testing {algorithm.get_name()} on {map_name}...")
                
                result = self.run_experiment(
                    algorithm=algorithm,
                    game_map=game_map,
                    map_type=map_name,
                    **kwargs
                )
                
                comparison_results.append(result)
        
        # Generate comparative summary
        summary = self._generate_comparison_summary(comparison_results)
        
        return {
            "comparison_results": comparison_results,
            "summary": summary,
            "algorithms_tested": [a.get_name() for a in algorithms],
            "maps_tested": map_names,
        }
    
    def _generate_comparison_summary(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from comparison results.
        
        Args:
            results: List of experiment results
            
        Returns:
            Summary dictionary
        """
        if not results:
            return {}
        
        # Group by algorithm
        by_algorithm = {}
        for result in results:
            algo_name = result["algorithm"]
            if algo_name not in by_algorithm:
                by_algorithm[algo_name] = []
            by_algorithm[algo_name].append(result)
        
        # Calculate averages for each algorithm
        summary = {}
        for algo_name, algo_results in by_algorithm.items():
            survival_rates = [
                r["survival_rate"]["survival_percentage"]
                for r in algo_results
            ]
            diversity_indices = [
                r["path_diversity"]["diversity_index"]
                for r in algo_results
            ]
            avg_times = [
                r["computational_cost"]["avg_time_ms"]
                for r in algo_results
            ]
            
            summary[algo_name] = {
                "avg_survival_rate": sum(survival_rates) / len(survival_rates) if survival_rates else 0,
                "avg_diversity_index": sum(diversity_indices) / len(diversity_indices) if diversity_indices else 0,
                "avg_computation_time_ms": sum(avg_times) / len(avg_times) if avg_times else 0,
                "experiments": len(algo_results),
            }
        
        return summary
    
    def generate_report(
        self,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a text report of evaluation results.
        
        Args:
            output_file: Optional file path to save report
            
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 80,
            "Apathion Pathfinding Evaluation Report",
            "=" * 80,
            "",
            f"Total Experiments: {len(self.results)}",
            "",
        ]
        
        if not self.results:
            report_lines.append("No results to report.")
            report = "\n".join(report_lines)
            return report
        
        # Summary by algorithm
        by_algorithm = {}
        for result in self.results:
            algo = result["algorithm"]
            if algo not in by_algorithm:
                by_algorithm[algo] = []
            by_algorithm[algo].append(result)
        
        for algo_name, algo_results in by_algorithm.items():
            report_lines.extend([
                f"\nAlgorithm: {algo_name}",
                "-" * 40,
            ])
            
            for result in algo_results:
                report_lines.extend([
                    f"  Map: {result.get('map_type', 'unknown')}",
                    f"    Survival Rate: {result['survival_rate']['survival_percentage']:.1f}%",
                    f"    Path Diversity: {result['path_diversity']['diversity_index']:.2f}",
                    f"    Avg Computation: {result['computational_cost']['avg_time_ms']:.2f} ms",
                    "",
                ])
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def export_results(self, prefix: str = "") -> Dict[str, str]:
        """
        Export all results and logs.
        
        Args:
            prefix: Optional prefix for filenames
            
        Returns:
            Dictionary of exported file paths
        """
        exported = self.logger.export_csv(prefix=prefix)
        
        # Also export JSON summary
        json_path = self.logger.export_json(prefix=prefix)
        exported["json_summary"] = json_path
        
        return exported
    
    def clear(self) -> None:
        """Clear all results and logs."""
        self.results.clear()
        self.logger.clear()

