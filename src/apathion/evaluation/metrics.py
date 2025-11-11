"""
Metrics module for performance evaluation of pathfinding algorithms.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter


def survival_rate(
    enemies_spawned: int,
    enemies_reached_goal: int
) -> Dict[str, Any]:
    """
    Calculate survival rate (percentage of enemies reaching goal).
    
    Args:
        enemies_spawned: Total number of enemies spawned
        enemies_reached_goal: Number of enemies that reached the goal
        
    Returns:
        Dictionary with survival rate metrics
    """
    if enemies_spawned == 0:
        return {
            "metric": "survival_rate",
            "enemies_spawned": 0,
            "enemies_reached_goal": 0,
            "survival_rate": 0.0,
            "survival_percentage": 0.0,
        }
    
    rate = enemies_reached_goal / enemies_spawned
    percentage = rate * 100.0
    
    return {
        "metric": "survival_rate",
        "enemies_spawned": enemies_spawned,
        "enemies_reached_goal": enemies_reached_goal,
        "enemies_defeated": enemies_spawned - enemies_reached_goal,
        "survival_rate": rate,
        "survival_percentage": percentage,
    }


def damage_efficiency(
    enemies_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate damage efficiency (average damage taken per surviving unit).
    
    Args:
        enemies_data: List of enemy data dictionaries with 'damage_taken' and 'reached_goal'
        
    Returns:
        Dictionary with damage efficiency metrics
    """
    if not enemies_data:
        return {
            "metric": "damage_efficiency",
            "survivors": 0,
            "avg_damage_taken": 0.0,
            "total_damage": 0.0,
        }
    
    survivors = [e for e in enemies_data if e.get("reached_goal", False)]
    all_damage = [e.get("damage_taken", 0.0) for e in enemies_data]
    survivor_damage = [e.get("damage_taken", 0.0) for e in survivors]
    
    avg_damage_all = np.mean(all_damage) if all_damage else 0.0
    avg_damage_survivors = np.mean(survivor_damage) if survivor_damage else 0.0
    total_damage = sum(all_damage)
    
    return {
        "metric": "damage_efficiency",
        "total_enemies": len(enemies_data),
        "survivors": len(survivors),
        "avg_damage_taken": avg_damage_survivors,
        "avg_damage_all_enemies": avg_damage_all,
        "total_damage": total_damage,
        "min_damage": min(all_damage) if all_damage else 0.0,
        "max_damage": max(all_damage) if all_damage else 0.0,
    }


def path_diversity(
    paths: List[List[Tuple[int, int]]]
) -> Dict[str, Any]:
    """
    Calculate path diversity using Shannon entropy.
    
    Higher entropy indicates more diverse routing choices.
    
    Args:
        paths: List of paths, where each path is a list of (x, y) positions
        
    Returns:
        Dictionary with path diversity metrics
    """
    if not paths:
        return {
            "metric": "path_diversity",
            "num_paths": 0,
            "unique_paths": 0,
            "shannon_entropy": 0.0,
            "diversity_index": 0.0,
        }
    
    # Convert paths to hashable strings for counting
    path_strings = [str(path) for path in paths]
    path_counts = Counter(path_strings)
    
    # Calculate Shannon entropy
    total_paths = len(path_strings)
    unique_path_count = len(path_counts)
    
    # If only one unique path, diversity is 0
    if unique_path_count <= 1:
        shannon_entropy = 0.0
        diversity_index = 0.0
    else:
    probabilities = [count / total_paths for count in path_counts.values()]
    shannon_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
    
    # Diversity index (normalized entropy)
        # Maximum entropy occurs when all paths are equally likely
        max_entropy = np.log2(unique_path_count)
    diversity_index = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
    
    return {
        "metric": "path_diversity",
        "num_paths": total_paths,
        "unique_paths": len(path_counts),
        "shannon_entropy": shannon_entropy,
        "diversity_index": diversity_index,
        "most_common_path_usage": max(path_counts.values()) / total_paths if path_counts else 0.0,
    }


def adaptation_speed(
    decision_logs: List[Dict[str, Any]],
    change_event_time: float
) -> Dict[str, Any]:
    """
    Calculate adaptation speed (frames to converge on new optimal path after change).
    
    Args:
        decision_logs: List of pathfinding decision records with timestamps
        change_event_time: Time when the environment changed (e.g., tower placed)
        
    Returns:
        Dictionary with adaptation speed metrics
    """
    if not decision_logs:
        return {
            "metric": "adaptation_speed",
            "change_event_time": change_event_time,
            "frames_to_adapt": 0,
            "time_to_adapt": 0.0,
        }
    
    # Find decisions after the change event
    post_change_decisions = [
        d for d in decision_logs
        if d.get("timestamp", 0) >= change_event_time
    ]
    
    if not post_change_decisions:
        return {
            "metric": "adaptation_speed",
            "change_event_time": change_event_time,
            "frames_to_adapt": 0,
            "time_to_adapt": 0.0,
            "decisions_analyzed": 0,
        }
    
    # PLACEHOLDER: Simple metric counting decisions until path stabilizes
    # TODO: Implement sophisticated convergence detection
    # - Track when paths stop changing significantly
    # - Measure quality improvement over time
    
    frames_to_adapt = len(post_change_decisions)
    first_decision_time = post_change_decisions[0].get("timestamp", change_event_time)
    time_to_adapt = first_decision_time - change_event_time
    
    return {
        "metric": "adaptation_speed",
        "change_event_time": change_event_time,
        "frames_to_adapt": frames_to_adapt,
        "time_to_adapt": time_to_adapt,
        "decisions_analyzed": len(post_change_decisions),
    }


def computational_cost(
    decision_logs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate computational cost (average ms per pathfinding decision).
    
    Args:
        decision_logs: List of pathfinding decision records with 'decision_time'
        
    Returns:
        Dictionary with computational cost metrics
    """
    if not decision_logs:
        return {
            "metric": "computational_cost",
            "decisions": 0,
            "avg_time_ms": 0.0,
            "total_time_ms": 0.0,
        }
    
    decision_times = [
        d.get("decision_time_ms", 0.0)
        for d in decision_logs
    ]
    
    avg_time = np.mean(decision_times) if decision_times else 0.0
    total_time = sum(decision_times)
    min_time = min(decision_times) if decision_times else 0.0
    max_time = max(decision_times) if decision_times else 0.0
    median_time = np.median(decision_times) if decision_times else 0.0
    
    return {
        "metric": "computational_cost",
        "decisions": len(decision_logs),
        "avg_time_ms": avg_time,
        "median_time_ms": median_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "total_time_ms": total_time,
        "std_dev_ms": np.std(decision_times) if decision_times else 0.0,
    }


def strategic_depth(
    tower_configs: List[List[Tuple[int, int]]],
    behavior_changes: List[bool]
) -> Dict[str, Any]:
    """
    Calculate strategic depth (number of tower configurations that change behavior).
    
    Args:
        tower_configs: List of tower configurations (each is list of positions)
        behavior_changes: List of booleans indicating if behavior changed for each config
        
    Returns:
        Dictionary with strategic depth metrics
    """
    if not tower_configs:
        return {
            "metric": "strategic_depth",
            "configurations_tested": 0,
            "configurations_causing_change": 0,
            "strategic_depth_ratio": 0.0,
        }
    
    total_configs = len(tower_configs)
    configs_with_change = sum(behavior_changes[:total_configs])
    
    ratio = configs_with_change / total_configs if total_configs > 0 else 0.0
    
    return {
        "metric": "strategic_depth",
        "configurations_tested": total_configs,
        "configurations_causing_change": configs_with_change,
        "strategic_depth_ratio": ratio,
        "strategic_depth_percentage": ratio * 100.0,
    }


def aggregate_metrics(
    metrics_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate multiple metric dictionaries into a summary.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_list:
        return {"aggregated_metrics": {}}
    
    # Group metrics by type
    by_type = {}
    for metric in metrics_list:
        metric_type = metric.get("metric", "unknown")
        if metric_type not in by_type:
            by_type[metric_type] = []
        by_type[metric_type].append(metric)
    
    # Aggregate each type
    aggregated = {}
    for metric_type, metric_group in by_type.items():
        aggregated[metric_type] = {
            "count": len(metric_group),
            "data": metric_group,
        }
    
    return {
        "aggregated_metrics": aggregated,
        "total_metrics": len(metrics_list),
        "metric_types": list(by_type.keys()),
    }

