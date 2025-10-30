"""
Logger module for data collection and CSV export.
"""

from typing import List, Dict, Any, Optional, Tuple
import csv
import json
from pathlib import Path
from datetime import datetime


class GameLogger:
    """
    Logger for collecting game state, pathfinding decisions, and performance metrics.
    
    Logs three types of data matching requirements.md:
    1. Game State Data (runtime input)
    2. Pathfinding Decision Logs
    3. Performance Metrics
    
    Attributes:
        session_id: Unique identifier for this logging session
        output_dir: Directory where logs are saved
        frame_logs: Buffer for frame-by-frame game state
        decision_logs: Buffer for pathfinding decisions
        metric_logs: Buffer for performance metrics
    """
    
    def __init__(self, output_dir: str = "data/logs", session_id: Optional[str] = None):
        """
        Initialize game logger.
        
        Args:
            output_dir: Directory to save log files
            session_id: Unique session identifier (auto-generated if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id
        
        self.frame_logs: List[Dict[str, Any]] = []
        self.decision_logs: List[Dict[str, Any]] = []
        self.metric_logs: List[Dict[str, Any]] = []
        
        self._frame_count = 0
    
    def log_frame(
        self,
        frame: int,
        game_time: float,
        enemies: List[Dict[str, Any]],
        towers: List[Dict[str, Any]],
        **kwargs
    ) -> None:
        """
        Log game state for a single frame.
        
        Corresponds to "Game State Data" table in requirements.md.
        
        Args:
            frame: Frame number
            game_time: Game time in seconds
            enemies: List of enemy state dictionaries
            towers: List of tower state dictionaries
            **kwargs: Additional state information
        """
        # Extract damage zones from towers
        damage_zones = []
        for tower in towers:
            zone = {
                "position": tower.get("position"),
                "range": tower.get("range"),
                "dps": tower.get("damage", 0) * tower.get("attack_rate", 0),
            }
            damage_zones.append(zone)
        
        # Log each enemy's state
        for enemy in enemies:
            log_entry = {
                "frame": frame,
                "game_time": game_time,
                "enemy_id": enemy.get("id"),
                "position": enemy.get("position"),
                "health": enemy.get("health"),
                "damage_taken": enemy.get("damage_taken", 0),
                "is_alive": enemy.get("is_alive", True),
                "tower_count": len(towers),
                "damage_zone_count": len(damage_zones),
            }
            log_entry.update(kwargs)
            self.frame_logs.append(log_entry)
        
        self._frame_count += 1
    
    def log_decision(
        self,
        timestamp: float,
        algorithm: str,
        enemy_id: str,
        chosen_path: List[Tuple[int, int]],
        alternative_paths: int,
        decision_time_ms: float,
        **kwargs
    ) -> None:
        """
        Log a pathfinding decision.
        
        Corresponds to "Pathfinding Decision Logs" table in requirements.md.
        
        Args:
            timestamp: Game time when decision was made
            algorithm: Name of pathfinding algorithm
            enemy_id: ID of enemy making the decision
            chosen_path: Path that was selected
            alternative_paths: Number of alternative paths considered
            decision_time_ms: Time taken to compute path (milliseconds)
            **kwargs: Additional decision metadata
        """
        log_entry = {
            "timestamp": timestamp,
            "algorithm": algorithm,
            "enemy_id": enemy_id,
            "chosen_path": str(chosen_path),  # Convert to string for CSV
            "path_length": len(chosen_path),
            "alternative_paths": alternative_paths,
            "decision_time_ms": decision_time_ms,
        }
        log_entry.update(kwargs)
        self.decision_logs.append(log_entry)
    
    def log_wave_results(
        self,
        wave: int,
        algorithm: str,
        enemies_spawned: int,
        enemies_survived: int,
        avg_damage: float,
        path_diversity: float,
        total_cpu_time: float,
        **kwargs
    ) -> None:
        """
        Log results for a completed wave.
        
        Corresponds to "Performance Metrics" table in requirements.md.
        
        Args:
            wave: Wave number
            algorithm: Name of pathfinding algorithm
            enemies_spawned: Total enemies in wave
            enemies_survived: Number that reached goal
            avg_damage: Average damage taken per enemy
            path_diversity: Path diversity metric
            total_cpu_time: Total CPU time for pathfinding (seconds)
            **kwargs: Additional performance metrics
        """
        survival_percentage = (
            (enemies_survived / enemies_spawned * 100)
            if enemies_spawned > 0 else 0
        )
        
        log_entry = {
            "wave": wave,
            "algorithm": algorithm,
            "enemies_spawned": enemies_spawned,
            "enemies_survived": enemies_survived,
            "survival_percentage": survival_percentage,
            "avg_damage": avg_damage,
            "path_diversity": path_diversity,
            "cpu_time_seconds": total_cpu_time,
        }
        log_entry.update(kwargs)
        self.metric_logs.append(log_entry)
    
    def export_csv(self, prefix: str = "") -> Dict[str, str]:
        """
        Export all logs to CSV files.
        
        Args:
            prefix: Optional prefix for filenames
            
        Returns:
            Dictionary mapping log type to filepath
        """
        exported_files = {}
        
        # Export frame logs
        if self.frame_logs:
            filepath = self._export_to_csv(
                self.frame_logs,
                f"{prefix}game_state_{self.session_id}.csv"
            )
            exported_files["frame_logs"] = filepath
        
        # Export decision logs
        if self.decision_logs:
            filepath = self._export_to_csv(
                self.decision_logs,
                f"{prefix}decisions_{self.session_id}.csv"
            )
            exported_files["decision_logs"] = filepath
        
        # Export metric logs
        if self.metric_logs:
            filepath = self._export_to_csv(
                self.metric_logs,
                f"{prefix}metrics_{self.session_id}.csv"
            )
            exported_files["metric_logs"] = filepath
        
        return exported_files
    
    def _export_to_csv(self, data: List[Dict[str, Any]], filename: str) -> str:
        """
        Export data to a CSV file.
        
        Args:
            data: List of dictionaries to export
            filename: Name of the CSV file
            
        Returns:
            Full path to the exported file
        """
        if not data:
            return ""
        
        filepath = self.output_dir / filename
        
        # Get all unique keys from all dictionaries
        fieldnames = set()
        for entry in data:
            fieldnames.update(entry.keys())
        fieldnames = sorted(fieldnames)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        return str(filepath)
    
    def export_json(self, prefix: str = "") -> str:
        """
        Export all logs to a single JSON file.
        
        Args:
            prefix: Optional prefix for filename
            
        Returns:
            Path to the exported JSON file
        """
        filepath = self.output_dir / f"{prefix}session_{self.session_id}.json"
        
        data = {
            "session_id": self.session_id,
            "frame_count": self._frame_count,
            "frame_logs": self.frame_logs,
            "decision_logs": self.decision_logs,
            "metric_logs": self.metric_logs,
        }
        
        with open(filepath, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=2)
        
        return str(filepath)
    
    def clear(self) -> None:
        """Clear all log buffers."""
        self.frame_logs.clear()
        self.decision_logs.clear()
        self.metric_logs.clear()
        self._frame_count = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of logged data.
        
        Returns:
            Dictionary with log counts and statistics
        """
        return {
            "session_id": self.session_id,
            "frames_logged": self._frame_count,
            "frame_entries": len(self.frame_logs),
            "decisions_logged": len(self.decision_logs),
            "metrics_logged": len(self.metric_logs),
            "output_directory": str(self.output_dir),
        }

