"""
Configuration module for game parameters and experiment settings.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class MapConfig:
    """Configuration for map generation."""
    width: int = 30
    height: int = 20
    map_type: str = "simple"  # "simple", "branching", "open_arena", "dynamic_maze"
    obstacle_density: float = 0.0  # 0.0 to 1.0
    baseline_path: Optional[List[List[int]]] = None  # Optional fixed path as baseline [[x1,y1], [x2,y2], ...]
    obstacle_regions: Optional[List[List[int]]] = None  # Rectangular obstacle regions [[x1, y1, x2, y2], ...]
    spawn_points: Optional[List[List[int]]] = None  # Spawn points [[x1, y1], [x2, y2], ...]
    goal_positions: Optional[List[List[int]]] = None  # Goal positions [[x1, y1], [x2, y2], ...]
    

@dataclass
class EnemyConfig:
    """Configuration for enemy spawning and behavior."""
    enemies_per_wave: int = 10
    wave_count: int = 5
    spawn_delay: float = 1.0  # Seconds between enemy spawns
    enemy_types: List[str] = field(default_factory=lambda: ["normal"])
    enemy_speed_multiplier: float = 1.0


@dataclass
class TowerConfig:
    """Configuration for tower placement and stats."""
    initial_towers: int = 3
    tower_types: List[str] = field(default_factory=lambda: ["basic", "sniper", "rapid", "area"])
    allow_dynamic_placement: bool = True
    tower_damage_multiplier: float = 1.0
    tower_range_multiplier: float = 1.0
    initial_tower_placements: Optional[List[Dict[str, Any]]] = None  # List of {position: [x, y], type: str}


@dataclass
class AStarConfig:
    """Configuration for A* pathfinding algorithm."""
    name: str = "A*-Enhanced"
    alpha: float = 0.5  # Weight for damage cost
    beta: float = 0.3   # Weight for congestion cost
    diagonal_movement: bool = True
    use_enhanced: bool = True  # Use enhanced mode (damage + congestion), False for basic A*
    replan_frequency: int = 10  # Frames between replanning


@dataclass
class ACOConfig:
    """Configuration for ACO pathfinding algorithm."""
    name: str = "ACO"
    num_ants: int = 10
    evaporation_rate: float = 0.1
    deposit_strength: float = 1.0
    alpha: float = 1.0  # Pheromone importance
    beta: float = 2.0   # Heuristic importance
    gamma: float = 0.5  # Damage avoidance weight


@dataclass
class DQNConfig:
    """Configuration for DQN pathfinding algorithm."""
    name: str = "DQN"
    state_size: int = 32  # Feature vector size (updated to include directional walkability)
    action_size: int = 8  # 8 directional movements
    use_cache: bool = True
    cache_duration: int = 5
    model_path: Optional[str] = None
    plan_full_path: bool = True  # If True, generate complete path; if False, step-by-step
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.0003
    gamma: float = 0.99  # Discount factor
    buffer_size: int = 100000  # Replay buffer size
    learning_starts: int = 1000  # Steps before learning starts
    target_update_interval: int = 1000  # Steps between target network updates
    train_freq: int = 4  # Update frequency
    gradient_steps: int = 1  # Gradient steps per update
    
    # Exploration parameters
    exploration_fraction: float = 0.5  # Fraction of training for epsilon decay (increased for better exploration)
    exploration_initial_eps: float = 1.0  # Initial epsilon
    exploration_final_eps: float = 0.05  # Final epsilon
    
    # Network architecture
    net_arch: List[int] = field(default_factory=lambda: [128, 128])  # Hidden layers
    
    # Device and performance
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Training settings
    max_episode_steps: int = 500  # Max steps per episode
    verbose: int = 1  # Verbosity level


@dataclass
class FixedConfig:
    """Configuration for Fixed path algorithm."""
    name: str = "Fixed-Path"
    # baseline_path is stored in MapConfig, not here


@dataclass
class VisualizationConfig:
    """Configuration for pygame visualization."""
    window_width: int = 1200
    window_height: int = 800
    show_grid: bool = True
    show_tower_ranges: bool = True
    show_health_bars: bool = True
    show_pheromones: bool = True  # For ACO visualization
    show_paths: bool = True  # Show enemy paths in debug mode


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and logging."""
    log_directory: str = "data/logs"
    results_directory: str = "data/results"
    enable_frame_logging: bool = True
    enable_decision_logging: bool = True
    log_frequency: int = 1  # Log every N frames
    export_csv: bool = True
    export_json: bool = True


@dataclass
class GameConfig:
    """Main game configuration combining all settings."""
    # Sub-configurations
    map: MapConfig = field(default_factory=MapConfig)
    enemies: EnemyConfig = field(default_factory=EnemyConfig)
    towers: TowerConfig = field(default_factory=TowerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Algorithm-specific configurations
    astar: AStarConfig = field(default_factory=AStarConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    fixed: FixedConfig = field(default_factory=FixedConfig)
    
    # Game settings
    target_fps: int = 60
    simulation_speed: float = 1.0
    random_seed: Optional[int] = None
    
    # Algorithm selection
    algorithm: str = "astar"  # "astar", "aco", "dqn", "fixed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameConfig":
        """Create configuration from dictionary."""
        # Extract sub-configs
        map_config = MapConfig(**data.get("map", {}))
        enemy_config = EnemyConfig(**data.get("enemies", {}))
        tower_config = TowerConfig(**data.get("towers", {}))
        eval_config = EvaluationConfig(**data.get("evaluation", {}))
        vis_config = VisualizationConfig(**data.get("visualization", {}))
        
        # Extract algorithm-specific configs
        astar_config = AStarConfig(**data.get("astar", {}))
        aco_config = ACOConfig(**data.get("aco", {}))
        dqn_config = DQNConfig(**data.get("dqn", {}))
        fixed_config = FixedConfig(**data.get("fixed", {}))
        
        # Create main config
        config = cls(
            map=map_config,
            enemies=enemy_config,
            towers=tower_config,
            evaluation=eval_config,
            visualization=vis_config,
            astar=astar_config,
            aco=aco_config,
            dqn=dqn_config,
            fixed=fixed_config,
        )
        
        # Update other fields
        for key, value in data.items():
            if key not in ["map", "enemies", "towers", "evaluation", "visualization", 
                          "astar", "aco", "dqn", "fixed"]:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_json(cls, filepath: str) -> "GameConfig":
        """Load configuration from JSON file."""
        path = Path(filepath)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def default_astar(cls) -> "GameConfig":
        """Create default configuration for A* experiments."""
        config = cls()
        config.algorithm = "astar"
        config.enemies.enemies_per_wave = 50
        config.enemies.wave_count = 10
        return config
    
    @classmethod
    def default_aco(cls) -> "GameConfig":
        """Create default configuration for ACO experiments."""
        config = cls()
        config.algorithm = "aco"
        config.enemies.enemies_per_wave = 50
        config.enemies.wave_count = 10
        return config
    
    @classmethod
    def default_dqn(cls) -> "GameConfig":
        """Create default configuration for DQN experiments."""
        config = cls()
        config.algorithm = "dqn"
        config.enemies.enemies_per_wave = 10  # Fewer for DQN
        config.enemies.wave_count = 5
        config.target_fps = 30  # Lower FPS for DQN
        return config


@dataclass
class ExperimentConfig:
    """Configuration for comparative experiments."""
    name: str = "experiment"
    description: str = ""
    
    # Algorithms to compare
    algorithms: List[str] = field(default_factory=lambda: ["astar", "aco"])
    
    # Maps to test
    map_types: List[str] = field(default_factory=lambda: ["simple", "branching", "open_arena"])
    
    # Test parameters
    num_runs: int = 3  # Repetitions per configuration
    waves_per_run: int = 10
    enemies_per_wave: int = 30
    
    # Tower configurations to test
    tower_scenarios: List[List[tuple]] = field(default_factory=list)
    
    # Algorithm-specific configs
    astar: AStarConfig = field(default_factory=AStarConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    fixed: FixedConfig = field(default_factory=FixedConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        astar_config = AStarConfig(**data.get("astar", {}))
        aco_config = ACOConfig(**data.get("aco", {}))
        dqn_config = DQNConfig(**data.get("dqn", {}))
        fixed_config = FixedConfig(**data.get("fixed", {}))
        
        config = cls(
            astar=astar_config,
            aco=aco_config,
            dqn=dqn_config,
            fixed=fixed_config,
        )
        
        # Update other fields
        for key, value in data.items():
            if key not in ["astar", "aco", "dqn", "fixed"]:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_json(cls, filepath: str) -> "ExperimentConfig":
        """Load from JSON file."""
        path = Path(filepath)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, filepath: str) -> None:
        """Save to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined experiment configurations matching requirements.md
EXPERIMENT_PRESETS = {
    "baseline": ExperimentConfig(
        name="baseline",
        description="Baseline comparison of all algorithms on simple maps",
        algorithms=["astar", "aco"],
        map_types=["simple"],
        num_runs=3,
        waves_per_run=5,
        enemies_per_wave=30,
    ),
    
    "full_comparison": ExperimentConfig(
        name="full_comparison",
        description="Full comparison across all map types",
        algorithms=["astar", "aco", "dqn"],
        map_types=["simple", "branching", "open_arena"],
        num_runs=5,
        waves_per_run=10,
        enemies_per_wave=50,
    ),
    
    "performance_test": ExperimentConfig(
        name="performance_test",
        description="Performance scaling test with varying enemy counts",
        algorithms=["astar", "aco"],
        map_types=["simple"],
        num_runs=3,
        waves_per_run=10,
        enemies_per_wave=100,
    ),
}


def get_experiment_preset(preset_name: str) -> ExperimentConfig:
    """
    Get a predefined experiment configuration.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        ExperimentConfig instance
        
    Raises:
        KeyError: If preset name not found
    """
    if preset_name not in EXPERIMENT_PRESETS:
        available = ", ".join(EXPERIMENT_PRESETS.keys())
        raise KeyError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return EXPERIMENT_PRESETS[preset_name]

