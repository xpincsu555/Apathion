"""Test path validation functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apathion.game.map import Map
from apathion.config import GameConfig


def test_simple_map_validation():
    """Test path validation on simple map."""
    print("=" * 60)
    print("Test 1: Simple Map Path Validation")
    print("=" * 60)
    
    # Create simple map
    game_map = Map.create_simple_map()
    
    # Test valid path (straight line from spawn to goal)
    valid_path = [(x, 10) for x in range(30)]
    is_valid, error = game_map.validate_path(valid_path)
    print(f"Valid path test: {'PASS' if is_valid else 'FAIL'}")
    if not is_valid:
        print(f"  Error: {error}")
    
    # Test invalid path (starts at wrong location)
    invalid_path = [(5, 5)] + [(x, 10) for x in range(6, 30)]
    is_valid, error = game_map.validate_path(invalid_path)
    print(f"Invalid start test: {'PASS' if not is_valid else 'FAIL'}")
    print(f"  Expected error: {error}")
    
    # Test discontinuous path
    discontinuous_path = [(0, 10), (5, 10), (10, 10), (29, 10)]
    is_valid, error = game_map.validate_path(discontinuous_path)
    print(f"Discontinuous path test: {'PASS' if not is_valid else 'FAIL'}")
    print(f"  Expected error: {error}")
    print()


def test_branching_map_validation():
    """Test path validation on branching map."""
    print("=" * 60)
    print("Test 2: Branching Map Path Validation")
    print("=" * 60)
    
    # Create branching map
    game_map = Map.create_branching_map()
    
    # Test middle path (should avoid obstacles)
    middle_path = [
        (0, 10), (1, 10), (2, 10), (3, 10), (4, 11), (5, 11), (6, 11), (7, 11),
        (8, 11), (9, 11), (10, 11), (11, 11), (12, 11), (13, 11), (14, 10),
        (15, 10), (16, 10), (17, 10), (18, 10), (19, 10), (20, 10), (21, 10),
        (22, 10), (23, 10), (24, 10), (25, 10), (26, 10), (27, 10), (28, 10), (29, 10)
    ]
    is_valid, error = game_map.validate_path(middle_path)
    print(f"Middle route test: {'PASS' if is_valid else 'FAIL'}")
    if not is_valid:
        print(f"  Error: {error}")
    
    # Test path through obstacle (should fail)
    obstacle_path = [(x, 10) for x in range(30)]  # Straight line, may hit obstacles
    is_valid, error = game_map.validate_path(obstacle_path)
    print(f"Obstacle avoidance test: {'PASS' if not is_valid else 'FAIL'}")
    if not is_valid:
        print(f"  Expected error: {error}")
    
    print()


def test_config_loading():
    """Test loading config with baseline path."""
    print("=" * 60)
    print("Test 3: Config Loading with Baseline Path")
    print("=" * 60)
    
    # Load example config
    config_path = Path(__file__).parent.parent / "configs" / "example_game.json"
    if config_path.exists():
        config = GameConfig.from_json(str(config_path))
        print(f"Loaded config: {config_path.name}")
        print(f"  Map type: {config.map.map_type}")
        print(f"  Baseline path: {len(config.map.baseline_path) if config.map.baseline_path else 0} points")
        
        # Create map and validate
        if config.map.baseline_path:
            game_map = Map.create_simple_map()
            path_tuples = [(p[0], p[1]) for p in config.map.baseline_path]
            is_valid, error = game_map.validate_path(path_tuples)
            print(f"  Path validation: {'PASS' if is_valid else 'FAIL'}")
            if not is_valid:
                print(f"    Error: {error}")
    else:
        print(f"Config file not found: {config_path}")
    
    # Load branching config
    branching_path = Path(__file__).parent.parent / "configs" / "branching_map.json"
    if branching_path.exists():
        config = GameConfig.from_json(str(branching_path))
        print(f"\nLoaded config: {branching_path.name}")
        print(f"  Map type: {config.map.map_type}")
        print(f"  Baseline path: {len(config.map.baseline_path) if config.map.baseline_path else 0} points")
        
        # Create map and validate
        if config.map.baseline_path:
            game_map = Map.create_branching_map()
            path_tuples = [(p[0], p[1]) for p in config.map.baseline_path]
            is_valid, error = game_map.validate_path(path_tuples)
            print(f"  Path validation: {'PASS' if is_valid else 'FAIL'}")
            if not is_valid:
                print(f"    Error: {error}")
    else:
        print(f"Config file not found: {branching_path}")
    
    print()


def test_map_structure():
    """Test branching map structure."""
    print("=" * 60)
    print("Test 4: Branching Map Structure")
    print("=" * 60)
    
    game_map = Map.create_branching_map()
    print(f"Map dimensions: {game_map.width}x{game_map.height}")
    print(f"Spawn points: {game_map.spawn_points}")
    print(f"Goal positions: {game_map.goal_positions}")
    
    # Count obstacles
    obstacle_count = (game_map.grid == 1).sum()
    print(f"Total obstacles: {obstacle_count}")
    print(f"Obstacle density: {obstacle_count / (game_map.width * game_map.height):.2%}")
    
    print("\nMap visualization (. = walkable, # = obstacle):")
    for y in range(game_map.height):
        row = ""
        for x in range(game_map.width):
            if (x, y) in game_map.spawn_points:
                row += "S"
            elif (x, y) in game_map.goal_positions:
                row += "G"
            elif game_map.grid[y, x] == 1:
                row += "#"
            else:
                row += "."
        print(row)
    
    print()


if __name__ == "__main__":
    test_simple_map_validation()
    test_branching_map_validation()
    test_config_loading()
    test_map_structure()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

