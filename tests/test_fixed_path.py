"""Test fixed path pathfinder functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apathion.pathfinding.fixed import FixedPathfinder
from apathion.game.map import Map
from apathion.config import GameConfig


def test_fixed_pathfinder_basic():
    """Test basic fixed pathfinder functionality."""
    print("=" * 60)
    print("Test 1: Basic Fixed Pathfinder")
    print("=" * 60)
    
    # Create a simple path
    test_path = [(0, 10), (1, 10), (2, 10), (3, 10), (4, 10)]
    
    # Create pathfinder
    pathfinder = FixedPathfinder(baseline_path=test_path)
    print(f"Pathfinder name: {pathfinder.get_name()}")
    
    # Test find_path (should skip the start point and return rest of path)
    result = pathfinder.find_path((0, 10), (4, 10))
    print(f"Path length: {len(result)}")
    print(f"Path start: {result[0] if result else None}")
    print(f"Expected to skip (0, 10) and start from (1, 10)")
    
    # Should return path starting from next waypoint after start
    expected = test_path[1:]  # Skip first point since enemy is already there
    assert result == expected, f"Path should be {expected}, got {result}"
    assert result[0] == (1, 10), "Should start from next waypoint"
    print("✓ Basic functionality works\n")


def test_fixed_pathfinder_with_config():
    """Test fixed pathfinder with config file."""
    print("=" * 60)
    print("Test 2: Fixed Pathfinder with Config")
    print("=" * 60)
    
    # Load config with baseline path
    config_path = Path(__file__).parent.parent / "configs" / "branching_map.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
    
    config = GameConfig.from_json(str(config_path))
    
    # Convert baseline path to tuples
    baseline_path = [(p[0], p[1]) for p in config.map.baseline_path]
    
    # Create pathfinder
    pathfinder = FixedPathfinder(baseline_path=baseline_path)
    print(f"Loaded baseline path with {len(baseline_path)} points")
    
    # Get path
    result = pathfinder.find_path((0, 10), (29, 10))
    print(f"Returned path length: {len(result)}")
    print(f"Path start: {result[0]}")
    print(f"Path end: {result[-1]}")
    
    # Validate against map
    game_map = Map.create_branching_map()
    pathfinder.update_state(game_map, [])
    
    is_valid = pathfinder.validate_path()
    print(f"Path validation: {'PASS' if is_valid else 'FAIL'}")
    assert is_valid, "Path should be valid"
    print("✓ Config-based path works\n")


def test_fixed_pathfinder_cli_integration():
    """Test CLI integration with fixed pathfinder."""
    print("=" * 60)
    print("Test 3: CLI Integration")
    print("=" * 60)
    
    from apathion.cli import ApathionCLI
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "example_game.json"
    config = GameConfig.from_json(str(config_path))
    
    cli = ApathionCLI()
    
    # Create fixed pathfinder via CLI
    pathfinder = cli._create_pathfinder("fixed", game_config=config)
    print(f"Created pathfinder: {pathfinder.get_name()}")
    
    # Test path
    result = pathfinder.find_path((0, 10), (29, 10))
    print(f"Path length: {len(result)}")
    print(f"Path start: {result[0]}")
    print(f"Path end: {result[-1]}")
    
    # Path should have one less point since it skips the start position
    expected_length = len(config.map.baseline_path) - 1
    assert len(result) == expected_length, f"Path should have {expected_length} points"
    print("✓ CLI integration works\n")


if __name__ == "__main__":
    test_fixed_pathfinder_basic()
    test_fixed_pathfinder_with_config()
    test_fixed_pathfinder_cli_integration()
    
    print("=" * 60)
    print("All fixed path tests completed successfully!")
    print("=" * 60)

