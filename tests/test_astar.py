"""Test A* pathfinding implementation."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apathion.pathfinding.astar import AStarPathfinder
from apathion.game.map import Map
from apathion.game.tower import Tower, TowerType


def test_basic_astar():
    """Test basic A* pathfinding (only g(n) and h(n))."""
    print("=" * 60)
    print("Test 1: Basic A* (g(n) + h(n) only)")
    print("=" * 60)
    
    # Create pathfinder in basic mode
    pathfinder = AStarPathfinder(
        name="A*-Basic",
        use_enhanced=False,
        diagonal_movement=True
    )
    
    # Create simple map
    game_map = Map.create_simple_map()
    pathfinder.update_state(game_map, [])
    
    # Find path from spawn to goal
    start = game_map.spawn_points[0]
    goal = game_map.goal_positions[0]
    
    print(f"Finding path from {start} to {goal}")
    path = pathfinder.find_path(start, goal)
    
    print(f"Path found with {len(path)} steps")
    print(f"First 5 steps: {path[:5]}")
    print(f"Last 5 steps: {path[-5:]}")
    
    # Validate path
    is_valid, error = game_map.validate_path(path)
    print(f"Path validation: {'PASS' if is_valid else 'FAIL'}")
    if not is_valid:
        print(f"  Error: {error}")
    
    # Calculate path cost
    cost = pathfinder.calculate_path_cost(path)
    print(f"Path cost (distance): {cost:.2f}")
    
    print()


def test_enhanced_astar():
    """Test enhanced A* with damage and congestion costs."""
    print("=" * 60)
    print("Test 2: Enhanced A* (with damage and congestion)")
    print("=" * 60)
    
    # Create pathfinder in enhanced mode
    pathfinder = AStarPathfinder(
        name="A*-Enhanced",
        alpha=0.5,
        beta=0.3,
        use_enhanced=True,
        diagonal_movement=True
    )
    
    # Create simple map
    game_map = Map.create_simple_map()
    
    # Add tower in the middle to create a damage zone
    tower = Tower(
        id="tower_1",
        position=(15, 10),
        tower_type=TowerType.BASIC,
        damage=10,
        range=5.0,
        attack_rate=1.0
    )
    
    pathfinder.update_state(game_map, [tower])
    
    # Find path from spawn to goal
    start = game_map.spawn_points[0]
    goal = game_map.goal_positions[0]
    
    print(f"Finding path from {start} to {goal}")
    print(f"Tower at {tower.position} with range {tower.range}")
    path = pathfinder.find_path(start, goal)
    
    print(f"Path found with {len(path)} steps")
    print(f"First 5 steps: {path[:5]}")
    print(f"Last 5 steps: {path[-5:]}")
    
    # Validate path
    is_valid, error = game_map.validate_path(path)
    print(f"Path validation: {'PASS' if is_valid else 'FAIL'}")
    if not is_valid:
        print(f"  Error: {error}")
    
    # Calculate path cost
    cost = pathfinder.calculate_path_cost(path)
    print(f"Path cost (distance): {cost:.2f}")
    
    # Check if path avoids tower
    distances_to_tower = []
    for pos in path:
        dist = ((pos[0] - tower.position[0])**2 + (pos[1] - tower.position[1])**2)**0.5
        distances_to_tower.append(dist)
    
    min_dist = min(distances_to_tower)
    print(f"Minimum distance to tower: {min_dist:.2f}")
    print(f"Tower range: {tower.range:.2f}")
    
    print()


def test_branching_map():
    """Test A* on branching map with obstacles."""
    print("=" * 60)
    print("Test 3: A* on Branching Map")
    print("=" * 60)
    
    # Create pathfinder
    pathfinder = AStarPathfinder(
        name="A*-Basic",
        use_enhanced=False,
        diagonal_movement=True
    )
    
    # Create branching map
    game_map = Map.create_branching_map()
    pathfinder.update_state(game_map, [])
    
    # Find path from spawn to goal
    start = game_map.spawn_points[0]
    goal = game_map.goal_positions[0]
    
    print(f"Finding path from {start} to {goal}")
    path = pathfinder.find_path(start, goal)
    
    print(f"Path found with {len(path)} steps")
    
    # Validate path
    is_valid, error = game_map.validate_path(path)
    print(f"Path validation: {'PASS' if is_valid else 'FAIL'}")
    if not is_valid:
        print(f"  Error: {error}")
    
    # Calculate path cost
    cost = pathfinder.calculate_path_cost(path)
    print(f"Path cost (distance): {cost:.2f}")
    
    # Verify path avoids obstacles
    hits_obstacle = False
    for pos in path:
        if not game_map.is_walkable(pos[0], pos[1]):
            hits_obstacle = True
            print(f"  ERROR: Path goes through obstacle at {pos}")
    
    if not hits_obstacle:
        print("  Path successfully avoids all obstacles")
    
    print()


def test_heuristic():
    """Test Euclidean heuristic calculation."""
    print("=" * 60)
    print("Test 4: Euclidean Heuristic")
    print("=" * 60)
    
    pathfinder = AStarPathfinder()
    
    # Test cases
    test_cases = [
        ((0, 0), (3, 4), 5.0),      # 3-4-5 triangle
        ((0, 0), (1, 1), 1.414),    # Diagonal
        ((0, 0), (5, 0), 5.0),      # Horizontal
        ((0, 0), (0, 5), 5.0),      # Vertical
        ((5, 5), (10, 10), 7.071),  # Diagonal
    ]
    
    all_passed = True
    for start, goal, expected in test_cases:
        result = pathfinder.calculate_heuristic(start, goal)
        passed = abs(result - expected) < 0.01
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {start} -> {goal}: {result:.3f} (expected {expected:.3f}) [{status}]")
    
    print(f"\nAll heuristic tests: {'PASS' if all_passed else 'FAIL'}")
    print()


def test_mode_switching():
    """Test switching between basic and enhanced modes at runtime."""
    print("=" * 60)
    print("Test 5: Runtime Mode Switching")
    print("=" * 60)
    
    # Create pathfinder in enhanced mode
    pathfinder = AStarPathfinder(
        name="A*-Enhanced",
        alpha=0.5,
        beta=0.3,
        use_enhanced=True
    )
    
    # Create map with tower
    game_map = Map.create_simple_map()
    tower = Tower(
        id="tower_1",
        position=(15, 10),
        tower_type=TowerType.BASIC,
        damage=10,
        range=5.0,
        attack_rate=1.0
    )
    pathfinder.update_state(game_map, [tower])
    
    start = game_map.spawn_points[0]
    goal = game_map.goal_positions[0]
    
    # Find path in enhanced mode
    print("Finding path in enhanced mode...")
    path_enhanced = pathfinder.find_path(start, goal)
    cost_enhanced = pathfinder.calculate_path_cost(path_enhanced)
    print(f"  Steps: {len(path_enhanced)}, Cost: {cost_enhanced:.2f}")
    
    # Find path in basic mode (runtime override)
    print("Finding path in basic mode (runtime override)...")
    path_basic = pathfinder.find_path(start, goal, use_enhanced=False)
    cost_basic = pathfinder.calculate_path_cost(path_basic)
    print(f"  Steps: {len(path_basic)}, Cost: {cost_basic:.2f}")
    
    print(f"\nPath difference: {abs(len(path_enhanced) - len(path_basic))} steps")
    print(f"Cost difference: {abs(cost_enhanced - cost_basic):.2f}")
    
    print()


def test_pathfinder_config():
    """Test pathfinder configuration and logging."""
    print("=" * 60)
    print("Test 6: Pathfinder Configuration")
    print("=" * 60)
    
    # Test basic A*
    basic = AStarPathfinder(
        name="A*-Basic",
        use_enhanced=False
    )
    print("Basic A* config:")
    print(f"  {basic.to_dict()}")
    
    # Test enhanced A*
    enhanced = AStarPathfinder(
        name="A*-Enhanced",
        alpha=0.5,
        beta=0.3,
        use_enhanced=True
    )
    print("\nEnhanced A* config:")
    print(f"  {enhanced.to_dict()}")
    
    print()


if __name__ == "__main__":
    test_basic_astar()
    test_enhanced_astar()
    test_branching_map()
    test_heuristic()
    test_mode_switching()
    test_pathfinder_config()
    
    print("=" * 60)
    print("All A* tests completed!")
    print("=" * 60)

