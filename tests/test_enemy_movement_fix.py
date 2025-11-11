"""
Test to verify that enemies with fixed pathfinding move correctly.

This test addresses the bug where enemies would stay at the start point
without moving when using the fixed path algorithm.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from apathion.pathfinding.fixed import FixedPathfinder
from apathion.game.enemy import Enemy
from apathion.game.game import GameState
from apathion.game.map import Map


def test_enemy_movement_with_fixed_path():
    """Test that enemies move correctly with fixed pathfinding."""
    # Create a simple baseline path
    baseline_path = [
        (0, 10), (0, 9), (0, 8), (0, 7), (0, 6),
        (0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0),
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)
    ]
    
    # Create pathfinder with baseline path
    pathfinder = FixedPathfinder(baseline_path=baseline_path)
    
    # Create game map
    game_map = Map.create_simple_map()
    pathfinder.update_state(game_map, [])
    
    # Create game state
    game_state = GameState(game_map)
    game_state.start()
    
    # Spawn an enemy at the start of the path
    enemies = game_state.spawn_wave(num_enemies=1)
    assert len(enemies) == 1
    enemy = enemies[0]
    
    # Get path for the enemy
    start = (int(enemy.position[0]), int(enemy.position[1]))
    goal = baseline_path[-1]
    path = pathfinder.find_path(start, goal)
    
    # Verify path is not empty
    assert len(path) > 0, "Path should not be empty"
    
    # Set the path for the enemy
    enemy.set_path(path)
    
    # Verify enemy has a path
    assert len(enemy.current_path) > 0, "Enemy should have a non-empty path"
    
    # Get initial position
    initial_pos = enemy.position
    
    # Update game state (simulate movement)
    delta_time = 0.1  # 100ms
    game_state.update(delta_time)
    
    # Verify enemy has moved
    assert enemy.position != initial_pos, "Enemy should have moved from start position"
    
    # Verify enemy is moving in the right direction (toward first waypoint)
    first_waypoint = path[0]
    dx = first_waypoint[0] - initial_pos[0]
    dy = first_waypoint[1] - initial_pos[1]
    
    # Enemy should be moving toward the first waypoint
    moved_dx = enemy.position[0] - initial_pos[0]
    moved_dy = enemy.position[1] - initial_pos[1]
    
    # Check that the movement direction is correct (using dot product)
    if dx != 0 or dy != 0:
        # Normalize expected direction
        expected_length = (dx ** 2 + dy ** 2) ** 0.5
        expected_dx = dx / expected_length
        expected_dy = dy / expected_length
        
        # Normalize actual movement
        moved_length = (moved_dx ** 2 + moved_dy ** 2) ** 0.5
        if moved_length > 0:
            actual_dx = moved_dx / moved_length
            actual_dy = moved_dy / moved_length
            
            # Dot product should be close to 1 (same direction)
            dot_product = expected_dx * actual_dx + expected_dy * actual_dy
            assert dot_product > 0.99, f"Enemy should move in correct direction (dot={dot_product})"


def test_enemy_continues_moving():
    """Test that enemy continues moving along the path."""
    # Create a simple baseline path
    baseline_path = [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    
    # Create pathfinder
    pathfinder = FixedPathfinder(baseline_path=baseline_path)
    
    # Create map and game state
    game_map = Map.create_simple_map()
    pathfinder.update_state(game_map, [])
    game_state = GameState(game_map)
    game_state.start()
    
    # Create enemy at start
    enemy = Enemy.create_normal("test_enemy", (0.0, 5.0))
    game_state.enemies.append(enemy)
    
    # Set path
    start = (0, 5)
    goal = (5, 5)
    path = pathfinder.find_path(start, goal)
    enemy.set_path(path)
    
    # Track positions over multiple updates
    positions = [enemy.position]
    
    # Simulate 10 updates
    for _ in range(10):
        game_state.update(0.1)
        positions.append(enemy.position)
    
    # Verify enemy moved forward (x should increase)
    assert positions[-1][0] > positions[0][0], "Enemy should move forward along x-axis"
    
    # Verify y coordinate stays roughly constant (moving horizontally)
    assert abs(positions[-1][1] - positions[0][1]) < 0.5, "Enemy should stay on horizontal path"


def test_waypoint_type_conversion():
    """Test that integer waypoints are correctly converted to floats."""
    enemy = Enemy.create_normal("test_enemy", (0.0, 0.0))
    
    # Set a path with integer coordinates
    path = [(1, 0), (2, 0), (3, 0)]
    enemy.set_path(path)
    
    # Get next waypoint (returns int tuple)
    waypoint = enemy.get_next_waypoint()
    assert waypoint == (1, 0)
    
    # Convert to float as done in game.py
    waypoint_float = (float(waypoint[0]), float(waypoint[1]))
    
    # Move toward the waypoint
    reached = enemy.move(waypoint_float, 0.5)  # Move for 0.5 seconds
    
    # Verify enemy moved
    assert enemy.position != (0.0, 0.0), "Enemy should have moved"
    
    # If reached, verify position is correct
    if reached:
        assert enemy.position == (1.0, 0.0), "Enemy should be at waypoint"


if __name__ == "__main__":
    test_enemy_movement_with_fixed_path()
    test_enemy_continues_moving()
    test_waypoint_type_conversion()
    print("All tests passed!")

