"""Visual comparison of Basic A* vs Enhanced A* pathfinding."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apathion.pathfinding.astar import AStarPathfinder
from apathion.game.map import Map
from apathion.game.tower import Tower, TowerType


def visualize_path_on_map(game_map, path, towers, title):
    """Print a visual representation of the path on the map."""
    print(f"\n{title}")
    print("=" * 60)
    
    # Create visualization grid
    grid = []
    for y in range(game_map.height):
        row = []
        for x in range(game_map.width):
            if (x, y) in game_map.spawn_points:
                row.append('S')
            elif (x, y) in game_map.goal_positions:
                row.append('G')
            elif (x, y) in path:
                row.append('*')
            elif game_map.grid[y, x] == 1:
                row.append('#')
            elif any((x, y) == t.position for t in towers):
                row.append('T')
            else:
                # Check if in tower range
                in_range = False
                for t in towers:
                    dx = x - t.position[0]
                    dy = y - t.position[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist <= t.range:
                        in_range = True
                        break
                row.append('.' if not in_range else '·')
        grid.append(''.join(row))
    
    # Print grid
    for row in grid:
        print(row)
    
    print(f"\nLegend: S=Start, G=Goal, *=Path, T=Tower, #=Obstacle, ·=Tower Range, .=Empty")


def main():
    """Run visual comparison between Basic and Enhanced A*."""
    print("=" * 60)
    print("Visual Comparison: Basic A* vs Enhanced A*")
    print("=" * 60)
    
    # Create map
    game_map = Map.create_simple_map()
    
    # Add towers to create damage zones
    towers = [
        Tower(
            id="tower_1",
            position=(10, 10),
            tower_type=TowerType.BASIC,
            damage=15,
            range=4.0,
            attack_rate=1.0
        ),
        Tower(
            id="tower_2",
            position=(20, 10),
            tower_type=TowerType.BASIC,
            damage=15,
            range=4.0,
            attack_rate=1.0
        ),
    ]
    
    start = game_map.spawn_points[0]
    goal = game_map.goal_positions[0]
    
    print(f"\nMap: {game_map.width}x{game_map.height}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Towers: {len(towers)}")
    for i, tower in enumerate(towers, 1):
        print(f"  Tower {i}: position={tower.position}, range={tower.range}, damage={tower.damage}")
    
    # Test Basic A*
    print("\n" + "=" * 60)
    print("BASIC A* (shortest path, ignores towers)")
    print("=" * 60)
    
    basic_astar = AStarPathfinder(
        name="A*-Basic",
        use_enhanced=False,
        diagonal_movement=True
    )
    basic_astar.update_state(game_map, towers)
    basic_path = basic_astar.find_path(start, goal)
    basic_cost = basic_astar.calculate_path_cost(basic_path)
    
    print(f"\nPath length: {len(basic_path)} steps")
    print(f"Path cost (distance): {basic_cost:.2f}")
    
    # Calculate damage exposure for basic path
    basic_damage = 0
    for pos in basic_path:
        basic_damage += basic_astar.estimate_damage_at_position(pos)
    print(f"Total damage exposure: {basic_damage:.2f}")
    
    visualize_path_on_map(game_map, basic_path, towers, "Basic A* Path Visualization")
    
    # Test Enhanced A*
    print("\n" + "=" * 60)
    print("ENHANCED A* (balances distance and damage)")
    print("=" * 60)
    
    enhanced_astar = AStarPathfinder(
        name="A*-Enhanced",
        alpha=0.5,
        beta=0.3,
        use_enhanced=True,
        diagonal_movement=True
    )
    enhanced_astar.update_state(game_map, towers)
    enhanced_path = enhanced_astar.find_path(start, goal)
    enhanced_cost = enhanced_astar.calculate_path_cost(enhanced_path)
    
    print(f"\nPath length: {len(enhanced_path)} steps")
    print(f"Path cost (distance): {enhanced_cost:.2f}")
    
    # Calculate damage exposure for enhanced path
    enhanced_damage = 0
    for pos in enhanced_path:
        enhanced_damage += enhanced_astar.estimate_damage_at_position(pos)
    print(f"Total damage exposure: {enhanced_damage:.2f}")
    
    visualize_path_on_map(game_map, enhanced_path, towers, "Enhanced A* Path Visualization")
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Path length difference: {len(enhanced_path) - len(basic_path)} steps")
    print(f"Distance cost difference: {enhanced_cost - basic_cost:.2f}")
    print(f"Damage exposure difference: {enhanced_damage - basic_damage:.2f}")
    print(f"Damage reduction: {((basic_damage - enhanced_damage) / basic_damage * 100):.1f}%")
    
    print("\nConclusion:")
    if enhanced_damage < basic_damage:
        print("  ✓ Enhanced A* successfully reduced damage exposure")
        print(f"  ✓ Trade-off: {enhanced_cost - basic_cost:.2f} extra distance for safety")
    else:
        print("  - Both paths have similar damage profiles")
    
    print("\n" + "=" * 60)
    print("Visual comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

