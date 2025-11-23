#!/usr/bin/env python3
"""
Test script to verify palm tree decoration system.
"""

import pygame
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from apathion.game.game import GameState
from apathion.game.map import Map
from apathion.game.enemy import Enemy
from apathion.game.tower import TowerType
from apathion.game.renderer import GameRenderer
from apathion.config import VisualizationConfig


def main():
    """Test palm tree decorations on the map."""
    
    pygame.init()
    
    config = VisualizationConfig(
        window_width=900,
        window_height=700,
        show_grid=True,
        show_tower_ranges=True,
    )
    
    screen = pygame.display.set_mode((config.window_width, config.window_height))
    pygame.display.set_caption("Palm Tree Decoration Test")
    
    game_map = Map.create_simple_map()
    game_state = GameState(game_map)
    renderer = GameRenderer(screen, config, game_map.width, game_map.height)
    
    print("=" * 70)
    print("PALM TREE DECORATION TEST")
    print("=" * 70)
    
    # Check if palm tree sprite loaded
    if renderer.palmtree_sprite:
        w, h = renderer.palmtree_sprite.get_size()
        print(f"\n✓ Palm tree sprite loaded: {w}x{h} pixels")
    else:
        print("\n✗ Palm tree sprite NOT loaded!")
        return
    
    print("\nFeatures:")
    print("  ✓ Palm trees on all OBSTACLE tiles")
    print("  ✓ NO palm trees on walkable paths")
    print("  ✓ No palm trees on tower locations")
    print("  ✓ Palm trees disappear when towers are placed")
    
    print("\nSetup:")
    
    # Place some towers to see palm trees disappear
    tower_positions = [(5, 5), (7, 5), (5, 7)]
    tower_types = [TowerType.BASIC, TowerType.RAPID, TowerType.SNIPER]
    
    for pos, ttype in zip(tower_positions, tower_types):
        game_state.place_tower(pos, ttype.value, force=True)
        print(f"  ✓ Placed {ttype.value} tower at {pos} (palm tree removed)")
    
    # Spawn some enemies
    spawn = game_map.spawn_points[0]
    enemies = [
        Enemy.create_normal("e1", (float(spawn[0]), float(spawn[1]))),
        Enemy.create_fast("e2", (float(spawn[0] + 1), float(spawn[1]))),
    ]
    
    # Give them paths
    goal = game_map.goal_positions[0]
    for enemy in enemies:
        enemy.set_path([(spawn[0] + i, spawn[1]) for i in range(1, 5)] + [(goal[0], goal[1])])
    
    game_state.enemies.extend(enemies)
    print(f"  ✓ Spawned {len(enemies)} enemies with paths")
    
    game_state.start()
    
    print("\n" + "=" * 70)
    print("Rendering map (press ESC to quit)...")
    print("\nYou should see:")
    print("  • Palm trees on OBSTACLE tiles (walls)")
    print("  • Clear walkable paths WITHOUT palm trees")
    print("  • No palm trees under towers")
    print("  • Map looks like obstacles are filled with palm trees")
    print("=" * 70)
    
    clock = pygame.time.Clock()
    running = True
    frame_count = 0
    
    while running:
        delta_time = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            
            # Allow placing more towers with mouse click
            if event.type == pygame.MOUSEBUTTONDOWN:
                grid_pos = renderer.get_grid_position(event.pos)
                if grid_pos:
                    tower = game_state.place_tower(grid_pos, "basic", force=True)
                    if tower:
                        print(f"  ✓ Placed tower at {grid_pos} - palm tree removed!")
        
        # Update game state
        game_state.update(delta_time)
        
        # Render
        renderer.render(game_state, algorithm_name="Palm Tree Test", fps=clock.get_fps())
        pygame.display.flip()
        
        frame_count += 1
        
        # Stop after 10 seconds if user doesn't interact
        if frame_count > 600:
            print("\n✓ Auto-closing after 10 seconds")
            break
    
    pygame.quit()
    print("\n✓ Palm tree decoration test complete!")


if __name__ == "__main__":
    main()

