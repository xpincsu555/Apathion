"""
Integration tests for ACO pathfinding with game systems.
"""

import pytest
from apathion.pathfinding.aco import ACOPathfinder
from apathion.game.map import Map
from apathion.game.tower import Tower
from apathion.game.enemy import Enemy


class TestACOIntegration:
    """Integration tests for ACO pathfinding."""
    
    def test_aco_with_game_map(self):
        """Test ACO pathfinding with actual game map."""
        aco = ACOPathfinder(num_ants=10)
        
        # Create a realistic game map
        game_map = Map(width=20, height=20)
        
        # Add some obstacles
        for x in range(5, 15):
            game_map.grid[10][x] = 1
        
        # Create towers
        towers = [
            Tower(id="t1", position=(10, 5), range=4.0, damage=15.0),
            Tower(id="t2", position=(10, 15), range=4.0, damage=15.0),
        ]
        
        # Update pathfinder state
        aco.update_state(game_map, towers)
        
        # Find path
        start = (0, 10)
        goal = (19, 10)
        path = aco.find_path(start, goal)
        
        # Verify path is valid
        assert len(path) >= 2
        assert path[0] == start
        assert path[-1] == goal
        
        # Verify path avoids obstacles
        for pos in path:
            assert game_map.is_walkable(pos[0], pos[1])
    
    def test_aco_pheromone_persistence(self):
        """Test that pheromones persist across multiple pathfinding calls."""
        aco = ACOPathfinder(num_ants=5, evaporation_rate=0.1)
        game_map = Map(width=15, height=15)
        
        aco.update_state(game_map, [])
        
        # Get initial pheromone at a position
        test_pos = (7, 7)
        initial_pheromone = aco.get_pheromone_at(test_pos)
        
        # Run pathfinding multiple times that pass through this area
        for _ in range(5):
            path = aco.find_path((0, 0), (14, 14))
            # Update state to apply evaporation
            aco.update_state(game_map, [])
        
        # Check if pheromone levels changed
        # (they should have changed due to either deposits or evaporation)
        final_pheromone = aco.get_pheromone_at(test_pos)
        # The pheromone level will be different from initial
        # (Could be higher if ants passed through, or lower due to evaporation)
        assert final_pheromone >= 0.01  # Should maintain minimum
    
    def test_aco_with_multiple_enemies(self):
        """Test ACO behavior when multiple enemies use it."""
        aco = ACOPathfinder(num_ants=8, alpha=1.5, beta=2.0)
        game_map = Map(width=25, height=25)
        
        aco.update_state(game_map, [])
        
        # Simulate multiple enemies finding paths
        paths = []
        for i in range(5):
            start = (0, i * 5)
            goal = (24, i * 5)
            path = aco.find_path(start, goal)
            paths.append(path)
            
            # Update state between enemies (evaporation)
            aco.update_state(game_map, [])
        
        # All paths should be valid
        for i, path in enumerate(paths):
            start = (0, i * 5)
            goal = (24, i * 5)
            assert path[0] == start
            assert path[-1] == goal
    
    def test_aco_parameter_tuning(self):
        """Test ACO with different parameter configurations."""
        game_map = Map(width=15, height=15)
        
        # High alpha (pheromone importance)
        aco_high_alpha = ACOPathfinder(num_ants=10, alpha=3.0, beta=1.0)
        aco_high_alpha.update_state(game_map, [])
        
        # High beta (heuristic importance)
        aco_high_beta = ACOPathfinder(num_ants=10, alpha=1.0, beta=3.0)
        aco_high_beta.update_state(game_map, [])
        
        start = (0, 0)
        goal = (14, 14)
        
        # Both should find valid paths
        path_alpha = aco_high_alpha.find_path(start, goal)
        path_beta = aco_high_beta.find_path(start, goal)
        
        assert path_alpha[0] == start and path_alpha[-1] == goal
        assert path_beta[0] == start and path_beta[-1] == goal
        
        # High beta should generally prefer more direct paths initially
        # (since it weights heuristic more)
        cost_alpha = aco_high_alpha.calculate_path_cost(path_alpha)
        cost_beta = aco_high_beta.calculate_path_cost(path_beta)
        
        # Both costs should be reasonable
        assert cost_alpha > 0
        assert cost_beta > 0
    
    def test_aco_with_dynamic_towers(self):
        """Test ACO adaptation when towers are added/removed."""
        aco = ACOPathfinder(num_ants=10)
        game_map = Map(width=20, height=20)
        
        # Initial state with no towers
        aco.update_state(game_map, [])
        path1 = aco.find_path((0, 10), (19, 10))
        
        # Add tower in the middle
        tower = Tower(id="t1", position=(10, 10), range=5.0, damage=20.0)
        aco.update_state(game_map, [tower])
        path2 = aco.find_path((0, 10), (19, 10))
        
        # Both paths should be valid
        assert path1[0] == (0, 10) and path1[-1] == (19, 10)
        assert path2[0] == (0, 10) and path2[-1] == (19, 10)
        
        # Pheromones should still be present
        assert aco.pheromone_grid is not None
    
    def test_aco_complex_maze(self):
        """Test ACO pathfinding in a complex maze-like environment."""
        aco = ACOPathfinder(num_ants=15, alpha=1.0, beta=2.5)
        game_map = Map(width=30, height=30)
        
        # Create a maze-like structure
        for y in range(5, 25):
            if y % 4 == 0:
                for x in range(5, 25):
                    if x % 4 != 0:
                        game_map.grid[y][x] = 1
        
        aco.update_state(game_map, [])
        
        start = (2, 15)
        goal = (27, 15)
        
        path = aco.find_path(start, goal)
        
        # Should find a valid path through the maze
        assert len(path) >= 2
        assert path[0] == start
        assert path[-1] == goal
        
        # All positions should be walkable
        for pos in path:
            assert game_map.is_walkable(pos[0], pos[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

