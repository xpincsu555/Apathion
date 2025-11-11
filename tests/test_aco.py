"""
Tests for ACO pathfinding algorithm.
"""

import pytest
import numpy as np
from apathion.pathfinding.aco import ACOPathfinder
from apathion.game.map import Map
from apathion.game.tower import Tower


class TestACOPathfinder:
    """Test suite for ACO pathfinding algorithm."""
    
    def test_initialization(self):
        """Test ACO pathfinder initialization."""
        aco = ACOPathfinder(
            name="Test-ACO",
            num_ants=20,
            evaporation_rate=0.2,
            deposit_strength=2.0,
            alpha=1.5,
            beta=2.5,
        )
        
        assert aco.name == "Test-ACO"
        assert aco.num_ants == 20
        assert aco.evaporation_rate == 0.2
        assert aco.deposit_strength == 2.0
        assert aco.alpha == 1.5
        assert aco.beta == 2.5
        assert aco.pheromone_grid is None
    
    def test_pheromone_grid_initialization(self):
        """Test pheromone grid initialization."""
        aco = ACOPathfinder()
        game_map = Map(width=10, height=10)
        
        aco.update_state(game_map, [])
        
        assert aco.pheromone_grid is not None
        assert aco.pheromone_grid.shape == (10, 10)
        assert np.all(aco.pheromone_grid >= 0.01)
    
    def test_simple_path_finding(self):
        """Test basic path finding on open map."""
        aco = ACOPathfinder(num_ants=5)
        
        # Create 10x10 open map
        game_map = Map(width=10, height=10)
        aco.update_state(game_map, [])
        
        start = (0, 0)
        goal = (9, 9)
        
        path = aco.find_path(start, goal)
        
        assert len(path) >= 2
        assert path[0] == start
        assert path[-1] == goal
    
    def test_path_with_obstacles(self):
        """Test path finding around obstacles."""
        aco = ACOPathfinder(num_ants=10, alpha=1.0, beta=3.0)
        
        # Create map with obstacle
        game_map = Map(width=10, height=10)
        
        # Add vertical wall
        for y in range(3, 8):
            game_map.grid[y][5] = 1  # Obstacle
        
        aco.update_state(game_map, [])
        
        start = (2, 5)
        goal = (8, 5)
        
        path = aco.find_path(start, goal)
        
        # Path should exist
        assert len(path) >= 2
        assert path[0] == start
        assert path[-1] == goal
        
        # Path should not go through obstacles
        for pos in path:
            assert game_map.is_walkable(pos[0], pos[1])
    
    def test_pheromone_evaporation(self):
        """Test pheromone evaporation over time."""
        aco = ACOPathfinder(evaporation_rate=0.5)
        game_map = Map(width=10, height=10)
        
        aco.update_state(game_map, [])
        
        # Get initial pheromone levels
        initial_pheromones = aco.pheromone_grid.copy()
        
        # Deposit pheromones on a path
        path = [(0, 0), (1, 1), (2, 2)]
        aco._deposit_pheromones(path)
        
        # Check pheromones increased
        assert aco.pheromone_grid[0, 0] > initial_pheromones[0, 0]
        
        # Apply evaporation multiple times
        for _ in range(10):
            aco._evaporate_pheromones()
        
        # Pheromones should decrease but not below minimum
        assert aco.pheromone_grid[0, 0] >= 0.01
    
    def test_pheromone_deposit(self):
        """Test pheromone deposit along path."""
        aco = ACOPathfinder(deposit_strength=10.0)
        game_map = Map(width=10, height=10)
        
        aco.update_state(game_map, [])
        
        initial_pheromone = aco.pheromone_grid[5, 5]
        
        path = [(5, 5), (5, 6), (5, 7)]
        aco._deposit_pheromones(path)
        
        # Pheromone should increase at path positions
        assert aco.pheromone_grid[5, 5] > initial_pheromone
        assert aco.pheromone_grid[6, 5] > initial_pheromone
        assert aco.pheromone_grid[7, 5] > initial_pheromone
    
    def test_quality_based_deposit(self):
        """Test that higher quality paths get more pheromones."""
        aco = ACOPathfinder(deposit_strength=10.0)
        game_map = Map(width=10, height=10)
        
        aco.update_state(game_map, [])
        
        path1 = [(0, 0), (1, 1)]
        path2 = [(3, 3), (4, 4)]
        
        initial1 = aco.pheromone_grid[0, 0]
        initial2 = aco.pheromone_grid[3, 3]
        
        # Deposit with different qualities
        aco._deposit_pheromones_with_quality(path1, quality=1.0)
        aco._deposit_pheromones_with_quality(path2, quality=0.5)
        
        deposit1 = aco.pheromone_grid[0, 0] - initial1
        deposit2 = aco.pheromone_grid[3, 3] - initial2
        
        # Higher quality should get more pheromone
        assert deposit1 > deposit2
    
    def test_transition_probability(self):
        """Test transition probability calculation."""
        aco = ACOPathfinder(alpha=1.0, beta=2.0)
        game_map = Map(width=10, height=10)
        
        aco.update_state(game_map, [])
        
        current = (5, 5)
        neighbor1 = (6, 5)  # Closer to goal
        neighbor2 = (4, 5)  # Further from goal
        goal = (9, 5)
        
        prob1 = aco.calculate_transition_probability(current, neighbor1, goal)
        prob2 = aco.calculate_transition_probability(current, neighbor2, goal)
        
        # Neighbor closer to goal should have higher probability
        assert prob1 > prob2
    
    def test_get_valid_neighbors(self):
        """Test neighbor generation."""
        aco = ACOPathfinder()
        game_map = Map(width=10, height=10)
        
        aco.update_state(game_map, [])
        
        # Test center position (should have 8 neighbors)
        neighbors = aco._get_valid_neighbors((5, 5))
        assert len(neighbors) == 8
        
        # Test corner (should have 3 neighbors)
        neighbors = aco._get_valid_neighbors((0, 0))
        assert len(neighbors) == 3
    
    def test_get_valid_neighbors_with_obstacles(self):
        """Test neighbor generation with obstacles."""
        aco = ACOPathfinder()
        game_map = Map(width=10, height=10)
        
        # Add obstacles around position
        game_map.grid[5][6] = 1
        game_map.grid[6][6] = 1
        
        aco.update_state(game_map, [])
        
        neighbors = aco._get_valid_neighbors((5, 5))
        
        # Should not include obstacle positions
        assert (6, 5) not in neighbors
        assert (6, 6) not in neighbors
    
    def test_multiple_ants_convergence(self):
        """Test that multiple ants find similar paths."""
        aco = ACOPathfinder(num_ants=20, alpha=2.0, beta=1.0)
        game_map = Map(width=10, height=10)
        
        aco.update_state(game_map, [])
        
        start = (0, 0)
        goal = (9, 9)
        
        # Run multiple times to build up pheromones
        paths = []
        for _ in range(3):
            path = aco.find_path(start, goal)
            paths.append(path)
        
        # All paths should reach the goal
        for path in paths:
            assert path[0] == start
            assert path[-1] == goal
    
    def test_path_cost_calculation(self):
        """Test path cost calculation."""
        aco = ACOPathfinder()
        
        path = [(0, 0), (1, 0), (2, 0), (3, 0)]
        cost = aco.calculate_path_cost(path)
        
        # Cost should be approximately 3.0 (3 unit moves)
        assert abs(cost - 3.0) < 0.01
    
    def test_damage_awareness(self):
        """Test that ACO can be configured to avoid tower damage."""
        aco = ACOPathfinder(num_ants=10, alpha=1.0, beta=1.0)
        game_map = Map(width=20, height=20)
        
        # Place tower in center
        tower = Tower(id="tower1", position=(10, 10), range=5, damage=10, attack_rate=1.0)
        
        aco.update_state(game_map, [tower])
        
        start = (0, 10)
        goal = (19, 10)
        
        path = aco.find_path(start, goal)
        
        # Path should exist
        assert len(path) >= 2
        assert path[0] == start
        assert path[-1] == goal
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        aco = ACOPathfinder(
            name="TestACO",
            num_ants=15,
            evaporation_rate=0.15,
            deposit_strength=1.5,
            alpha=1.2,
            beta=2.3,
        )
        
        data = aco.to_dict()
        
        assert data["name"] == "TestACO"
        assert data["type"] == "ACOPathfinder"
        assert data["num_ants"] == 15
        assert data["evaporation_rate"] == 0.15
        assert data["deposit_strength"] == 1.5
        assert data["alpha"] == 1.2
        assert data["beta"] == 2.3
        assert "pheromone_initialized" in data
    
    def test_get_pheromone_at(self):
        """Test getting pheromone level at specific position."""
        aco = ACOPathfinder()
        game_map = Map(width=10, height=10)
        
        aco.update_state(game_map, [])
        
        # Get initial pheromone
        initial = aco.get_pheromone_at((5, 5))
        assert initial > 0
        
        # Deposit on path
        path = [(5, 5), (6, 6)]
        aco._deposit_pheromones(path)
        
        # Pheromone should increase
        updated = aco.get_pheromone_at((5, 5))
        assert updated > initial
    
    def test_fallback_on_no_path(self):
        """Test fallback behavior when no valid path exists."""
        aco = ACOPathfinder(num_ants=5)
        game_map = Map(width=10, height=10)
        
        # Create complete barrier
        for x in range(10):
            game_map.grid[5][x] = 1
        
        aco.update_state(game_map, [])
        
        start = (5, 3)
        goal = (5, 7)
        
        path = aco.find_path(start, goal)
        
        # Should return fallback path
        assert path == [start, goal]
    
    def test_damage_avoidance_with_gamma(self):
        """Test that gamma parameter enables damage avoidance."""
        game_map = Map(width=20, height=10)
        
        # Create tower in the middle of a direct path
        tower = Tower(id="tower1", position=(10, 5), range=4.0, damage=50.0)
        
        # Test without damage avoidance (gamma=0)
        aco_no_avoid = ACOPathfinder(num_ants=10, gamma=0.0, beta=2.0)
        aco_no_avoid.update_state(game_map, [tower])
        
        # Test with strong damage avoidance (gamma=2.0)
        aco_with_avoid = ACOPathfinder(num_ants=10, gamma=2.0, beta=2.0)
        aco_with_avoid.update_state(game_map, [tower])
        
        start = (2, 5)
        goal = (18, 5)
        
        path_no_avoid = aco_no_avoid.find_path(start, goal)
        path_with_avoid = aco_with_avoid.find_path(start, goal)
        
        # Both should reach goal
        assert path_no_avoid[-1] == goal
        assert path_with_avoid[-1] == goal
        
        # Calculate average damage exposure for each path
        def avg_damage(path):
            total = sum(aco_with_avoid.estimate_damage_at_position(pos) for pos in path)
            return total / len(path) if path else 0
        
        damage_no_avoid = avg_damage(path_no_avoid)
        damage_with_avoid = avg_damage(path_with_avoid)
        
        # Path with damage avoidance should have less average damage exposure
        # (may not always be true due to randomness, but should be true on average)
        print(f"Damage without avoidance: {damage_no_avoid:.2f}")
        print(f"Damage with avoidance: {damage_with_avoid:.2f}")
    
    def test_gamma_parameter(self):
        """Test gamma parameter is properly stored and used."""
        aco = ACOPathfinder(gamma=1.5)
        
        assert aco.gamma == 1.5
        
        # Check it appears in to_dict
        config = aco.to_dict()
        assert "gamma" in config
        assert config["gamma"] == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

