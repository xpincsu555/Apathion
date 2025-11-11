# ACO (Ant Colony Optimization) Pathfinding Implementation

## Overview

The ACO pathfinding algorithm is a nature-inspired optimization technique based on the foraging behavior of ants. It uses pheromone trails to guide path selection, enabling adaptive and emergent pathfinding behavior.

## Algorithm Description

### Core Concept

Ant Colony Optimization simulates the behavior of ant colonies that find optimal paths to food sources through pheromone communication:

1. **Multiple Ants**: Each pathfinding call simulates multiple "ants" exploring different paths
2. **Pheromone Trails**: Successful paths deposit virtual pheromones on grid cells
3. **Probabilistic Selection**: Future ants prefer paths with stronger pheromone trails
4. **Evaporation**: Pheromones decay over time, allowing adaptation to changes
5. **Quality Feedback**: Better paths (shorter) deposit more pheromones

### Key Features

- **Swarm Intelligence**: Emergent optimal behavior from simple agent interactions
- **Exploration vs Exploitation**: Balance between following known paths and discovering new ones
- **Adaptive**: Automatically adjusts to map changes through evaporation
- **Probabilistic**: Non-deterministic pathfinding with diverse solutions
- **Memory**: Pheromone grid maintains history of successful paths

## Implementation Details

### Class: `ACOPathfinder`

Location: `src/apathion/pathfinding/aco.py`

### Constructor Parameters

```python
ACOPathfinder(
    name: str = "ACO",
    num_ants: int = 10,
    evaporation_rate: float = 0.1,
    deposit_strength: float = 1.0,
    alpha: float = 1.0,
    beta: float = 2.0,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "ACO" | Identifier for the pathfinder |
| `num_ants` | int | 10 | Number of ants per pathfinding iteration |
| `evaporation_rate` | float | 0.1 | Pheromone decay rate (0-1, higher = faster decay) |
| `deposit_strength` | float | 1.0 | Base amount of pheromone deposited |
| `alpha` | float | 1.0 | Pheromone importance weight |
| `beta` | float | 2.0 | Heuristic (distance) importance weight |

### Algorithm Flow

```
1. Initialize pheromone grid with uniform values
2. For each ant (num_ants times):
   a. Start from initial position
   b. While not at goal:
      - Get valid neighbors
      - Calculate transition probabilities using pheromones and heuristic
      - Select next position probabilistically
      - Add to path
   c. Evaluate path quality
   d. Deposit pheromones proportional to path quality
3. Return best path found
4. Apply evaporation to all pheromones
```

### Transition Probability Formula

The probability of moving from current position to a neighbor is calculated as:

```
P(neighbor) = (τ^α) × (η^β) / Σ[(τ^α) × (η^β)]
```

Where:
- `τ` (tau) = pheromone level at neighbor position
- `η` (eta) = heuristic value (inverse distance to goal)
- `α` (alpha) = pheromone importance weight
- `β` (beta) = heuristic importance weight

### Pheromone Dynamics

#### Deposit
```
deposit_amount = (deposit_strength / path_length) × quality
quality = 1.0 / (path_cost + 1.0)
```

Shorter paths deposit more pheromone per cell.

#### Evaporation
```
pheromone[x, y] *= (1.0 - evaporation_rate)
pheromone[x, y] = max(pheromone[x, y], 0.01)  # Minimum threshold
```

## Usage Examples

### Basic Usage

```python
from apathion.pathfinding.aco import ACOPathfinder
from apathion.game.map import Map

# Create pathfinder
aco = ACOPathfinder(num_ants=10)

# Initialize with map
game_map = Map(width=20, height=20)
aco.update_state(game_map, [])

# Find path
start = (0, 0)
goal = (19, 19)
path = aco.find_path(start, goal)
```

### With Custom Parameters

```python
# High exploration (follow pheromones strongly)
aco_explore = ACOPathfinder(
    num_ants=20,
    alpha=3.0,  # Strong pheromone influence
    beta=1.0,   # Weak heuristic influence
    evaporation_rate=0.05  # Slow evaporation
)

# High exploitation (prefer direct paths)
aco_exploit = ACOPathfinder(
    num_ants=10,
    alpha=1.0,   # Weak pheromone influence
    beta=3.0,    # Strong heuristic influence
    evaporation_rate=0.2  # Fast evaporation
)
```

### With Tower Awareness

```python
from apathion.game.tower import Tower

# Create pathfinder
aco = ACOPathfinder(num_ants=15)

# Add towers
towers = [
    Tower(id="t1", position=(10, 10), range=5.0, damage=20.0)
]

# Update state with towers
aco.update_state(game_map, towers)

# Pathfinding will consider tower damage zones
path = aco.find_path(start, goal)
```

### Runtime Parameter Override

```python
# Use more ants for difficult pathfinding
path = aco.find_path(
    start,
    goal,
    num_ants=30,          # Override default
    max_iterations=2000   # Allow longer searches
)
```

## Parameter Tuning Guide

### Alpha (α) - Pheromone Importance

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.0 - 1.0 | Weak pheromone influence | Exploring new paths, changing environments |
| 1.0 - 2.0 | Balanced | General purpose |
| 2.0 - 4.0 | Strong pheromone influence | Converging to known good paths |

### Beta (β) - Heuristic Importance

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.0 - 1.0 | Weak heuristic influence | Trust pheromones over distance |
| 1.0 - 2.0 | Balanced | General purpose |
| 2.0 - 4.0 | Strong heuristic influence | Prefer direct paths initially |

### Number of Ants

| Range | Behavior | Performance |
|-------|----------|-------------|
| 5-10 | Fast, less exploration | Quick pathfinding |
| 10-20 | Balanced | General purpose |
| 20-50 | Thorough exploration | Better paths, slower |

### Evaporation Rate

| Range | Behavior | Use Case |
|-------|----------|----------|
| 0.01-0.1 | Slow decay | Stable environments |
| 0.1-0.3 | Moderate decay | Dynamic environments |
| 0.3-0.5 | Fast decay | Rapidly changing maps |

## Performance Characteristics

### Time Complexity

- **Per ant**: O(N × K) where N is path length, K is neighbors per node
- **Full search**: O(A × N × K) where A is number of ants
- **Average case**: O(A × sqrt(W² + H²) × 8) for W×H map

### Space Complexity

- **Pheromone grid**: O(W × H)
- **Path storage**: O(N) per path
- **Total**: O(W × H + A × N)

### Performance Tips

1. **Reduce num_ants** for faster pathfinding
2. **Increase beta** for more direct paths
3. **Increase evaporation_rate** for faster adaptation
4. **Cache paths** when map doesn't change frequently

## Comparison with Other Algorithms

| Feature | ACO | A* | DQN |
|---------|-----|----|----|
| Optimality | Near-optimal | Optimal | Learned |
| Speed | Moderate | Fast | Moderate |
| Memory | High | Moderate | High |
| Adaptability | High | Low | High |
| Deterministic | No | Yes | No |
| Setup | Simple | Simple | Complex |

## Advanced Features

### Pheromone Visualization

```python
# Get pheromone level at position
pheromone = aco.get_pheromone_at((x, y))

# Access full grid
if aco.pheromone_grid is not None:
    max_pheromone = aco.pheromone_grid.max()
    min_pheromone = aco.pheromone_grid.min()
```

### Custom Quality Functions

The algorithm can be extended with custom path quality metrics:

```python
# Current: quality = 1.0 / (cost + 1.0)
# Could incorporate: damage, congestion, time, etc.
```

### Multi-Objective Optimization

ACO can optimize multiple objectives:
- Path length
- Damage exposure
- Congestion avoidance
- Resource collection

## Testing

Run ACO tests:
```bash
python -m pytest tests/test_aco.py -v
python -m pytest tests/test_aco_integration.py -v
```

Run demonstration:
```bash
python examples/aco_demo.py
```

## API Reference

### Main Methods

#### `find_path(start, goal, **kwargs)`

Find path from start to goal using ACO algorithm.

**Parameters:**
- `start: Tuple[int, int]` - Starting (x, y) position
- `goal: Tuple[int, int]` - Goal (x, y) position
- `**kwargs` - Optional parameters:
  - `num_ants: int` - Override number of ants
  - `max_iterations: int` - Maximum iterations per ant (default 1000)

**Returns:**
- `List[Tuple[int, int]]` - Path from start to goal

#### `update_state(game_map, towers)`

Update pathfinder with current game state.

**Parameters:**
- `game_map: Map` - Current game map
- `towers: List[Tower]` - Active towers

#### `get_pheromone_at(position)`

Get pheromone level at a specific position.

**Parameters:**
- `position: Tuple[int, int]` - Grid position

**Returns:**
- `float` - Pheromone level

#### `calculate_transition_probability(current, neighbor, goal)`

Calculate probability of transitioning from current to neighbor.

**Parameters:**
- `current: Tuple[int, int]` - Current position
- `neighbor: Tuple[int, int]` - Neighbor position
- `goal: Tuple[int, int]` - Goal position

**Returns:**
- `float` - Transition probability

## References

### Academic Background

1. Dorigo, M., & Stützle, T. (2004). "Ant Colony Optimization." MIT Press.
2. Dorigo, M., & Di Caro, G. (1999). "Ant colony optimization: a new meta-heuristic."
3. Stützle, T., & Hoos, H. H. (2000). "MAX–MIN ant system." Future Generation Computer Systems.

### Implementation Notes

- Uses 8-directional movement (cardinal + diagonal)
- Maintains minimum pheromone threshold to prevent stagnation
- Quality-based pheromone deposit favors shorter paths
- Probabilistic selection allows diverse path exploration

## Future Enhancements

Potential improvements:
- [ ] Elite ant strategy (extra pheromones for best path)
- [ ] Local search optimization of found paths
- [ ] Adaptive parameter tuning
- [ ] Multi-colony cooperation
- [ ] Rank-based pheromone deposit
- [ ] 3D pheromone visualization
- [ ] Path smoothing post-processing

## License

This implementation is part of the Apathion project and follows the same license.

