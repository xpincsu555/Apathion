# A* Pathfinding Implementation

## Overview

This document describes the A* pathfinding algorithm implementation for the Apathion project. The implementation provides both **Basic A*** and **Enhanced A*** variants with different cost functions.

## CLI Algorithm Names

The implementation can be accessed via three algorithm names in the CLI:

| Algorithm Name | Mode | Logged As | Description |
|---------------|------|-----------|-------------|
| `astar_basic` | Basic | "A*-Basic" | Uses only g(n) + h(n), finds shortest path |
| `astar_enhanced` | Enhanced | "A*-Enhanced" | Adds damage and congestion costs |
| `astar` | Enhanced | "A*-Enhanced" | Default, same as astar_enhanced |

**Quick Example:**
```bash
# Compare both variants
apathion evaluate --algorithms=astar_basic,astar_enhanced --maps=simple
```

## Features

### 1. Basic A* Algorithm
- Uses only `g(n)` (cost from start) and `h(n)` (heuristic to goal)
- Cost function: `f(n) = g(n) + h(n)`
- Provides optimal shortest path without considering external factors
- Logged as `"A*-Basic"` in experiment results

### 2. Enhanced A* Algorithm
- Extends basic A* with additional cost components
- Cost function: `f(n) = g(n) + h(n) + α*damage(n) + β*congestion(n)`
- Considers:
  - **Damage zones**: Avoids areas with high tower damage exposure
  - **Congestion**: Can incorporate enemy density (placeholder for future implementation)
- Logged as `"A*-Enhanced"` in experiment results

### 3. Euclidean Heuristic
- Uses Euclidean distance for heuristic calculation
- Formula: `h(n) = sqrt((x - goal_x)² + (y - goal_y)²)`
- Admissible and consistent for grid-based pathfinding with diagonal movement

## Implementation Details

### Core Algorithm
The implementation follows the standard A* algorithm:

1. Initialize open set (priority queue) with start node
2. While open set is not empty:
   - Pop node with lowest f-score
   - If goal reached, reconstruct path
   - For each neighbor:
     - Calculate tentative g-score
     - Add enhanced costs (if enabled)
     - Update if better path found
3. Return path or fallback to straight line

### Movement
- Supports both cardinal (N, S, E, W) and diagonal movement
- Movement cost is calculated using Euclidean distance
- Respects obstacle boundaries and map edges

### Cost Calculation

#### Basic Mode (`use_enhanced=False`)
```python
movement_cost = euclidean_distance(current, neighbor)
g_score = g_score[current] + movement_cost
```

#### Enhanced Mode (`use_enhanced=True`)
```python
movement_cost = euclidean_distance(current, neighbor)
if alpha > 0:
    movement_cost += alpha * damage_at_position(neighbor)
if beta > 0:
    movement_cost += beta * congestion_at_position(neighbor)
g_score = g_score[current] + movement_cost
```

## Configuration

### In Code
```python
from apathion.pathfinding.astar import AStarPathfinder

# Basic A*
basic_astar = AStarPathfinder(
    name="A*-Basic",
    use_enhanced=False,
    diagonal_movement=True
)

# Enhanced A*
enhanced_astar = AStarPathfinder(
    name="A*-Enhanced",
    alpha=0.5,        # Weight for damage cost
    beta=0.3,         # Weight for congestion cost
    use_enhanced=True,
    diagonal_movement=True
)
```

### In JSON Config
```json
{
  "astar": {
    "name": "A*-Enhanced",
    "alpha": 0.5,
    "beta": 0.3,
    "diagonal_movement": true,
    "use_enhanced": true
  }
}
```

### Runtime Override
You can override the mode at runtime when calling `find_path`:
```python
# Use basic A* for this specific path, even if pathfinder is enhanced
path = pathfinder.find_path(start, goal, use_enhanced=False)

# Override weights
path = pathfinder.find_path(start, goal, alpha=0.8, beta=0.2)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "A*-Enhanced" | Algorithm name for logging |
| `alpha` | float | 0.5 | Weight for damage cost component |
| `beta` | float | 0.3 | Weight for congestion cost component |
| `diagonal_movement` | bool | True | Allow diagonal movement |
| `use_enhanced` | bool | True | Use enhanced cost function |

## Example Configs

### Basic A* Configuration
See `configs/astar_basic.json`:
```json
{
  "name": "astar_basic",
  "description": "Basic A* using only g(n) and h(n) costs",
  "algorithms": ["astar"],
  "astar": {
    "name": "A*-Basic",
    "alpha": 0.0,
    "beta": 0.0,
    "diagonal_movement": true,
    "use_enhanced": false
  }
}
```

### Enhanced A* Configuration
See `configs/astar_comparison.json`:
```json
{
  "name": "astar_basic_vs_enhanced",
  "description": "Comparison between basic and enhanced A*",
  "algorithms": ["astar"],
  "astar": {
    "name": "A*-Enhanced",
    "alpha": 0.5,
    "beta": 0.3,
    "diagonal_movement": true,
    "use_enhanced": true
  }
}
```

## Testing

Run the comprehensive A* test suite:
```bash
python tests/test_astar.py
```

Test coverage includes:
1. Basic A* pathfinding
2. Enhanced A* with damage avoidance
3. Pathfinding on maps with obstacles
4. Euclidean heuristic accuracy
5. Runtime mode switching
6. Configuration serialization

## Usage Examples

### Running with Algorithm Names

The CLI supports three algorithm names:

```bash
# Basic A* (shortest path, no damage avoidance)
apathion play --algorithm=astar_basic

# Enhanced A* (damage avoidance)
apathion play --algorithm=astar_enhanced

# Default "astar" uses enhanced mode
apathion play --algorithm=astar
```

### Running with Config Files

```bash
# Basic A*
apathion play --config configs/astar_basic.json

# Enhanced A*
apathion play --config configs/astar_comparison.json
```

### Comparing Both Modes

```bash
# Compare using algorithm names
apathion evaluate --algorithms=astar_basic,astar_enhanced --maps=simple,branching

# Or use separate config files
apathion evaluate --config configs/astar_basic.json
apathion evaluate --config configs/astar_comparison.json
```

Results will be logged with different algorithm names:
- `"A*-Basic"` for basic mode
- `"A*-Enhanced"` for enhanced mode

## Performance Characteristics

### Time Complexity
- **Best case**: O(b^d) where b is branching factor, d is depth
- **Worst case**: O(b^d) with large damage/congestion weights
- **Typical**: Very efficient for grid-based maps

### Space Complexity
- O(b^d) for storing nodes in open and closed sets

### Optimality
- **Basic A***: Guaranteed optimal path (shortest distance)
- **Enhanced A***: Optimal for composite cost function (distance + damage + congestion)

## Algorithm Comparison

| Feature | Basic A* | Enhanced A* |
|---------|----------|-------------|
| Cost function | g(n) + h(n) | g(n) + h(n) + α*damage + β*congestion |
| Path optimality | Shortest distance | Balanced distance/safety |
| Tower awareness | No | Yes |
| Use case | Baseline comparison | Intelligent enemy behavior |
| Log name | "A*-Basic" | "A*-Enhanced" |

## Future Enhancements

1. **Congestion Implementation**: Currently a placeholder, could track enemy density
2. **Dynamic Replanning**: Replan when towers are placed or destroyed
3. **Adaptive Weights**: Adjust α and β based on enemy health/speed
4. **Jump Point Search**: Optimization for uniform cost grids
5. **Hierarchical A***: For very large maps

## References

- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Transactions on Systems Science and Cybernetics.
- Euclidean distance heuristic for admissibility with diagonal movement
- Standard A* implementation patterns from game development literature

