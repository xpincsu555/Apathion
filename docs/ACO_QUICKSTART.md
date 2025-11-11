# ACO Pathfinding Quick Start Guide

## What is ACO?

Ant Colony Optimization (ACO) is a nature-inspired pathfinding algorithm that simulates how ants find optimal paths using pheromone trails. Multiple "ants" explore paths and deposit pheromones, with better paths receiving stronger trails that guide future pathfinding.

## Quick Start

### 1. Basic Usage in Code

```python
from apathion.pathfinding.aco import ACOPathfinder
from apathion.game.map import Map

# Create pathfinder
aco = ACOPathfinder(num_ants=10)

# Initialize with map
game_map = Map(width=20, height=20)
aco.update_state(game_map, [])

# Find path
path = aco.find_path((0, 0), (19, 19))
print(f"Found path with {len(path)} steps")
```

### 2. Using the CLI

```bash
# Run interactive game with ACO
apathion play --algorithm=aco --map_type=branching --waves=5

# Run with custom config file
apathion play --config_file=configs/aco_example.json

# Run experiment comparing algorithms
apathion evaluate --config_file=configs/example_experiment.json
```

### 3. Custom Configuration

Create a config file (e.g., `my_aco_config.json`):

```json
{
  "algorithm": "aco",
  "pathfinding": {
    "aco": {
      "name": "My-ACO",
      "num_ants": 20,
      "alpha": 1.5,
      "beta": 2.0,
      "evaporation_rate": 0.15
    }
  }
}
```

## Key Parameters

| Parameter | What it does | Tip |
|-----------|-------------|-----|
| `num_ants` | More ants = more exploration | Start with 10-15 |
| `alpha` | Pheromone importance | Higher = follow trails more |
| `beta` | Distance importance | Higher = prefer direct paths |
| `evaporation_rate` | How fast trails fade | Higher = adapt faster |

## Common Use Cases

### Fast Pathfinding
```python
# Use fewer ants and high beta for quick, direct paths
aco = ACOPathfinder(num_ants=5, beta=3.0)
```

### Thorough Exploration
```python
# Use more ants and balanced parameters
aco = ACOPathfinder(num_ants=20, alpha=1.5, beta=1.5)
```

### Adaptive to Changes
```python
# High evaporation for dynamic environments
aco = ACOPathfinder(evaporation_rate=0.3)
```

## Running the Demo

See ACO in action with visualizations:

```bash
cd /path/to/Apathion
source .venv/bin/activate
python examples/aco_demo.py
```

## Testing

Run tests to verify your ACO setup:

```bash
# Unit tests
python -m pytest tests/test_aco.py -v

# Integration tests
python -m pytest tests/test_aco_integration.py -v

# All ACO tests
python -m pytest tests/test_aco*.py -v
```

## Understanding the Output

When ACO runs, you'll see:
- **Path length**: Number of steps in the path
- **Path cost**: Total distance (lower is better)
- **Pheromone trails**: Stronger where ants found good paths

### Convergence Example
```
Iteration 1: Path cost = 45.04
Iteration 2: Path cost = 38.21
Iteration 3: Path cost = 24.80  ← Getting better!
```

## Troubleshooting

### Problem: Paths seem random
**Solution**: Increase `beta` (heuristic importance)

### Problem: All paths are the same
**Solution**: Decrease `alpha` or increase `evaporation_rate`

### Problem: Pathfinding is slow
**Solution**: Reduce `num_ants` or increase `beta`

### Problem: Paths don't adapt to changes
**Solution**: Increase `evaporation_rate`

## Next Steps

1. Read the full documentation: `docs/ACO_IMPLEMENTATION.md`
2. Try the interactive demo: `python examples/aco_demo.py`
3. Experiment with parameters in `configs/aco_example.json`
4. Compare ACO with A* using experiments

## API Quick Reference

```python
# Create pathfinder
aco = ACOPathfinder(
    num_ants=10,
    alpha=1.0,
    beta=2.0,
    evaporation_rate=0.1,
    deposit_strength=1.0
)

# Update state
aco.update_state(game_map, towers)

# Find path
path = aco.find_path(start, goal)

# Get pheromone level
pheromone = aco.get_pheromone_at((x, y))

# Get configuration
config = aco.to_dict()
```

## Additional Resources

- **Full Documentation**: `docs/ACO_IMPLEMENTATION.md`
- **Demo Script**: `examples/aco_demo.py`
- **Config Example**: `configs/aco_example.json`
- **Unit Tests**: `tests/test_aco.py`
- **Integration Tests**: `tests/test_aco_integration.py`

## Example Workflow

```bash
# 1. Run the demo to see ACO in action
python examples/aco_demo.py

# 2. Play an interactive game
apathion play --algorithm=aco --map_type=branching

# 3. Create your own config
cp configs/aco_example.json configs/my_aco.json
# Edit my_aco.json with your preferred settings

# 4. Test your configuration
apathion play --config_file=configs/my_aco.json

# 5. Run experiments
apathion evaluate --config_file=configs/my_experiment.json
```

## Getting Help

If you need more information:
- Check `docs/ACO_IMPLEMENTATION.md` for detailed documentation
- Run `python examples/aco_demo.py` to see visual examples
- Look at `tests/test_aco.py` for usage examples
- Review `configs/aco_example.json` for configuration examples

---

**Ready to go!** Start with `apathion play --algorithm=aco` to see ACO in action! 🐜

