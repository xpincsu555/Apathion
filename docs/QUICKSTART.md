# Apathion Quick Start Guide

This guide will help you get started with the Apathion framework in minutes.

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the project directory
cd Apathion

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Your First Experiment

### 1. Run a Simple Game

```bash
# Run a basic game with A* pathfinding
apathion play

# Output will show:
# - Map created
# - Towers placed
# - Waves spawned and completed
# - Final statistics
```

### 2. Try Different Algorithms

```bash
# Test ACO algorithm
apathion play --algorithm=aco --waves=5 --enemies=20

# Test on a branching map
apathion play --algorithm=astar --map_type=branching --waves=10
```

### 3. Run Comparative Evaluation

```bash
# Compare A* and ACO on simple maps
apathion evaluate --algorithms=astar,aco --maps=simple --waves=5 --runs=2

# This will:
# - Run both algorithms on the specified map
# - Repeat each configuration 2 times
# - Generate performance metrics
# - Export results to data/results/
```

### 4. Use a Predefined Experiment

```bash
# Run the baseline comparison preset
apathion evaluate --preset=baseline

# Available presets:
# - baseline: Quick comparison
# - full_comparison: Comprehensive test
# - performance_test: Scaling test
```

### 5. Use Configuration Files

```bash
# Run with a custom config file
apathion play --config_file=configs/example_game.json

# Run evaluation with experiment config
apathion evaluate --config_file=configs/example_experiment.json
```

## Understanding the Output

### Game Statistics

After running `apathion play`, you'll see:

```
Game Statistics:
  Total enemies spawned: 50
  Enemies defeated: 30
  Enemies escaped: 20
  Survival rate: 40.0%
```

### Evaluation Results

After running `apathion evaluate`, you'll see a detailed report:

```
Algorithm: A*-Enhanced
  Map: simple
    Survival Rate: 65.2%
    Path Diversity: 2.35
    Avg Computation: 0.85 ms
```

### Log Files

Results are automatically saved:

```
data/
├── logs/
│   ├── game_state_20240101_120000.csv
│   ├── decisions_20240101_120000.csv
│   └── metrics_20240101_120000.csv
└── results/
    └── session_20240101_120000.json
```

## Next Steps

### Customize Your Experiment

1. **Edit configuration files** in `configs/` directory
2. **Modify algorithm parameters** (alpha, beta, etc.)
3. **Create custom tower placements**
4. **Design new map layouts**

### Implement Full Algorithms

The current implementation provides placeholders. To implement full algorithms:

1. **A* Search**: Complete `src/apathion/pathfinding/astar.py`
   - Implement priority queue-based search
   - Add composite cost function
   - Implement replanning logic

2. **ACO**: Complete `src/apathion/pathfinding/aco.py`
   - Implement ant simulation
   - Add probabilistic path selection
   - Implement pheromone updates

3. **DQN**: Complete `src/apathion/pathfinding/dqn.py`
   - Implement neural network architecture
   - Add training pipeline
   - Implement experience replay

### Add Visualization

To visualize the game in real-time:

1. Integrate Pygame rendering
2. Display enemies, towers, and paths
3. Show pheromone trails (for ACO)
4. Add debug overlays

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Reinstall dependencies
uv sync

# Or if using pip
pip install -e .
```

### No Module Named 'apathion'

Make sure you're in the activated virtual environment:

```bash
source .venv/bin/activate
```

### Permission Errors for Log Files

Ensure the data directories exist:

```bash
mkdir -p data/logs data/results
```

## Example Workflow

Here's a complete example workflow for running experiments:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Test individual algorithms
apathion play --algorithm=astar --waves=5
apathion play --algorithm=aco --waves=5

# 3. Run comparative evaluation
apathion evaluate --algorithms=astar,aco --maps=simple,branching --runs=3

# 4. Check results
ls -lh data/results/

# 5. Analyze logs (placeholder - to be implemented)
apathion analyze --log_dir=data/logs
```

## Tips

1. **Start small**: Use fewer waves and enemies while testing
2. **Use presets**: Start with `--preset=baseline` for quick tests
3. **Check logs**: Review CSV files to understand algorithm behavior
4. **Tune parameters**: Adjust alpha/beta values in config files
5. **Compare systematically**: Run multiple repetitions for reliable results

## Getting Help

For detailed information about any command:

```bash
apathion <command> -- --help
```

Examples:

```bash
apathion play -- --help
apathion evaluate -- --help
```

## Additional Resources

- See `README.md` for full documentation
- Check `requirements.md` for project specifications
- Review `configs/` for example configurations
- Explore `src/apathion/` for implementation details

