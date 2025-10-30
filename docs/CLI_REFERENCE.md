# Apathion CLI Quick Reference

## Installation

```bash
uv sync && source .venv/bin/activate
```

---

## Commands

### `apathion play`
Run an interactive game session

**Basic Usage:**
```bash
apathion play
```

**Common Examples:**
```bash
# Test A* on branching map
apathion play --algorithm=astar --map_type=branching --waves=10 --enemies=30

# Test ACO on open arena
apathion play --algorithm=aco --map_type=open_arena --waves=5 --enemies=50

# Use config file
apathion play --config_file=configs/example_game.json
```

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--algorithm` | str | astar | Algorithm: astar, aco, dqn, fixed |
| `--map_type` | str | simple | Map: simple, branching, open_arena |
| `--waves` | int | 5 | Number of waves |
| `--enemies` | int | 10 | Enemies per wave |
| `--config_file` | str | None | Path to JSON config |

---

### `apathion evaluate`
Run comparative evaluation experiments

**Basic Usage:**
```bash
apathion evaluate
```

**Common Examples:**
```bash
# Compare A* and ACO
apathion evaluate --algorithms=astar,aco --maps=simple,branching --waves=10 --runs=3

# Use baseline preset
apathion evaluate --preset=baseline

# Full comparison with custom output
apathion evaluate --preset=full_comparison --output=results/exp1

# Custom evaluation
apathion evaluate --algorithms=astar --maps=simple --waves=20 --enemies=100 --runs=5
```

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--algorithms` | list[str] | [astar,aco] | Comma-separated algorithms |
| `--maps` | list[str] | [simple,branching] | Comma-separated map types |
| `--waves` | int | 5 | Waves per experiment |
| `--enemies` | int | 30 | Enemies per wave |
| `--runs` | int | 3 | Repetitions per config |
| `--preset` | str | None | Preset: baseline, full_comparison, performance_test |
| `--config_file` | str | None | Path to experiment config JSON |
| `--output` | str | data/results | Output directory |

**Presets:**
- `baseline`: Quick comparison (3 runs, simple map)
- `full_comparison`: Comprehensive test (5 runs, all maps)
- `performance_test`: Scaling test (100 enemies)

---

### `apathion train`
Train DQN model *(placeholder)*

**Basic Usage:**
```bash
apathion train --episodes=5000
```

**Common Examples:**
```bash
# Train on branching map
apathion train --episodes=5000 --map_type=branching --save_path=models/branching_dqn.pth

# Train with config
apathion train --config_file=configs/dqn_training.json
```

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--episodes` | int | 1000 | Training episodes |
| `--map_type` | str | simple | Training map type |
| `--save_path` | str | models/dqn_model.pth | Model save path |
| `--config_file` | str | None | Training config JSON |

---

### `apathion analyze`
Analyze logged results *(placeholder)*

**Basic Usage:**
```bash
apathion analyze --log_dir=data/logs
```

**Common Examples:**
```bash
# Analyze with report output
apathion analyze --log_dir=data/logs --output=analysis_report.txt

# Analyze specific session
apathion analyze --log_dir=data/logs/session_20240101
```

**Options:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--log_dir` | str | data/logs | Log directory path |
| `--output` | str | None | Output report file |

---

## Configuration Files

### Game Configuration (`configs/example_game.json`)

```json
{
  "algorithm": "astar",
  "target_fps": 60,
  "map": {
    "width": 30,
    "height": 20,
    "map_type": "simple"
  },
  "enemies": {
    "enemies_per_wave": 20,
    "wave_count": 5
  },
  "towers": {
    "initial_towers": 3
  }
}
```

### Experiment Configuration (`configs/example_experiment.json`)

```json
{
  "name": "my_experiment",
  "algorithms": ["astar", "aco"],
  "map_types": ["simple", "branching"],
  "num_runs": 3,
  "waves_per_run": 10,
  "enemies_per_wave": 30
}
```

---

## Common Workflows

### Quick Test
```bash
# Test A*
apathion play --algorithm=astar --waves=3

# Test ACO  
apathion play --algorithm=aco --waves=3

# Compare both
apathion evaluate --algorithms=astar,aco --maps=simple --runs=2
```

### Baseline Comparison
```bash
# Run baseline preset
apathion evaluate --preset=baseline

# Check results
ls -lh data/results/
```

### Full Evaluation
```bash
# Full comparison across all maps
apathion evaluate --preset=full_comparison

# Or custom comprehensive test
apathion evaluate \
  --algorithms=astar,aco,dqn \
  --maps=simple,branching,open_arena \
  --waves=20 \
  --enemies=50 \
  --runs=5 \
  --output=results/comprehensive_test
```

### Performance Testing
```bash
# Test with many enemies
apathion evaluate \
  --algorithms=astar,aco \
  --maps=simple \
  --waves=10 \
  --enemies=100 \
  --runs=3
```

---

## Output Files

### Log Files (`data/logs/`)
- `game_state_TIMESTAMP.csv` - Frame-by-frame game state
- `decisions_TIMESTAMP.csv` - Pathfinding decisions
- `metrics_TIMESTAMP.csv` - Performance metrics
- `session_TIMESTAMP.json` - Full session data

### Result Files (`data/results/`)
- Experiment-specific CSV and JSON files
- Generated reports

---

## Help

Get detailed help for any command:

```bash
apathion <command> -- --help
```

Examples:
```bash
apathion play -- --help
apathion evaluate -- --help
apathion train -- --help
apathion analyze -- --help
```

---

## Tips

1. **Start Small**: Use `--waves=3 --enemies=10` while testing
2. **Use Presets**: Start with `--preset=baseline`
3. **Check Logs**: Review CSV files to understand behavior
4. **Multiple Runs**: Use `--runs=5` for statistical reliability
5. **Custom Configs**: Create JSON configs for complex experiments

---

## Troubleshooting

### Command Not Found
```bash
# Activate environment
source .venv/bin/activate

# Or reinstall
uv sync
```

### Import Errors
```bash
# Reinstall in editable mode
pip install -e .
```

### Permission Errors
```bash
# Ensure directories exist
mkdir -p data/logs data/results models
```

---

## Algorithm Parameters

### A* (`astar`)
- `alpha`: Damage cost weight (default: 0.5)
- `beta`: Congestion cost weight (default: 0.3)
- `diagonal_movement`: Allow diagonals (default: true)

### ACO (`aco`)
- `num_ants`: Ants per iteration (default: 10)
- `evaporation_rate`: Pheromone decay (default: 0.1)
- `alpha`: Pheromone importance (default: 1.0)
- `beta`: Heuristic importance (default: 2.0)

### DQN (`dqn`)
- `state_size`: State vector size (default: 64)
- `action_size`: Number of actions (default: 8)
- `use_cache`: Enable caching (default: true)
- `cache_duration`: Cache frames (default: 5)

---

## Map Types

- `simple`: Open map with minimal obstacles
- `branching`: Map with 2-3 route choices
- `open_arena`: Maximum routing freedom
- `dynamic_maze`: Walls appear/disappear *(to be implemented)*

---

## Quick Reference Card

```bash
# Installation
uv sync && source .venv/bin/activate

# Quick test
apathion play --algorithm=astar --waves=3

# Quick evaluation
apathion evaluate --preset=baseline

# Custom evaluation
apathion evaluate --algorithms=astar,aco --maps=simple,branching --runs=3

# Check results
ls data/results/

# Get help
apathion <command> -- --help
```

