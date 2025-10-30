# Apathion
Adaptive Pathfinding Enemies in Tower Defense Games

An experimental framework for evaluating adaptive pathfinding algorithms in tower defense games, including A*, ACO, and DQN approaches.

## Overview

This project implements and compares three pathfinding approaches for tower defense games:

1. **Enhanced A* Search** - Extends standard A* with composite cost functions
2. **Ant Colony Optimization (ACO)** - Enables swarm intelligence through pheromone trails
3. **Deep Q-Network (DQN)** - Reinforcement learning approach for learned policies

## Project Structure

```
Apathion/
├── pyproject.toml          # uv project configuration
├── .python-version          # Python version (3.11)
├── src/
│   └── apathion/
│       ├── cli.py          # Fire-based CLI entry point
│       ├── config.py       # Configuration management
│       ├── game/           # Core game entities
│       │   ├── map.py      # Grid-based map
│       │   ├── enemy.py    # Enemy entities
│       │   ├── tower.py    # Tower entities
│       │   └── game.py     # Game state management
│       ├── pathfinding/    # Pathfinding algorithms
│       │   ├── base.py     # Abstract pathfinder interface
│       │   ├── astar.py    # A* implementation
│       │   ├── aco.py      # ACO implementation
│       │   └── dqn.py      # DQN implementation
│       └── evaluation/     # Evaluation infrastructure
│           ├── metrics.py  # Performance metrics
│           ├── logger.py   # Data logging
│           └── evaluator.py # Comparison framework
├── tests/                  # Test suite
└── data/                   # Logs and results
    ├── logs/
    └── results/
```

## Installation

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Setup with uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd Apathion

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Setup with pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

## Usage

Apathion provides a command-line interface using Fire. All commands are accessible through the `apathion` command.

### Basic Commands

#### Play Mode
Run an interactive game session:

```bash
# Basic usage
apathion play

# With custom parameters
apathion play --algorithm=astar --map_type=branching --waves=10 --enemies=30

# Using a config file
apathion play --config_file=configs/my_config.json
```

**Options:**
- `--algorithm`: Pathfinding algorithm (astar, aco, dqn) [default: astar]
- `--map_type`: Map type (simple, branching, open_arena) [default: simple]
- `--waves`: Number of waves [default: 5]
- `--enemies`: Enemies per wave [default: 10]
- `--config_file`: Path to JSON config file

#### Evaluate Mode
Run comparative experiments:

```bash
# Compare algorithms on different maps
apathion evaluate --algorithms=astar,aco --maps=simple,branching --waves=10

# Use a predefined experiment preset
apathion evaluate --preset=full_comparison

# Custom evaluation with multiple runs
apathion evaluate --algorithms=astar,aco,dqn --maps=simple --runs=5 --output=results/exp1
```

**Options:**
- `--algorithms`: Comma-separated list of algorithms
- `--maps`: Comma-separated list of map types
- `--waves`: Waves per experiment [default: 5]
- `--enemies`: Enemies per wave [default: 30]
- `--runs`: Number of repetitions [default: 3]
- `--preset`: Use predefined preset (baseline, full_comparison, performance_test)
- `--output`: Output directory [default: data/results]

**Available Presets:**
- `baseline`: Quick comparison on simple maps
- `full_comparison`: Comprehensive test across all map types
- `performance_test`: Scaling test with high enemy counts

#### Train Mode (Placeholder)
Train a DQN model:

```bash
apathion train --episodes=5000 --map_type=branching --save_path=models/my_model.pth
```

**Note:** DQN training is currently a placeholder and requires implementation of the training pipeline.

#### Analyze Mode (Placeholder)
Analyze logged results:

```bash
apathion analyze --log_dir=data/logs --output=analysis_report.txt
```

## Configuration

### Configuration Files

Configuration can be managed through JSON files. Example:

```json
{
  "algorithm": "astar",
  "target_fps": 60,
  "map": {
    "width": 30,
    "height": 20,
    "map_type": "branching"
  },
  "enemies": {
    "enemies_per_wave": 50,
    "wave_count": 10,
    "enemy_types": ["normal", "fast"]
  },
  "towers": {
    "initial_towers": 3,
    "tower_types": ["basic", "sniper"]
  }
}
```

### Algorithm-Specific Configuration

Each algorithm has specific parameters that can be tuned:

**A* Parameters:**
- `alpha`: Weight for damage cost [default: 0.5]
- `beta`: Weight for congestion cost [default: 0.3]
- `diagonal_movement`: Allow diagonal moves [default: true]

**ACO Parameters:**
- `num_ants`: Ants per iteration [default: 10]
- `evaporation_rate`: Pheromone evaporation [default: 0.1]
- `alpha`: Pheromone importance [default: 1.0]
- `beta`: Heuristic importance [default: 2.0]

**DQN Parameters:**
- `state_size`: State vector size [default: 64]
- `action_size`: Number of actions [default: 8]
- `use_cache`: Enable decision caching [default: true]
- `cache_duration`: Frames to cache [default: 5]

## Development Status

### Current Implementation Status

✅ **Completed:**
- Project structure and dependency management
- Core game entities (Map, Enemy, Tower, GameState)
- Abstract pathfinder interface
- Algorithm placeholders (A*, ACO, DQN)
- Evaluation metrics framework
- Data logging infrastructure
- CLI with Fire
- Configuration management system

⚠️ **Placeholders (To Be Implemented):**
- Full A* pathfinding algorithm
- Complete ACO implementation
- DQN neural network and training pipeline
- Actual game simulation loop
- Pygame visualization
- Advanced metrics (adaptation speed, strategic depth)

### Next Steps

1. **Implement full A* algorithm** with priority queue and composite cost functions
2. **Implement ACO** with pheromone updates and probabilistic path selection
3. **Add game simulation loop** for realistic evaluation
4. **Implement DQN training** using PyTorch or stable-baselines3
5. **Add visualization** using Pygame for debugging and demonstrations
6. **Extend metrics** with sophisticated analysis tools

## Evaluation Metrics

The framework collects and analyzes the following metrics:

- **Survival Rate:** Percentage of enemies reaching the goal
- **Damage Efficiency:** Average damage taken per surviving unit
- **Path Diversity:** Shannon entropy of route distribution
- **Adaptation Speed:** Time to converge on new optimal paths
- **Computational Cost:** Average ms per pathfinding decision
- **Strategic Depth:** Number of tower configurations affecting behavior

## Data Output

All experiments generate structured logs in CSV and JSON formats:

1. **Game State Logs** - Frame-by-frame game state
2. **Pathfinding Decision Logs** - Algorithm decisions with timing
3. **Performance Metrics** - Wave-level performance statistics

Logs are saved to `data/logs/` by default and can be analyzed using the `analyze` command.

## Team

- **Xiaoqin Pi**
- **Weiyuan Ding**

## License

MIT License - see LICENSE file for details.

## References

Based on the technical requirements and evaluation plan in `requirements.md`.
