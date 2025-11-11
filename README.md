# Apathion

**Adaptive Pathfinding Enemies in Tower Defense Games**

A research framework for evaluating adaptive pathfinding algorithms in tower defense games. Implements Fixed baseline, Enhanced A* with damage-aware costs, and Ant Colony Optimization with swarm intelligence.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/xpincsu555/Apathion.git
cd Apathion

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Run Interactive Game

```bash
# A* algorithm (default)
apathion play --algorithm astar --map_type branching --waves 5

# ACO with swarm intelligence
apathion play --algorithm aco --map_type branching --waves 5

# Fixed baseline
apathion play --algorithm fixed --map_type branching --waves 5
```

**Controls:** Click to place towers, T to change tower type, Space to pause, Tab to toggle view, ESC to quit.

## Algorithms

**Fixed-Path Baseline** - Traditional tower defense pathfinding following a predetermined route.

**Enhanced A*** - Extends classical A* with composite cost function balancing distance, damage exposure, and congestion: $f(n) = g(n) + h(n) + \alpha \cdot damage(n) + \beta \cdot congestion(n)$

**Ant Colony Optimization (ACO)** - Swarm intelligence using pheromone trails for diverse, adaptive pathfinding with probabilistic path selection.

## Run Experiments

### Batch Experiments

```bash
# Run all experiments (~30-45 minutes)
bash scripts/run_report_experiments.sh

# Or run individual experiments
python scripts/run_experiments.py --config configs/experiments/baseline_comparison.json
python scripts/run_experiments.py --config configs/experiments/full_comparison.json
```

### Analyze Results

```bash
# Auto-detects latest results and generates charts
apathion analyze --plots --output_dir report_figures

# Or specify a session file
apathion analyze --session_file data/results/experiment_session_20251111_120000.json --plots
```

Generates:
- `survival_rates.png` - Survival rate comparison
- `decision_times.png` - Computation time comparison
- `summary_table.csv` - Statistics table

## Results

Our pilot evaluation shows:

| Algorithm | Survival Rate | Path Diversity | Avg Decision Time |
|-----------|--------------|----------------|-------------------|
| Fixed-Path | 88.00% | 0.000 | 0.003 ms |
| A*-Enhanced | **100.00%** | 0.000 | 0.520 ms |
| ACO-Swarm | 69.48% | --- | 49.025 ms |

**Key Findings:**
- A* achieves perfect survival (100%) with real-time performance (<5ms)
- ACO requires parameter tuning to improve survival rate
- All algorithms tested on branching map with multiple path choices

## Project Structure

```
src/apathion/
├── pathfinding/          # Algorithm implementations
│   ├── fixed.py          # Fixed baseline
│   ├── astar.py          # Enhanced A*
│   └── aco.py            # Ant Colony Optimization
├── game/                 # Game engine (Pygame)
│   ├── map.py            # Grid-based map
│   ├── enemy.py          # Enemy entities
│   ├── tower.py          # Tower entities
│   └── game_loop.py      # Game loop & rendering
├── evaluation/           # Experiment framework
│   ├── headless_simulator.py  # Batch experiments
│   ├── metrics.py        # Performance metrics
│   └── logger.py         # Data logging
└── cli.py                # Command-line interface

configs/experiments/      # Experiment configurations
data/results/            # Experiment results (CSV/JSON)
```

## Documentation

- **Pilot Evaluation Report:** `pilot_evaluation_report.tex` - Research findings and analysis
- **Running Experiments:** `docs/RUNNING_EXPERIMENTS.md` - Detailed experiment guide
- **Implementation Docs:**
  - `docs/ASTAR_IMPLEMENTATION.md` - Enhanced A* details
  - `docs/ACO_IMPLEMENTATION.md` - ACO algorithm details
  - `docs/CLI_REFERENCE.md` - Command-line usage
  - `docs/GAMEPLAY_GUIDE.md` - Interactive play guide

## Team

**Xiaoqin Pi** - ACO implementation, parameter optimization  
**Weiyuan Ding** - A* implementation, evaluation framework

## License

MIT License - see LICENSE file for details.
