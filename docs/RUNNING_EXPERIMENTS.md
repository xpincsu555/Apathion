# Running Experiments Guide

This guide explains how to run batch experiments for the Apathion project to generate data for your evaluation report.

**Note:** All experiments use the **branching map** configuration from `configs/branching_map.json`. Simple single-path maps are excluded as they don't demonstrate the benefits of adaptive pathfinding algorithms.

## Quick Start

### 1. Run a Preset Experiment

The fastest way to get started is to use one of the preset experiment configurations:

```bash
# From the project root directory
python scripts/run_experiments.py --config configs/experiments/baseline_comparison.json
```

### 2. Available Experiment Presets

We provide several pre-configured experiments designed for different analysis goals:

#### Baseline Comparison
**Purpose:** Establish baseline performance and show improvement over fixed paths  
**File:** `configs/experiments/baseline_comparison.json`  
**Algorithms:** Fixed, A*  
**Map:** Branching  
**Duration:** ~5-10 minutes  

```bash
python scripts/run_experiments.py --config configs/experiments/baseline_comparison.json
```

#### Full Algorithm Comparison
**Purpose:** Main comparison for the report across all algorithms  
**File:** `configs/experiments/full_comparison.json`  
**Algorithms:** Fixed, A*, ACO  
**Map:** Branching  
**Duration:** ~15-25 minutes  

```bash
python scripts/run_experiments.py --config configs/experiments/full_comparison.json
```

#### Performance Stress Test
**Purpose:** Test real-time performance requirements with high enemy counts  
**File:** `configs/experiments/performance_stress.json`  
**Algorithms:** Fixed, A*, ACO  
**Duration:** ~10-15 minutes  

```bash
python scripts/run_experiments.py --config configs/experiments/performance_stress.json
```

#### Path Diversity Analysis
**Purpose:** Demonstrate ACO's diverse path generation vs A*'s deterministic paths  
**File:** `configs/experiments/path_diversity_test.json`  
**Algorithms:** A*, ACO  
**Duration:** ~10-15 minutes  

```bash
python scripts/run_experiments.py --config configs/experiments/path_diversity_test.json
```

### 3. Custom Experiments

You can also run experiments with custom parameters:

```bash
# Compare all algorithms on simple and branching maps
python scripts/run_experiments.py \
  --algorithms fixed,astar,aco \
  --maps simple,branching \
  --runs 5 \
  --waves 10 \
  --enemies 30 \
  --output data/results/my_experiment
```

## Understanding the Results

After running an experiment, you'll find the following files in `data/results/`:

### 1. `experiment_session_YYYYMMDD_HHMMSS.json`
Complete experiment data including:
- Full configuration for each run
- Wave-by-wave results
- Summary statistics per algorithm

### 2. `experiment_metrics_YYYYMMDD_HHMMSS.csv`
CSV file with key metrics for each experiment run:
- Algorithm name
- Map type
- Survival rate (%)
- Average decision time (ms)
- Max decision time (ms)
- Simulation time

**Use this for:** Creating charts and tables in your report

### 3. `experiment_decisions_YYYYMMDD_HHMMSS.csv`
Wave-level decision data:
- Per-wave pathfinding times
- Useful for analyzing performance over time

## Analyzing Results

### Generate Summary Statistics

```bash
# Analyze the most recent session (auto-detects)
apathion analyze --plots

# Or specify a specific session file
apathion analyze \
  --session_file data/results/experiment_session_YYYYMMDD_HHMMSS.json \
  --plots \
  --output_dir report_figures
```

This will output:
- Average survival rates per algorithm
- Average computation times
- Success criteria evaluation (≥25% improvement, <5ms per decision)
- Visualization plots (if --plots flag used)

Generates:
- `report_figures/survival_rates.png` - Survival rate comparison bar chart
- `report_figures/decision_times.png` - Computation time comparison
- `report_figures/summary_table.csv` - Formatted statistics table

## Recommended Experiments for Report

For your Pilot Evaluation Report, we recommend running these experiments in order:

### 1. Baseline Comparison (Required)
Shows that adaptive pathfinding improves over fixed paths.

```bash
python scripts/run_experiments.py --config configs/experiments/baseline_comparison.json
```

**Expected Results:**
- A* should achieve >25% higher survival rate than Fixed
- Decision time should be <5ms for real-time performance

### 2. Full Comparison (Required)
Main results showing all three algorithms across map types.

```bash
python scripts/run_experiments.py --config configs/experiments/full_comparison.json
```

**Expected Results:**
- ACO should show diverse paths (multiple route choices)
- A* should be fastest but deterministic
- All should outperform fixed baseline

### 3. Performance Stress Test (Optional but Recommended)
Demonstrates scalability to 50+ enemies.

```bash
python scripts/run_experiments.py --config configs/experiments/performance_stress.json
```

**Expected Results:**
- All algorithms maintain <5ms average decision time
- 60 FPS target is maintained with 50+ enemies

### 4. Path Diversity Analysis (For Discussion)
Shows ACO's strategic depth through path diversity.

```bash
python scripts/run_experiments.py --config configs/experiments/path_diversity_test.json
```

**Expected Results:**
- ACO achieves diversity index ≥2.0
- A* shows index ~1.0 (deterministic single path)

## Tips for Running Experiments

1. **Run experiments in order** - Start with baseline, then build up to full comparison
2. **Check disk space** - Experiments generate CSV/JSON files (~5-10 MB per session)
3. **Monitor progress** - The script shows real-time progress for each experiment
4. **Multiple runs** - Use at least 5 runs per configuration for statistical significance
5. **Save session IDs** - Note the session ID (timestamp) for each experiment to analyze results later

## Troubleshooting

### "No module named 'apathion'"
Make sure you're running from the project root and have installed the package:
```bash
pip install -e .
# or with uv:
uv pip install -e .
```

### "Map configuration not found"
Ensure `configs/branching_map.json` exists. All experiments use this branching map configuration which includes:
- Map layout with multiple paths
- Obstacle regions
- Baseline path for Fixed algorithm
- Tower placements

### Experiments take too long
Reduce the number of runs or waves:
```bash
python scripts/run_experiments.py \
  --algorithms fixed,astar \
  --maps simple \
  --runs 3 \
  --waves 5 \
  --enemies 20
```

## What to Include in Your Report

From these experiments, you should be able to generate:

1. **Algorithm Implementation section:**
   - Show the three algorithms are fully implemented
   - Reference the actual code files

2. **Pilot Metrics section:**
   - Tables showing survival rates, computation times
   - Charts comparing algorithms across maps
   - Path diversity metrics

3. **Success Cases:**
   - A* and ACO outperform fixed baseline
   - Real-time performance achieved (<5ms)
   - Path diversity demonstrated

4. **Failure Cases:**
   - ACO replanning delay when towers placed (trade-off discussion)
   - Any scenarios where adaptive algorithms didn't improve
   - Performance bottlenecks with very high enemy counts

5. **Next Steps:**
   - Optimizing ACO replanning
   - Implementing congestion avoidance
   - Adding DQN implementation

