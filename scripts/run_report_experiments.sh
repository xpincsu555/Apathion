#!/bin/bash
# Refactored script to run comprehensive pathfinding algorithm experiments
#
# This script evaluates four pathfinding algorithms (Fixed, A*, ACO, DQN) on:
# - Survival rate: percentage of enemies reaching the goal
# - Path diversity: Shannon entropy of route distribution
# - Adaptation speed: frames to converge after tower placement
# - Computational cost: average milliseconds per pathfinding decision
#
# Maps tested: branching_map.json and open_arena.json
#
# Usage: bash scripts/run_report_experiments.sh
#
# Estimated total time: 45-60 minutes

set -e  # Exit on error

echo "================================================================"
echo "Apathion - Comprehensive Algorithm Evaluation"
echo "================================================================"
echo ""
echo "This script will run 3 experiments comparing 4 algorithms:"
echo "  - Fixed Path (baseline)"
echo "  - A* (heuristic search)"
echo "  - ACO (swarm intelligence)"
echo "  - DQN (reinforcement learning)"
echo ""
echo "Metrics evaluated:"
echo "  1. Survival rate: % of enemies reaching goal"
echo "  2. Path diversity: Shannon entropy of routes"
echo "  3. Adaptation speed: frames to adapt after tower placement"
echo "  4. Computational cost: average ms per decision"
echo ""
echo "Experiments:"
echo "  1. Full Algorithm Comparison (All 4) - 20-25 min"
echo "  2. Path Diversity Test - 15-20 min"
echo "  3. Performance Stress Test (100 enemies/wave) - 15-20 min"
echo ""
echo "Maps: branching_map and open_arena"
echo "Total estimated time: 50-60 minutes"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Change to project root
cd "$(dirname "$0")/.."

echo ""
echo "================================================================"
echo "Experiment 1/3: Full Algorithm Comparison"
echo "================================================================"
echo "Testing all algorithms: Fixed, A*, ACO, and DQN"
python scripts/run_experiments.py --config configs/experiments/full_comparison.json

echo ""
echo "================================================================"
echo "Experiment 2/3: Path Diversity Analysis"
echo "================================================================"
echo "Measuring path diversity across algorithms"
python scripts/run_experiments.py --config configs/experiments/path_diversity_test.json

echo ""
echo "================================================================"
echo "Experiment 3/3: Performance Stress Test"
echo "================================================================"
echo "Testing computational cost with 100 enemies per wave"
python scripts/run_experiments.py --config configs/experiments/performance_stress.json

echo ""
echo "================================================================"
echo "All Experiments Complete!"
echo "================================================================"
echo ""
echo "Your results are saved in: data/results/"
echo ""
echo "Result files:"
echo "  - experiment_metrics_*.csv: Detailed metrics for each run"
echo "  - experiment_summary_*.csv: Aggregated statistics by algorithm"
echo "  - experiment_session_*.json: Full experiment data"
echo ""
echo "Metrics evaluated:"
echo "  - survival_rate_percent: % of enemies that reached the goal"
echo "  - path_diversity_entropy: Shannon entropy of path choices"
echo "  - adaptation_frames: Frames to adapt after tower placement"
echo "  - computational_cost_ms: Average decision time in milliseconds"
echo ""
echo "Next steps:"
echo "1. Analyze the CSV files to generate charts and tables"
echo "2. Use the metrics for your evaluation report"
echo "3. Compare algorithms across different metrics and maps"
echo ""
echo "See docs/RUNNING_EXPERIMENTS.md for more details."

