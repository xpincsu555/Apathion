#!/bin/bash
# Quick-start script to run all experiments needed for the Pilot Evaluation Report
#
# This script runs a comprehensive set of experiments that will provide
# all the data needed for your milestone deliverables.
#
# All experiments use the branching map from configs/branching_map.json
#
# Usage: bash scripts/run_report_experiments.sh
#
# Estimated total time: 30-45 minutes

set -e  # Exit on error

echo "================================================================"
echo "Apathion - Pilot Evaluation Experiments (Branching Map)"
echo "================================================================"
echo ""
echo "This script will run 4 experiments to generate data for your report:"
echo "  1. Baseline Comparison (5-10 min)"
echo "  2. Full Algorithm Comparison (15-25 min)"
echo "  3. Performance Stress Test (10-15 min)"
echo "  4. Path Diversity Analysis (10-15 min)"
echo ""
echo "All experiments use the branching map (multiple paths)."
echo "Total estimated time: 30-45 minutes"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Change to project root
cd "$(dirname "$0")/.."

echo ""
echo "================================================================"
echo "Experiment 1/4: Baseline Comparison"
echo "================================================================"
echo "Comparing Fixed baseline vs A* to establish improvement"
python scripts/run_experiments.py --config configs/experiments/baseline_comparison.json

echo ""
echo "================================================================"
echo "Experiment 2/4: Full Algorithm Comparison"
echo "================================================================"
echo "Testing Fixed, A*, and ACO across multiple map types"
python scripts/run_experiments.py --config configs/experiments/full_comparison.json

echo ""
echo "================================================================"
echo "Experiment 3/4: Performance Stress Test"
echo "================================================================"
echo "Testing real-time performance with 100 enemies per wave"
python scripts/run_experiments.py --config configs/experiments/performance_stress.json

echo ""
echo "================================================================"
echo "Experiment 4/4: Path Diversity Analysis"
echo "================================================================"
echo "Demonstrating ACO's diverse path generation"
python scripts/run_experiments.py --config configs/experiments/path_diversity_test.json

echo ""
echo "================================================================"
echo "All Experiments Complete!"
echo "================================================================"
echo ""
echo "Your results are saved in: data/results/"
echo ""
echo "Next steps:"
echo "1. Analyze the results to generate charts and tables:"
echo "   apathion analyze --plots --output_dir report_figures"
echo "2. Use the generated data for your Pilot Evaluation Report"
echo "3. Charts and tables will be in: report_figures/"
echo ""
echo "See docs/RUNNING_EXPERIMENTS.md for more details."

