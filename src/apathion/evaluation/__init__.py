"""
Evaluation module for performance metrics, logging, and comparative analysis.
"""

from apathion.evaluation.metrics import (
    survival_rate,
    damage_efficiency,
    path_diversity,
    adaptation_speed,
    computational_cost,
)
from apathion.evaluation.logger import GameLogger
from apathion.evaluation.evaluator import Evaluator

__all__ = [
    "survival_rate",
    "damage_efficiency",
    "path_diversity",
    "adaptation_speed",
    "computational_cost",
    "GameLogger",
    "Evaluator",
]

