"""
Apathion - Adaptive Pathfinding Enemies in Tower Defense Games

An experimental framework for evaluating adaptive pathfinding algorithms
in tower defense games, including A*, ACO, and DQN approaches.
"""

__version__ = "0.1.0"
__authors__ = ["Xiaoqin Pi", "Weiyuan Ding"]

from apathion.game.game import GameState
from apathion.game.map import Map
from apathion.game.enemy import Enemy
from apathion.game.tower import Tower

__all__ = [
    "GameState",
    "Map",
    "Enemy",
    "Tower",
]

