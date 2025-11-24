"""
Reward profile configurations for DQN training.

Different profiles optimize for different objectives:
- SPEED: Fastest path to goal (current default)
- SURVIVAL: Highest survival rate, avoid damage
- BALANCED: Mix of both
"""

from enum import Enum
from typing import Dict, Any


class RewardProfile(Enum):
    """Reward profile types."""
    SPEED = "speed"
    SURVIVAL = "survival"
    BALANCED = "balanced"


def get_reward_weights(profile: RewardProfile) -> Dict[str, Any]:
    """
    Get reward weights for a given profile.
    
    Args:
        profile: Reward profile type
        
    Returns:
        Dictionary of reward weights
    """
    if profile == RewardProfile.SPEED:
        return {
            "name": "Speed-Optimized",
            "description": "Prioritizes reaching goal quickly",
            "goal_reward": 1000.0,
            "health_bonus_multiplier": 20.0,  # Small bonus for health
            "death_penalty": 2.0,              # Small penalty for death
            "step_penalty": 0.001,             # Encourages shorter paths
            "damage_penalty_multiplier": 0.05, # Small damage penalty
            "progress_reward_multiplier": 0.1,
            "spawn_distance_bonus_multiplier": 0.01,
        }
    
    elif profile == RewardProfile.SURVIVAL:
        return {
            "name": "Survival-Optimized",
            "description": "Prioritizes staying alive and minimizing damage",
            "goal_reward": 1000.0,
            "health_bonus_multiplier": 500.0,   # HUGE bonus for high health
            "death_penalty": 100.0,             # HUGE penalty for death
            "step_penalty": 0.0001,             # Much smaller (length doesn't matter much)
            "damage_penalty_multiplier": 20.0,   # Large damage penalty
            "progress_reward_multiplier": 0.5,  # Increased from 0.02 - stronger goal-seeking incentive
            "spawn_distance_bonus_multiplier": 0.005,
        }
    
    elif profile == RewardProfile.BALANCED:
        return {
            "name": "Balanced",
            "description": "Balance between speed and survival",
            "goal_reward": 1000.0,
            "health_bonus_multiplier": 100.0,   # Significant health bonus
            "death_penalty": 20.0,              # Significant death penalty
            "step_penalty": 0.0005,             # Moderate step penalty
            "damage_penalty_multiplier": 0.2,   # Moderate damage penalty
            "progress_reward_multiplier": 0.08,
            "spawn_distance_bonus_multiplier": 0.008,
        }
    
    else:
        raise ValueError(f"Unknown profile: {profile}")


def calculate_reward_comparison():
    """Calculate reward for different scenarios across profiles."""
    scenarios = {
        "Fast path (30 steps, 20% health)": {
            "steps": 30,
            "health_ratio": 0.2,
            "damage_per_step": 3.0,
            "died": False,
        },
        "Safe path (50 steps, 100% health)": {
            "steps": 50,
            "health_ratio": 1.0,
            "damage_per_step": 0.5,
            "died": False,
        },
        "Risky path (25 steps, DIED)": {
            "steps": 25,
            "health_ratio": 0.0,
            "damage_per_step": 5.0,
            "died": True,
        },
    }
    
    print("=" * 80)
    print("REWARD PROFILE COMPARISON")
    print("=" * 80)
    
    for profile in [RewardProfile.SPEED, RewardProfile.BALANCED, RewardProfile.SURVIVAL]:
        weights = get_reward_weights(profile)
        print(f"\n{profile.value.upper()} Profile: {weights['name']}")
        print("-" * 80)
        
        for scenario_name, scenario in scenarios.items():
            reward = 0.0
            
            if scenario["died"]:
                reward -= weights["death_penalty"]
            else:
                reward += weights["goal_reward"]
                reward += scenario["health_ratio"] * weights["health_bonus_multiplier"]
            
            reward -= scenario["steps"] * weights["step_penalty"]
            
            # Approximate damage penalty
            total_damage = scenario["damage_per_step"] * scenario["steps"] * 0.1
            reward -= (total_damage / 100.0) * weights["damage_penalty_multiplier"]
            
            print(f"  {scenario_name:40s}: {reward:8.2f}")
        
        # Calculate which is best
        fast_r = 1000 + 0.2*weights["health_bonus_multiplier"] - 30*weights["step_penalty"] - (90/100)*weights["damage_penalty_multiplier"]
        safe_r = 1000 + 1.0*weights["health_bonus_multiplier"] - 50*weights["step_penalty"] - (25/100)*weights["damage_penalty_multiplier"]
        
        diff = safe_r - fast_r
        pct = (diff / fast_r) * 100
        
        print(f"\n  Safe path advantage: {diff:+.2f} points ({pct:+.1f}%)")
        if pct < 5:
            print(f"  ⚠️  Difference too small - agent may prefer speed")
        else:
            print(f"  ✓ Clear incentive for survival")


if __name__ == "__main__":
    calculate_reward_comparison()

