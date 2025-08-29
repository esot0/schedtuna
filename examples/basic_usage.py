#!/usr/bin/env python3
"""
Basic usage example for RL SCX Params
====================================

This example shows the simplest way to use the RL optimizer.
"""

from rl_scx_params import optimize_scheduler

# Quick optimization with default settings
results = optimize_scheduler(
    scheduler_name="scx_flashyspark",
    episodes=50,
    algorithm="ppo"
)

print("Optimization complete!")
print(f"Results: {results}")
