#!/usr/bin/env python3
"""
Config file usage example for RL SCX Params
===========================================

This example shows how to use configuration files.
"""

from rl_scx_params import RLSchedulerOptimizer, OptimizerConfig

# Method 1: Load from YAML config file
optimizer_yaml = RLSchedulerOptimizer("flashyspark_config.yaml")
results_yaml = optimizer_yaml.train()

# Method 2: Load from JSON config file
optimizer_json = RLSchedulerOptimizer("config.json")
results_json = optimizer_json.train()

# Method 3: Load config, modify it, then use it
config = OptimizerConfig.from_file("minimal_config.yaml")
config.episodes = 75  # Override the value from file
config.learning_rate = 0.0005

optimizer_modified = RLSchedulerOptimizer(config)
results_modified = optimizer_modified.train()

# Method 4: Create config programmatically and save it
new_config = OptimizerConfig(
    scheduler_name="scx_rusty",
    algorithm="sac",
    episodes=150,
    optimize_metric="latency"
)
new_config.to_file("my_custom_config.yaml")
print("Saved new configuration to my_custom_config.yaml")
