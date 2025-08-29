#!/usr/bin/env python3
"""
Advanced usage example for RL SCX Params
=======================================

This example shows how to use the full API with custom configurations.
"""

from rl_scx_params import RLSchedulerOptimizer, OptimizerConfig

# Method 1: Create optimizer with dictionary config
config_dict = {
    "scheduler_name": "scx_flashyspark",
    "algorithm": "ppo",
    "episodes": 100,
    "learning_rate": 0.001,
    "optimize_metric": "throughput",
    "experiment_name": "my_custom_experiment"
}

optimizer = RLSchedulerOptimizer(config_dict)

# Get information about the scheduler
scheduler_info = optimizer.get_scheduler_info()
print("Scheduler Information:")
print(f"  Name: {scheduler_info['scheduler_name']}")
print(f"  Description: {scheduler_info['description']}")
print(f"  Number of parameters: {len(scheduler_info['parameters'])}")
print("\nAvailable parameters:")
for param_name, param_info in scheduler_info['parameters'].items():
    print(f"  - {param_name}: {param_info['type']} (default: {param_info['default']})")


print("\nStarting training...")
results = optimizer.train()
print(f"Training complete! Results: {results}")

# Test the best parameters
if 'experiment_path' in results:
    print("\nTesting best parameters...")
    test_results = optimizer.test(results['experiment_path'], test_runs=10)
    print(f"Test results: {test_results}")
