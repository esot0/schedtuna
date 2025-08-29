#!/usr/bin/env python3
"""
Custom scheduler parameters example
===================================

This example shows how to define your own scheduler parameters
if you're working with a custom or modified scheduler.
"""

from rl_scx_params import RLSchedulerOptimizer

# Create optimizer
optimizer = RLSchedulerOptimizer({
    "scheduler_name": "my_custom_scheduler",
    "episodes": 100,
    "algorithm": "ppo"
})

# Define custom parameters for your scheduler
optimizer.define_scheduler_params({
    # Boolean parameters
    'enable_turbo': {
        'type': 'boolean',
        'default': False,
        'description': 'Enable turbo boost mode',
        'command_arg': '--turbo'
    },
    
    # Integer parameters
    'slice_duration': {
        'type': 'integer',
        'default': 10000,
        'min_value': 1000,
        'max_value': 50000,
        'description': 'Time slice duration in microseconds',
        'command_arg': '--slice-duration'
    },
    
    # Float parameters
    'load_threshold': {
        'type': 'float',
        'default': 0.75,
        'min_value': 0.0,
        'max_value': 1.0,
        'description': 'CPU load threshold for load balancing',
        'command_arg': '--load-threshold'
    },
    
    # Categorical parameters
    'scheduling_mode': {
        'type': 'categorical',
        'default': 'balanced',
        'choices': ['performance', 'balanced', 'powersave'],
        'description': 'Overall scheduling mode',
        'command_arg': '--mode'
    }
})

# Check the defined parameters
info = optimizer.get_scheduler_info()
print("Custom scheduler parameters:")
for name, spec in info['parameters'].items():
    print(f"  {name}: {spec}")

# Train with the custom parameters
results = optimizer.train()
print(f"\nTraining complete! Results: {results}")
