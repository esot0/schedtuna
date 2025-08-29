#!/usr/bin/env python3
"""
Test script to demonstrate workload optimization workflow
"""

import json
from pathlib import Path
from rl_scx_params import RLSchedulerOptimizer, OptimizerConfig

def test_workload_optimization():
    """Test the workload optimization functionality."""
    
    print("=" * 60)
    print("Testing RL SCX Params Workload Optimization")
    print("=" * 60)
    
    # Test 1: Create optimized parameters for latency-sensitive workload
    print("\n1. Creating optimized parameters for latency-sensitive workload...")
    
    config = OptimizerConfig(
        scheduler_name="scx_flashyspark",
        workload_type="latency_sensitive",
        episodes=2,  # Very short for testing
        baseline_runs=1,
        optimize_metric="pp_tokens_per_sec"
    )
    
    optimizer = RLSchedulerOptimizer(config)
    
    # Define minimal parameters for testing
    optimizer.define_scheduler_params({
        'slice_us': {
            'type': 'integer',
            'default': 4096,
            'min_value': 1024,
            'max_value': 8192,
            'command_arg': '--slice-us'
        },
        'sticky_cpu': {
            'type': 'boolean',
            'default': False,
            'command_arg': '--sticky-cpu'
        }
    })
    
    # This would normally train, but for testing we'll create dummy results
    print("  - Simulating training (in real usage, this would run RL training)...")
    
    # Create dummy optimized parameters with ALL parameters
    dummy_params = {
        "latency_sensitive": {
            "parameters": {
                # Boolean parameters
                "sticky_cpu": False,
                "direct_dispatch": False,
                "aggressive_gpu_tasks": False,
                "local_pcpu": True,
                "no_wake_sync": False,
                "slice_lag_scaling": True,
                "local_kthreads": True,
                "stay_with_kthread": False,
                "native_priority": False,
                "tickless_sched": False,
                "timer_kick": False,
                
                # Time slice parameters
                "slice_us": 2048,
                "slice_us_min": 128,
                "slice_us_lag": 2048,
                "run_us_lag": 16384,
                
                # Other numeric parameters
                "cpu_busy_thresh": 50,
                "max_avg_nvcsw": 256
            },
            "reward": 125.3,
            "optimize_metric": "pp_tokens_per_sec",
            "experiment_path": "dummy_path",
            "timestamp": "2024-01-20T10:30:00"
        },
        "cpu_intensive": {
            "parameters": {
                # Boolean parameters
                "sticky_cpu": False,
                "direct_dispatch": True,
                "aggressive_gpu_tasks": False,
                "local_pcpu": False,
                "no_wake_sync": False,
                "slice_lag_scaling": False,
                "local_kthreads": False,
                "stay_with_kthread": False,
                "native_priority": False,
                "tickless_sched": False,
                "timer_kick": False,
                
                # Time slice parameters
                "slice_us": 8192,
                "slice_us_min": 512,
                "slice_us_lag": 4096,
                "run_us_lag": 65536,
                
                # Other numeric parameters
                "cpu_busy_thresh": 80,
                "max_avg_nvcsw": 64
            },
            "reward": 98.7,
            "optimize_metric": "tg_tokens_per_sec",
            "experiment_path": "dummy_path",
            "timestamp": "2024-01-20T11:00:00"
        }
    }
    
    # Save dummy parameters
    params_file = Path(config.params_output_file).expanduser()
    params_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(params_file, 'w') as f:
        json.dump(dummy_params, f, indent=2)
    
    print(f"  - Saved optimized parameters to: {params_file}")
    
    # Test 2: Read back the parameters
    print("\n2. Reading optimized parameters...")
    
    with open(params_file, 'r') as f:
        loaded_params = json.load(f)
    
    for workload_type, data in loaded_params.items():
        print(f"\n  Workload: {workload_type}")
        print(f"  - Reward: {data['reward']}")
        print(f"  - Optimized for: {data['optimize_metric']}")
        print(f"  - Parameters:")
        for param, value in data['parameters'].items():
            print(f"    - {param}: {value}")
    
    print("\n3. Integration with scx_flashyspark scheduler:")
    print("  - The scheduler will automatically load these parameters")
    print("  - When it detects a workload type change, it applies the corresponding parameters")
    print("  - If ML-optimized parameters are available, they override defaults")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_workload_optimization()
