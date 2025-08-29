#!/usr/bin/env python3
"""
Example script to optimize scheduler parameters for different workload types
"""

from rl_scx_params import RLSchedulerOptimizer, OptimizerConfig
import sys

def optimize_for_workload(workload_type: str, episodes: int = 50):
    """Optimize scheduler parameters for a specific workload type."""
    
    print(f"\nOptimizing for {workload_type} workload...")
    
    # Create configuration for the workload
    config = OptimizerConfig(
        scheduler_name="scx_flashyspark",
        workload_type=workload_type,
        episodes=episodes,
        baseline_runs=5,
        learning_rate=1e-3,
        save_frequency=10,
        # Use appropriate metric based on workload
        optimize_metric="pp_tokens_per_sec" if workload_type in ["latency_sensitive", "gpu_intensive"] else "tg_tokens_per_sec"
    )
    
    # Create optimizer
    optimizer = RLSchedulerOptimizer(config)
    
    # Define scheduler parameters to optimize
    params = {
        'slice_us': {
            'type': 'integer',
            'default': 4096,
            'min_value': 512 if workload_type == "latency_sensitive" else 2048,
            'max_value': 8192 if workload_type == "latency_sensitive" else 32768,
            'description': 'Time slice in microseconds',
            'command_arg': '--slice-us'
        },
        'cpu_busy_thresh': {
            'type': 'integer',
            'default': 65,
            'min_value': 30,
            'max_value': 90,
            'description': 'CPU utilization threshold',
            'command_arg': '--cpu-busy-thresh'
        },
        'sticky_cpu': {
            'type': 'boolean',
            'default': workload_type == "cache_sensitive",
            'description': 'Enable CPU stickiness',
            'command_arg': '--sticky-cpu'
        },
        'direct_dispatch': {
            'type': 'boolean',
            'default': workload_type in ["cpu_intensive", "gpu_intensive"],
            'description': 'Enable direct dispatch',
            'command_arg': '--direct-dispatch'
        }
    }
    
    if workload_type in ["latency_sensitive", "cache_sensitive"]:
        params['local_pcpu'] = {
            'type': 'boolean',
            'default': True,
            'description': 'Prioritize per-CPU tasks',
            'command_arg': '--local-pcpu'
        }
    
    if workload_type == "gpu_intensive":
        params['aggressive_gpu_tasks'] = {
            'type': 'boolean',
            'default': True,
            'description': 'GPU task prioritization',
            'command_arg': '--aggressive-gpu-tasks'
        }
    
    optimizer.define_scheduler_params(params)
    
    # Run optimization
    try:
        results = optimizer.train()
        print(f"\nOptimization complete for {workload_type}!")
        print(f"Best parameters: {results['best_params']}")
        print(f"Best reward: {results['best_reward']:.2f}")
        print(f"Results saved to: {config.params_output_file}")
        return results
    except Exception as e:
        print(f"Error during optimization: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: optimize_workloads.py <workload_type> [episodes]")
        print("\nWorkload types:")
        print("  - latency_sensitive")
        print("  - cpu_intensive")
        print("  - cache_sensitive")
        print("  - gpu_intensive")
        print("  - mixed")
        sys.exit(1)
    
    workload_type = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    valid_workloads = ["latency_sensitive", "cpu_intensive", "cache_sensitive", "gpu_intensive", "mixed"]
    if workload_type not in valid_workloads:
        print(f"Error: Invalid workload type '{workload_type}'")
        print(f"Valid types: {', '.join(valid_workloads)}")
        sys.exit(1)
    
    optimize_for_workload(workload_type, episodes)

if __name__ == "__main__":
    main()
