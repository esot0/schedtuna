# scx_flashyspark RL Parameter Optimizer

This directory contains a reinforcement learning system for optimizing scx_flashyspark scheduler parameters to improve llama-bench performance.

## Overview

The system uses reinforcement learning (PPO or SAC algorithms) to automatically find optimal boolean parameter combinations for the scx_flashyspark scheduler. It evaluates performance using real llama-bench runs and can optimize for either prompt processing (pp) or text generation (tg) metrics.

## Key Features

- **Comprehensive Parameter Space**: Optimizes all 20 boolean parameters available in scx_flashyspark
- **Real Performance Evaluation**: Uses actual llama-bench runs for performance measurement
- **Flexible Optimization**: Choose between optimizing prompt processing or text generation performance
- **Multiple RL Algorithms**: Supports both PPO and SAC algorithms
- **Experiment Tracking**: Comprehensive logging, visualization, and result storage
- **Safety Mechanisms**: Automatic cleanup of scheduler processes and timeout handling

## Parameter Space

The system optimizes 20 boolean parameters:

### Core Scheduling Behavior
- `slice_lag_scaling`: Dynamic slice lag scaling based on CPU utilization
- `tickless`: Tickless mode - infinite time slices with preemption only on contention
- `rr_sched`: Round-robin scheduling with fixed time slices
- `no_builtin_idle`: Disable in-kernel idle CPU selection policy
- `local_pcpu`: Enable prioritization of per-CPU tasks
- `direct_dispatch`: Always allow direct dispatch to idle CPUs
- `sticky_cpu`: Enable CPU stickiness to reduce task migrations
- `native_priority`: Use native Linux priority range instead of normalization
- `local_kthreads`: Enable per-CPU kthread prioritization
- `no_wake_sync`: Disable direct dispatch during synchronous wakeups

### Advanced Features
- `stay_with_kthread`: Keep tasks on CPUs where kthreads are running (experimental)
- `aggressive_gpu_tasks`: GPU task mode - only GPU tasks can use big/performance cores
- `workload_aware_scheduling`: Make CPU selection decisions based on workload type
- `timer_kick`: Use BPF timer instead of scx_kick_cpu for task kicking
- `more_dsqs`: Use multiple dispatch queues to prioritize tasks on specific cores

### System Optimizations
- `disable_l2`: Disable L2 cache awareness optimizations
- `disable_l3`: Disable L3 cache awareness optimizations
- `disable_smt`: Disable SMT (simultaneous multithreading) awareness
- `disable_numa`: Disable NUMA rebalancing optimizations
- `cpufreq`: Enable CPU frequency control (requires schedutil governor)

## Quick Start

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Basic Training:**
```bash
# Train PPO agent optimizing prompt processing performance
python rl_experiments/main.py --algorithm ppo --episodes 50 --optimize-metric pp

# Train SAC agent optimizing text generation performance
python rl_experiments/main.py --algorithm sac --episodes 50 --optimize-metric tg
```

3. **Test Specific Parameters:**
```bash
# Test aggressive GPU mode with timer kick
python rl_experiments/test_params.py --aggressive-gpu-tasks --timer-kick

# Test workload-aware scheduling with cache optimizations disabled
python rl_experiments/test_params.py --workload-aware-scheduling --disable-l2 --disable-l3
```

## Usage Examples

### Training Examples

```bash
# Standard PPO training optimizing prompt processing
python rl_experiments/main.py --algorithm ppo --episodes 100 --optimize-metric pp

# SAC training with custom parameters
python rl_experiments/main.py --algorithm sac --episodes 75 --lr 1e-4 --optimize-metric tg

# Training with baseline evaluation
python rl_experiments/main.py --algorithm ppo --episodes 50 --baseline-runs 5

# Custom experiment name and model path
python rl_experiments/main.py --algorithm ppo --episodes 50 \
    --experiment-name "gpu_optimization" \
    --model-path "/path/to/custom/model.gguf"
```

### Testing Examples

```bash
# Test GPU-focused configuration
python rl_experiments/test_params.py --aggressive-gpu-tasks --sticky-cpu --stay-with-kthread

# Test high-performance configuration
python rl_experiments/test_params.py --direct-dispatch --timer-kick --more-dsqs --cpufreq

# Test cache-disabled configuration
python rl_experiments/test_params.py --disable-l2 --disable-l3 --disable-smt

# Test with increased runs for statistical significance
python rl_experiments/test_params.py --workload-aware-scheduling --runs 10
```

## Command Line Arguments

### Main Training Script (`main.py`)

- `--algorithm {ppo,sac}`: RL algorithm to use (default: ppo)
- `--episodes N`: Number of training episodes (default: 50)
- `--optimize-metric {pp,tg}`: Optimize prompt processing or text generation (default: pp)
- `--baseline-runs N`: Number of baseline evaluation runs (default: 3)
- `--model-path PATH`: Path to llama model file
- `--experiment-name NAME`: Custom experiment name
- `--timeout N`: Benchmark timeout in seconds (default: 300)
- `--learning-rate FLOAT`: Learning rate for RL agent (default: 3e-4)
- `--update-frequency N`: Agent update frequency in episodes (default: 10)
- `--save-frequency N`: Checkpoint save frequency in episodes (default: 25)

### Parameter Testing Script (`test_params.py`)

All 20 boolean parameters are available as command-line flags:

#### Core Scheduling
- `--slice-lag-scaling`: Dynamic slice lag scaling
- `--tickless`: Tickless mode
- `--rr-sched`: Round-robin scheduling
- `--no-builtin-idle`: Disable built-in idle selection
- `--local-pcpu`: Enable tasks prioritization
- `--direct-dispatch`: Allow direct dispatch to idle CPUs
- `--sticky-cpu`: Enable CPU stickiness
- `--native-priority`: Use native task priorities
- `--local-kthreads`: Per-CPU kthread prioritization
- `--no-wake-sync`: Disable direct dispatch during sync wakeups

#### Advanced Features
- `--stay-with-kthread`: Keep tasks on CPUs where kthreads are running
- `--aggressive-gpu-tasks`: Aggressive GPU task mode
- `--workload-aware-scheduling`: Workload-aware scheduling mode
- `--timer-kick`: Use BPF timer for task kicking
- `--more-dsqs`: Use multiple dispatch queues

#### System Optimizations
- `--disable-l2`: Disable L2 cache awareness
- `--disable-l3`: Disable L3 cache awareness
- `--disable-smt`: Disable SMT awareness
- `--disable-numa`: Disable NUMA rebalancing
- `--cpufreq`: Enable CPU frequency control

#### Testing Options
- `--runs N`: Number of test runs (default: 3)
- `--use-environment`: Use RL environment for testing

## Output and Results

### Training Results
- **Experiment logs**: Detailed logs in `rl_experiments/[experiment_name]/experiment.log`
- **Episode data**: JSON and CSV files with per-episode results
- **Best parameters**: `best_parameters.json` contains optimal configuration found
- **Checkpoints**: PyTorch model checkpoints saved periodically
- **Visualizations**: Performance plots and training curves

### Key Metrics
- **PP tokens/second**: Prompt processing throughput from llama-bench
- **TG tokens/second**: Text generation throughput from llama-bench
- **Reward**: Combined metric based on selected optimization target
- **Success rate**: Percentage of successful benchmark runs
- **Execution time**: Time taken for each benchmark run

### Example Output Format
```
| model | size | params | backend | ngl | test | t/s |
| llama 70B IQ4_NL | ... | ... | CUDA | 99 | pp512 | 257.21 ± 10.67 |
| llama 70B IQ4_NL | ... | ... | CUDA | 99 | tg128 | 5.09 ± 0.32 |
```

## File Structure

```
rl_experiments/
├── main.py              # Main RL training script
├── test_params.py       # Manual parameter testing utility
├── requirements.txt     # Python dependencies
├── README.md           # This documentation
└── [experiment_name]/  # Generated experiment results
    ├── experiment.log   # Detailed training logs
    ├── best_parameters.json # Optimal parameters found
    ├── episode_results.json # Complete episode data
    ├── episode_results.csv  # Episode data in CSV format
    ├── baseline_results.json # Baseline performance
    ├── performance_plots.png # Training visualizations
    └── checkpoint_*.pt  # Model checkpoints
```

## System Requirements

- **OS**: Linux (scheduler requires kernel support)
- **Python**: 3.8+ with ML libraries (PyTorch, Gymnasium, etc.)
- **Hardware**: CUDA-capable GPU recommended for llama-bench
- **Permissions**: sudo access required for scheduler operations
- **Storage**: ~1GB for experiment results and model checkpoints

## Integration with scx_flashyspark

The system directly integrates with the scx_flashyspark scheduler by:

1. **Dynamic Parameter Application**: Translates RL actions to scheduler command-line arguments
2. **Real-time Evaluation**: Runs actual scheduler instances with llama-bench workloads
3. **Performance Parsing**: Extracts metrics from llama-bench table output
4. **Safe Operation**: Automatic cleanup of scheduler processes and timeout handling

## Advanced Usage

### Custom Reward Functions
The reward function can be modified by editing the `BenchmarkResult.get_reward()` method to incorporate additional metrics or different optimization objectives.

### Extended Training
For production optimization, consider running longer training sessions:
```bash
python rl_experiments/main.py --algorithm ppo --episodes 500 --save-frequency 50
```

### Hyperparameter Tuning
Experiment with different learning rates and update frequencies:
```bash
python rl_experiments/main.py --algorithm sac --lr 1e-3 --update-frequency 5
```

## Troubleshooting

### Common Issues
- **Scheduler startup failures**: Check sudo permissions and scheduler binary path
- **llama-bench timeouts**: Increase `--timeout` value or check model path
- **Memory issues**: Reduce batch sizes or use smaller models
- **Permission errors**: Ensure proper sudo access for scheduler operations

### Debug Mode
Enable verbose logging for troubleshooting:
```bash
python rl_experiments/main.py --verbose --algorithm ppo --episodes 10
```

## Performance Tips

1. **Start Small**: Begin with 25-50 episodes to verify system operation
2. **Use Baselines**: Always run baseline evaluation for comparison
3. **Monitor Resources**: Watch CPU, memory, and GPU utilization during training
4. **Checkpoint Frequently**: Use appropriate `--save-frequency` for long runs
5. **Choose Metrics Wisely**: Optimize the metric most relevant to your workload (pp vs tg)
