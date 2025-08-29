# Schedtuna

A comprehensive reinforcement learning framework for optimizing Linux scheduler parameters across multiple scheduler types and parameter configurations.

## Overview

This system uses reinforcement learning algorithms (PPO, SAC) to automatically discover optimal parameter configurations for various Linux schedulers.


## Supported Schedulers

### scx_flashyspark
Full-featured scheduler with comprehensive parameter support:

#### Boolean Parameters
- `slice_lag_scaling`: Dynamic slice lag scaling based on CPU utilization
- `rr_sched`: Round-robin scheduling with fixed time slices
- `no_builtin_idle`: Disable in-kernel idle CPU selection policy
- `local_pcpu`: Enable prioritization of per-CPU tasks
- `direct_dispatch`: Always allow direct dispatch to idle CPUs
- `sticky_cpu`: Enable CPU stickiness to reduce task migrations
- `stay_with_kthread`: Keep tasks on CPUs where kthreads are running
- `native_priority`: Use native Linux priority range
- `local_kthreads`: Enable per-CPU kthread prioritization
- `no_wake_sync`: Disable direct dispatch during synchronous wakeups
- `aggressive_gpu_tasks`: GPU task mode for performance cores
- `timer_kick`: Use BPF timer for task kicking

#### Numeric Parameters
- `slice_us`: Base time slice duration (1000-100000 μs)
- `cpu_util_threshold`: CPU utilization threshold (0.1-1.0)

#### Categorical Parameters
- `scheduling_policy`: Overall policy mode (default, performance, powersave, latency)

### scx_rusty
Basic scheduler support with essential parameters:
- `direct`: Enable direct dispatch
- `kick`: Enable kicking mechanism
- `slice_us`: Time slice duration (1000-50000 μs)

## Quick Start

### Installation

1. **Setup Environment:**
```bash
./setup.sh
```

2. **Activate Virtual Environment:**
```bash
source rl_scx_params/bin/activate
```

### Using the Python API (Recommended)

The easiest way to use rl_scx_params is through the Python API:

```python
from rl_scx_params import optimize_scheduler

# Quick optimization with default settings
results = optimize_scheduler(
    scheduler_name="scx_flashyspark",
    episodes=50,
    algorithm="ppo"
)
```

See the [API Usage](#api-usage) section for more details.

### Using the Command Line

You can also use the traditional command-line interface:

1. **Optimize scx_flashyspark:**
```bash
# Quick test run
python main.py --scheduler scx_flashyspark --episodes 10 --baseline-runs 2

# Full optimization
python main.py --scheduler scx_flashyspark --algorithm ppo --episodes 100
```

2. **Optimize scx_rusty:**
```bash
python main.py --scheduler scx_rusty --algorithm sac --episodes 50
```

3. **Custom Configuration:**
```bash
python main.py --scheduler scx_flashyspark \
    --algorithm ppo \
    --episodes 200 \
    --optimize-metric tg_tokens_per_sec \
    --learning-rate 1e-4 \
    --experiment-name "gpu_workload_optimization"
```

## Advanced Features

### Experience Replay and Pre-training

Leverage historical experiment data to accelerate learning:

```bash
# Analyze historical patterns
python experience_replay.py --statistics --param-combinations 15

# Pre-train an agent
python experience_replay.py --pretrain-agent --algorithm ppo --pretrain-epochs 20

# Use pre-trained model
python main.py --scheduler scx_flashyspark \
    --pretrained-model pretrained_ppo_model.pt \
    --episodes 100
```

### Parameter Analysis

Understand which parameters work best:

```bash
# Analyze parameter effectiveness
python pretrain.py --scheduler scx_flashyspark --action analyze

# Recompute rewards with new function
python pretrain.py --scheduler scx_flashyspark --action recompute
```

### Custom Benchmarks

Use different benchmark tools:

```bash
python main.py --scheduler scx_flashyspark \
    --benchmark-cmd "/path/to/custom/benchmark" \
    --model-path "/path/to/model.gguf"
```

## API Usage

The rl_scx_params package provides a clean Python API for programmatic usage:

### Basic Usage

```python
from rl_scx_params import optimize_scheduler

# Quick optimization with minimal configuration
results = optimize_scheduler(
    scheduler_name="scx_flashyspark",
    episodes=50,
    algorithm="ppo"
)
```

### Using the RLSchedulerOptimizer Class

```python
from rl_scx_params import RLSchedulerOptimizer

# Create optimizer with dictionary configuration
optimizer = RLSchedulerOptimizer({
    "scheduler_name": "scx_flashyspark",
    "algorithm": "ppo",
    "episodes": 100,
    "learning_rate": 0.001,
    "optimize_metric": "throughput"
})

# Get scheduler information
info = optimizer.get_scheduler_info()
print(f"Optimizing {info['scheduler_name']} with {len(info['parameters'])} parameters")

# Train the agent
results = optimizer.train()

# Test the best parameters
test_results = optimizer.test(results['experiment_path'], test_runs=10)
```

### Using Configuration Files

Create a YAML configuration file (`config.yaml`):

```yaml
scheduler_name: scx_flashyspark
algorithm: ppo
episodes: 100
learning_rate: 0.001
optimize_metric: throughput
experiment_name: my_experiment
```

Then use it in your code:

```python
from rl_scx_params import RLSchedulerOptimizer

# Load from config file
optimizer = RLSchedulerOptimizer("config.yaml")
results = optimizer.train()
```

You can also use JSON configuration files:

```json
{
  "scheduler_name": "scx_flashyspark",
  "algorithm": "sac",
  "episodes": 200,
  "learning_rate": 0.0001,
  "optimize_metric": "latency"
}
```

### Defining Custom Scheduler Parameters

If you're working with a custom scheduler, you can define your own parameters:

```python
optimizer = RLSchedulerOptimizer()

# Define custom parameters
optimizer.define_scheduler_params({
    'enable_turbo': {
        'type': 'boolean',
        'default': False,
        'description': 'Enable turbo boost mode',
        'command_arg': '--turbo'
    },
    'slice_duration': {
        'type': 'integer',
        'default': 10000,
        'min_value': 1000,
        'max_value': 50000,
        'description': 'Time slice duration in microseconds',
        'command_arg': '--slice-duration'
    },
    'load_threshold': {
        'type': 'float',
        'default': 0.75,
        'min_value': 0.0,
        'max_value': 1.0,
        'description': 'CPU load threshold',
        'command_arg': '--load-threshold'
    },
    'scheduling_mode': {
        'type': 'categorical',
        'default': 'balanced',
        'choices': ['performance', 'balanced', 'powersave'],
        'description': 'Overall scheduling mode',
        'command_arg': '--mode'
    }
})

# Train with custom parameters
results = optimizer.train()
```

### Example Scripts

Check the `examples/` directory for complete working examples:

- `basic_usage.py` - Simple optimization example
- `advanced_usage.py` - Full API demonstration
- `config_file_usage.py` - Using configuration files
- `custom_scheduler_params.py` - Custom scheduler parameters

### Configuration Options

All configuration options available in OptimizerConfig:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `scheduler_name` | str | "scx_flashyspark" | Scheduler to optimize |
| `algorithm` | str | "ppo" | RL algorithm ("ppo" or "sac") |
| `episodes` | int | 50 | Number of training episodes |
| `baseline_runs` | int | 3 | Number of baseline evaluation runs |
| `learning_rate` | float | 1e-3 | Learning rate for RL agent |
| `update_frequency` | int | 5 | Agent update frequency (episodes) |
| `save_frequency` | int | 25 | Checkpoint save frequency |
| `benchmark_cmd` | str | None | Custom benchmark command |
| `model_path` | str | None | Path to model file |
| `timeout` | int | 300 | Benchmark timeout (seconds) |
| `optimize_metric` | str | "throughput" | Metric to optimize |
| `reward_scaling` | float | 1.0 | Reward scaling factor |
| `experiment_name` | str | None | Custom experiment name |
| `use_gpu` | bool | True | Use GPU if available |
| `pretrained_model_path` | str | None | Path to pretrained model |

## Command Line Reference

### Main Training Script (`main.py`)

**Core Options:**
- `--scheduler {scx_flashyspark,scx_rusty}`: Scheduler to optimize
- `--algorithm {ppo,sac}`: RL algorithm to use
- `--episodes N`: Number of training episodes
- `--optimize-metric {pp_tokens_per_sec,tg_tokens_per_sec}`: Optimization target

**Training Configuration:**
- `--learning-rate FLOAT`: Learning rate for RL agent
- `--update-frequency N`: Agent update frequency
- `--save-frequency N`: Checkpoint save frequency
- `--baseline-runs N`: Number of baseline evaluation runs

**System Configuration:**
- `--model-path PATH`: Path to model file for benchmarking
- `--benchmark-cmd PATH`: Path to benchmark executable
- `--timeout N`: Benchmark timeout in seconds
- `--no-gpu`: Force CPU usage

**Experiment Management:**
- `--experiment-name NAME`: Custom experiment name
- `--pretrained-model PATH`: Pre-trained model to load
- `--reward-scaling FLOAT`: Reward scaling factor

### Experience Replay Script (`experience_replay.py`)

**Data Management:**
- `--results-dir PATH`: Directory containing experiment results
- `--experiment-patterns PATTERN [PATTERN ...]`: Experiment name patterns to include
- `--max-episodes-per-exp N`: Maximum episodes per experiment

**Analysis Options:**
- `--statistics`: Show detailed statistics
- `--param-combinations N`: Number of top parameter combinations to analyze
- `--recalculate`: Recalculate rewards with new function

**Pre-training:**
- `--pretrain-agent`: Pre-train a new agent
- `--algorithm {ppo,sac}`: Algorithm for pre-training
- `--pretrain-epochs N`: Number of pre-training epochs

### Analysis Script (`pretrain.py`)

**Actions:**
- `--action {analyze,pretrain,recompute}`: Action to perform
- `--scheduler SCHEDULER`: Scheduler to analyze

## Output and Results

### Experiment Structure
```
results/
├── scx_flashyspark_ppo_100ep_20241201_120000/
│   ├── experiment.log              # Detailed training logs
│   ├── best_parameters.json        # Optimal parameters found
│   ├── episode_results.json        # Complete episode data
│   ├── episode_results.csv         # Episode data in CSV format
│   ├── baseline_results.json       # Baseline performance
│   ├── performance_plots.png       # Training visualizations
│   └── checkpoint_episode_50.pt    # Model checkpoints
└── scx_rusty_sac_50ep_20241201_130000/
    └── ...
```

### Key Metrics

**Performance Metrics:**
- **PP tokens/second**: Prompt processing throughput
- **TG tokens/second**: Text generation throughput
- **Execution time**: Benchmark completion time
- **Success rate**: Percentage of successful runs

**Training Metrics:**
- **Reward**: Combined optimization metric
- **Policy loss**: RL algorithm policy loss
- **Value loss**: Value function loss
- **Exploration**: Current exploration rate

## Adding New Schedulers

The framework is designed for easy extension. To add a new scheduler:

1. **Define Scheduler Configuration:**
```python
def get_scx_newscheduler_config() -> SchedulerConfig:
    parameters = {
        'enable_feature': ParameterSpec(
            'enable_feature', ParameterType.BOOLEAN, False,
            description="Enable special feature",
            command_arg="--enable-feature"
        ),
        'time_slice': ParameterSpec(
            'time_slice', ParameterType.INTEGER, 10000,
            min_value=1000, max_value=50000,
            description="Time slice in microseconds",
            command_arg="--time-slice"
        ),
        'policy_mode': ParameterSpec(
            'policy_mode', ParameterType.CATEGORICAL, 'balanced',
            choices=['balanced', 'performance', 'efficiency'],
            description="Scheduling policy mode",
            command_arg="--policy"
        )
    }
    
    return SchedulerConfig(
        name="scx_newscheduler",
        binary_path="/path/to/scx_newscheduler",
        parameters=parameters,
        description="New scheduler with custom parameters"
    )
```

2. **Register Configuration:**
```python
# In get_scheduler_config() function
configs = {
    'scx_flashyspark': get_scx_flashyspark_config,
    'scx_rusty': get_scx_rusty_config,
    'scx_newscheduler': get_scx_newscheduler_config,  # Add this line
}
```

3. **Update Command Line Options:**
```python
parser.add_argument(
    "--scheduler", "-S",
    choices=["scx_flashyspark", "scx_rusty", "scx_newscheduler"],  # Add new scheduler
    default="scx_flashyspark",
    help="Scheduler to optimize"
)
```

## System Requirements

**Operating System:**
- Linux with sched_ext support
- Kernel 6.12+ recommended




**Storage:**
- 2GB+ free space for experiments and models
- Additional space for benchmark models

## Performance Tips

1. **Start Small**: Begin with 25-50 episodes to verify system operation
2. **Use Baselines**: Always run baseline evaluation for comparison
3. **Monitor Resources**: Watch CPU, memory, and GPU utilization
4. **Leverage History**: Use experience replay for faster convergence
5. **Choose Metrics Wisely**: Optimize for your specific workload requirements
6. **Checkpoint Frequently**: Use appropriate save frequency for long runs



### Debug Mode

Enable verbose logging:
```bash
python main.py --verbose --scheduler scx_flashyspark --episodes 10
```




