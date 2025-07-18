# Using Historical Experiment Data to Accelerate Training

This guide explains how to leverage your previous experiment results to improve your RL agent's training speed and performance.

## Overview

You've run many experiments and accumulated valuable data in `fs_results/`. Instead of starting from scratch each time, you can now:

1. **Re-calculate rewards** for all historical episodes using your updated reward function
2. **Pre-train agents** using historical data for faster convergence
3. **Analyze patterns** to understand what parameter combinations work best
4. **Continue training** from pre-trained models

## Quick Start

### 1. Basic Usage: Re-calculate Rewards and Analyze

```bash
# Load historical data and recalculate rewards with new reward function
cd /home/nvidia/rl_experiments
python experience_replay.py --optimize-metric pp_tokens_per_sec --reward-scaling 1.5
```

### 2. Pre-train an Agent

```bash
# Pre-train an agent using historical data
python experience_replay.py --pretrain-agent --algorithm ppo --pretrain-epochs 20
```

### 3. Start Training with Pre-trained Model

```bash
# Use the pre-trained model in a new training run
python main.py --algorithm ppo --episodes 100 --pretrained-model pretrained_ppo_model.pt
```

## Detailed Usage

### Command Line Options for `experience_replay.py`

| Option | Description | Example |
|--------|-------------|---------|
| `--experiment-patterns` | Filter experiments by name patterns | `--experiment-patterns ppo_500ep ppo_200ep` |
| `--max-episodes-per-exp` | Limit episodes loaded per experiment | `--max-episodes-per-exp 200` |
| `--optimize-metric` | Metric for reward calculation | `--optimize-metric pp_tokens_per_sec` |
| `--reward-scaling` | Scale rewards more/less extreme | `--reward-scaling 2.0` |
| `--baseline-pp` | Custom PP baseline | `--baseline-pp 10200` |
| `--baseline-tg` | Custom TG baseline | `--baseline-tg 145` |
| `--pretrain-agent` | Pre-train a new agent | `--pretrain-agent` |
| `--algorithm` | Algorithm for pre-training | `--algorithm ppo` |
| `--pretrain-epochs` | Pre-training epochs | `--pretrain-epochs 15` |

### Example Workflows

#### Workflow 1: Experiment with New Reward Function

```bash
# 1. Load data from your best experiments and try new reward scaling
python experience_replay.py \
    --experiment-patterns ppo_500ep \
    --optimize-metric pp_tokens_per_sec \
    --reward-scaling 2.0 \
    --baseline-pp 10200 \
    --baseline-tg 145

# 2. Check the output for best parameter combinations
# 3. Use insights to modify your main reward function
```

#### Workflow 2: Fast Training with Pre-trained Agent

```bash
# 1. Pre-train agent with historical data
python experience_replay.py \
    --pretrain-agent \
    --algorithm ppo \
    --pretrain-epochs 20 \
    --max-episodes-per-exp 150

# 2. Use pre-trained model for new training
python main.py \
    --algorithm ppo \
    --episodes 50 \
    --learning-rate 3e-4 \
    --pretrained-model pretrained_ppo_model.pt \
    --experiment-name "fast_training_with_pretraining"
```

#### Workflow 3: Comprehensive Analysis

```bash
# 1. Analyze all historical data to find patterns
python experience_replay.py \
    --optimize-metric pp_tokens_per_sec \
    --output-file comprehensive_analysis.json

# 2. Run the example analysis script
python example_usage.py
```

## Understanding the Output

### Re-calculated Rewards Analysis

When you run experience replay, you'll see output like:

```
Loading historical experiment data...
Found 15 experiment directories to process
Successfully loaded 3426 historical experiences

Using baselines - PP: 10156.34, TG: 143.67
Recalculated rewards for 3426 experiences

Reward Statistics:
Original rewards - Mean: 98.45, Std: 12.34
New rewards - Mean: 156.78, Std: 23.45
Best reward: 287.65

Top 5 parameter combinations:
 1. Reward: 287.7 | PP: 10645.3 | TG: 147.8 | Enabled: aggressive_gpu_tasks, native_priority, slice_lag_scaling...
 2. Reward: 273.2 | PP: 10534.1 | TG: 146.9 | Enabled: local_kthreads, sticky_cpu, aggressive_gpu_tasks...
```

This tells you:
- How many historical experiences were processed
- How the new reward function compares to the old one
- Which parameter combinations perform best with your new reward function

### Pre-training Output

During pre-training, you'll see:

```
Starting pre-training with historical data...
Epoch 0/20 - Policy Loss: 2.3456, Value Loss: 1.2345
Epoch 5/20 - Policy Loss: 1.8756, Value Loss: 0.9234
...
Pre-training completed!
Pre-trained model saved to: pretrained_ppo_model.pt
Pre-training used 3426 historical experiences
```

The decreasing loss values indicate the agent is learning from historical data.

## How It Works

### 1. Data Loading
- Reads `episode_results.json` files from your experiment directories
- Extracts parameter combinations, performance metrics, and original rewards
- Converts parameters to the state/action format used by your RL agent

### 2. Reward Re-calculation
- Uses your current `get_reward()` function from `BenchmarkResult`
- Applies new baselines, scaling factors, and optimization metrics
- Preserves original performance metrics while updating reward values

### 3. Pre-training
- Creates training batches from historical experiences
- Trains the agent's policy and value networks using supervised learning
- Uses a reduced learning rate to avoid overfitting to historical data

### 4. Analysis
- Groups experiences by performance quartiles
- Identifies which parameters appear most frequently in high-performing configurations
- Provides statistical comparisons between different performance levels

## Tips for Best Results

### 1. Choose Good Historical Data
- Focus on longer experiments (500+ episodes) for more mature data
- Include experiments with different hyperparameters for diversity
- Avoid experiments that failed early or had technical issues

### 2. Reward Function Tuning
- Start with conservative reward scaling (0.5-2.0) to avoid extreme values
- Use appropriate baselines - either calculated from your data or known good values
- Consider which metric (`pp_tokens_per_sec` vs `tg_tokens_per_sec`) matters most for your use case

### 3. Pre-training Strategy
- Use 10-30 epochs for pre-training (more can lead to overfitting)
- Reduce learning rate during actual training when using pre-trained models
- Consider using fewer episodes in your main training since you start with experience

### 4. Combining Approaches
- Use analysis to understand patterns, then pre-train based on insights
- Start with pre-trained models and use different reward functions to fine-tune
- Save multiple pre-trained models with different optimization focuses

## Troubleshooting

### No Historical Data Found
- Check that `fs_results/` contains experiment directories
- Verify `episode_results.json` files exist in the experiment directories
- Use `--experiment-patterns` to target specific experiments

### Pre-training Fails
- Ensure you have successful experiences in your historical data
- Check that your virtual environment has all required packages
- Try reducing batch size or number of epochs

### Rewards Look Wrong
- Verify your baseline values are reasonable
- Check that your reward scaling factor isn't too extreme
- Compare with original rewards to ensure the direction makes sense

## Files Created

| File | Description |
|------|-------------|
| `processed_experiences.json` | Historical data with recalculated rewards |
| `pretrained_ppo_model.pt` | Pre-trained PPO agent |
| `pretrained_sac_model.pt` | Pre-trained SAC agent |
| `historical_experiences_recomputed.json` | Example output from analysis |

## Advanced Usage

### Custom Baselines
If you know your system's typical performance:

```bash
python experience_replay.py \
    --baseline-pp 10400 \
    --baseline-tg 148 \
    --reward-scaling 1.0
```

### Focus on Specific Experiments
To analyze only your best experiments:

```bash
python experience_replay.py \
    --experiment-patterns ppo_500ep_20250717 ppo_200ep_20250716 \
    --max-episodes-per-exp 300
```

### Different Optimization Targets
To optimize for text generation instead of prompt processing:

```bash
python experience_replay.py \
    --optimize-metric tg_tokens_per_sec \
    --reward-scaling 1.5 \
    --pretrain-agent
```

## Next Steps

1. **Experiment with reward functions** - Use the analysis to understand what works
2. **Pre-train agents** - Get a head start on training with your historical knowledge
3. **Iterate quickly** - Use pre-trained models to test new ideas faster
4. **Build experience databases** - Save processed experiences for future use

This system transforms your historical experiment data from static logs into active training resources, potentially reducing the time needed to find optimal configurations by 50-80%.
