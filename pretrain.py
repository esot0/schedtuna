#!/usr/bin/env python3
"""
Pre-training and Analysis Tools for Scheduler Optimization
=========================================================

Utilities for analyzing historical experiment data and pre-training RL agents
using supervised learning on successful parameter combinations.

Features:
- Load and analyze historical experiment results
- Pre-train agents using behavior cloning
- Analyze parameter effectiveness patterns
- Support for multiple scheduler types
"""

import os
import sys
import json
from typing import Optional, List
from pathlib import Path

def replace_old_rewards(results_dir: str = "results") -> int:
        """
        Replace old reward values in historical experiment files with recalculated rewards.

        Args:
            results_dir: Directory containing experiment results

        Returns:
            Number of experiment files processed
        """
        from experience_replay import ExperienceReplayManager
        import numpy as np
        import shutil

        manager = ExperienceReplayManager()
        num_loaded = manager.load_experiments()

        if num_loaded == 0:
            print("No historical data available")
            return 0

        # Recalculate rewards
        reward_params = {'optimize_metric': 'pp_tokens_per_sec', 'reward_scaling': 1.0}
        manager.recalculate_rewards(reward_function_params=reward_params, baseline_pp=10200, baseline_tg=141)

        if not Path(results_dir).exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # Find all experiment directories
        experiment_dirs = [d for d in Path(results_dir).iterdir() if d.is_dir()]


        print(f"Found {len(experiment_dirs)} experiment directories to process")

        total_loaded = 0
        for exp_dir in experiment_dirs:
            episodes_file = exp_dir / "episode_results.json"
            if not episodes_file.exists():
                continue

            print(f"Loading data from: {exp_dir.name}")
            try:
                with open(episodes_file, 'r') as f:
                    episodes_data = json.load(f)

                for episode in episodes_data:
                    for exp in manager.historical_experiences:
                        if episode["reward"] == exp.original_reward:
                            episode["reward"] = exp.reward
                            break

                with open(episodes_file, 'w') as f:
                    json.dump(episodes_data, f, indent=4)

                total_loaded += 1
            except Exception as e:
                print(f"Error loading data from {exp_dir.name}: {e}")
                continue

        return total_loaded


#  1: Load and analyze historical data with new reward function
def load_and_recompute_rewards(scheduler_name: str = "scx_flashyspark"):
    """
    Load historical data and recalculate rewards with updated reward function.
    
    Args:
        scheduler_name: Name of the scheduler to analyze
    """
    print("=== 1: Load and Recompute Rewards ===")

    from experience_replay import ExperienceReplayManager

    # Initialize manager
    manager = ExperienceReplayManager()

    # Load data from
    num_loaded = manager.load_experiments()

    if num_loaded == 0:
        print("No data found. Make sure you have experiments in fs_results/")
        return

    # Recalculate rewards with new parameters
    reward_params = {
        'optimize_metric': 'pp_tokens_per_sec',  # Or 'tg_tokens_per_sec'
        'reward_scaling': 2.0  # Make rewards more extreme
    }

    manager.recalculate_rewards(
        reward_function_params=reward_params,
        baseline_pp=10200,  # Custom baseline - or None to auto-calculate
        baseline_tg=141     # Custom baseline - or None to auto-calculate
    )

    # Analyze the results
    best_configs = manager.get_best_experiences(top_k=10)
    print(f"\nTop 10 configurations with new reward function:")
    for i, exp in enumerate(best_configs, 1):
        params_summary = []
        for key, value in exp.params.to_dict().items():
            if value:  # Only show enabled parameters
                params_summary.append(key)

        print(f"{i:2d}. Reward: {exp.reward:6.1f} | PP: {exp.performance_metrics['pp_tokens_per_sec']:7.1f} | "
              f"TG: {exp.performance_metrics['tg_tokens_per_sec']:6.1f} | Enabled: {', '.join(params_summary[:3])}...")

    # Save processed data for later use
    manager.save_processed_data("historical_experiences_recomputed.json")
    print(f"\nSaved {len(manager.historical_experiences)} experiences to historical_experiences_recomputed.json")


def pretrain_agent(scheduler_name: str = "scx_flashyspark"):
    """
    Create and pre-train an agent using historical data.
    
    Args:
        scheduler_name: Name of the scheduler to train for
        
    Returns:
        Path to the saved pre-trained model
    """
    print("\n===  2: Pre-train Agent ===")

    from experience_replay import ExperienceReplayManager, pretrain_agent_with_historical_data
    from main import PPOAgent, SchedulerEnvironment, DEVICE

    # Load and process historical data
    manager = ExperienceReplayManager()
    num_loaded = manager.load_experiments()

    if num_loaded == 0:
        print("No historical data available for pre-training")
        return None

    # Recalculate rewards
    reward_params = {'optimize_metric': 'pp_tokens_per_sec', 'reward_scaling': 2.0}
    manager.recalculate_rewards(reward_function_params=reward_params)

    env = SchedulerEnvironment(scheduler_name=scheduler_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, device=DEVICE)

    # Pre-train the agent
    print(f"Pre-training agent with {len([e for e in manager.historical_experiences if e.success])} historical experiences...")

    metrics = pretrain_agent_with_historical_data(
        agent=agent,
        experience_manager=manager,
        num_epochs=20,
        batch_size=32,
        learning_rate_multiplier=1.0
    )
    # Save pre-trained model
    import torch
    model_path = "pretrained_agent_from_history.pt"
    checkpoint = {
        'policy_net': agent.policy_net.state_dict(),
        'policy_optimizer': agent.policy_optimizer.state_dict(),
        'value_net': agent.value_net.state_dict(),
        'value_optimizer': agent.value_optimizer.state_dict(),
        'training_metrics': metrics,
        'num_historical_experiences': len([e for e in manager.historical_experiences if e.success])
    }
    torch.save(checkpoint, model_path)
    print(f"Pre-trained model saved to: {model_path}")

    return model_path


def continue_training_with_pretrained_model(pretrained_model_path):
    """
    Start a new training run using a pre-trained model
    """
    print("\n===  3: Continue Training with Pre-trained Model ===")

    if not pretrained_model_path or not os.path.exists(pretrained_model_path):
        print("No pre-trained model available")
        return

    from main import train_rl_agent

    print(f"Starting training with pre-trained model: {pretrained_model_path}")

    # Start training with the pre-trained model
    # Note: You would need to modify the main.py train_rl_agent function to accept pretrained_model_path
    # For now, this shows the concept

    success = train_rl_agent(
        algorithm="ppo",
        episodes=2000,  # Fewer episodes needed since we start with experience
        baseline_runs=3,
        learning_rate=1e-6,  # Slightly lower learning rate for fine-tuning
        experiment_name="pretrained_continuation",
        optimize_metric="pp_tokens_per_sec",
        reward_scaling=1.0,
        # pretrained_model_path=pretrained_model_path  # Add this parameter
    )

    print(f"Training completed: {success}")


def analyze_historical_patterns(scheduler_name: str = "scx_flashyspark"):
    """
    Analyze patterns in historical data to understand parameter effectiveness.
    
    Args:
        scheduler_name: Name of the scheduler to analyze
    """
    print("\n===  4: Analyze Historical Patterns ===")

    from experience_replay import ExperienceReplayManager
    import numpy as np

    manager = ExperienceReplayManager()
    num_loaded = manager.load_experiments()

    if num_loaded == 0:
        print("No historical data available")
        return

    # Recalculate rewards
    reward_params = {'optimize_metric': 'pp_tokens_per_sec', 'reward_scaling': 1.0}
    manager.recalculate_rewards(reward_function_params=reward_params, baseline_pp=10200, baseline_tg=141)

    # Analyze parameter effectiveness
    successful_experiences = [exp for exp in manager.historical_experiences if exp.success]

    if not successful_experiences:
        print("No successful experiences found")
        return

    # Group by reward quartiles
    rewards = [exp.reward for exp in successful_experiences]
    q95 = np.percentile(rewards, 95)
    q50 = np.percentile(rewards, 50)
    q5 = np.percentile(rewards, 5)

    pp_q95 = np.percentile([exp.performance_metrics['pp_tokens_per_sec'] for exp in successful_experiences], 95)
    tg_q95 = np.percentile([exp.performance_metrics['tg_tokens_per_sec'] for exp in successful_experiences], 95)
    pp_q75 = np.percentile([exp.performance_metrics['pp_tokens_per_sec'] for exp in successful_experiences], 75)
    tg_q75 = np.percentile([exp.performance_metrics['tg_tokens_per_sec'] for exp in successful_experiences], 75)
    tg_q50 = np.percentile([exp.performance_metrics['tg_tokens_per_sec'] for exp in successful_experiences], 50)
    pp_q50 = np.percentile([exp.performance_metrics['pp_tokens_per_sec'] for exp in successful_experiences], 50)
    pp_q25 = np.percentile([exp.performance_metrics['pp_tokens_per_sec'] for exp in successful_experiences], 25)
    tg_q25 = np.percentile([exp.performance_metrics['tg_tokens_per_sec'] for exp in successful_experiences], 25)
    pp_q5 = np.percentile([exp.performance_metrics['pp_tokens_per_sec'] for exp in successful_experiences], 5)
    tg_q5 = np.percentile([exp.performance_metrics['tg_tokens_per_sec'] for exp in successful_experiences], 5)


    high_performers = [exp for exp in successful_experiences if exp.reward >= q95]
    medium_performers = [exp for exp in successful_experiences if q50 <= exp.reward < q95]
    low_performers = [exp for exp in successful_experiences if exp.reward <= q5]

    highest_performers_pp = [exp for exp in high_performers if exp.performance_metrics['pp_tokens_per_sec'] >= pp_q95]
    highest_performers_tg = [exp for exp in high_performers if exp.performance_metrics['tg_tokens_per_sec'] >= tg_q95]
    highest_performers_combined = [exp for exp in high_performers if exp.performance_metrics['pp_tokens_per_sec'] >= pp_q95 and exp.performance_metrics['tg_tokens_per_sec'] >= tg_q95]
    high_performers_pp = [exp for exp in high_performers if exp.performance_metrics['pp_tokens_per_sec'] >= pp_q75]
    medium_performers_pp = [exp for exp in medium_performers if exp.performance_metrics['pp_tokens_per_sec'] >= pp_q50 and exp.performance_metrics['pp_tokens_per_sec'] <= pp_q75]
    medium_performers_tg = [exp for exp in medium_performers if exp.performance_metrics['tg_tokens_per_sec'] >= tg_q50 and exp.performance_metrics['tg_tokens_per_sec'] <= tg_q75]
    high_performers_tg = [exp for exp in high_performers if exp.performance_metrics['tg_tokens_per_sec'] >= tg_q75 and exp.performance_metrics['tg_tokens_per_sec'] <= tg_q95]
    low_performers_pp = [exp for exp in low_performers if exp.performance_metrics['pp_tokens_per_sec'] <= pp_q25 and exp.performance_metrics['pp_tokens_per_sec'] >= pp_q5]
    low_performers_tg = [exp for exp in low_performers if exp.performance_metrics['tg_tokens_per_sec'] <= tg_q25 and exp.performance_metrics['tg_tokens_per_sec'] >= tg_q5]
    lowest_performers_pp = [exp for exp in low_performers if exp.performance_metrics['pp_tokens_per_sec'] <= pp_q5]
    lowest_performers_tg = [exp for exp in low_performers if exp.performance_metrics['tg_tokens_per_sec'] <= tg_q5]

    print(f"Performance Analysis ({len(successful_experiences)} total experiences):")
    print(f"High performers (top 5%): {len(high_performers)} experiences")
    print(f"Medium performers (50-95%): {len(medium_performers)} experiences")
    print(f"Low performers (bottom 5%): {len(low_performers)} experiences")

    print(f"Highest pp performers (top 5%): {len(highest_performers_pp)} experiences")
    print(f"Highest tg performers (top 5%): {len(highest_performers_tg)} experiences")
    print(f"Highest combined performers (top 5%): {len(highest_performers_combined)} experiences")
    print(f"High pp performers (top 25%): {len(high_performers_pp)} experiences")
    print(f"High tg performers (top 25%): {len(high_performers_tg)} experiences")
    print(f"Low pp performers (bottom 25%): {len(low_performers_pp)} experiences")
    print(f"Low tg performers (bottom 25%): {len(low_performers_tg)} experiences")
    print(f"Lowest pp performers (bottom 5%): {len(lowest_performers_pp)} experiences")
    print(f"Lowest tg performers (bottom 5%): {len(lowest_performers_tg)} experiences")

    # Analyze parameter patterns
    def analyze_parameter_group(experiences, group_name):
        if not experiences:
            return

        print(f"\n{group_name} parameter patterns:")
        param_counts = {}

        for exp in experiences:
            for param_name, param_value in exp.params.to_dict().items():
                if param_value:  # Parameter is enabled
                    param_counts[param_name] = param_counts.get(param_name, 0) + 1

        # Sort by frequency
        sorted_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)
        total = len(experiences)

        for param_name, count in sorted_params[:8]:  # Top 8 parameters
            percentage = (count / total) * 100
            print(f"  {param_name:20s}: {count:3d}/{total:3d} ({percentage:5.1f}%)")

    analyze_parameter_group(high_performers, "HIGH PERFORMERS")
    analyze_parameter_group(medium_performers, "MEDIUM PERFORMERS")
    analyze_parameter_group(low_performers, "LOW PERFORMERS")
    analyze_parameter_group(highest_performers_pp, "HIGHEST PERFORMERS PP")
    analyze_parameter_group(highest_performers_tg, "HIGHEST PERFORMERS TG")
    analyze_parameter_group(highest_performers_combined, "HIGHEST PERFORMERS COMBINED")
    analyze_parameter_group(high_performers_pp, "HIGH PERFORMERS PP")
    analyze_parameter_group(high_performers_tg, "HIGH PERFORMERS TG")
    analyze_parameter_group(medium_performers_pp, "MEDIUM PERFORMERS PP")
    analyze_parameter_group(medium_performers_tg, "MEDIUM PERFORMERS TG")
    analyze_parameter_group(low_performers_pp, "LOW PERFORMERS PP")
    analyze_parameter_group(low_performers_tg, "LOW PERFORMERS TG")
    analyze_parameter_group(lowest_performers_pp, "LOWEST PERFORMERS PP")
    analyze_parameter_group(lowest_performers_tg, "LOWEST PERFORMERS TG")
    # Performance statistics
    print(f"\nPerformance Statistics:")
    print(f"High performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in high_performers]):.1f}")
    print(f"High performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in high_performers]):.1f}")
    print(f"Low performers  - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in low_performers]):.1f}")
    print(f"Low performers  - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in low_performers]):.1f}")

    print(f"Highest pp performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in highest_performers_pp]):.1f}")
    print(f"Highest pp performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in highest_performers_pp]):.1f}")
    print(f"Highest tg performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in highest_performers_tg]):.1f}")
    print(f"Highest tg performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in highest_performers_tg]):.1f}")
    print(f"Highest combined performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in highest_performers_combined]):.1f}")
    print(f"Highest combined performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in highest_performers_combined]):.1f}")
    print(f"High pp performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in high_performers_pp]):.1f}")
    print(f"High pp performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in high_performers_pp]):.1f}")
    print(f"Medium pp performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in medium_performers_pp]):.1f}")
    print(f"Medium pp performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in medium_performers_pp]):.1f}")
    print(f"Medium tg performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in medium_performers_tg]):.1f}")
    print(f"Medium tg performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in medium_performers_tg]):.1f}")
    print(f"Low pp performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in low_performers_pp]):.1f}")
    print(f"Low pp performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in low_performers_pp]):.1f}")


    print(f"High tg performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in high_performers_tg]):.1f}")
    print(f"High tg performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in high_performers_tg]):.1f}")
    print(f"Low tg performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in low_performers_tg]):.1f}")
    print(f"Low tg performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in low_performers_tg]):.1f}")
    print(f"Lowest pp performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in lowest_performers_pp]):.1f}")
    print(f"Lowest pp performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in lowest_performers_pp]):.1f}")
    print(f"Lowest tg performers - Avg PP: {np.mean([e.performance_metrics['pp_tokens_per_sec'] for e in lowest_performers_tg]):.1f}")
    print(f"Lowest tg performers - Avg TG: {np.mean([e.performance_metrics['tg_tokens_per_sec'] for e in lowest_performers_tg]):.1f}")


def main():
    """
    Run all s to demonstrate the capabilities
    """

    try:
        #  1: Load and recompute rewards
        # load_and_recompute_rewards()

        # # #  2: Pre-train agent
        # pretrained_model_path = pretrain_agent()

        # # #  3: Continue training (conceptual)
        # continue_training_with_pretrained_model(pretrained_model_path)

        #  4: Analyze patterns
        print("Historical Data Experience Replay s")
        print("=" * 50)
        analyze_historical_patterns()

        # print("\n" + "=" * 50)
        # print("s completed successfully!")
        # print("\nNext steps:")

    except Exception as e:
        print(f"Error running s: {e}")
        print("Make sure you have historical experiment data in fs_results/")


if __name__ == "__main__":
    main()
