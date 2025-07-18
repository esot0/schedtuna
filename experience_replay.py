#!/home/nvidia/rl_experiments/venv/bin/python3
"""
Experience Replay from Historical Data
=====================================

This script enables you to leverage data from previous experiments by:
1. Loading historical episode results from previous experiments
2. Re-calculating rewards using a new/updated reward function
3. Using this data to pre-train or warm-start your RL agent

This can significantly accelerate learning by giving the agent a head start
based on all the parameter combinations you've already tested.
"""

import os
import json
import numpy as np
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from main import ParamWrapper

# Import necessary classes from main.py
import sys
sys.path.append('/home/nvidia/rl_experiments')
from main import (
    BenchmarkResult, SchedulerParams, FlashySparkEnvironment,
    PPOAgent, SACAgent, DEVICE
)

@dataclass
class HistoricalExperience:
    """A single historical experience that can be used for training"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    params: SchedulerParams
    original_reward: float
    original_filename: str
    performance_metrics: Dict[str, float]
    success: bool

class ExperienceReplayManager:
    """Manages loading and processing historical experiment data"""

    def __init__(self, results_dir: str = "/home/nvidia/rl_experiments/fs_results"):
        self.results_dir = Path(results_dir)
        self.historical_experiences: List[HistoricalExperience] = []
        self.env = None

    def load_experiments(self, experiment_patterns: Optional[List[str]] = None, max_episodes_per_exp: Optional[int] = None) -> int:
        """
        Load historical data from previous experiments

        Args:
            experiment_patterns: List of experiment name patterns to include (None = all)
            max_episodes_per_exp: Maximum episodes to load per experiment (None = all)

        Returns:
            Number of experiences loaded
        """
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        # Find all experiment directories
        experiment_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]

        # Filter by patterns if specified
        if experiment_patterns:
            filtered_dirs = []
            for pattern in experiment_patterns:
                filtered_dirs.extend([d for d in experiment_dirs if pattern in d.name])
            experiment_dirs = filtered_dirs

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

                # Limit episodes if specified
                if max_episodes_per_exp:
                    episodes_data = episodes_data[:max_episodes_per_exp]

                for episode_data in episodes_data:
                    experience = self._convert_episode_to_experience(episode_data, exp_dir)
                    if experience:
                        self.historical_experiences.append(experience)
                        total_loaded += 1

            except Exception as e:
                print(f"Error loading {exp_dir.name}: {e}")
                continue

        print(f"Successfully loaded {total_loaded} historical experiences")
        return total_loaded

    def _convert_episode_to_experience(self, episode_data: Dict, exp_dir: Path) -> Optional[HistoricalExperience]:
        """Convert episode JSON data to HistoricalExperience"""
        try:
                        # Extract parameters - handle missing fields by filtering
            params_dict = episode_data.get('params', {})

            # Filter out parameters that don't exist in current SchedulerParams
            # This handles compatibility with older experiments
            valid_params = {}
            try:
                # Try to create a dummy SchedulerParams to see what fields are valid
                dummy_params = SchedulerParams()
                valid_fields = set(dummy_params.__dict__.keys())

                for key, value in params_dict.items():
                    if key in valid_fields:
                        # Ensure we convert to proper boolean
                        if isinstance(value, (bool, int, str)):
                            bool_value = bool(value) if not isinstance(value, str) else value.lower() == 'true'
                            valid_params[key] = bool_value

                params = SchedulerParams(**valid_params)

            except Exception as param_error:
                print(f"Error creating SchedulerParams: {param_error}")
                return None

            # Extract result data
            result_data = episode_data.get('result', {})
            if not result_data.get('success', False):
                return None  # Skip failed runs

            # Create BenchmarkResult for reward calculation
            benchmark_result = BenchmarkResult(
                pp_tokens_per_sec=result_data.get('pp_tokens_per_sec', 0),
                tg_tokens_per_sec=result_data.get('tg_tokens_per_sec', 0),
                success=result_data.get('success', False),
                error_msg=result_data.get('error_msg', ''),
                raw_output=result_data.get('raw_output', ''),
                execution_time=result_data.get('execution_time', 0)
            )

            # Get environment for state/action conversion (create if needed)
            if self.env is None:
                self.env = FlashySparkEnvironment()

            # Convert parameters to state and action using correct method names
            state = self.env._get_observation(params)
            action = self.env._normalize_params(params)  # Use correct method name

            # For next_state, we'll use the same state (since we don't have sequential data)
            # In actual RL training, next_state would be after the action is taken
            next_state = state.copy()

            return HistoricalExperience(
                state=state,
                action=action,
                reward=episode_data.get('reward', 0),  # Will be recalculated
                next_state=next_state,
                params=params,
                original_reward=episode_data.get('reward', 0),
                original_filename = exp_dir.name,
                performance_metrics={
                    'pp_tokens_per_sec': benchmark_result.pp_tokens_per_sec,
                    'tg_tokens_per_sec': benchmark_result.tg_tokens_per_sec,
                    'execution_time': benchmark_result.execution_time
                },
                success=benchmark_result.success
            )

        except Exception as e:
            print(f"Error converting episode data: {e}")
            return None

    def recalculate_rewards(self,
                          reward_function_params: Dict[str, Any],
                          baseline_pp: Optional[float] = None,
                          baseline_tg: Optional[float] = None) -> int:
        """
        Re-calculate rewards for all historical experiences using new reward function

        Args:
            reward_function_params: Parameters for the reward function
            baseline_pp: Baseline PP performance (calculated from data if None)
            baseline_tg: Baseline TG performance (calculated from data if None)

        Returns:
            Number of experiences with recalculated rewards
        """
        if not self.historical_experiences:
            print("No historical experiences loaded")
            return 0

        # Calculate baselines from data if not provided
        if baseline_pp is None or baseline_tg is None:
            pp_values = [exp.performance_metrics['pp_tokens_per_sec']
                        for exp in self.historical_experiences if exp.success]
            tg_values = [exp.performance_metrics['tg_tokens_per_sec']
                        for exp in self.historical_experiences if exp.success]

            if baseline_pp is None:
                baseline_pp = np.mean(pp_values) if pp_values else 10000.0
            if baseline_tg is None:
                baseline_tg = np.mean(tg_values) if tg_values else 140.0

        print(f"Using baselines - PP: {baseline_pp:.2f}, TG: {baseline_tg:.2f}")

        # Recalculate rewards
        recalculated = 0
        for experience in self.historical_experiences:
            if not experience.success:
                continue

            # Create BenchmarkResult for reward calculation
            result = BenchmarkResult(
                pp_tokens_per_sec=experience.performance_metrics['pp_tokens_per_sec'],
                tg_tokens_per_sec=experience.performance_metrics['tg_tokens_per_sec'],
                success=experience.success,
                error_msg='',
                raw_output='',
                execution_time=experience.performance_metrics['execution_time']
            )

            # Calculate new reward
            new_reward = result.get_reward(
                optimize_metric=reward_function_params.get('optimize_metric', 'pp_tokens_per_sec'),
                baseline_pp=baseline_pp,
                baseline_tg=baseline_tg,
                params=experience.params,
                reward_scaling=reward_function_params.get('reward_scaling', 1.0)
            )

            experience.reward = new_reward
            recalculated += 1

        print(f"Recalculated rewards for {recalculated} experiences")
        return recalculated

    def create_training_batches(self, batch_size: int = 32) -> List[Tuple]:
        """
        Create training batches from historical experiences

        Returns:
            List of (states, actions, rewards, next_states) tuples
        """
        if not self.historical_experiences:
            return []

        # Filter successful experiences
        valid_experiences = [exp for exp in self.historical_experiences if exp.success]

        # Create batches
        batches = []
        for i in range(0, len(valid_experiences), batch_size):
            batch_experiences = valid_experiences[i:i + batch_size]

            states = np.array([exp.state for exp in batch_experiences])
            actions = np.array([exp.action for exp in batch_experiences])
            rewards = np.array([exp.reward for exp in batch_experiences])
            next_states = np.array([exp.next_state for exp in batch_experiences])

            batches.append((states, actions, rewards, next_states))

        return batches

    def get_best_experiences(self, top_k: int = 100) -> List[HistoricalExperience]:
        """Get the top-k best performing experiences by reward"""
        valid_experiences = [exp for exp in self.historical_experiences if exp.success]
        return sorted(valid_experiences, key=lambda x: x.reward, reverse=True)[:top_k]

    def get_best_pp(self, top_k: int = 100) -> List[HistoricalExperience]:
        """Get the top-k best performing experiences by pp_tokens_per_sec"""
        valid_experiences = [exp for exp in self.historical_experiences if exp.success]
        return sorted(valid_experiences, key=lambda x: x.performance_metrics['pp_tokens_per_sec'], reverse=True)[:top_k]

    def get_best_tg(self, top_k: int = 100) -> List[HistoricalExperience]:
        """Get the top-k best performing experiences by tg_tokens_per_sec"""
        valid_experiences = [exp for exp in self.historical_experiences if exp.success]
        return sorted(valid_experiences, key=lambda x: x.performance_metrics['tg_tokens_per_sec'], reverse=True)[:top_k]

    def analyze_parameter_combinations(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze the most common parameter combinations in historical data

        Returns:
            Dictionary with statistics about parameter combinations
        """
        from collections import defaultdict
        def powerset(iterable, min_length: int = 1):
            """Generate all subsets of an iterable"""
            from itertools import chain, combinations
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(min_length, len(s)+1))

        param_combinations = defaultdict(list)
        # Group experiences by parameter combination
        for exp in self.historical_experiences:
            # Create a hashable representation of parameters
            parameters_on = powerset([p for p in asdict(exp.params).items() if p[1]], min_length=3)
            for parameter_combo in parameters_on:
               param_key = tuple(parameter_combo)
               param_combinations[param_key].append(exp)

        # Calculate statistics for each combination
        combination_stats = []
        for param_key, experiences in param_combinations.items():
            successful_exps = [exp for exp in experiences if exp.success]
            total_count = len(experiences)
            success_count = len(successful_exps)
            success_rate = success_count / total_count if total_count > 0 else 0

            if successful_exps:
                avg_reward = np.mean([exp.reward for exp in successful_exps])
                avg_pp = np.mean([exp.performance_metrics['pp_tokens_per_sec'] for exp in successful_exps])
                avg_tg = np.mean([exp.performance_metrics['tg_tokens_per_sec'] for exp in successful_exps])
                avg_exec_time = np.mean([exp.performance_metrics['execution_time'] for exp in successful_exps])

                combination_stats.append({
                    'param_tuple': param_key,
                    'total_count': total_count,
                    'success_count': success_count,
                    'success_rate': success_rate,
                    'avg_reward': avg_reward,
                    'avg_pp_tokens_per_sec': avg_pp,
                    'avg_tg_tokens_per_sec': avg_tg,
                    'avg_execution_time': avg_exec_time,
                    'experiments': [exp.original_filename for exp in experiences]
                })

        # Sort by frequency (total_count)
        combination_stats.sort(key=lambda x: x['total_count'], reverse=True)

        return {
            'total_combinations': len(combination_stats),
            'total_experiences': len(self.historical_experiences),
            'top_combinations': combination_stats[:top_k],
            'all_combinations': combination_stats
        }

    def print_parameter_combination_analysis(self, top_k: int = 10):
        """Print detailed analysis of parameter combinations"""
        stats = self.analyze_parameter_combinations(top_k)

        print(f"\n{'='*60}")
        print(f"PARAMETER COMBINATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Total unique parameter combinations: {stats['total_combinations']}")
        print(f"Total experiences analyzed: {stats['total_experiences']}")

        print(f"\nTop {top_k} Most Frequent Parameter Combinations:")
        print(f"{'='*80}")

        for i, combo in enumerate(stats['top_combinations'], 1):
            print(f"\n{i}. Frequency: {combo['total_count']} times")
            print(f"   Success Rate: {combo['success_rate']:.1%} ({combo['success_count']}/{combo['total_count']})")

            if combo['success_count'] > 0:
                print(f"   Performance Averages:")
                print(f"     Reward: {combo['avg_reward']:.2f}")
                print(f"     PP Tokens/sec: {combo['avg_pp_tokens_per_sec']:.1f}")
                print(f"     TG Tokens/sec: {combo['avg_tg_tokens_per_sec']:.1f}")
                print(f"     Execution Time: {combo['avg_execution_time']:.2f}s")

            print(f"   Parameters:")
            print(f"     {combo['param_tuple']}")
            # Show which experiments used this combination
            exp_files = list(set(combo['experiments']))  # Remove duplicates
            if len(exp_files) <= 3:
                print(f"   Used in experiments: {', '.join(exp_files)}")
            else:
                print(f"   Used in experiments: {', '.join(exp_files[:3])} ... (and {len(exp_files)-3} others)")

    def save_processed_data(self, output_file: str):
        """Save processed experiences for later use"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable_data = []
        for exp in self.historical_experiences:
            exp_data = {
                'state': exp.state.tolist(),
                'action': exp.action.tolist(),
                'reward': exp.reward,
                'next_state': exp.next_state.tolist(),
                'params': asdict(exp.params),
                'original_reward': exp.original_reward,
                'performance_metrics': exp.performance_metrics,
                'success': exp.success
            }
            serializable_data.append(exp_data)

        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"Saved {len(serializable_data)} processed experiences to {output_path}")




def pretrain_agent_with_historical_data(
    agent,
    experience_manager: ExperienceReplayManager,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate_multiplier: float = 0.1
) -> Dict[str, List[float]]:
    """
    Pre-train an agent using historical experiences with supervised learning

    Args:
        agent: The RL agent (PPO or SAC)
        experience_manager: Manager containing historical experiences
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate_multiplier: Scale learning rate for pre-training

    Returns:
        Training metrics
    """
    print("Starting pre-training with historical data...")

    # Get training batches
    batches = experience_manager.create_training_batches(batch_size)
    if not batches:
        print("No training batches available")
        return {}

    # Temporarily reduce learning rate for pre-training
    original_lr = agent.policy_optimizer.param_groups[0]['lr']
    pretrain_lr = original_lr * learning_rate_multiplier

    for param_group in agent.policy_optimizer.param_groups:
        param_group['lr'] = pretrain_lr

    metrics = {'policy_loss': [], 'value_loss': []}

    for epoch in range(num_epochs):
        epoch_policy_loss = []
        epoch_value_loss = []

        for states, actions, rewards, next_states in batches:
            # Convert to tensors
            states_tensor = torch.FloatTensor(states).to(DEVICE)
            actions_tensor = torch.FloatTensor(actions).to(DEVICE)
            rewards_tensor = torch.FloatTensor(rewards).to(DEVICE)
            next_states_tensor = torch.FloatTensor(next_states).to(DEVICE)

            # ===== SUPERVISED LEARNING APPROACH =====
            # For pretraining, we want the agent to learn to predict the historical actions
            # given the historical states (behavior cloning / imitation learning)

            # Policy Loss: Supervised learning to predict historical actions
            policy_output = agent.policy_net(states_tensor)

            if policy_output.shape[1] == agent.action_dim * 2:  # PPO/SAC style (mean + log_std)
                # Extract mean and log_std
                mean = policy_output[:, :agent.action_dim]
                log_std = policy_output[:, agent.action_dim:]
                std = torch.exp(log_std.clamp(-20, 2))  # Clamp for stability

                # Create Gaussian distribution
                from torch.distributions import Normal
                dist = Normal(mean, std)

                # Negative log likelihood loss (maximize likelihood of historical actions)
                policy_loss = -dist.log_prob(actions_tensor).sum(dim=-1).mean()

            else:  # Direct action prediction
                # MSE loss between predicted and historical actions
                policy_loss = torch.nn.functional.mse_loss(policy_output, actions_tensor)

            # Value Loss: Learn to predict rewards (value function learning)
            if hasattr(agent, 'value_net'):
                values = agent.value_net(states_tensor).squeeze()
                # Use rewards as targets for supervised learning
                value_loss = torch.nn.functional.mse_loss(values, rewards_tensor)
            else:
                value_loss = torch.tensor(0.0).to(DEVICE)

            # Update policy network
            agent.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 0.5)
            agent.policy_optimizer.step()

            # Update value network if it exists
            if hasattr(agent, 'value_net') and hasattr(agent, 'value_optimizer'):
                agent.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.value_net.parameters(), 0.5)
                agent.value_optimizer.step()

            epoch_policy_loss.append(policy_loss.item())
            epoch_value_loss.append(value_loss.item())

        # Track metrics
        avg_policy_loss = np.mean(epoch_policy_loss)
        avg_value_loss = np.mean(epoch_value_loss)

        metrics['policy_loss'].append(avg_policy_loss)
        metrics['value_loss'].append(avg_value_loss)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} - Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}")

    # Restore original learning rate
    for param_group in agent.policy_optimizer.param_groups:
        param_group['lr'] = original_lr

    print("Pre-training completed!")
    print(f"Final Policy Loss: {metrics['policy_loss'][-1]:.4f}")
    print(f"Final Value Loss: {metrics['value_loss'][-1]:.4f}")

    return metrics

def diagnose_pretraining_data(experience_manager: ExperienceReplayManager, agent):
    """
    Diagnose potential issues with pretraining data
    """
    print("\n=== PRETRAINING DATA DIAGNOSTICS ===")

    # Check if we have any valid experiences
    valid_experiences = [exp for exp in experience_manager.historical_experiences if exp.success]
    print(f"Valid experiences: {len(valid_experiences)}")

    if not valid_experiences:
        print("❌ No valid experiences found!")
        return

    # Get a sample batch to check data
    batches = experience_manager.create_training_batches(batch_size=min(32, len(valid_experiences)))
    if not batches:
        print("❌ No training batches created!")
        return

    states, actions, rewards, next_states = batches[0]

    print(f"📊 Data shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")

    print(f"📊 Data ranges:")
    print(f"  States - min: {states.min():.3f}, max: {states.max():.3f}")
    print(f"  Actions - min: {actions.min():.3f}, max: {actions.max():.3f}")
    print(f"  Rewards - min: {rewards.min():.3f}, max: {rewards.max():.3f}")

    # Check if actions are all the same (would cause 0 loss)
    unique_actions = np.unique(actions.reshape(-1, actions.shape[-1]), axis=0)
    print(f"📊 Unique action combinations: {len(unique_actions)}")

    if len(unique_actions) == 1:
        print("❌ All actions are identical! This will cause zero policy loss.")
        print(f"   Action: {unique_actions[0]}")

    # Test policy network output
    with torch.no_grad():
        states_tensor = torch.FloatTensor(states[:5]).to(DEVICE)
        policy_output = agent.policy_net(states_tensor)
        print(f"📊 Policy network output shape: {policy_output.shape}")
        print(f"   Expected: ({states.shape[0]}, {agent.action_dim * 2}) for PPO/SAC")

        if policy_output.shape[1] == agent.action_dim * 2:
            mean = policy_output[:, :agent.action_dim]
            log_std = policy_output[:, agent.action_dim:]
            print(f"   Mean range: {mean.min():.3f} to {mean.max():.3f}")
            print(f"   Log_std range: {log_std.min():.3f} to {log_std.max():.3f}")

    print("=== END DIAGNOSTICS ===\n")

def main():
    parser = argparse.ArgumentParser(description="Experience Replay from Historical Data")
    parser.add_argument("--results-dir", default="/home/nvidia/rl_experiments/fs_results",
                       help="Directory containing experiment results")
    parser.add_argument("--experiment-patterns", nargs="*",
                       help="Experiment name patterns to include (e.g., 'ppo_500ep')")
    parser.add_argument("--max-episodes-per-exp", type=int,
                       help="Maximum episodes to load per experiment")
    parser.add_argument("--optimize-metric", default="pp_tokens_per_sec",
                       choices=["pp_tokens_per_sec", "tg_tokens_per_sec"],
                       help="Metric to optimize for reward calculation")
    parser.add_argument("--reward-scaling", type=float, default=1.0,
                       help="Reward scaling factor")
    parser.add_argument("--baseline-pp", type=float, help="Baseline PP performance")
    parser.add_argument("--baseline-tg", type=float, help="Baseline TG performance")
    parser.add_argument("--output-file", default="processed_experiences.json",
                       help="File to save processed experiences")
    parser.add_argument("--pretrain-agent", action="store_true",
                       help="Pre-train a new agent with historical data")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo",
                       help="RL algorithm for pre-training")
    parser.add_argument("--pretrain-epochs", type=int, default=10,
                       help="Number of pre-training epochs")
    parser.add_argument("--param-combinations", type=int, default=10,
                       help="Number of top parameter combinations to analyze")
    parser.add_argument("--save-attempted-combinations", action="store_true",
                       help="Save attempted combinations to a file")
    parser.add_argument("--statistics", action="store_true",
                       help="Show statistics")
    parser.add_argument("--recalculate", action="store_true",
                       help="Recalculate rewards")

    args = parser.parse_args()

    # Initialize experience manager
    manager = ExperienceReplayManager(args.results_dir)

    # Load historical data
    print("Loading historical experiment data...")
    num_loaded = manager.load_experiments(
        experiment_patterns=args.experiment_patterns,
        max_episodes_per_exp=args.max_episodes_per_exp
    )

    if num_loaded == 0:
        print("No historical data loaded. Exiting.")
        return

    if args.recalculate:
        print("Recalculating rewards with new reward function...")
        reward_params = {
        'optimize_metric': args.optimize_metric,
        'reward_scaling': args.reward_scaling
        }
        manager.recalculate_rewards(
            reward_function_params=reward_params,
            baseline_pp=args.baseline_pp,
            baseline_tg=args.baseline_tg
        )

    if args.statistics:
        experiences = [exp for exp in manager.historical_experiences if exp.success]
        if experiences:
            rewards = [exp.reward for exp in experiences]
            original_rewards = [exp.original_reward for exp in experiences]

            print(f"\nReward Statistics:")
            print(f"Original rewards - Mean: {np.mean(original_rewards):.2f}, Std: {np.std(original_rewards):.2f}")
            print(f"New rewards - Mean: {np.mean(rewards):.2f}, Std: {np.std(rewards):.2f}")
            print(f"Best reward: {max(rewards):.2f}")

            def print_best(header: str, experiences: List[HistoricalExperience] = []):
                print(f"\n{header}\n{"=" * 10}\n")
                for i, exp in enumerate(experiences, 1):
                    print(f"{i}. Reward: {exp.reward:.2f}, PP: {exp.performance_metrics['pp_tokens_per_sec']:.1f}, "
                        f"TG: {exp.performance_metrics['tg_tokens_per_sec']:.1f}m, Params: {exp.params}, Filename: {exp.original_filename}, Execution Time: {exp.performance_metrics['execution_time']:.2f}s")


            # Show best configurations
            best_experiences = manager.get_best_experiences(top_k=5)
            print_best(f"Top 5 Rewards:", best_experiences)
            print_best(f"Top 5 PP:", manager.get_best_pp(top_k=5))
            print_best(f"Top 5 TG:", manager.get_best_tg(top_k=5))

            # Show parameter combination analysis
            manager.print_parameter_combination_analysis(top_k=args.param_combinations)

        # Save processed data
        manager.save_processed_data(args.output_file)

    if args.save_attempted_combinations:
        l = [exp.params for exp in manager.historical_experiences]
        ParamWrapper.save_attempted_combinations(l)
        print(len(ParamWrapper.list_unattempted_combinations()))
    # Pre-train agent if requested
    if args.pretrain_agent:
        print(f"\nInitializing {args.algorithm.upper()} agent for pre-training...")

        # Create dummy environment to get dimensions
        env = FlashySparkEnvironment()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Initialize agent
        if args.algorithm.lower() == "ppo":
            agent = PPOAgent(state_dim, action_dim, device=DEVICE)
        else:
            agent = SACAgent(state_dim, action_dim, device=DEVICE)

        # Run diagnostics to help debug any issues
        diagnose_pretraining_data(manager, agent)

        # Pre-train
        metrics = pretrain_agent_with_historical_data(
            agent, manager,
            num_epochs=args.pretrain_epochs,
            batch_size=32
        )

        # Save pre-trained model
        pretrained_model_path = f"pretrained_{args.algorithm}_model.pt"
        checkpoint = {
            'policy_net': agent.policy_net.state_dict(),
            'policy_optimizer': agent.policy_optimizer.state_dict(),
            'training_metrics': metrics,
            'num_historical_experiences': len(experiences)
        }

        if hasattr(agent, 'value_net'):
            checkpoint['value_net'] = agent.value_net.state_dict()
            checkpoint['value_optimizer'] = agent.value_optimizer.state_dict()

        torch.save(checkpoint, pretrained_model_path)
        print(f"Pre-trained model saved to: {pretrained_model_path}")

        print(f"Pre-training used {len(experiences)} historical experiences")
        print("You can now use this pre-trained model to start new training runs!")


if __name__ == "__main__":
    main()
