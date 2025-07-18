#!/home/nvidia/rl_experiments/venv/bin/python3
"""
Reinforcement Learning Optimizer for scx_flashyspark Scheduler Parameters
========================================================================

WHAT THIS PROGRAM DOES:
This program uses Artificial Intelligence (specifically "Reinforcement Learning")
to automatically find the best settings for a Linux scheduler called "scx_flashyspark".

Think of it like having an AI assistant that:
1. Tries different combinations of scheduler settings
2. Runs performance tests to see how fast the system runs
3. Learns which settings work better than others
4. Gradually gets better at picking good settings
5. Eventually finds the optimal configuration

REINFORCEMENT LEARNING EXPLAINED:
- Reinforcement Learning (RL) is like training a pet with rewards and punishments
- The AI "agent" tries different actions (scheduler settings)
- It gets "rewards" for good performance and "penalties" for bad performance
- Over time, it learns to pick actions that get higher rewards
- This is similar to how humans learn - through trial and error with feedback

KEY CONCEPTS:
- Agent: The AI that makes decisions about which settings to try
- Environment: The computer system running the scheduler and benchmarks
- State: Current information about system performance and settings
- Action: A choice of scheduler parameter values to test
- Reward: A score based on how well the system performed
- Episode: One complete test of a set of parameters

Usage:
    python rl_experiments/main.py --algorithm ppo --episodes 100
    python rl_experiments/main.py --algorithm sac --episodes 50 --baseline-runs 3
"""

import os
import sys
import json
import operator
import time
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import signal
import atexit
import random

# Add the scripts directory to the Python path
sys.path.insert(0, '/home/nvidia/scripts')

# GPU Memory Management Utilities
def setup_gpu_optimizations():
    """Setup GPU optimizations for better performance"""
    if not HAS_ML_LIBS or DEVICE is None or DEVICE.type != "cuda":
        return

    try:
        # Clear cache to start fresh
        torch.cuda.empty_cache()

        # Enable cuDNN auto-tuning for better performance
        torch.backends.cudnn.benchmark = True

        # Set memory fraction to prevent OOM (use 90% of available memory)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = int(total_memory * 0.1)  # Reserve 10%
        torch.cuda.set_per_process_memory_fraction(0.9)

        print(f"GPU optimizations enabled:")
        print(f"  - Memory fraction: 90% (~{(total_memory * 0.9) / 1e9:.1f} GB)")
        print(f"  - cuDNN auto-tuning: enabled")
        print(f"  - Cache cleared: ready for training")

    except Exception as e:
        print(f"Warning: Could not setup GPU optimizations: {e}")

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    if not HAS_ML_LIBS or DEVICE is None or DEVICE.type != "cuda":
        return "CPU mode"

    try:
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"GPU memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total"
    except:
        return "GPU memory info unavailable"

def cleanup_gpu_memory():
    """Clean up GPU memory to prevent accumulation"""
    if not HAS_ML_LIBS or DEVICE is None or DEVICE.type != "cuda":
        return

    try:
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    except:
        pass

try:
    import gymnasium as gym
    from gymnasium import spaces
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Normal
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_ML_LIBS = True
    HAS_MATPLOTLIB = True

    # GPU Detection and Setup
    if torch.cuda.is_available():
        print("GPU detected")
        DEVICE = torch.device("cuda")
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        DEVICE = torch.device("cpu")
        print("No GPU detected, using CPU")

except ImportError as e:
    print(f"Warning: Missing ML libraries - {e}")
    print("Install with: pip install gymnasium torch matplotlib seaborn")
    HAS_ML_LIBS = False
    HAS_MATPLOTLIB = False
    DEVICE = None

# Check for pandas separately since it's optional for some features
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Import our benchmark utility functions
try:
    from parse_benchmark_results import BenchmarkResultsParser
except ImportError:
    print("Warning: Could not import benchmark parser. Some functionality may be limited.")
    BenchmarkResultsParser = None


@dataclass
class SchedulerParams:
    """
    WHAT THIS IS: A container for all the settings we can adjust in the scheduler

    Think of this like the "control panel" for the scheduler with various switches/knobs.
    Each parameter is like a toggle switch that can be ON (True) or OFF (False).

    The AI will try different combinations of these switches to see which ones
    make the system run faster. There are 14 different switches, so there are
    2^14 = 16,384 possible combinations to explore!

    WHAT EACH SETTING DOES (in simple terms):
    """
    # Boolean flags for scheduler behavior (True = ON, False = OFF)
    slice_lag_scaling: bool = False    # Adjust timing based on system load
    rr_sched: bool = False            # Share CPU time equally between tasks
    no_builtin_idle: bool = False     # Disable automatic CPU selection
    local_pcpu: bool = False          # Prioritize certain types of tasks
    direct_dispatch: bool = False     # Skip the scheduling queue sometimes
    sticky_cpu: bool = False          # Keep tasks on the same CPU core
    stay_with_kthread: bool = False   # Keep tasks near kernel threads
    native_priority: bool = False     # Use Linux's default task priorities
    local_kthreads: bool = False      # Prioritize system tasks per CPU
    no_wake_sync: bool = False        # Change how tasks wake up other tasks
    aggressive_gpu_tasks: bool = False # Give GPU tasks the fastest CPU cores
    timer_kick: bool = False          # Use different timing mechanism
    params = {}

    # def __init__(self, **kwargs):
    #     # Initialize all parameters with default values first
    #     self.slice_lag_scaling = False
    #     self.rr_sched = False
    #     self.no_builtin_idle = False
    #     self.local_pcpu = False
    #     self.direct_dispatch = False
    #     self.sticky_cpu = False
    #     self.stay_with_kthread = False
    #     self.native_priority = False
    #     self.local_kthreads = False
    #     self.no_wake_sync = False
    #     self.aggressive_gpu_tasks = False
    #     self.timer_kick = False
    #     self.params = {}

    #     # Now update with provided kwargs
    #     for key, value in kwargs.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #             self.params[key] = value
    #         else:
    #             raise ValueError(f"Invalid parameter: {key}")

    def __hash__(self):
        return hash(json.dumps(asdict(self), indent=4))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return json.dumps(asdict(self), indent=4, sort_keys=True)

    def _to_command_args(self) -> List[str]:
        """Convert parameters to command line arguments for scx_flashyspark"""
        args = []

        # Add boolean flags
        if self.slice_lag_scaling:
            args.append("--slice-lag-scaling")
        if self.rr_sched:
            args.append("--rr-sched")
        if self.no_builtin_idle:
            args.append("--no-builtin-idle")
        if self.local_pcpu:
            args.append("--local-pcpu")
        if self.direct_dispatch:
            args.append("--direct-dispatch")
        if self.sticky_cpu:
            args.append("--sticky-cpu")
        if self.stay_with_kthread:
            args.append("--stay-with-kthread")
        if self.native_priority:
            args.append("--native-priority")
        if self.local_kthreads:
            args.append("--local-kthreads")
        if self.no_wake_sync:
            args.append("--no-wake-sync")
        if self.aggressive_gpu_tasks:
            args.append("--aggressive-gpu-tasks")
        if self.timer_kick:
            args.append("--timer-kick")
        return args

class ParamWrapper:
    @staticmethod
    def append_attempted_combinations(params):
        """Append a parameter combination to the attempted combinations file"""

        # need to debug (not really appending)
        try:
            with open("attempted_combinations.log", "r") as f:
                content = f.read()
                if content:
                    p = eval(content)
                else:
                    p = set()
        except FileNotFoundError:
            p = set()
        except Exception:
            p = set()  # If file is corrupted, start fresh

        # Add new combinations
        if isinstance(params, SchedulerParams):
            p.add(str(params))
        elif isinstance(params, str):
            p.add(params)
        elif isinstance(params, set):
            p.update(params)
        else:
            raise ValueError(f"Invalid parameter type: {type(params)}")

        # Write back to file
        with open("attempted_combinations.log", "w") as f:
            f.write(str(p))

    @staticmethod
    def save_attempted_combinations(params: list[SchedulerParams], filename: Optional[str] = "attempted_combinations"):

        """
        Save attempted parameter combinations to a file. Currently only tested for boolean parameters.
        """
        attempted_combinations = set()

        for p in params:
            attempted_combinations.add(str(p))

        with open("attempted_combinations.log", "w") as f:
            f.write(str(attempted_combinations))

    @staticmethod
    def list_all_combinations(params: list[str] = []):
        from itertools import product
        """List all possible combinations of the boolean parameters"""
                # Use the same boolean parameter list as defined in FlashySparkEnvironment
        bool_params = [
            'slice_lag_scaling', 'rr_sched', 'no_builtin_idle',
            'local_pcpu', 'direct_dispatch', 'sticky_cpu', 'stay_with_kthread',
            'native_priority', 'local_kthreads', 'no_wake_sync', 'aggressive_gpu_tasks',
            'timer_kick'
        ]

        all_combinations = set()
        for p in product([True, False], repeat=len(bool_params)):
            param_dict = dict(zip(bool_params, p))
            key = json.dumps(param_dict, indent=4, sort_keys=True)
            all_combinations.add(key)
        return all_combinations

    @staticmethod
    def list_unattempted_combinations():
        """
        List all parameter combinations that have not been attempted. Currently only works for boolean parameters.
        """
        from itertools import product
        try:
            with open("attempted_combinations.log", "r") as f:
                attempted_combinations = eval(f.read())
        except FileNotFoundError:
            attempted_combinations = set()

        all_combinations = ParamWrapper.list_all_combinations()
        # Calculate unattempted combinations
        unattempted_combinations = all_combinations - attempted_combinations

        with open("unchecked_combinations", "w") as f:
            f.write(str(unattempted_combinations))

        return unattempted_combinations

    @staticmethod
    def generate_all_combinations(params: list[SchedulerParams]):
        """Generate all possible combinations of the boolean parameters"""
        from itertools import product

    @staticmethod
    def get_unattempted_combinations():
        """Get all unchecked combinations"""
        try:
            with open("unchecked_combinations", "r") as f:
                unchecked_combinations = eval(f.read())
        except FileNotFoundError:
            unchecked_combinations = set()
        return unchecked_combinations

    @staticmethod
    def get_attempted_combinations(params: list[SchedulerParams]):
        """Get all attempted combinations"""
        try:
            with open("attempted_combinations.log", "r") as f:
                attempted_combinations = eval(f.read())
        except FileNotFoundError:
            attempted_combinations = set()
        return attempted_combinations

    @staticmethod
    def params_as_list(params: SchedulerParams ):
        """Convert a SchedulerParams object to a list"""
        return [""]

@dataclass
class BenchmarkResult:
    """
    WHAT THIS IS: The results from testing how fast the system runs

    When we test a set of scheduler parameters, we run a benchmark (performance test)
    to see how fast the system can process text using a language model (like ChatGPT).
    This class stores all the information about how that test went.

    WHAT THE METRICS MEAN:
    - pp_tokens_per_sec: How fast we can process prompts (input text)
    - tg_tokens_per_sec: How fast we can generate responses (output text)
    - Higher numbers = better performance = faster system

    Think of it like testing a car: pp speed is how fast it accelerates,
    tg speed is how fast it can cruise. We want both to be as high as possible!
    """
    pp_tokens_per_sec: float = 0.0    # Prompt processing speed (higher = better)
    tg_tokens_per_sec: float = 0.0    # Text generation speed (higher = better)
    success: bool = False             # Did the test complete without errors?
    error_msg: str = ""               # What went wrong if it failed?
    raw_output: str = ""              # The raw text output from the benchmark
    execution_time: float = 0.0       # How long the test took to run

    def get_reward(self, optimize_metric: str = "pp", baseline_pp: Optional[float] = None,
                   baseline_tg: Optional[float] = None, params: Optional['SchedulerParams'] = None,
                   reward_scaling: float = 1.0) -> float:
        """
        ENHANCED REWARD FUNCTION: Calculates a more informative "reward score" for the AI

        This enhanced version provides stronger gradients and better discrimination
        between configurations to help the RL agent learn more effectively.

        Args:
            optimize_metric: Whether to focus on "pp" (prompt) or "tg" (text generation) speed
            baseline_pp: Baseline PP performance for normalization
            baseline_tg: Baseline TG performance for normalization
            params: The scheduler settings that were tested
            reward_scaling: Multiplier to make rewards more/less extreme
        """
        if not self.success:
            # Graduated penalties based on error type
            if "timeout" in self.error_msg.lower():
                return -50.0  # Moderate penalty for timeouts
            elif "failed to start" in self.error_msg.lower():
                return -200.0  # High penalty for configuration errors
            else:
                return -100.0  # Standard penalty for other failures

        # Use reasonable defaults if no baseline provided
        pp_baseline = baseline_pp if baseline_pp is not None and baseline_pp > 0 else 10000.0
        tg_baseline = baseline_tg if baseline_tg is not None and baseline_tg > 0 else 140.0

        # Calculate raw performance ratios
        pp_ratio = self.pp_tokens_per_sec / pp_baseline
        tg_ratio = self.tg_tokens_per_sec / tg_baseline

        # Use logarithmic scaling for better gradient discrimination
        # This provides stronger signals for improvements and avoids reward plateaus
        pp_log_reward = np.log(max(pp_ratio, 0.1)) * 50.0  # Log scaling amplifies differences
        tg_log_reward = np.log(max(tg_ratio, 0.1)) * 50.0

        # Weight the primary metric more heavily
        if optimize_metric == "pp_tokens_per_sec":
            primary_reward = pp_log_reward * 4.0      # 4x weight for primary metric
            secondary_reward = tg_log_reward * 1.0    # 1x weight for secondary
        else:  # optimize_metric == "tg"
            primary_reward = tg_log_reward * 4.0      # 4x weight for primary metric
            secondary_reward = pp_log_reward * 1.0    # 1x weight for secondary

        # Base reward from performance
        base_reward = primary_reward + secondary_reward

        # Enhanced execution time bonus/penalty with more discrimination
        if self.execution_time > 0:
            if self.execution_time < 9.25:
                base_reward += 10.0    # Significant bonus for very fast execution
            elif self.execution_time < 9.35:
                base_reward += 5.0     # Medium bonus for fast execution
            elif self.execution_time > 9.45:
                base_reward -= 10.0    # Penalty for slow execution
 # Higher penalty for very slow execution

        # Bonus for achieving high absolute performance (not just relative)
        absolute_performance_bonus = 0.0
        if self.pp_tokens_per_sec > pp_baseline * 1.1:  # 10% improvement
            absolute_performance_bonus += 15.0
        if self.tg_tokens_per_sec > tg_baseline * 1.1:  # 10% improvement
            absolute_performance_bonus += 15.0

        base_reward += absolute_performance_bonus

        # Bonus for balanced performance (both metrics improved)
        if pp_ratio > 1.0 and tg_ratio > 1.0:
            balance_bonus = min(pp_ratio, tg_ratio) * 10.0  # Reward balanced improvement
            base_reward += balance_bonus

        # Apply scaling
        final_reward = base_reward * reward_scaling

        # More conservative clipping to prevent extreme values while preserving gradients
        #final_reward = max(-300.0, min(300.0, final_reward))

        return final_reward

    def _calculate_parameter_bonus(self, params: 'SchedulerParams') -> float:
        """Calculate bonus/penalty based on parameter combinations"""
        bonus = 0.0


        if params.direct_dispatch and params.no_builtin_idle:
            bonus -= 100.0  # Potentially problematic combination

        # Too many aggressive features enabled
        aggressive_count = sum([
            params.aggressive_gpu_tasks,
            params.direct_dispatch,
            params.slice_lag_scaling,
            params.timer_kick,
        ])
        if aggressive_count >= 4:
            bonus -= 200.0  # Penalty for too many aggressive features

        # Reward certain beneficial combinations
        if params.local_pcpu and params.sticky_cpu:
            bonus += 50.0  # Good for latency-sensitive workloads

        if params.native_priority and params.local_kthreads:
            bonus += 30.0  # Good for system responsiveness

        # Bonus for conservative configurations that typically work well
        conservative_features = sum([
            params.local_pcpu,
            params.sticky_cpu,
            params.local_kthreads,
            params.native_priority
        ])
        if conservative_features >= 3 and aggressive_count <= 1:
            bonus += 75.0  # Bonus for balanced conservative approach

        return bonus


class FlashySparkEnvironment(gym.Env):
    """
    WHAT THIS IS: The "Environment" where the AI operates

    In reinforcement learning, we need an "environment" for the AI agent to interact with.
    Think of this like a video game world or a simulation where the AI can take actions
    and see what happens.

    OUR ENVIRONMENT IS:
    - The computer system running the scheduler
    - The AI can change scheduler settings (actions)
    - The AI observes performance metrics (state)
    - The AI gets reward/penalty based on performance

    ANALOGY: Imagine teaching someone to drive a car
    - Environment = the car and road
    - Actions = steering wheel, gas pedal, brake
    - State = speed, direction, obstacles ahead
    - Reward = +1 for staying on road, -10 for crashing

    OUR VERSION:
    - Actions = choosing scheduler parameter combinations
    - State = current performance metrics and settings
    - Reward = based on how fast the system runs
    - Goal = find the parameter combination that makes the system fastest
    """

    def __init__(self, model_path: Optional[str] = None, timeout: int = 300, optimize_metric: str = "pp_tokens_per_sec", reward_scaling: float = 1.0):
        super().__init__()

        self.model_path = model_path or "/home/nvidia/llama.cpp/models/Llama-3.2-1B-Instruct-Q6_K.gguf"
        self.llama_bench = "/home/nvidia/llama.cpp/build/bin/llama-bench"
        self.scheduler_path = "/home/nvidia/bin/scx_flashyspark"
        self.timeout = timeout
        self.optimize_metric = optimize_metric  # "pp" or "tg"
        self.reward_scaling = reward_scaling

        # Track current scheduler process for cleanup
        self.current_scheduler_process = None
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.best_params = None

        # Fixed baseline performance (calculated once and reused throughout training)
        self.fixed_baseline_pp = None
        self.fixed_baseline_tg = None
        self.baseline_calculated = False

        # Boolean parameters only (12 parameters)
        self.bool_params = [
            'slice_lag_scaling', 'rr_sched', 'no_builtin_idle',
            'local_pcpu', 'direct_dispatch', 'sticky_cpu', 'stay_with_kthread',
            'native_priority', 'local_kthreads', 'no_wake_sync', 'aggressive_gpu_tasks',
             'timer_kick',
        ]

        # Define action space: discrete values for boolean flags only
        # 13 boolean parameters
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Observation space: current parameters + recent performance metrics
        # 12 params + 5 performance metrics (reward, pp_tokens/s, tg_tokens/s, success_rate, execution_time)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(17,), dtype=np.float32
        )

        # Performance history for observation state (but NOT for baseline calculation)
        self.performance_history = []
        self.max_history = 10

        # Simple parameter combination tracking
        self.unattempted_params = ParamWrapper.list_unattempted_combinations()  # Store SchedulerParams objects for ParamWrapper.save_attempted_combinations

        # Setup signal handlers for cleanup
        self._setup_cleanup()

    def _setup_cleanup(self):
        """Setup cleanup handlers"""
        def cleanup_handler(signum, frame):
            self._cleanup()
            sys.exit(1)

        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        atexit.register(self._cleanup)

    def _cleanup(self):
        """Clean up any running scheduler processes"""
        if self.current_scheduler_process:
            try:
                self.current_scheduler_process.terminate()
                self.current_scheduler_process.wait(timeout=5)
            except:
                try:
                    self.current_scheduler_process.kill()
                except:
                    pass
            finally:
                self.current_scheduler_process = None

        # Kill any remaining scheduler processes
        try:
            subprocess.run(["sudo", "pkill", "-f", "scx_flashyspark"],
                         capture_output=True, timeout=10)
        except:
            pass

    def _normalize_params(self, params: SchedulerParams) -> np.ndarray:
        """Normalize scheduler parameters to [-1, 1] range"""
        normalized = []

        # Add boolean parameters as -1/1
        for param_name in self.bool_params:
            value = getattr(params, param_name)
            normalized.append(1.0 if value else -1.0)

        return np.array(normalized, dtype=np.float32)

    def _denormalize_action(self, action: np.ndarray) -> SchedulerParams:
        """Convert normalized action to SchedulerParams"""
        params = SchedulerParams()

        # Set boolean parameters
        for i, param_name in enumerate(self.bool_params):
            value = action[i] > 0.0  # Convert to boolean
            setattr(params, param_name, value)

        return params

    def _get_observation(self, params: SchedulerParams) -> np.ndarray:
        """Get current observation state"""
        # Start with normalized parameters
        obs = self._normalize_params(params).tolist()

        # Add performance metrics from recent history
        if self.performance_history:
            recent_performance = self.performance_history[-5:]  # Last 5 runs
            avg_reward = np.mean([p['reward'] for p in recent_performance])
            avg_pp_tokens = np.mean([p['pp_tokens_per_sec'] for p in recent_performance])
            avg_tg_tokens = np.mean([p['tg_tokens_per_sec'] for p in recent_performance])
            success_rate = np.mean([p['success'] for p in recent_performance])
            avg_exec_time = np.mean([p['execution_time'] for p in recent_performance])

            # Normalize performance metrics
            obs.extend([
                np.tanh(avg_reward / 500.0),           # Reward (higher scale for tokens/s)
                np.tanh(avg_pp_tokens / 500.0),        # PP tokens/s
                np.tanh(avg_tg_tokens / 50.0),         # TG tokens/s (typically lower)
                2.0 * success_rate - 1.0,             # Success rate [-1,1]
                np.tanh(avg_exec_time / 120.0)         # Execution time
            ])
        else:
            # No history yet, use neutral values
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment to start of new episode"""
        super().reset(seed=seed)

        self._cleanup()  # Ensure clean state
        self.episode_count += 1

        # Start with default parameters
        default_params = SchedulerParams()
        observation = self._get_observation(default_params)

        return observation, {}

    def step(self, action):
        """Execute one step in the environment - use RL guidance to select from unattempted combinations"""
        # Convert agent's action to target parameters (what the agent wants to try)
        agent_target_params = self._denormalize_action(action)

        # Find the closest unattempted combination to what the agent wants
        closest_unattempted = self._find_closest_unattempted_params(agent_target_params)

        if closest_unattempted is not None:
            # Use the closest unattempted combination to the agent's preference
            params = closest_unattempted
            print(f"Using RL-guided unattempted combination")
        else:
            # Fall back to agent's exact choice if no unattempted combinations left
            params = agent_target_params
            print("No unattempted combinations left, using agent's exact choice")

        # Run benchmark with these parameters
        result = self._run_benchmark(params)

        # Calculate baseline performance for relative scoring
        baseline_pp = self.fixed_baseline_pp
        baseline_tg = self.fixed_baseline_tg

        # Calculate reward with enhanced discrimination
        reward = result.get_reward(self.optimize_metric, baseline_pp, baseline_tg, params, self.reward_scaling)

        # Update performance history
        perf_record = {
            'reward': reward,
            'pp_tokens_per_sec': result.pp_tokens_per_sec,
            'tg_tokens_per_sec': result.tg_tokens_per_sec,
            'success': result.success,
            'execution_time': result.execution_time,
            'params': asdict(params),
            'baseline_performance': baseline_pp # Store baseline for reward calculation
        }
        self.performance_history.append(perf_record)

        # Keep history bounded
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)

        # Track best performance
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params

        # Get next observation
        observation = self._get_observation(params)

        # Episode is never really "done" in this continuous optimization
        terminated = False
        truncated = False

        info = {
            'reward': reward,
            'params': asdict(params),
            'result': asdict(result),
            'best_reward': self.best_reward,
            'episode': self.episode_count
        }

        return observation, reward, terminated, truncated, info

    def _run_benchmark(self, params: SchedulerParams) -> BenchmarkResult:
        """Run llama-bench with given scheduler parameters"""
        result = BenchmarkResult()

        try:
            print(f"Testing parameters: {params}")

            # Start scheduler with parameters
            scheduler_cmd = ["sudo", self.scheduler_path] + params._to_command_args()
            print(f"Starting scheduler: {' '.join(scheduler_cmd)}")

            start_time = time.time()

            # Start the scheduler
            self.current_scheduler_process = subprocess.Popen(
                scheduler_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for scheduler to initialize
            time.sleep(3)

            # Check if scheduler started successfully
            if self.current_scheduler_process.poll() is not None:
                stdout, stderr = self.current_scheduler_process.communicate()
                result.error_msg = f"Scheduler failed to start: {stderr}"
                result.execution_time = time.time() - start_time
                return result

            # Run llama-bench
            print("Running llama-bench...")
            bench_result = subprocess.run(
                [self.llama_bench, "-m", self.model_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            result.execution_time = time.time() - start_time
            result.raw_output = bench_result.stdout

            if bench_result.returncode == 0:
                # Parse benchmark output for metrics
                result.success = True
                result.pp_tokens_per_sec, result.tg_tokens_per_sec = self._parse_benchmark_output(bench_result.stdout)
            else:
                result.error_msg = f"Benchmark failed: {bench_result.stderr}"

        except subprocess.TimeoutExpired:
            result.error_msg = "Benchmark timed out"
            result.execution_time = self.timeout

        except Exception as e:
            result.error_msg = f"Unexpected error: {str(e)}"
            result.execution_time = time.time() - start_time if 'start_time' in locals() else 0

        finally:
            # Always cleanup scheduler
            self._cleanup()

        return result

    def _parse_benchmark_output(self, output: str) -> Tuple[float, float]:
        """Parse llama-bench output to extract pp and tg metrics from table format

        Expected format:
        | model                    | size | params | backend | ngl | test | t/s |
        | llama 70B IQ4_NL         | ... | ... | CUDA | 99 | pp512 | 257.21 ± 10.67 |
        | llama 70B IQ4_NL         | ... | ... | CUDA | 99 | tg128 | 5.09 ± 0.32 |
        """
        pp_tokens_per_sec = 0.0
        tg_tokens_per_sec = 0.0

        try:
            import re

            for line_num, line in enumerate(output.split('\n')):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Skip header line (contains 'model' and 'test' etc)
                if 'model' in line.lower() and 'test' in line.lower():
                    continue

                # Skip separator lines (mostly dashes)
                if re.match(r'^\s*\|[\s\-\:]*\|', line):
                    continue

                # Skip build info and other non-table lines
                if not line.startswith('|') or line.count('|') < 6:
                    continue

                # Process data lines in table format
                parts = [p.strip() for p in line.split('|')]

                if len(parts) >= 8:  # Ensure we have enough columns (including empty strings)
                    test_type = parts[6].strip()  # test column
                    tokens_per_sec_str = parts[7].strip()  # t/s column


                    # Extract numeric value (ignore ± error if present)
                    # Look for pattern like "10142.17 ± 193.31" or just "10142.17"
                    match = re.search(r'(\d+(?:\.\d+)?)', tokens_per_sec_str)
                    if match:
                        tokens_per_sec = float(match.group(1))

                        # Determine if this is pp (prompt processing) or tg (text generation)
                        if 'pp' in test_type.lower():
                            pp_tokens_per_sec = max(pp_tokens_per_sec, tokens_per_sec)
                        elif 'tg' in test_type.lower():
                            tg_tokens_per_sec = max(tg_tokens_per_sec, tokens_per_sec)
        except Exception as e:
            print(f"Warning: Error parsing benchmark output: {e}")
            print(f"Output was: {output[:500]}...")  # Show first 500 chars for debugging

        print(f"Parsed performance: PP={pp_tokens_per_sec:.2f} tokens/sec, TG={tg_tokens_per_sec:.2f} tokens/sec")
        return pp_tokens_per_sec, tg_tokens_per_sec

    def set_fixed_baseline(self, baseline_pp: float, baseline_tg: float):
        """Set the fixed baseline values that will be used throughout training"""
        self.fixed_baseline_pp = baseline_pp
        self.fixed_baseline_tg = baseline_tg
        self.baseline_calculated = True
        print(f"Fixed baseline set: PP={baseline_pp:.2f} tokens/sec, TG={baseline_tg:.2f} tokens/sec")

    def _convert_string_to_params(self, params_str: str) -> SchedulerParams:
        """Convert a parameter string back to SchedulerParams object"""
        try:
            params_dict = json.loads(params_str)
            params = SchedulerParams()
            for key, value in params_dict.items():
                if hasattr(params, key):
                    setattr(params, key, bool(value))
            return params
        except Exception as e:
            print(f"Warning: Could not convert params string to SchedulerParams: {e}")
            # Return default params as fallback
            return SchedulerParams()

    def _find_closest_unattempted_params(self, target_params: SchedulerParams) -> Optional[SchedulerParams]:
        """Find and remove the closest unattempted combination to the target"""
        if not self.unattempted_params:
            return None

        target_vector = self._normalize_params(target_params)
        best_match = None
        best_distance = float('inf')
        best_params_str = None

        for params_str in self.unattempted_params:
            candidate_params = self._convert_string_to_params(params_str)
            candidate_vector = self._normalize_params(candidate_params)

            distance = np.linalg.norm(target_vector - candidate_vector)

            if distance < best_distance:
                best_distance = distance
                best_match = candidate_params
                best_params_str = params_str

        if best_params_str:
            self.unattempted_params.remove(best_params_str)
            print(f"Selected closest unattempted combination (distance: {best_distance:.3f}). Remaining: {len(self.unattempted_params)}")

        return best_match


# ============================================================================
# NEURAL NETWORK CLASSES - THE AI'S "BRAIN"
# ============================================================================

class MLPNetwork(nn.Module):
    """
    WHAT THIS IS: The AI's "brain" - a neural network with GPU acceleration

    A neural network is loosely inspired by how neurons work in a real brain.
    It's made of layers of artificial "neurons" that process information.

    HOW IT WORKS:
    1. Input layer receives information (like performance metrics)
    2. Hidden layers process and transform that information
    3. Output layer produces decisions (like which parameters to try)

    ANALOGY: Think of it like a decision-making committee:
    - Input layer = information gatherers
    - Hidden layers = advisors who discuss and analyze
    - Output layer = final decision makers

    Each "neuron" is like a person who:
    - Receives information from others
    - Applies their own judgment (weights/biases)
    - Passes their conclusion to the next layer

    The AI learns by adjusting these "judgments" based on rewards/penalties.

    GPU ACCELERATION: When available, all computations run on GPU for 5-10x speedup!
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 256], device=None):
        super().__init__()

        self.device = device or DEVICE or torch.device("cpu")

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Move to GPU if available
        self.to(self.device)
        if self.device.type == "cuda":
            print(f"Neural network moved to GPU: {self.device}")

    def forward(self, x):
        # Ensure input is on the correct device
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        return self.network(x)


class PPOAgent:
    """
    WHAT THIS IS: One type of AI algorithm called "PPO" (Proximal Policy Optimization) with GPU acceleration

    PPO is a specific method for training AI agents. Think of it as a "learning strategy"
    that tells the AI how to improve based on experience.

    WHY PPO IS GOOD:
    - Stable: doesn't make wild changes that break learning
    - Efficient: learns from experience without wasting data
    - Proven: works well for many different types of problems

    HOW PPO WORKS (simplified):
    1. Try some actions and see what rewards you get
    2. Figure out which actions led to good rewards
    3. Adjust your decision-making to do more of the good actions
    4. But don't change too drastically at once (that's the "proximal" part)

    THE AI HAS TWO "BRAINS":
    1. Policy Network: "What action should I take?" (the decision maker)
    2. Value Network: "How good is my current situation?" (the evaluator)

    ANALOGY: Like a student with a tutor
    - Student (policy) tries different study methods
    - Tutor (value) evaluates how well each method works
    - Both learn together to find the best study strategy

    GPU ACCELERATION: Neural networks run on GPU for 5-10x faster training!
    """

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, device=None):
        if not HAS_ML_LIBS:
            raise ImportError("ML libraries required for PPO agent")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or DEVICE or torch.device("cpu")

        # Policy network (actor) - moved to GPU
        self.policy_net = MLPNetwork(state_dim, action_dim * 2, device=self.device)  # mean and std for each action
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Value network (critic) - moved to GPU
        self.value_net = MLPNetwork(state_dim, 1, device=self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # PPO hyperparameters - tuned for better learning and faster convergence
        self.clip_ratio = 0.2
        self.value_loss_coeff = 0.5
        self.entropy_coeff = 0.05  # Higher entropy for better exploration when stuck
        self.gamma = 0.9   # Lower for faster learning in episodic tasks
        self.gae_lambda = 0.85  # Lower for less bias, faster adaptation

        # Training data storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        if self.device.type == "cuda":
            print(f"PPO Agent initialized on GPU: {self.device}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def get_action(self, state: np.ndarray, deterministic: bool = False, exploration_noise: float = 0.1):
        """Sample action from policy with better exploration and GPU acceleration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_output = self.policy_net(state_tensor)
            mean = policy_output[:, :self.action_dim]
            log_std = policy_output[:, self.action_dim:]
            std = torch.exp(log_std.clamp(-20, 2))  # Clamp for numerical stability

            if deterministic:
                action = mean
            else:
                # Add exploration noise to encourage exploration
                exploration_std = std + exploration_noise
                dist = Normal(mean, exploration_std)
                action = dist.sample()

            # Clamp action to [-1, 1]
            action = torch.tanh(action)

        return action.squeeze(0).cpu().numpy()

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition for training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_advantages(self):
        """Compute GAE advantages"""
        advantages = []
        returns = []

        last_advantage = 0
        last_return = 0

        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]

            delta = self.rewards[i] + self.gamma * next_value - self.values[i]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            advantages.insert(0, last_advantage)

            last_return = self.rewards[i] + self.gamma * last_return
            returns.insert(0, last_return)

        return advantages, returns

    def update(self):
        """Update policy and value networks with GPU acceleration"""
        if len(self.states) == 0:
            return {}

        # Convert to tensors and move to GPU
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_advantages()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of updates (all on GPU for speed)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(10):  # 10 epochs
            # Get current policy predictions (GPU accelerated)
            policy_output = self.policy_net(states)
            mean = policy_output[:, :self.action_dim]
            log_std = policy_output[:, self.action_dim:]
            std = torch.exp(log_std.clamp(-20, 2))

            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()

            # Compute ratio and clipped objective (GPU accelerated)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # Value loss (GPU accelerated)
            current_values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(current_values, returns)

            # Total loss
            total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy

            # Update networks (gradients computed on GPU)
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)

            self.policy_optimizer.step()
            self.value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Clear stored transitions
        self.clear_buffer()

        return {
            'policy_loss': total_policy_loss / 10,
            'value_loss': total_value_loss / 10,
            'entropy': total_entropy / 10,
            'device': str(self.device)
        }

    def clear_buffer(self):
        """Clear stored transitions"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


class SACAgent:
    """Soft Actor-Critic agent for continuous control with GPU acceleration"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, device=None):
        if not HAS_ML_LIBS:
            raise ImportError("ML libraries required for SAC agent")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or DEVICE or torch.device("cpu")

        # Networks - all moved to GPU
        self.policy_net = MLPNetwork(state_dim, action_dim * 2, device=self.device)
        self.q1_net = MLPNetwork(state_dim + action_dim, 1, device=self.device)
        self.q2_net = MLPNetwork(state_dim + action_dim, 1, device=self.device)
        self.value_net = MLPNetwork(state_dim, 1, device=self.device)
        self.target_value_net = MLPNetwork(state_dim, 1, device=self.device)

        # Copy parameters to target network
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # SAC hyperparameters
        self.gamma = 0.99
        self.tau = 0.005  # Target network update rate
        self.alpha = 0.2   # Temperature parameter

        # Replay buffer
        self.buffer_size = 10000
        self.batch_size = 64
        self.buffer = []

        if self.device.type == "cuda":
            print(f"SAC Agent initialized on GPU: {self.device}")

    def get_action(self, state: np.ndarray, deterministic: bool = False, exploration_noise: float = 0.1):
        """Sample action from policy with GPU acceleration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_output = self.policy_net(state_tensor)
            mean = policy_output[:, :self.action_dim]
            log_std = policy_output[:, self.action_dim:]
            std = torch.exp(log_std.clamp(-20, 2))

            if deterministic:
                action = torch.tanh(mean)
            else:
                dist = Normal(mean, std)
                sample = dist.sample()
                action = torch.tanh(sample)

        return action.squeeze(0).cpu().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        transition = (state, action, reward, next_state, done)

        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def update(self):
        """Update SAC networks with GPU acceleration"""
        if len(self.buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Move all tensors to GPU
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Update Q networks (GPU accelerated)
        current_q1 = self.q1_net(torch.cat([states, actions], dim=1)).squeeze()
        current_q2 = self.q2_net(torch.cat([states, actions], dim=1)).squeeze()

        with torch.no_grad():
            next_value = self.target_value_net(next_states).squeeze()
            target_q = rewards + self.gamma * (1 - dones) * next_value

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update value network (GPU accelerated)
        current_value = self.value_net(states).squeeze()

        # Get policy distribution
        policy_output = self.policy_net(states)
        mean = policy_output[:, :self.action_dim]
        log_std = policy_output[:, self.action_dim:]
        std = torch.exp(log_std.clamp(-20, 2))

        dist = Normal(mean, std)
        policy_sample = dist.sample()
        policy_action = torch.tanh(policy_sample)
        log_prob = dist.log_prob(policy_sample).sum(axis=-1)
        log_prob -= torch.log(1 - policy_action.pow(2) + 1e-6).sum(axis=-1)

        q1_policy = self.q1_net(torch.cat([states, policy_action], dim=1)).squeeze()
        q2_policy = self.q2_net(torch.cat([states, policy_action], dim=1)).squeeze()
        min_q_policy = torch.min(q1_policy, q2_policy)

        target_value = min_q_policy - self.alpha * log_prob
        value_loss = F.mse_loss(current_value, target_value.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network (GPU accelerated)
        policy_loss = (self.alpha * log_prob - min_q_policy).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target network
        self._soft_update(self.value_net, self.target_value_net)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'device': str(self.device)
        }

    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)


class ExperimentLogger:
    """Logger for tracking experiment results and generating reports"""

    def __init__(self, experiment_name: str, output_dir: str = "/home/nvidia/rl_experiments/fs_results", optimize_metric: str = "pp_tokens_per_sec"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimize_metric = optimize_metric

        # Create experiment subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Data storage
        self.episode_data = []
        self.best_params = None
        self.attempted_params = set()
        self.best_reward = float('-inf')

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.exp_dir / "experiment.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)

    def log_episode(self, episode: int, reward: float, params: Dict, result: Dict, training_metrics: Optional[Dict] = None):
        """Log episode results"""
        episode_record = {
            'episode': episode,
            'reward': reward,
            'params': params,
            'result': result,
            'training_metrics': training_metrics or {},
            'timestamp': datetime.now().isoformat()
        }

        self.episode_data.append(episode_record)
        self.attempted_params.add(str(params))

        # Update best parameters
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params.copy()

            self.logger.info(f"New best reward: {reward:.3f}")
            self.logger.info(f"Best parameters: {params}")

        # Log episode summary
        self.logger.info(f"Episode {episode}: Reward={reward:.3f}, Success={result.get('success', False)}")

        # Save intermediate results every 10 episodes. Is this necessary since I already save every save_frequency episodes?
        if episode % 10 == 0:
            self.save_results(optimize_metric=self.optimize_metric)

    def save_results(self, optimize_metric: str = "pp_tokens_per_sec"):
        import bisect
        """Save experiment results to files"""
        # Convert episode data to JSON-serializable format
        serializable_episode_data = []
        for episode_record in self.episode_data:
            serializable_record = episode_record.copy()
            # Convert any SchedulerParams objects to dicts with standard Python types
            if 'params' in serializable_record:
                if hasattr(serializable_record['params'], '__dict__'):
                    # Convert dataclass to dict and ensure boolean values are standard Python bools
                    params_dict = asdict(serializable_record['params'])
                    serializable_record['params'] = {k: bool(v) for k, v in params_dict.items()}
                else:
                    # Already a dict, just ensure bools are standard Python bools
                    serializable_record['params'] = {k: bool(v) for k, v in serializable_record['params'].items()}
            bisect.insort(serializable_episode_data, serializable_record, key=lambda x: -x['result'][optimize_metric])


        # Save full episode data as JSON
        results_file = self.exp_dir / "episode_results.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_episode_data, f, indent=2)

        # Save best parameters - convert to standard dict with Python bools
        best_params_file = self.exp_dir / "best_parameters.json"
        with open(best_params_file, 'w') as f:
            best_params_dict = None
            if self.best_params:
                if hasattr(self.best_params, '__dict__'):
                    # Convert dataclass to dict
                    params_dict = asdict(self.best_params)
                    best_params_dict = {k: bool(v) for k, v in params_dict.items()}
                else:
                    # Already a dict
                    best_params_dict = {k: bool(v) for k, v in self.best_params.items()}

            json.dump({
                'best_reward': float(self.best_reward),
                'best_params': best_params_dict,
                'total_episodes': len(self.episode_data)
            }, f, indent=2)

        # Save CSV for analysis
        if HAS_PANDAS:
            # Create DataFrame with flattened data for better analysis
            csv_data = []
            for record in serializable_episode_data:
                flattened_record = {
                    'episode': record['episode'],
                    'reward': record['reward'],
                    'params': record['params'],
                    'pp_tokens_per_sec': record['result'].get('pp_tokens_per_sec', None),
                    'tg_tokens_per_sec': record['result'].get('tg_tokens_per_sec', None),
                    'success': record['result'].get('success', None),
                    'error_msg': record['result'].get('error_msg', ''),
                    'execution_time': record['result'].get('execution_time', None),
                    'training_metrics': record['training_metrics'],
                    'timestamp': record['timestamp']
                }
                csv_data.append(flattened_record)
            df = pd.DataFrame(csv_data)

            df.sort_values(by=optimize_metric, inplace=True, ascending=False)
            csv_file = self.exp_dir / "episode_results.csv"
            df.to_csv(csv_file, index=False)

            ParamWrapper.append_attempted_combinations(self.attempted_params)
            self.attempted_params.clear()

        self.logger.info(f"Results saved to {self.exp_dir}")

    def generate_plots(self):
        """Generate performance plots"""
        if not HAS_MATPLOTLIB or len(self.episode_data) < 2:
            return

        try:
            episodes = [d['episode'] for d in self.episode_data]
            rewards = [d['reward'] for d in self.episode_data]

            plt.figure(figsize=(12, 8))

            # Reward plot
            plt.subplot(2, 2, 1)
            plt.plot(episodes, rewards)
            plt.title('Reward vs Episode')
            plt.xlabel('Episode')
            plt.ylabel('Reward')

            # Moving average
            if len(rewards) > 10:
                window = min(20, len(rewards) // 4)
                moving_avg = pd.Series(rewards).rolling(window=window).mean()
                plt.plot(episodes, moving_avg, 'r-', label=f'Moving avg ({window})')
                plt.legend()

            # Success rate plot
            plt.subplot(2, 2, 2)
            success_rates = [d['result'].get('success', False) for d in self.episode_data]
            if len(success_rates) > 10:
                window = min(20, len(success_rates) // 4)
                success_avg = pd.Series(success_rates).rolling(window=window).mean()
                plt.plot(episodes, success_avg)
            plt.title('Success Rate vs Episode')
            plt.xlabel('Episode')
            plt.ylabel('Success Rate')

            # Throughput plot
            plt.subplot(2, 2, 3)
            throughputs = [d['result'].get('throughput', 0) for d in self.episode_data]
            plt.plot(episodes, throughputs)
            plt.title('Throughput vs Episode')
            plt.xlabel('Episode')
            plt.ylabel('Throughput')

            # Parameter evolution (slice_us)
            plt.subplot(2, 2, 4)
            slice_us_values = [d['params'].get('slice_us', 0) for d in self.episode_data]
            plt.plot(episodes, slice_us_values)
            plt.title('slice_us Parameter Evolution')
            plt.xlabel('Episode')
            plt.ylabel('slice_us')

            plt.tight_layout()
            plot_file = self.exp_dir / "performance_plots.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Plots saved to {plot_file}")

        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {e}")


def run_baseline_evaluation(env: FlashySparkEnvironment, num_runs: int = 5) -> Dict:
    """Run baseline evaluation with default parameters"""
    print(f"Running baseline evaluation with {num_runs} runs...")

    baseline_results = []
    default_params = SchedulerParams()

    for i in range(num_runs):
        print(f"Baseline run {i+1}/{num_runs}")
        result = env._run_benchmark(default_params)
        # For baseline, use simple scoring without relative performance (no scaling)
        baseline_reward = result.get_reward("pp_tokens_per_sec", None, None, default_params, 1.0)  # Use pp by default for baseline
        baseline_results.append({
            'reward': baseline_reward,
            'pp_tokens_per_sec': result.pp_tokens_per_sec,
            'tg_tokens_per_sec': result.tg_tokens_per_sec,
            'success': result.success,
            'execution_time': result.execution_time
        })

    # Calculate statistics
    successful_runs = [r for r in baseline_results if r['success']]

    if successful_runs:
        baseline_stats = {
            'mean_reward': np.mean([r['reward'] for r in successful_runs]),
            'std_reward': np.std([r['reward'] for r in successful_runs]),
            'mean_pp_tokens_per_sec': np.mean([r['pp_tokens_per_sec'] for r in successful_runs]),
            'mean_tg_tokens_per_sec': np.mean([r['tg_tokens_per_sec'] for r in successful_runs]),
            'success_rate': len(successful_runs) / len(baseline_results),
            'num_runs': len(baseline_results),
            'successful_runs': len(successful_runs)
        }
    else:
        baseline_stats = {
            'mean_reward': -100.0,
            'std_reward': 0.0,
            'mean_pp_tokens_per_sec': 0.0,
            'mean_tg_tokens_per_sec': 0.0,
            'success_rate': 0.0,
            'num_runs': len(baseline_results),
            'successful_runs': 0
        }

    print(f"Baseline evaluation completed:")
    print(f"  Success rate: {baseline_stats['success_rate']:.2%}")
    print(f"  Mean reward: {baseline_stats['mean_reward']:.3f} ± {baseline_stats['std_reward']:.3f}")
    print(f"  Mean PP tokens/s: {baseline_stats['mean_pp_tokens_per_sec']:.3f}")
    print(f"  Mean TG tokens/s: {baseline_stats['mean_tg_tokens_per_sec']:.3f}")

    return baseline_stats


def load_pretrained_model(agent, model_path: str, algorithm: str):
    """Load a pre-trained model into the agent"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Load policy network
    if 'policy_net' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['policy_net'])

    # Load policy optimizer
    if 'policy_optimizer' in checkpoint:
        agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])

    # Load algorithm-specific networks
    if algorithm == "ppo":
        if 'value_net' in checkpoint and hasattr(agent, 'value_net'):
            agent.value_net.load_state_dict(checkpoint['value_net'])
        if 'value_optimizer' in checkpoint and hasattr(agent, 'value_optimizer'):
            agent.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
    elif algorithm == "sac":
        if 'q1_net' in checkpoint and hasattr(agent, 'q1_net'):
            agent.q1_net.load_state_dict(checkpoint['q1_net'])
        if 'q2_net' in checkpoint and hasattr(agent, 'q2_net'):
            agent.q2_net.load_state_dict(checkpoint['q2_net'])
        if 'value_net' in checkpoint and hasattr(agent, 'value_net'):
            agent.value_net.load_state_dict(checkpoint['value_net'])
        if 'target_value_net' in checkpoint and hasattr(agent, 'target_value_net'):
            agent.target_value_net.load_state_dict(checkpoint['target_value_net'])

    print(f"Loaded pre-trained model with {checkpoint.get('num_historical_experiences', 'unknown')} historical experiences")


def train_rl_agent(
    algorithm: str = "ppo",
    episodes: int = 100,
    baseline_runs: int = 5,
    model_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    timeout: int = 300,
    learning_rate: float = 1e-6,
    update_frequency: int = 10,
    save_frequency: int = 10,
    optimize_metric: str = "pp_tokens_per_sec",
    reward_scaling: float = 1.0,
    use_gpu: bool = True,
    pretrained_model_path: Optional[str] = None
):
    """
    MAIN TRAINING FUNCTION - This is where the AI actually learns!

    This function orchestrates the entire learning process. Think of it as the
    "conductor" of an orchestra, coordinating all the different parts.

    THE LEARNING PROCESS:
    1. SETUP: Create the AI agent and environment
    2. BASELINE: Test default settings to see how well the system normally runs
    3. EPISODES: Run many rounds where the AI tries different settings
    4. LEARNING: After each round, the AI updates its knowledge
    5. TRACKING: Save progress and find the best settings discovered

    WHAT HAPPENS IN EACH EPISODE:
    1. AI chooses scheduler parameter settings (based on what it has learned)
    2. We apply those settings to the scheduler
    3. We run a performance test (benchmark)
    4. We calculate a reward score based on performance
    5. AI updates its "brain" to get better at choosing good settings

    ANALOGY: Like training an athlete
    - Episodes = practice sessions
    - Each practice, try different techniques (parameter combinations)
    - Coach evaluates performance (reward function)
    - Athlete adjusts technique based on feedback (neural network learning)
    - Over time, athlete gets better at choosing winning strategies

    PARAMETERS EXPLAINED:
    - episodes: How many practice sessions (more = better learning but takes longer)
    - learning_rate: How fast the AI learns (too fast = unstable, too slow = inefficient)
    - algorithm: Which learning method to use (PPO or SAC)
    """

    if not HAS_ML_LIBS:
        print("Error: Required ML libraries not installed.")
        print("Install with: pip install gymnasium torch matplotlib seaborn numpy pandas")
        return False

    # Set experiment name
    if experiment_name is None:
        experiment_name = f"flashyspark_{algorithm}_{episodes}ep"

    print(f"Starting RL optimization experiment: {experiment_name}")
    print(f"Algorithm: {algorithm}")
    print(f"Episodes: {episodes}")
    print(f"Model: {model_path or 'default'}")
    print("=" * 60)

    # Override device if GPU disabled
    global DEVICE
    if not use_gpu and DEVICE and DEVICE.type == "cuda":
        DEVICE = torch.device("cpu")
        print("GPU disabled by user, using CPU")

    # Setup GPU optimizations
    setup_gpu_optimizations()

    # STEP 1: Initialize environment (the "world" where AI operates)
    env = FlashySparkEnvironment(model_path=model_path, timeout=timeout, optimize_metric=optimize_metric, reward_scaling=reward_scaling)

    # These numbers define the AI's "input/output dimensions"
    # state_dim = how much information the AI sees about the current situation
    # action_dim = how many different controls/parameters the AI can adjust
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize logger
    logger = ExperimentLogger(experiment_name, optimize_metric=optimize_metric)
    logger.logger.info(f"Initializing RL experiment with {algorithm} algorithm")

    try:
        # Run baseline evaluation
        if baseline_runs > 0:
            logger.logger.info("Running baseline evaluation...")
            baseline_stats = run_baseline_evaluation(env, baseline_runs)

            # Set fixed baseline for the environment
            env.set_fixed_baseline(baseline_stats['mean_pp_tokens_per_sec'], baseline_stats['mean_tg_tokens_per_sec'])

            # Save baseline results
            baseline_file = logger.exp_dir / "baseline_results.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline_stats, f, indent=2)
        else:
            baseline_stats = None

        # Initialize RL agent with GPU support
        logger.logger.info(f"Initializing {algorithm.upper()} agent...")
        if algorithm.lower() == "ppo":
            agent = PPOAgent(state_dim, action_dim, lr=learning_rate, device=DEVICE)
        elif algorithm.lower() == "sac":
            agent = SACAgent(state_dim, action_dim, lr=learning_rate, device=DEVICE)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Load pre-trained model if specified
        if pretrained_model_path:
            logger.logger.info(f"Loading pre-trained model from: {pretrained_model_path}")
            load_pretrained_model(agent, pretrained_model_path, algorithm.lower())
            logger.logger.info("Pre-trained model loaded successfully!")

        # STEP 4: Training loop - This is where the AI learns!
        logger.logger.info("Starting training loop...")
        state, _ = env.reset()  # Start in a clean state

        # Add learning improvements
        recent_rewards = []  # Track recent rewards for smoothing
        exploration_schedule = []  # Track exploration for analysis
        best_recent_avg = float('-inf')  # Track best recent average
        episodes_without_improvement = 0  # Early stopping detection
        stuck_counter = 0  # Track how long we've been stuck

        for episode in range(1, episodes + 1):
            logger.logger.info(f"\n--- Episode {episode}/{episodes} ---")

            # ENHANCED DYNAMIC EXPLORATION with adaptive restarts
            base_exploration = max(0.3 * (1.0 - episode / episodes), 0.05)

            # Increase exploration when stuck
            if episodes_without_improvement > 15:
                stuck_penalty = min(episodes_without_improvement / 25, 1.0)  # Up to 100% boost
                base_exploration = min(base_exploration * (1 + stuck_penalty), 0.5)
                stuck_counter += 1

                # Random restart every 10 episodes when very stuck
                if stuck_counter > 10 and episode % 10 == 0:
                    base_exploration = 0.6  # Force high exploration
                    logger.logger.info(f"Random restart triggered - high exploration mode")
                    stuck_counter = 0

            exploration_noise = base_exploration
            exploration_schedule.append(exploration_noise)

                        # STEP 4a: AI decides what action to take (which parameters to try)
            # The AI looks at the current state and uses its neural network to decide
            # what scheduler parameters to test next

            # Add epsilon-greedy exploration for better parameter space coverage
            epsilon = min(0.1, exploration_noise)  # 20% chance of random action when exploring
            use_random_action = np.random.random() < epsilon and episode > 5

            if use_random_action:
                # Pure random exploration - sample random configuration
                action = np.random.uniform(-1, 1, agent.action_dim)
                value = 0.0
                log_prob = 0.0
                logger.logger.info(f"Random exploration action (ε={epsilon:.3f})")
            else:
                # Use agent's policy
                if algorithm.lower() == "ppo":
                    action = agent.get_action(state, exploration_noise=exploration_noise)  # PPO agent chooses parameters

                    # Get value estimate for PPO
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        value = agent.value_net(state_tensor).item()

                    # Calculate log probability for PPO
                    policy_output = agent.policy_net(state_tensor)
                    mean = policy_output[:, :action_dim]
                    log_std = policy_output[:, action_dim:]
                    std = torch.exp(log_std.clamp(-20, 2))
                    dist = Normal(mean, std + exploration_noise)  # Use same noise as action
                    log_prob = dist.log_prob(torch.FloatTensor(action).to(agent.device)).sum().item()

                else:  # SAC
                    action = agent.get_action(state, exploration_noise=exploration_noise)
                    value = 0.0  # SAC doesn't need value for storage
                    log_prob = 0.0  # SAC doesn't need log_prob for storage

            # STEP 4b: Test the chosen parameters and get results
            # This applies the AI's chosen settings to the scheduler, runs a benchmark,
            # and calculates a reward score based on performance
            next_state, reward, terminated, truncated, info = env.step(action)

            # REWARD SMOOTHING: Track recent rewards for analysis
            recent_rewards.append(reward)
            if len(recent_rewards) > 20:  # Keep last 20 episodes
                recent_rewards.pop(0)

            avg_recent_reward = np.mean(recent_rewards) if recent_rewards else reward

            # STEP 4c: Store this experience for learning
            # The AI remembers: "When I was in state X and took action Y, I got reward Z"
            # This memory will be used later to improve decision-making
            if algorithm.lower() == "ppo":
                agent.store_transition(state, action, reward, value, log_prob, terminated)
            else:  # SAC
                agent.store_transition(state, action, reward, next_state, terminated)

            # Enhanced logging with exploration info
            logger.log_episode(
                episode=episode,
                reward=reward,
                params=info['params'],
                result=info['result'],
                training_metrics={'exploration_noise': exploration_noise, 'avg_recent_reward': avg_recent_reward}
            )

            # STEP 4d: Update the AI's "brain" (neural network learning)
            # Periodically, the AI analyzes all its recent experiences and updates
            # its decision-making to be better at choosing good parameters
            training_metrics = {}
            if episode % update_frequency == 0:
                logger.logger.info("Updating agent...")
                logger.logger.info(get_gpu_memory_usage())  # Monitor GPU memory

                # This is where the actual "learning" happens!
                # The AI adjusts its neural network weights based on which actions
                # led to good vs bad rewards
                training_metrics = agent.update()

                if training_metrics:
                    logger.logger.info(f"Training metrics: {training_metrics}")

                # Clean up GPU memory periodically
                if episode % (update_frequency * 3) == 0:
                    cleanup_gpu_memory()
                    logger.logger.info("GPU memory cleaned up")

                # Log learning progress
                if len(recent_rewards) >= 10:
                    reward_trend = np.mean(recent_rewards[-10:]) - np.mean(recent_rewards[-20:-10]) if len(recent_rewards) >= 20 else 0
                    logger.logger.info(f"Recent reward trend: {reward_trend:+.3f}")
                    logger.logger.info(f"Exploration noise: {exploration_noise:.3f}")

            # EARLY STOPPING DETECTION: Check if learning has plateaued
            if len(recent_rewards) >= 15:  # Need enough data points
                current_avg = np.mean(recent_rewards[-10:])  # Average of last 10 episodes
                if current_avg > best_recent_avg + 1.0:  # Significant improvement threshold
                    best_recent_avg = current_avg
                    episodes_without_improvement = 0
                    logger.logger.info(f"New best recent average reward: {best_recent_avg:.3f}")
                else:
                    episodes_without_improvement += 1

                # Enhanced feedback when stuck with automatic adjustments
                if episodes_without_improvement >= 20:
                    logger.logger.warning(f"No improvement for {episodes_without_improvement} episodes.")
                    logger.logger.warning("Auto-adjustments enabled: higher exploration, more frequent updates")

                    # Reset counter less frequently to allow more aggressive exploration
                    if episodes_without_improvement >= 30:
                        episodes_without_improvement = 0  # Reset counter

            # Save checkpoint
            print(episode, save_frequency, episode % save_frequency)
            if episode % save_frequency == 0:
                logger.save_results(optimize_metric=logger.optimize_metric)
                logger.generate_plots()

                # Save agent checkpoint
                if hasattr(agent, 'policy_net'):
                    checkpoint = {
                        'episode': episode,
                        'policy_net': agent.policy_net.state_dict(),
                        'policy_optimizer': agent.policy_optimizer.state_dict(),
                        'exploration_schedule': exploration_schedule,
                        'recent_rewards': recent_rewards
                    }

                    if algorithm.lower() == "ppo":  # PPO
                        checkpoint.update({
                            'value_net': agent.value_net.state_dict(),
                            'value_optimizer': agent.value_optimizer.state_dict(),
                        })
                    elif algorithm.lower() == "sac":  # SAC
                        checkpoint.update({
                            'q1_net': agent.q1_net.state_dict(),
                            'q2_net': agent.q2_net.state_dict(),
                            'value_net': agent.value_net.state_dict(),
                            'target_value_net': agent.target_value_net.state_dict(),
                        })

                    checkpoint_file = logger.exp_dir / f"checkpoint_episode_{episode}.pt"
                    torch.save(checkpoint, checkpoint_file)
                    logger.logger.info(f"Checkpoint saved: {checkpoint_file}")

            # Update state for next iteration
            state = next_state

            # Enhanced progress reporting
            if episode % 5 == 0:
                recent_avg = np.mean(recent_rewards[-5:]) if len(recent_rewards) >= 5 else reward
                print(f"Episode {episode}/{episodes} - Current: {reward:.3f} - "
                      f"Recent Avg: {recent_avg:.3f} - Best: {env.best_reward:.3f} - "
                      f"Exploration: {exploration_noise:.3f}")

        # STEP 5: Save results and summary
        logger.save_results(optimize_metric=logger.optimize_metric)
        logger.generate_plots()

        # Print final summary
        logger.logger.info("\n" + "=" * 60)
        logger.logger.info("TRAINING COMPLETED - THE AI HAS LEARNED!")
        logger.logger.info("=" * 60)
        logger.logger.info(f"Total episodes: {episodes}")
        logger.logger.info(f"Best reward achieved: {env.best_reward:.3f}")
        logger.logger.info(f"Best parameters: {env.best_params}")
        logger.logger.info(f"Training device: {DEVICE}")
        logger.logger.info(get_gpu_memory_usage())



        # WHAT HAPPENED: The AI tried many different combinations of scheduler
        # parameters, learned which ones work better, and found the optimal settings!

        if baseline_stats and baseline_stats['successful_runs'] > 0:
            improvement = env.best_reward - baseline_stats['mean_reward']
            logger.logger.info(f"Improvement over baseline: {improvement:.3f}")
            logger.logger.info(f"Baseline reward: {baseline_stats['mean_reward']:.3f} ± {baseline_stats['std_reward']:.3f}")

        logger.logger.info(f"Results saved in: {logger.exp_dir}")

        return True

    except KeyboardInterrupt:
        logger.logger.info("Training interrupted by user")
        logger.save_results(optimize_metric=logger.optimize_metric)
        return False

    except Exception as e:
        logger.logger.error(f"Training failed with error: {e}")
        logger.save_results(optimize_metric=logger.optimize_metric)
        raise e

    finally:
        # Ensure cleanup
        env._cleanup()


def test_best_params(experiment_dir: str, num_test_runs: int = 10):
    """Test the best parameters found by RL optimization"""

    exp_path = Path(experiment_dir)
    best_params_file = exp_path / "best_parameters.json"

    if not best_params_file.exists():
        print(f"Error: Best parameters file not found at {best_params_file}")
        return False

    # Load best parameters
    with open(best_params_file, 'r') as f:
        best_data = json.load(f)

    best_params_dict = best_data['best_params']
    best_reward = best_data['best_reward']

    print(f"Testing best parameters from {experiment_dir}")
    print(f"Best training reward: {best_reward:.3f}")
    print(f"Parameters: {best_params_dict}")
    print("=" * 60)

    # Create scheduler params object
    params = SchedulerParams(**best_params_dict)

    # Initialize environment for testing
    env = FlashySparkEnvironment()

    # Run test episodes
    test_results = []
    for i in range(num_test_runs):
        print(f"Test run {i+1}/{num_test_runs}")
        result = env._run_benchmark(params)
        # For testing, use simple scoring without relative performance (no scaling)
        test_reward = result.get_reward("pp", None, None, params, 1.0)
        test_results.append({
            'reward': test_reward,
            'pp_tokens_per_sec': result.pp_tokens_per_sec,
            'tg_tokens_per_sec': result.tg_tokens_per_sec,
            'success': result.success,
            'execution_time': result.execution_time
        })

    # Calculate test statistics
    successful_tests = [r for r in test_results if r['success']]

    if successful_tests:
        test_stats = {
            'mean_reward': np.mean([r['reward'] for r in successful_tests]),
            'std_reward': np.std([r['reward'] for r in successful_tests]),
            'mean_pp_tokens_per_sec': np.mean([r['pp_tokens_per_sec'] for r in successful_tests]),
            'mean_tg_tokens_per_sec': np.mean([r['tg_tokens_per_sec'] for r in successful_tests]),
            'success_rate': len(successful_tests) / len(test_results),
            'num_runs': len(test_results),
            'successful_runs': len(successful_tests)
        }

        print(f"\nTest Results:")
        print(f"  Success rate: {test_stats['success_rate']:.2%}")
        print(f"  Mean reward: {test_stats['mean_reward']:.3f} ± {test_stats['std_reward']:.3f}")
        print(f"  Mean PP tokens/s: {test_stats['mean_pp_tokens_per_sec']:.3f}")
        print(f"  Mean TG tokens/s: {test_stats['mean_tg_tokens_per_sec']:.3f}")

        # Save test results
        test_results_file = exp_path / "test_results.json"
        with open(test_results_file, 'w') as f:
            json.dump({
                'test_stats': test_stats,
                'individual_results': test_results,
                'tested_params': best_params_dict
            }, f, indent=2)

        print(f"Test results saved to: {test_results_file}")

    else:
        print("ERROR: All test runs failed!")
        return False

    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RL Optimization for scx_flashyspark Scheduler Parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PPO agent for 100 episodes
  python rl_experiments/main.py --algorithm ppo --episodes 100

  # Train SAC agent with custom learning rate
  python rl_experiments/main.py --algorithm sac --episodes 50 --lr 1e-4

  # Test best parameters from previous experiment
  python rl_experiments/main.py --test /home/nvidia/rl_experiments/flashyspark_ppo_100ep_20241201_120000
        """
    )

    parser.add_argument(
        "--algorithm", "-a",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm to use (default: ppo)"
    )

    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=50,
        help="Number of training episodes (default: 50)"
    )

    parser.add_argument(
        "--baseline-runs", "-b",
        type=int,
        default=3,
        help="Number of baseline evaluation runs (default: 3)"
    )

    parser.add_argument(
        "--model-path", "-m",
        type=str,
        help="Path to llama model file (default: system default)"
    )

    parser.add_argument(
        "--experiment-name", "-n",
        type=str,
        help="Custom experiment name (default: auto-generated)"
    )

    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=300,
        help="Benchmark timeout in seconds (default: 300)"
    )

    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for RL agent (default: 1e-3, increased for better convergence)"
    )

    parser.add_argument(
        "--update-frequency", "-u",
        type=int,
        default=5,
        help="Agent update frequency in episodes (default: 5, more frequent for faster learning)"
    )

    parser.add_argument(
        "--save-frequency", "-s",
        type=int,
        default=25,
        help="Save/checkpoint frequency in episodes (default: 25)"
    )

    parser.add_argument(
        "--test",
        type=str,
        help="Test best parameters from experiment directory"
    )

    parser.add_argument(
        "--test-runs",
        type=int,
        default=10,
        help="Number of test runs when using --test (default: 10)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--optimize-metric",
        choices=["pp_tokens_per_sec", "tg_tokens_per_sec"],
        default="pp_tokens_per_sec",
        help="Metric to optimize: 'pp_tokens_per_sec' (prompt processing) or 'tg_tokens_per_sec' (text generation) (default: pp_tokens_per_sec)"
    )

    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
        help="Reward scaling factor to make differences more/less extreme (default: 1.0)"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Path to pre-trained model checkpoint to load before training"
    )

    args = parser.parse_args()

    # Setup basic logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        if args.test:
            # Test mode
            success = test_best_params(args.test, args.test_runs)
            return 0 if success else 1
        else:
            # Training mode
            success = train_rl_agent(
                algorithm=args.algorithm,
                episodes=args.episodes,
                baseline_runs=args.baseline_runs,
                model_path=args.model_path,
                experiment_name=args.experiment_name,
                timeout=args.timeout,
                learning_rate=args.learning_rate,
                update_frequency=args.update_frequency,
                save_frequency=args.save_frequency,
                optimize_metric=args.optimize_metric,
                reward_scaling=args.reward_scaling,
                use_gpu=not args.no_gpu,
                pretrained_model_path=args.pretrained_model
            )
            return 0 if success else 1

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
