#!/usr/bin/env python3
"""
RL Scheduler Optimizer API for scheduler parameter optimization.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import importlib

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. YAML config files will not be supported.")
    print("Install with: pip install pyyaml")

try:
    from .main import (
        SchedulerConfig,
        ParameterSpec,
        ParameterType,
        SchedulerParams,
        get_scheduler_config,
        train_rl_agent,
        test_best_parameters,
        HAS_ML_LIBS
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from main import (
        SchedulerConfig,
        ParameterSpec,
        ParameterType,
        SchedulerParams,
        get_scheduler_config,
        train_rl_agent,
        test_best_parameters,
        HAS_ML_LIBS
    )


@dataclass
class OptimizerConfig:
    """Configuration for the RL optimizer."""
    scheduler_name: str = "scx_flashyspark"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    algorithm: str = "ppo"
    episodes: int = 50
    baseline_runs: int = 3
    learning_rate: float = 1e-3
    update_frequency: int = 5
    save_frequency: int = 25
    
    benchmark_cmd: Optional[str] = None
    model_path: Optional[str] = None
    timeout: int = 300
    optimize_metric: str = "throughput"
    reward_scaling: float = 1.0
    
    experiment_name: Optional[str] = None
    use_gpu: bool = True
    pretrained_model_path: Optional[str] = None
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'OptimizerConfig':
        """Load configuration from a YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
                data = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(**data)
    
    def to_file(self, config_path: Union[str, Path]):
        """Save configuration to a YAML or JSON file."""
        config_path = Path(config_path)
        
        data = {
            'scheduler_name': self.scheduler_name,
            'scheduler_params': self.scheduler_params,
            'algorithm': self.algorithm,
            'episodes': self.episodes,
            'baseline_runs': self.baseline_runs,
            'learning_rate': self.learning_rate,
            'update_frequency': self.update_frequency,
            'save_frequency': self.save_frequency,
            'benchmark_cmd': self.benchmark_cmd,
            'model_path': self.model_path,
            'timeout': self.timeout,
            'optimize_metric': self.optimize_metric,
            'reward_scaling': self.reward_scaling,
            'experiment_name': self.experiment_name,
            'use_gpu': self.use_gpu,
            'pretrained_model_path': self.pretrained_model_path
        }
        
        with open(config_path, 'w') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            elif config_path.suffix == '.json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")


class RLSchedulerOptimizer:
    """Main API class for optimizing scheduler parameters using RL."""
    
    def __init__(self, config: Optional[Union[OptimizerConfig, str, Path, Dict]] = None):
        """Initialize the optimizer with configuration.
        
        Args:
            config: Can be:
                - OptimizerConfig object
                - Path to config file (YAML/JSON)
                - Dict with configuration values
                - None (uses defaults)
        """
        if config is None:
            self.config = OptimizerConfig()
        elif isinstance(config, OptimizerConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = OptimizerConfig.from_file(config)
        elif isinstance(config, dict):
            self.config = OptimizerConfig(**config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        
        self._setup_logging()
        
        if not HAS_ML_LIBS:
            raise ImportError("Machine learning libraries not available. Please install requirements.")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RLSchedulerOptimizer")
    
    def define_scheduler_params(self, params: Dict[str, Dict[str, Any]]):
        """Define custom scheduler parameters programmatically.
        
        Args:
            params: Dictionary mapping parameter names to their specifications.
                    Each spec should have: type, default, and optionally: 
                    min_value, max_value, choices, description, command_arg
        
        Example:
            optimizer.define_scheduler_params({
                'slice_us': {
                    'type': 'integer',
                    'default': 20000,
                    'min_value': 1000,
                    'max_value': 100000,
                    'description': 'Time slice in microseconds',
                    'command_arg': '--slice-us'
                },
                'enable_turbo': {
                    'type': 'boolean',
                    'default': False,
                    'description': 'Enable turbo mode',
                    'command_arg': '--turbo'
                }
            })
        """
        scheduler_params = {}
        
        for name, spec in params.items():
            param_type = spec.get('type', 'boolean')
            
            type_map = {
                'boolean': ParameterType.BOOLEAN,
                'integer': ParameterType.INTEGER,
                'float': ParameterType.FLOAT,
                'categorical': ParameterType.CATEGORICAL
            }
            
            if param_type not in type_map:
                raise ValueError(f"Invalid parameter type: {param_type}")
            
            scheduler_params[name] = ParameterSpec(
                name=name,
                param_type=type_map[param_type],
                default_value=spec.get('default'),
                min_value=spec.get('min_value'),
                max_value=spec.get('max_value'),
                choices=spec.get('choices'),
                description=spec.get('description', ''),
                command_arg=spec.get('command_arg')
            )
        
        self.custom_scheduler_config = SchedulerConfig(
            name=self.config.scheduler_name,
            binary_path=os.path.expanduser(f"~/bin/{self.config.scheduler_name}"),
            parameters=scheduler_params,
            description="Custom scheduler configuration"
        )
    
    def train(self, **kwargs) -> Dict[str, Any]:
        """Train the RL agent to optimize scheduler parameters.
        
        Args:
            **kwargs: Override any configuration parameters for this run
        
        Returns:
            Dictionary with training results including:
                - best_params: Best parameter configuration found
                - best_score: Best score achieved
                - experiment_path: Path to experiment results
                - training_history: Training metrics over episodes
        """
        config_dict = {
            'scheduler_name': self.config.scheduler_name,
            'algorithm': self.config.algorithm,
            'episodes': self.config.episodes,
            'baseline_runs': self.config.baseline_runs,
            'model_path': self.config.model_path,
            'benchmark_cmd': self.config.benchmark_cmd,
            'experiment_name': self.config.experiment_name,
            'timeout': self.config.timeout,
            'learning_rate': self.config.learning_rate,
            'update_frequency': self.config.update_frequency,
            'save_frequency': self.config.save_frequency,
            'optimize_metric': self.config.optimize_metric,
            'reward_scaling': self.config.reward_scaling,
            'use_gpu': self.config.use_gpu,
            'pretrained_model_path': self.config.pretrained_model_path
        }
        config_dict.update(kwargs)
        
        self.logger.info(f"Starting training with {config_dict['algorithm']} for {config_dict['episodes']} episodes")
        
        success = train_rl_agent(**config_dict)
        
        if not success:
            raise RuntimeError("Training failed")
        
        return {
            'success': True,
            'config': config_dict
        }
    
    def test(self, experiment_path: str, test_runs: int = 10) -> Dict[str, Any]:
        """Test the best parameters from a previous experiment.
        
        Args:
            experiment_path: Path to experiment directory
            test_runs: Number of test runs to perform
        
        Returns:
            Dictionary with test results
        """
        self.logger.info(f"Testing best parameters from {experiment_path}")
        
        test_best_parameters(experiment_path, test_runs)
        
        return {
            'experiment_path': experiment_path,
            'test_runs': test_runs
        }
    
    def get_scheduler_info(self) -> Dict[str, Any]:
        """Get information about the current scheduler configuration.
        
        Returns:
            Dictionary with scheduler name, parameters, and their specifications
        """
        if hasattr(self, 'custom_scheduler_config'):
            config = self.custom_scheduler_config
        else:
            config = get_scheduler_config(self.config.scheduler_name)
        
        params_info = {}
        for name, spec in config.parameters.items():
            params_info[name] = {
                'type': spec.param_type.name.lower(),
                'default': spec.default_value,
                'description': spec.description,
                'command_arg': spec.command_arg
            }
            
            if spec.min_value is not None:
                params_info[name]['min_value'] = spec.min_value
            if spec.max_value is not None:
                params_info[name]['max_value'] = spec.max_value
            if spec.choices:
                params_info[name]['choices'] = spec.choices
        
        return {
            'scheduler_name': config.name,
            'binary_path': config.binary_path,
            'description': config.description,
            'parameters': params_info
        }


def optimize_scheduler(
    scheduler_name: str = "scx_flashyspark",
    episodes: int = 50,
    algorithm: str = "ppo",
    config_file: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to quickly optimize a scheduler.
    
    Args:
        scheduler_name: Name of the scheduler to optimize
        episodes: Number of training episodes
        algorithm: RL algorithm to use ('ppo' or 'sac')
        config_file: Optional path to config file
        **kwargs: Additional configuration parameters
    
    Returns:
        Dictionary with optimization results
    """
    if config_file:
        optimizer = RLSchedulerOptimizer(config_file)
    else:
        config = OptimizerConfig(
            scheduler_name=scheduler_name,
            episodes=episodes,
            algorithm=algorithm,
            **kwargs
        )
        optimizer = RLSchedulerOptimizer(config)
    
    return optimizer.train()
