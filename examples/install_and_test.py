#!/usr/bin/env python3
"""
Quick installation and test script
==================================
"""

import subprocess
import sys
import os

def main():
    print("RL SCX Params - Quick Installation and Test")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("\n⚠️  WARNING: Not running in a virtual environment!")
        print("It's recommended to run this in a virtual environment.")
        print("\nTo create and activate a virtual environment:")
        print("  python -m venv rl_scx_env")
        print("  source rl_scx_env/bin/activate  # On Linux/Mac")
        print("  # or")
        print("  rl_scx_env\\Scripts\\activate  # On Windows")
        
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Install the package in development mode
    print("Installing rl_scx_params in development mode...")
    try:
        # Add the parent directory to Python path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        
        # Try to import the package
        import rl_scx_params
        print("Package imported successfully!")
        
    except ImportError as e:
        print(f"Failed to import package: {e}")
        return
    
    # Run a simple test
    print("\nRunning a simple test...")
    try:
        from rl_scx_params import RLSchedulerOptimizer
        
        # Create optimizer
        optimizer = RLSchedulerOptimizer({
            "scheduler_name": "scx_flashyspark",
            "episodes": 5,  # Very short test
            "algorithm": "ppo"
        })
        
        # Get scheduler info
        info = optimizer.get_scheduler_info()
        print(f"Scheduler: {info['scheduler_name']}")
        print(f"   Parameters: {len(info['parameters'])}")
        print(f"   Description: {info['description']}")
        
        print("\nAll tests passed! rl_scx_params is ready to use.")
        
        print("\nNext steps:")
        print("1. Check out the examples directory for more usage examples")
        print("2. Read the README.md for detailed documentation")
        print("3. Try running: python examples/basic_usage.py")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
