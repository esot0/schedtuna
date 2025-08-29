#!/usr/bin/env python3
"""
Setup script for rl_scx_params
==============================

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="rl_scx_params",
    version="0.1.0",
    author="Emily Soto",
    description="Reinforcement Learning for Linux Scheduler Parameter Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nvidia/rl_scx_params",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "yaml": ["pyyaml>=5.0"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "rl-scx-optimize=rl_scx_params.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rl_scx_params": ["examples/*"],
    },
)
