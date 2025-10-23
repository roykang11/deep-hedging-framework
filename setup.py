#!/usr/bin/env python3
"""
Deep Hedging Framework Setup
Professional setup script for the deep hedging research project
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deep-hedging-framework",
    version="1.0.0",
    author="Seojoon Kang",
    author_email="your.email@example.com",
    description="A comprehensive deep learning framework for option hedging using advanced neural networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deep-hedging-framework",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/deep-hedging-framework/issues",
        "Source": "https://github.com/yourusername/deep-hedging-framework",
        "Documentation": "https://github.com/yourusername/deep-hedging-framework/blob/main/docs/research_paper.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "torch>=1.12.0+cu113",
            "torchvision>=0.13.0+cu113",
            "torchaudio>=0.12.0+cu113",
        ],
    },
    entry_points={
        "console_scripts": [
            "deep-hedging=experiments.comprehensive_study:main",
            "deep-hedging-basic=experiments.basic_experiment:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    keywords=[
        "deep learning",
        "hedging",
        "options",
        "risk management",
        "neural networks",
        "quantitative finance",
        "machine learning",
        "pytorch",
        "lstm",
        "transformer",
        "attention",
    ],
    zip_safe=False,
)
