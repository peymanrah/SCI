from setuptools import setup, find_packages

setup(
    name="sci",
    version="0.1.0",
    description="Structural Causal Invariance for Compositional Generalization",
    author="SCI Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.2",
        "datasets>=2.14.0",
        "wandb>=0.16.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "tensorboard>=2.15.0",
        "einops>=0.7.0",
        "accelerate>=0.20.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
