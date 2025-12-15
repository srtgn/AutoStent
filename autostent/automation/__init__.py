"""
Automation Module

Automated 4C workflow generation including:
- YAML input file generation
- Batch processing
- HPC job submission
"""

from .yaml_generator import FourCYAMLGenerator
from .batch_processor import BatchProcessor
from .hpc_jobs import generate_slurm_job

__all__ = [
    "FourCYAMLGenerator",
    "BatchProcessor",
    "generate_slurm_job",
]

