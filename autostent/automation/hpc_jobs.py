"""
HPC Job Generation

Generate SLURM job scripts for batch execution on HPC systems.
"""

from pathlib import Path
from typing import Dict, Any, Optional


def generate_slurm_job(
    job_name: str,
    script_path: Path,
    python_script: Path,
    output_dir: Path,
    num_nodes: int = 1,
    num_tasks: int = 1,
    cpus_per_task: int = 1,
    memory_gb: int = 8,
    time_hours: int = 24,
    partition: str = "normal",
    additional_options: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Generate SLURM job script.
    
    Args:
        job_name: Job name
        script_path: Path to write SLURM script
        python_script: Path to Python script to execute
        output_dir: Output directory for logs
        num_nodes: Number of nodes
        num_tasks: Number of tasks
        cpus_per_task: CPUs per task
        memory_gb: Memory in GB
        time_hours: Time limit in hours
        partition: Partition name
        additional_options: Additional SLURM options
        
    Returns:
        Path to generated script
    """
    script_path = Path(script_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir / f"{job_name}_%j.out"}
#SBATCH --error={output_dir / f"{job_name}_%j.err"}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks={num_tasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory_gb}G
#SBATCH --time={time_hours}:00:00
#SBATCH --partition={partition}
"""
    
    if additional_options:
        for key, value in additional_options.items():
            script_content += f"#SBATCH --{key}={value}\n"
    
    script_content += f"""
# Load modules (adjust as needed)
# module load python/3.12
# module load gcc/11.2.0

# Activate virtual environment (if using)
# source /path/to/venv/bin/activate

# Run Python script
python {python_script} "$@"
"""
    
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    
    return script_path





