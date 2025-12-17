# AutoStent Quick Start Guide

## Installation

```bash
cd autostent
pip install -e .
```

## Basic Usage

### 1. Generate Stent Geometry

```python
from autostent.geometry import SplineStentGeometry, StentGeometryParameters

# Define parameters
params = StentGeometryParameters(
    diameter=10.0,  # mm
    length=20.0,  # mm
    strut_thickness=0.1,  # mm
    strut_width=0.15,  # mm
    num_struts=12,
    crown_height=1.0,  # mm
    crown_radius=0.5,  # mm
)

# Generate geometry
geometry = SplineStentGeometry(params)
points = geometry.sample_points(num_points_per_strut=100)

# Export mesh
from autostent.geometry import export_to_4c_mesh
mesh_path = export_to_4c_mesh(geometry, Path("stent.vtk"))
```

### 2. Run 4C Simulation

```python
from autostent.simulation import FourCSimulator, SimulationConfig
from autostent.automation import FourCYAMLGenerator

# Generate YAML
yaml_gen = FourCYAMLGenerator()
yaml_path = yaml_gen.generate(
    geometry_params=params,
    mesh_path=mesh_path,
    output_path=Path("input.4C.yaml"),
)

# Run simulation
simulator = FourCSimulator()
config = SimulationConfig(
    yaml_input_path=yaml_path,
    output_directory=Path("results"),
    timeout=600.0,
)
result = simulator.run_simulation(config)

# Check results
print(f"Success: {result.success}")
print(f"Max stress: {result.max_von_mises_stress} MPa")
print(f"Converged: {result.converged}")
```

### 3. RL Environment

```python
from autostent.rl import StentDesignEnv

# Create environment
env = StentDesignEnv(
    fourc_executable="fourc",  # Path to 4C executable
    max_episode_steps=50,
)

# Reset
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")

# Step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(f"Reward: {reward}")
print(f"Metrics: {info['metrics']}")
```

### 4. End-to-End Pipeline

```python
from autostent.pipelines import DesignPipeline
from autostent.data import load_patient_data

# Initialize pipeline
pipeline = DesignPipeline(fourc_executable="fourc")

# Load patient data (placeholder - extend for real data)
from autostent.data import PatientData
import numpy as np

patient_data = PatientData(
    patient_id="patient_001",
    vessel_centerline=np.array([[0, 0, 0], [0, 0, 30]]),
    vessel_diameter=10.0,
)

# Run pipeline
results = pipeline.run(patient_data)
print(f"Design metrics: {results['metrics']}")
```

### 5. Batch Processing

```python
from autostent.automation import BatchProcessor
from autostent.geometry import StentGeometryParameters

# Initialize processor
processor = BatchProcessor(fourc_executable="fourc")

# Define parameter sweep
base_params = StentGeometryParameters(
    diameter=10.0,
    length=20.0,
    strut_thickness=0.1,
    strut_width=0.15,
    num_struts=12,
    crown_height=1.0,
    crown_radius=0.5,
)

parameter_ranges = {
    "diameter": [9.0, 10.0, 11.0],
    "strut_thickness": [0.08, 0.1, 0.12],
}

# Run sweep
results = processor.run_parameter_sweep(
    parameter_ranges,
    base_params,
    output_dir=Path("sweep_results"),
)

print(f"Completed {len(results)} simulations")
```

### 6. HPC Job Generation

```python
from autostent.automation import generate_slurm_job

# Generate SLURM script
script_path = generate_slurm_job(
    job_name="stent_optimization",
    script_path=Path("job.sh"),
    python_script=Path("train_rl.py"),
    output_dir=Path("slurm_logs"),
    num_nodes=1,
    num_tasks=4,
    cpus_per_task=8,
    memory_gb=32,
    time_hours=24,
)

print(f"Generated job script: {script_path}")
# Submit with: sbatch job.sh
```

## Next Steps

1. **Extend Patient Data Loading**: Implement real DICOM/STL loaders in `data/patient_data.py`
2. **Customize Reward Function**: Modify `rl/stent_env.py` `_compute_reward()` method
3. **Add Result Parsing**: Implement VTK/VTU parsing in `simulation/result_parser.py`
4. **Train RL Agent**: Use any Gymnasium-compatible RL library (Stable-Baselines3, etc.)

## Requirements

- Python 3.10+
- 4C Multiphysics executable in PATH
- NumPy, SciPy, PyVista, Gymnasium

## Research Extensions

- Multi-objective optimization
- Uncertainty quantification
- Surrogate modeling
- Active learning strategies



