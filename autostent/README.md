# AutoStent - An Autonomous Design Assistant for Aneurysm Repair

Research-grade software framework for autonomous stent design optimization.

## Architecture

```
autostent/
├── data/          # Patient-specific data handling
├── geometry/      # Spline-based geometry generation
├── simulation/    # 4C FEM interface
├── rl/           # Reinforcement learning environment
├── automation/    # Workflow automation (YAML, batch, HPC)
├── evaluation/    # Biomechanical metrics
└── pipelines/     # End-to-end workflows
```

## Key Features

- **Spline-based Geometry**: Parametric B-spline representations for flexible stent design
- **4C FEM Integration**: Direct interface to 4C Multiphysics solver
- **Gymnasium RL Environment**: Standard RL interface for optimization
- **Automated Workflows**: Batch processing and HPC job generation
- **Patient-Specific**: End-to-end pipeline from patient data to optimized design

## Usage

### Basic Geometry Generation

```python
from autostent.geometry import SplineStentGeometry, StentGeometryParameters

params = StentGeometryParameters(
    diameter=10.0,
    length=20.0,
    strut_thickness=0.1,
    strut_width=0.15,
    num_struts=12,
    crown_height=1.0,
    crown_radius=0.5,
)

geometry = SplineStentGeometry(params)
points = geometry.sample_points()
```

### RL Environment

```python
from autostent.rl import StentDesignEnv

env = StentDesignEnv(
    fourc_executable="fourc",
    max_episode_steps=50,
)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### End-to-End Pipeline

```python
from autostent.pipelines import DesignPipeline
from autostent.data import load_patient_data

pipeline = DesignPipeline()
patient_data = load_patient_data(Path("patient_data.json"))
results = pipeline.run(patient_data)
```

## Requirements

- Python 3.10+
- NumPy, SciPy
- PyVista (for mesh export)
- Gymnasium (for RL)
- 4C Multiphysics (external solver)

## Installation

```bash
pip install -e .
```

## Research Focus

This framework is designed for research use. All components are:
- Modular and independently testable
- HPC-ready (batch execution, SLURM support)
- Physically meaningful (no mock data)
- Extensible for PhD-level research



