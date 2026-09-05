# AutoStent Architecture

## Overview

AutoStent is a research-grade software framework for autonomous stent design optimization. It integrates finite element multiphysics simulations (4C), reinforcement learning, and patient-specific modeling into a modular, HPC-ready system.

## Module Structure

### `/data`
**Purpose**: Patient-specific data handling

- `patient_data.py`: Data structures and loaders for patient geometries
- Abstracted interface for medical imaging data (DICOM, STL, etc.)
- Vessel centerline extraction and processing

### `/geometry`
**Purpose**: Spline-based geometry generation

- `spline_stent.py`: Parametric B-spline stent geometry generator
- `mesh_export.py`: Export to 4C-compatible mesh formats
- Control point manipulation for RL exploration
- Geometric validation and constraint enforcement

### `/simulation`
**Purpose**: 4C FEM interface

- `fourc_interface.py`: Execution and result parsing
- `result_parser.py`: Extract mechanical quantities from VTK/VTU outputs
- Robust error handling and logging
- Timeout and resource management

### `/rl`
**Purpose**: Reinforcement learning environment

- `stent_env.py`: Gymnasium-compatible environment
- `observation_space.py`: Observation encoding from simulation results
- `action_space.py`: Action space for parameter modification
- Each step: modify geometry → run FEM → compute reward

### `/automation`
**Purpose**: Workflow automation

- `yaml_generator.py`: Generate valid 4C YAML input files
- `batch_processor.py`: Parameter sweeps and batch execution
- `hpc_jobs.py`: SLURM job script generation
- Reproducible, script-driven workflows

### `/evaluation`
**Purpose**: Biomechanical metrics

- `metrics.py`: Compute performance metrics from FEM results
- Stress, displacement, strain analysis
- Safety factor calculations
- Convergence assessment

### `/pipelines`
**Purpose**: End-to-end workflows

- `design_pipeline.py`: Complete patient-to-design pipeline
- Integrates all modules into unified workflow
- Patient data → geometry → simulation → evaluation

## Design Principles

1. **Modularity**: Each module is independently testable
2. **Physical Validity**: All computations are physically meaningful
3. **HPC-Ready**: Non-interactive, batch-safe execution
4. **Research-Grade**: Extensible for research use
5. **No Mocks**: Real FEM integration, no fake data

## Data Flow

```
Patient Data
    ↓
Geometry Generation (Spline-based)
    ↓
Mesh Export (VTK/VTP)
    ↓
4C YAML Generation
    ↓
FEM Simulation (4C)
    ↓
Result Parsing (VTK/VTU)
    ↓
Biomechanical Metrics
    ↓
RL Reward Computation
    ↓
Design Optimization
```

## Integration Points

- **4C Solver**: External executable, called via subprocess
- **RL Framework**: Gymnasium-compatible, works with any RL library
- **HPC Systems**: SLURM job generation for batch execution
- **Medical Imaging**: Abstracted interface (extendable for DICOM, etc.)

## Extension Points

- Custom reward functions in `evaluation/metrics.py`
- Additional geometry generators in `geometry/`
- Alternative FEM solvers via `simulation/` interface
- Patient data loaders in `data/`





