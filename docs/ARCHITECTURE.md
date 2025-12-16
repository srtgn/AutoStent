# Architecture Overview

## Package Structure

```
4c-stent-rl/
├── src/
│   ├── fourc_automation/      # YAML generation and automation
│   │   ├── __init__.py
│   │   ├── yaml_generator.py   # Core YAML generation
│   │   ├── geometry.py         # Spline-based geometry
│   │   └── cli.py              # Command-line interface
│   │
│   └── gym_4c_stent/          # RL environment
│       ├── __init__.py
│       ├── env.py              # Gymnasium environment
│       └── simulator.py        # 4C simulation interface
│
├── examples/                   # Example scripts
├── tests/                      # Unit tests
└── docs/                      # Documentation
```

## Components

### 1. YAML Generator (`fourc_automation`)

**Purpose**: Automate generation of 4C YAML input files

**Key Classes**:
- `StentParameters`: Dataclass for stent design parameters
- `StentYAMLGenerator`: Generates YAML files from parameters
- `SplineGeometryGenerator`: Creates spline-based geometries

**Features**:
- Parameterized template generation
- Batch processing
- Validation
- RL integration helpers (state/action conversion)

### 2. RL Environment (`gym_4c_stent`)

**Purpose**: Gymnasium-compatible environment for RL training

**Key Classes**:
- `StentDesignEnv`: Main environment class
- `FourCSimulator`: Interface to 4C simulations
- `SimulationResult`: Results container

**Features**:
- Standard Gymnasium API
- Mock simulator for testing
- Multi-objective reward function
- Constraint handling
- Episode history tracking

## Data Flow

```
RL Agent
   │
   ├─> Action (parameter modifications)
   │
   ▼
StentDesignEnv
   │
   ├─> Convert action to parameters
   │
   ├─> StentYAMLGenerator.generate_yaml()
   │
   ├─> FourCSimulator.run_simulation()
   │
   ├─> Parse results
   │
   ├─> Calculate reward
   │
   └─> Return observation, reward, done, info
```

## State and Action Spaces

### Observation Space
- 7 normalized design parameters [0, 1]
- `[diameter, length, strut_thickness, strut_width, n_struts, crown_height, crown_radius]`

### Action Space
- 7 continuous actions [-1, 1]
- Modifications to design parameters
- Scaled to 10% max change per step

### Reward Function
Multi-objective optimization:
- **Stress**: Minimize (safety)
- **Compliance**: Maximize (flexibility)
- **Volume**: Minimize (efficiency)
- **Constraints**: Penalty for violations

## Integration Points

### With 4C Multiphysics
- YAML input files
- Subprocess execution
- VTK/JSON output parsing

### With RL Libraries
- Gymnasium-compatible
- Works with Stable-Baselines3, Ray RLlib, etc.
- Supports vectorized environments

### With QUEENS (Future)
- Uncertainty quantification
- Sensitivity analysis
- Bayesian optimization

## Extension Points

1. **Custom Reward Functions**: Modify `_calculate_reward()`
2. **Additional Constraints**: Extend `_check_constraints()`
3. **New Parameters**: Add to `StentParameters`
4. **Geometry Types**: Extend `SplineGeometryGenerator`
5. **Simulation Backends**: Implement custom `FourCSimulator`


