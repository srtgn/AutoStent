# Getting Started with 4C Stent RL

This guide will help you get started with the automated YAML generator and RL environment for 4C Multiphysics stent design optimization.

## Installation

### Prerequisites

- Python 3.10 or higher
- Conda (recommended) or pip
- 4C Multiphysics (optional, for real simulations)

### Step 1: Create Conda Environment

```bash
conda create -n 4c-stent-rl python=3.12
conda activate 4c-stent-rl
```

### Step 2: Install Dependencies

```bash
# Install from requirements
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Step 3: Verify Installation

```python
from fourc_automation import StentYAMLGenerator
from gym_4c_stent import StentDesignEnv

print("Installation successful!")
```

## Quick Start

### 1. Generate YAML Files

```python
from fourc_automation import StentYAMLGenerator, StentParameters

# Create generator
generator = StentYAMLGenerator()

# Define stent design
params = StentParameters(
    stent_diameter=10.0,  # mm
    stent_length=20.0,    # mm
    strut_thickness=0.1,  # mm
    number_of_struts=12,
)

# Generate YAML
yaml_dict = generator.generate_yaml(params, "stent_design.yaml")
```

### 2. Use RL Environment

```python
from gym_4c_stent import StentDesignEnv

# Create environment (with mock simulator for testing)
env = StentDesignEnv(use_mock_simulator=True)

# Reset and run
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

### 3. Train RL Agent

```python
from stable_baselines3 import PPO
from gym_4c_stent import StentDesignEnv

env = StentDesignEnv(use_mock_simulator=True)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("stent_ppo")
```

## Examples

See the `examples/` directory for complete examples:

- `basic_yaml_generation.py` - Generate YAML files
- `basic_rl_environment.py` - Use the RL environment
- `train_rl_agent.py` - Train an RL agent

## Next Steps

1. **Explore Examples**: Run the example scripts
2. **Customize Parameters**: Modify stent design parameters
3. **Integrate with 4C**: Set `use_mock_simulator=False` and configure 4C executable
4. **Train Agents**: Experiment with different RL algorithms
5. **Contribute**: Improve the code and submit pull requests

## Troubleshooting

### Import Errors

If you get import errors, make sure you've installed the package:
```bash
pip install -e .
```

### 4C Not Found

If using real 4C simulations, set the executable path:
```python
env = StentDesignEnv(
    fourc_executable="/path/to/fourc",
    use_mock_simulator=False,
)
```

### Mock Simulator

For testing without 4C, always use `use_mock_simulator=True`.




