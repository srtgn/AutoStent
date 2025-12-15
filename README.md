# AutoStent - Autonomous Stent Design with Reinforcement Learning

A research-grade software framework for autonomous stent design optimization using Reinforcement Learning and FEM simulation.

## Overview

AutoStent integrates:
- **Reinforcement Learning** (PPO) for design optimization
- **Physics-based simulation** (4C Multiphysics compatible)
- **Spline-based geometry generation**
- **Interactive web interfaces** for visualization

## Project Structure

```
├── autostent/           # Core Python package
│   ├── geometry/        # Spline-based stent geometry
│   ├── rl/              # Gymnasium RL environment
│   ├── simulation/      # 4C FEM interface + surrogate
│   ├── automation/      # YAML generation, batch processing
│   └── evaluation/      # Biomechanical metrics
├── notebooks/           # Jupyter notebooks for training
├── web_app/             # Streamlit web application
├── web_app_react/       # Modern Vue.js web application
├── src/                 # Additional source code
└── docs/                # Documentation
```

## Quick Start

### 1. Install Dependencies

```bash
cd autostent
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install jupyter matplotlib stable-baselines3 streamlit plotly
```

### 2. Run Jupyter Notebook

```bash
cd ..
jupyter notebook notebooks/stent_rl_training.ipynb
```

### 3. Run Web App (Streamlit)

```bash
streamlit run web_app/app.py
```

### 4. Run Modern Web App (Vue.js)

```bash
cd web_app_react
python3 -m http.server 3000
# Open http://localhost:3000
```

## Features

### RL Environment
- Gymnasium-compatible `StentDesignEnv`
- 12D observation space (geometry + simulation results)
- 7D action space (parameter modifications)
- Physics-based reward function

### Physics-Based Surrogate
When 4C is not available, uses simplified but physically-meaningful models:
- Strut thickness ↔ stress relationship
- Diameter ↔ hoop stress
- Number of struts ↔ load distribution

### Web Interfaces
- **Streamlit**: Python-native, easy to extend
- **Vue.js**: Modern dark UI with 3D visualization

## Requirements

- Python 3.10+
- NumPy, SciPy
- PyTorch (via stable-baselines3)
- Gymnasium
- PyVista (for mesh export)
- Streamlit, Plotly (for web UI)

## License

Research use only.
