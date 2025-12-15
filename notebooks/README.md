# Jupyter Notebooks for AutoStent

## Stent RL Training Notebook

**File**: `stent_rl_training.ipynb`

This notebook provides a complete walkthrough of using Reinforcement Learning for stent design optimization.

### What It Covers

1. **Problem Setup**: Understanding the stent design optimization challenge
2. **Environment Creation**: Setting up the RL environment
3. **Baseline Establishment**: Random search for comparison
4. **RL Training**: Training a PPO agent to optimize designs
5. **Evaluation**: Comparing RL agent vs random search
6. **Visualization**: Comprehensive plots and metrics
7. **Impact Analysis**: Understanding the real-world benefits

### How to Run

1. **Install dependencies**:
   ```bash
   pip install jupyter matplotlib stable-baselines3
   ```

2. **Activate autostent environment**:
   ```bash
   cd /Users/srtgn/Documents/Projects/4c/autostent
   source venv/bin/activate
   ```

3. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

4. **Open the notebook**: Navigate to `notebooks/stent_rl_training.ipynb`

### Expected Outputs

- Training progress logs
- Performance comparison plots
- Best design parameters
- Improvement metrics
- Saved model files (in `./models/stent_ppo/`)

### Notes

- The notebook uses a surrogate geometry backend for faster training
- Switch to real 4C simulations by setting `fourc_executable` parameter
- Training time depends on number of steps (default: 50,000)
- Results are saved automatically for later analysis

