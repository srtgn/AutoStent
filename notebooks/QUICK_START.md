# Quick Start: Running the RL Training Notebook

## Prerequisites

1. **Install Jupyter and dependencies**:
   ```bash
   cd autostent
   source venv/bin/activate
   pip install jupyter matplotlib stable-baselines3
   ```

2. **Verify autostent is installed**:
   ```bash
   python -c "import autostent; print('✓ autostent installed')"
   ```

## Running the Notebook

1. **Start Jupyter**:
   ```bash
   cd 4c
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `notebooks/stent_rl_training.ipynb`
   - Click to open

3. **Run cells sequentially**:
   - Use `Shift + Enter` to run each cell
   - Or use `Cell → Run All` to run everything

## What the Notebook Does

1. **Sets up the RL environment** - Creates Gymnasium-compatible environment
2. **Establishes baseline** - Runs random search for comparison
3. **Trains RL agent** - Uses PPO to learn optimal policy
4. **Evaluates performance** - Compares RL vs random search
5. **Visualizes results** - Creates comprehensive plots
6. **Analyzes impact** - Shows real-world benefits

## Expected Runtime

- **Baseline (random search)**: ~2-5 minutes
- **RL Training**: ~10-30 minutes (depending on timesteps)
- **Evaluation**: ~2-5 minutes
- **Total**: ~15-40 minutes

## Output Files

After running, you'll have:
- `stent_rl_results.png` - Visualization plots
- `./models/stent_ppo/` - Trained model files
- `./logs/stent_ppo/` - Training logs
- `./tensorboard_logs/` - TensorBoard logs (optional)

## Troubleshooting

### Import Errors
```bash
# Make sure autostent is installed
cd autostent
source venv/bin/activate
pip install -e .
```

### Missing stable-baselines3
```bash
pip install stable-baselines3
```

### Environment Errors
- The notebook uses surrogate geometry by default (fast)
- For real 4C simulations, set `fourc_executable="fourc"` in environment creation

## Next Steps

After running the notebook:
1. Analyze the results plots
2. Compare RL vs random search performance
3. Examine best design parameters
4. Consider hyperparameter tuning
5. Try with real 4C simulations





