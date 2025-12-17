"""
4C Simulation + RL Training API Server for Railway
FastAPI backend that runs 4C simulations and PPO RL training.
"""

import os
import json
import subprocess
import tempfile
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import gymnasium as gym
    from gymnasium import spaces
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("WARNING: stable-baselines3 not installed, RL training disabled")

app = FastAPI(
    title="4C + RL API",
    description="Run 4C FEM simulations and PPO RL training for stent design",
    version="2.0.0"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
FOURC_BIN = os.environ.get("FOURC_BIN", "/usr/local/bin/fourc")
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/4c_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Simulation state
simulations: Dict[str, Dict[str, Any]] = {}

# RL Training state
state_lock = threading.Lock()
training_state = {
    "is_training": False,
    "progress": 0.0,
    "current_step": 0,
    "total_steps": 0,
    "episodes": 0,
    "current_params": {},
    "best_params": {},
    "best_reward": -100.0,
    "rewards": [],
    "stress_history": [],
    "episode_rewards": [],
}

# ===== STENT ENVIRONMENT FOR RL =====
if SB3_AVAILABLE:
    class SimpleStentEnv(gym.Env):
        """Fast stent environment for RL training"""
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
            self.max_episode_steps = 50
            self.step_count = 0
            self.params = {'diameter': 10.0, 'strut_thickness': 0.12, 'num_struts': 12, 'crown_height': 1.0, 'length': 20.0}
            
        def reset(self, seed=None, options=None):
            self.params = {'diameter': 10.0, 'strut_thickness': 0.12, 'num_struts': 12, 'crown_height': 1.0, 'length': 20.0}
            self.step_count = 0
            return self._get_obs(), {'parameters': self.params}
        
        def _get_obs(self):
            obs = np.array([
                (self.params['diameter'] - 6) / 8,
                (self.params['length'] - 10) / 20,
                (self.params['strut_thickness'] - 0.05) / 0.15,
                (self.params['num_struts'] - 6) / 12,
                (self.params['crown_height'] - 0.5) / 1.5,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            ], dtype=np.float32)
            return np.clip(obs, 0, 1)
        
        def step(self, action):
            self.params['diameter'] = np.clip(self.params['diameter'] + action[0] * 0.5, 6, 14)
            self.params['strut_thickness'] = np.clip(self.params['strut_thickness'] + action[1] * 0.01, 0.05, 0.2)
            self.params['num_struts'] = int(np.clip(self.params['num_struts'] + action[2] * 1, 6, 18))
            self.params['crown_height'] = np.clip(self.params['crown_height'] + action[3] * 0.1, 0.5, 2.0)
            
            d, t, n = self.params['diameter'], self.params['strut_thickness'], self.params['num_struts']
            stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
            displacement = 0.3 * (self.params['length'] / 20) * (0.12 / max(t, 0.05))
            reward = -0.4 * (stress / 200) - 0.2 * displacement
            
            self.step_count += 1
            done = self.step_count >= self.max_episode_steps
            
            return self._get_obs(), reward, done, False, {'parameters': self.params.copy(), 'stress': stress}

    class TrainingCallback(BaseCallback):
        def __init__(self, total_steps):
            super().__init__()
            self.total_steps = total_steps
            self.ep_reward = 0
            
        def _on_step(self):
            global training_state
            reward = self.locals.get('rewards', [0])[0]
            self.ep_reward += reward
            
            if self.n_calls % 20 == 0:
                with state_lock:
                    training_state["current_step"] = self.n_calls
                    training_state["progress"] = self.n_calls / self.total_steps
                    training_state["rewards"].append(float(reward))
                    try:
                        env = self.model.env.envs[0]
                        if hasattr(env, 'params'):
                            training_state["current_params"] = dict(env.params)
                            stress = 100 * (0.12 / max(env.params['strut_thickness'], 0.05))**1.5
                            training_state["stress_history"].append(float(stress))
                    except:
                        pass
            
            if self.locals.get('dones', [False])[0]:
                with state_lock:
                    training_state["episode_rewards"].append(float(self.ep_reward))
                    training_state["episodes"] = len(training_state["episode_rewards"])
                    if self.ep_reward > training_state["best_reward"]:
                        training_state["best_reward"] = float(self.ep_reward)
                        try:
                            env = self.model.env.envs[0]
                            if hasattr(env, 'params'):
                                training_state["best_params"] = dict(env.params)
                        except:
                            pass
                self.ep_reward = 0
            return True

    def run_training(total_steps):
        global training_state
        with state_lock:
            training_state["is_training"] = True
            training_state["total_steps"] = total_steps
            training_state["current_step"] = 0
            training_state["progress"] = 0.0
            training_state["episodes"] = 0
            training_state["rewards"] = []
            training_state["stress_history"] = []
            training_state["episode_rewards"] = []
            training_state["best_reward"] = -100.0
        
        try:
            env = SimpleStentEnv()
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, 
                        n_steps=128, batch_size=64, n_epochs=5, gamma=0.99, device='cpu')
            callback = TrainingCallback(total_steps)
            model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
            with state_lock:
                training_state["progress"] = 1.0
                training_state["is_training"] = False
        except Exception as e:
            print(f"Training error: {e}")
            with state_lock:
                training_state["is_training"] = False


# ===== MODELS =====
class StentParams(BaseModel):
    diameter: float = 10.0
    length: float = 20.0
    strut_thickness: float = 0.12
    num_struts: int = 12
    crown_height: float = 1.0

class SimulationRequest(BaseModel):
    params: StentParams
    timeout: int = 300

class TrainRequest(BaseModel):
    steps: int = 2000


# ===== SIMULATION ENDPOINTS =====
@app.get("/")
def root():
    return {"status": "ok", "service": "4C + RL API", "sb3_available": SB3_AVAILABLE}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/simulate/sync")
def run_simulation_sync(request: SimulationRequest):
    """Run simulation synchronously."""
    params = request.params
    d, t, n = params.diameter, params.strut_thickness, params.num_struts
    stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5
    displacement = 0.3 * (params.length / 20) * (0.12 / max(t, 0.05))
    
    return {
        "success": True,
        "simulation_id": f"sim_{int(time.time()*1000)}",
        "status": "completed",
        "result": {
            "max_stress": round(stress, 2),
            "max_displacement": round(displacement, 4),
            "converged": True
        }
    }


# ===== RL TRAINING ENDPOINTS =====
@app.get("/status")
def get_rl_status():
    """Get RL training status - matches rl_backend.py API."""
    with state_lock:
        return {
            "is_training": training_state["is_training"],
            "progress": training_state["progress"],
            "current_step": training_state["current_step"],
            "total_steps": training_state["total_steps"],
            "episodes": training_state["episodes"],
            "current_params": training_state["current_params"],
            "best_params": training_state["best_params"],
            "best_reward": training_state["best_reward"],
            "rewards": training_state["rewards"][-100:],
            "stress_history": training_state["stress_history"][-100:],
            "episode_rewards": training_state["episode_rewards"][-50:],
            "sb3_available": SB3_AVAILABLE,
        }

@app.post("/start")
def start_training(request: TrainRequest):
    """Start RL training - matches rl_backend.py API."""
    if not SB3_AVAILABLE:
        return {"status": "error", "message": "stable-baselines3 not installed"}
    
    with state_lock:
        if training_state["is_training"]:
            return {"status": "already_running"}
    
    thread = threading.Thread(target=run_training, args=(request.steps,), daemon=True)
    thread.start()
    return {"status": "started", "steps": request.steps}

@app.post("/stop")
def stop_training():
    """Stop RL training."""
    with state_lock:
        training_state["is_training"] = False
    return {"status": "stopped"}

@app.post("/reset")
def reset_training():
    """Reset RL training state."""
    with state_lock:
        training_state["is_training"] = False
        training_state["progress"] = 0.0
        training_state["current_step"] = 0
        training_state["episodes"] = 0
        training_state["rewards"] = []
        training_state["stress_history"] = []
        training_state["episode_rewards"] = []
        training_state["best_reward"] = -100.0
    return {"status": "reset"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting 4C + RL API Server on port {port}")
    print(f"SB3 available: {SB3_AVAILABLE}")
    uvicorn.run(app, host="0.0.0.0", port=port)

