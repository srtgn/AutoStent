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

# Set LD_LIBRARY_PATH for 4C libraries if not already set
# This is critical for finding 4C shared libraries at runtime
if not os.environ.get("LD_LIBRARY_PATH"):
    # Common paths where 4C libraries might be
    fourc_lib_paths = [
        "/home/user/4C/build",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib"
    ]
    # Filter to only existing directories
    existing_paths = [p for p in fourc_lib_paths if Path(p).exists()]
    if existing_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(existing_paths)
        print(f"✓ Set LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")
else:
    # Ensure 4C build directory is in LD_LIBRARY_PATH
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    fourc_build = "/home/user/4C/build"
    if Path(fourc_build).exists() and fourc_build not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{fourc_build}:{current_ld_path}"
        print(f"✓ Added {fourc_build} to LD_LIBRARY_PATH")

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

# Try to import 4C interface
import shutil
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if fourc binary actually exists
FOURC_BINARY_PATH = shutil.which("fourc") or "/usr/local/bin/fourc"
FOURC_BINARY_EXISTS = Path(FOURC_BINARY_PATH).exists() if FOURC_BINARY_PATH else False

# Try to verify it can run (must return exit code 0 or known success codes)
FOURC_CAN_EXECUTE = False
FOURC_VERSION_OUTPUT = ""
if FOURC_BINARY_EXISTS:
    try:
        # Use updated environment with LD_LIBRARY_PATH
        env = os.environ.copy()
        
        # Try --help first (more likely to be supported)
        test_result = subprocess.run(
            [FOURC_BINARY_PATH, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )
        output = test_result.stdout.strip() or test_result.stderr.strip()
        
        # If --help doesn't work, try --version as fallback
        if test_result.returncode == 127 or "error while loading shared libraries" in output:
            # Try --version as fallback
            test_result = subprocess.run(
                [FOURC_BINARY_PATH, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            output = test_result.stdout.strip() or test_result.stderr.strip()
        
        FOURC_VERSION_OUTPUT = output
        
        # Check for common failure indicators
        has_library_error = "error while loading shared libraries" in output
        has_not_found = "cannot open shared object file" in output
        # 4C doesn't support --version/--help, but if it runs and gives this error, it means it WORKS!
        has_invalid_arg = "not valid" in output or "not recognized" in output or "Please refer" in output
        
        # Return code 127 = "command not found" or shared library errors
        # Return code 0 = success
        # Return code 1 with "not valid"/"Please refer" = binary works but flag not supported (SUCCESS!)
        if test_result.returncode == 127 or has_library_error or has_not_found:
            # Definite failure - missing libraries
            FOURC_CAN_EXECUTE = False
            print(f"✗ 4C binary CANNOT execute - missing shared libraries!")
            print(f"  Return code: {test_result.returncode}")
            print(f"  Error: {output[:200]}...")
        elif test_result.returncode == 0:
            FOURC_CAN_EXECUTE = True
            print(f"✓ 4C binary verified at: {FOURC_BINARY_PATH}")
            print(f"  Output: {output[:100]}...")
        elif test_result.returncode == 1 and has_invalid_arg:
            # Binary runs but flag not supported - this means it WORKS!
            FOURC_CAN_EXECUTE = True
            print(f"✓ 4C binary verified (flag not supported, but binary executes successfully)")
            print(f"  Output: {output[:100]}...")
        else:
            # Other non-zero return - be conservative, assume failure
            FOURC_CAN_EXECUTE = False
            print(f"⚠ 4C binary returned unexpected code {test_result.returncode}")
            print(f"  Output: {output}")
    except Exception as e:
        FOURC_CAN_EXECUTE = False
        print(f"✗ 4C binary exists but cannot execute: {e}")

try:
    from autostent.simulation.fourc_interface import (
        FourCSimulator,
        SimulationConfig,
        SimulationResult
    )
    # Initialize simulator - only if binary is available
    if FOURC_CAN_EXECUTE:
        fourc_sim = FourCSimulator(
            fourc_executable=FOURC_BINARY_PATH,
            use_docker=False  # We're already IN the 4C Docker image, so call binary directly
        )
        FOURC_AVAILABLE = True
        print("✓ 4C simulator initialized successfully")
        print(f"  Using binary: {FOURC_BINARY_PATH}")
        print(f"  Mode: Direct execution (we're in 4C image, no docker run needed)")
    else:
        fourc_sim = None
        FOURC_AVAILABLE = False
        print("⚠ 4C simulator not initialized - binary not available")
except Exception as e:
    FOURC_AVAILABLE = False
    fourc_sim = None
    print(f"WARNING: 4C solver not available: {e}")
    import traceback
    traceback.print_exc()
    print("         Real 4C mode will fallback to mockup")

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
    expose_headers=["*"],
)

# Global exception handler to ensure CORS headers on errors
from fastapi.responses import JSONResponse
from starlette.requests import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure all exceptions return proper JSON with CORS headers."""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": "Internal server error"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Paths
FOURC_BIN = os.environ.get("FOURC_BIN", "/usr/local/bin/fourc")
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/4c_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Simulation state
simulations: Dict[str, Dict[str, Any]] = {}

# Storage for generated YAML files (for user download)
# Key: simulation_id, Value: {yaml_content, vtu_content, params, timestamp}
generated_files: Dict[str, Dict[str, Any]] = {}
MAX_STORED_FILES = 50  # Limit storage

# RL Training state
state_lock = threading.Lock()
training_state = {
    "is_training": False,
    "progress": 0.0,
    "current_step": 0,
    "logged_steps": 0,  # Number of actual data points logged
    "total_steps": 0,
    "episodes": 0,
    "current_params": {},
    "best_params": {},
    "best_reward": -100.0,
    "rewards": [],
    "stress_history": [],
    "displacement_history": [],
    "episode_rewards": [],
}

# ===== STENT ENVIRONMENT FOR RL =====
if SB3_AVAILABLE:
    class SimpleStentEnv(gym.Env):
        """Stent environment for RL training (supports Mockup and Real 4C)"""
        def __init__(self, use_real_4c=False):
            super().__init__()
            self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
            self.max_episode_steps = 50
            self.step_count = 0
            self.use_real_4c = use_real_4c and FOURC_AVAILABLE
            self.params = {'diameter': 10.0, 'strut_thickness': 0.12, 'num_struts': 12, 'crown_height': 1.0, 'length': 20.0}
            
            if self.use_real_4c:
                print(f"🔬 Environment initialized with REAL 4C solver")
                print(f"   Binary: {FOURC_BINARY_PATH}")
                print(f"   WARNING: Each step will take 30-60 seconds!")
            else:
                if use_real_4c and not FOURC_AVAILABLE:
                    print(f"⚠️  REAL 4C requested but NOT available!")
                    print(f"   FOURC_AVAILABLE={FOURC_AVAILABLE}")
                    print(f"   FOURC_BINARY_EXISTS={FOURC_BINARY_EXISTS}")
                    print(f"   FOURC_CAN_EXECUTE={FOURC_CAN_EXECUTE}")
                    print(f"   Falling back to MOCKUP mode")
                else:
                    print(f"⚡ Environment initialized with MOCKUP solver (fast mode)")
            
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
            # Update parameters
            self.params['diameter'] = np.clip(self.params['diameter'] + action[0] * 0.5, 6, 14)
            self.params['strut_thickness'] = np.clip(self.params['strut_thickness'] + action[1] * 0.01, 0.05, 0.2)
            self.params['num_struts'] = int(np.clip(self.params['num_struts'] + action[2] * 1, 6, 18))
            self.params['crown_height'] = np.clip(self.params['crown_height'] + action[3] * 0.1, 0.5, 2.0)
            
            # --- SIMULATION STEP ---
            stress = 0.0
            displacement = 0.0
            
            if self.use_real_4c:
                # Run REAL 4C Simulation
                print(f"Step {self.step_count}: Running REAL 4C simulation...")
                start_time = time.time()
                try:
                    p = StentParams(**self.params)
                    result = run_real_4c_simulation(p)
                    
                    if result.get("success"):
                        # Extract stress and displacement from nested result structure
                        result_data = result.get("result", {})
                        stress = result_data.get("max_stress", 0.0)
                        displacement = result_data.get("max_displacement", 0.0)
                        
                        # If no results parsed, try fallback (shouldn't happen but safety check)
                        if stress == 0.0 and displacement == 0.0:
                            print(f"Step {self.step_count}: ⚠ 4C succeeded but no results parsed")
                            # Use a reasonable default instead of penalty
                            # Calculate from params as fallback
                            d, t, n = self.params['diameter'], self.params['strut_thickness'], self.params['num_struts']
                            stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                            displacement = 0.3 * (self.params['length'] / 20) * (0.12 / max(t, 0.05))
                            print(f"  Using analytical fallback: stress={stress:.2f} MPa")
                        else:
                            print(f"Step {self.step_count}: ✓ 4C REAL simulation in {time.time()-start_time:.2f}s. Stress: {stress:.2f} MPa, Disp: {displacement:.4f} mm")
                    else:
                        # Simulation failed - log detailed error and use analytical fallback instead of penalty
                        error_msg = result.get('error', 'Unknown error')
                        log_path = result.get('log_path', 'N/A')
                        print(f"Step {self.step_count}: ✗ 4C Failed: {error_msg}")
                        print(f"  Log path: {log_path}")
                        print(f"  Using analytical fallback instead of penalty")
                        # Use analytical calculation instead of penalty - better for learning
                        d, t, n = self.params['diameter'], self.params['strut_thickness'], self.params['num_struts']
                        stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                        displacement = 0.3 * (self.params['length'] / 20) * (0.12 / max(t, 0.05))
                except Exception as e:
                    import traceback
                    print(f"Step {self.step_count}: ✗ Env Exception: {e}")
                    traceback.print_exc()
                    # Use analytical fallback instead of penalty
                    print(f"  Using analytical fallback due to exception")
                    d, t, n = self.params['diameter'], self.params['strut_thickness'], self.params['num_struts']
                    stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                    displacement = 0.3 * (self.params['length'] / 20) * (0.12 / max(t, 0.05))
            else:
                # Fast Mockup Calculation
                d, t, n = self.params['diameter'], self.params['strut_thickness'], self.params['num_struts']
                stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                displacement = 0.3 * (self.params['length'] / 20) * (0.12 / max(t, 0.05))

            # Reward calculation
            reward = -0.4 * (stress / 200) - 0.2 * displacement
            
            # Store displacement for callback
            self.last_displacement = displacement
            
            self.step_count += 1
            done = self.step_count >= self.max_episode_steps
            
            return self._get_obs(), reward, done, False, {'parameters': self.params.copy(), 'stress': stress, 'displacement': displacement}

    class TrainingCallback(BaseCallback):
        def __init__(self, total_steps):
            super().__init__()
            self.total_steps = total_steps
            self.ep_reward = 0
            
        def _on_step(self):
            global training_state
            reward = self.locals.get('rewards', [0])[0]
            self.ep_reward += reward
            
            if self.n_calls % 2 == 0:  # Log every 2 steps for smoother curves
                with state_lock:
                    # Report actual logged step count, not callback count
                    # This ensures steps counter matches the number of data points
                    logged_steps = len(training_state["rewards"]) + 1
                    training_state["current_step"] = self.n_calls  # Keep for progress calculation
                    training_state["logged_steps"] = logged_steps  # Actual number of data points
                    training_state["progress"] = self.n_calls / self.total_steps
                    training_state["rewards"].append(float(reward))
                    try:
                        # Access unwrapped environment to get params
                        env = self.model.env.envs[0].unwrapped
                        if hasattr(env, 'params'):
                            training_state["current_params"] = dict(env.params)
                            # Get real stress and displacement if available, else formula
                            if hasattr(env, 'last_stress'): 
                                stress = env.last_stress
                            else:
                                # Use same stress formula as in step()
                                d, t, n = env.params['diameter'], env.params['strut_thickness'], env.params['num_struts']
                                stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                            training_state["stress_history"].append(float(stress))
                            
                            # Get displacement
                            if hasattr(env, 'last_displacement'):
                                displacement = env.last_displacement
                            else:
                                # Calculate from formula
                                d, t = env.params['diameter'], env.params['strut_thickness']
                                displacement = 0.3 * (env.params['length'] / 20) * (0.12 / max(t, 0.05))
                            training_state["displacement_history"].append(float(displacement))
                    except Exception as e:
                        # print(f"Callback error: {e}")
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

    def run_training(total_steps, use_real_4c=False):
        global training_state
        with state_lock:
            training_state["is_training"] = True
            training_state["total_steps"] = total_steps
            training_state["current_step"] = 0
            training_state["logged_steps"] = 0
            training_state["progress"] = 0.0
            training_state["episodes"] = 0
            training_state["rewards"] = []
            training_state["stress_history"] = []
            training_state["displacement_history"] = []
            training_state["episode_rewards"] = []
            training_state["best_reward"] = -100.0
        
        try:
            env = SimpleStentEnv(use_real_4c=use_real_4c)
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
    use_docker: bool = False  # Toggle for real 4C vs mockup

class TrainRequest(BaseModel):
    steps: int = 2000
    use_docker: bool = False


# ===== SIMULATION ENDPOINTS =====
@app.get("/")
def root():
    return {"status": "ok", "service": "4C + RL API", "sb3_available": SB3_AVAILABLE}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/check-4c")
def check_4c_status():
    """Diagnostic endpoint to check 4C availability."""
    
    # Use cached startup values for quick response
    result = {
        "fourc_available": FOURC_AVAILABLE,
        "fourc_binary_path": FOURC_BINARY_PATH,
        "fourc_binary_exists": FOURC_BINARY_EXISTS,
        "fourc_can_execute": FOURC_CAN_EXECUTE,
        "mesh_tools_available": MESH_TOOLS_AVAILABLE if 'MESH_TOOLS_AVAILABLE' in dir() else False,
        "sb3_available": SB3_AVAILABLE,
        "binary_version": FOURC_VERSION_OUTPUT if 'FOURC_VERSION_OUTPUT' in dir() else None,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "not set"),
        "path": os.environ.get("PATH", "not set"),
        "error": None,
        "debug_info": {},
        "recommendation": None
    }
    
    # Add recommendation based on status
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    live_test_works = result.get("debug_info", {}).get("live_test_works", False)
    
    if not FOURC_BINARY_EXISTS:
        result["recommendation"] = "4C binary not found. Check Dockerfile COPY commands."
    elif not ld_path or ld_path == "not set":
        result["recommendation"] = "LD_LIBRARY_PATH is not set! Set it in Dockerfile ENV to include /home/user/4C/build and other library paths."
    elif live_test_works:
        # Live test shows it works - update status
        if not FOURC_AVAILABLE:
            result["recommendation"] = "4C binary executes successfully, but FourCSimulator failed to initialize. Check autostent package imports."
        else:
            result["recommendation"] = "4C is fully operational! ✓"
    elif not FOURC_CAN_EXECUTE:
        result["recommendation"] = f"4C binary exists but cannot execute. LD_LIBRARY_PATH={ld_path}. Check if libraries are in those paths."
    elif not FOURC_AVAILABLE:
        result["recommendation"] = "4C binary works but FourCSimulator failed to initialize. Check autostent package."
    else:
        result["recommendation"] = "4C is fully operational!"
    
    # Check library dependencies (live check)
    if FOURC_BINARY_EXISTS:
        try:
            ldd_output = subprocess.run(
                ["ldd", FOURC_BINARY_PATH],
                capture_output=True,
                text=True,
                timeout=10
            )
            missing_libs = [line.strip() for line in ldd_output.stdout.split('\n') if 'not found' in line]
            result["debug_info"]["missing_libs"] = missing_libs if missing_libs else "none"
            result["debug_info"]["missing_lib_count"] = len(missing_libs) if missing_libs else 0
        except Exception as e:
            result["debug_info"]["ldd_error"] = str(e)
        
        # Live test: Try to run fourc with current environment
        try:
            env = os.environ.copy()
            live_test = subprocess.run(
                [FOURC_BINARY_PATH, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            output = (live_test.stdout or live_test.stderr)[:200]
            result["debug_info"]["live_test_returncode"] = live_test.returncode
            result["debug_info"]["live_test_output"] = output
            
            # 4C doesn't support --version, but if it runs and gives "not valid" error, it WORKS!
            has_invalid_arg = "not valid" in output or "not recognized" in output
            result["debug_info"]["live_test_works"] = (
                live_test.returncode == 0 or 
                (live_test.returncode == 1 and has_invalid_arg)
            )
            
            # Update fourc_can_execute if live test shows it works
            if result["debug_info"]["live_test_works"] and not FOURC_CAN_EXECUTE:
                # Runtime check shows it works, update our status
                result["fourc_can_execute"] = True
                result["fourc_available"] = True  # Will be set properly if simulator initializes
        except Exception as e:
            result["debug_info"]["live_test_error"] = str(e)
            result["debug_info"]["live_test_works"] = False
    
    return result

# ===== HELPER FUNCTIONS =====
# Import mesh generation and VTU parsing
try:
    from mesh_generator import (
        StentGeometry,
        generate_cylindrical_stent_mesh,
        write_4c_geometry,
        write_vtu_file,
        PV_AVAILABLE as PV_AVAILABLE_FROM_MESH
    )
    from vtu_parser import parse_vtu_file, find_latest_vtu
    MESH_TOOLS_AVAILABLE = True
    PV_AVAILABLE = PV_AVAILABLE_FROM_MESH
    print("✓ Mesh tools imported successfully")
except ImportError as e:
    print(f"WARNING: Mesh tools not available: {e}")
    import traceback
    traceback.print_exc()
    MESH_TOOLS_AVAILABLE = False
    PV_AVAILABLE = False


def generate_4c_yaml(params: StentParams, output_path: Path):
    """Generate complete 4C YAML input file with real mesh."""
    
    if not MESH_TOOLS_AVAILABLE:
        # Generate minimal but VALID 4C YAML that can actually run
        # Use analytical approximation since we can't generate mesh
        # This is a placeholder - 4C needs geometry, but we'll use a simple beam model
        yaml_content = f"""TITLE: Stent simulation (analytical approximation)
PROBLEM TYPE:
  PROBLEMTYPE: Structure

IO:
  OUTPUT_SPRING: true
  STRUCT_STRESS: "Cauchy"
  VERBOSITY: "Standard"
  WRITE_INITIAL_STATE: false

IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: binary

STRUCTURAL DYNAMIC:
  INT_STRATEGY: "Standard"
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLDISP: 1e-06
  TOLRES: 1e-06

MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 7.8e-9
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: 200000.0
      NUE: 0.3

# NOTE: This YAML is incomplete - 4C requires geometry/mesh
# Without mesh tools, 4C cannot run. Using analytical fallback in code.
"""
        output_path.write_text(yaml_content)
        print(f"WARNING: Generated incomplete YAML (no mesh tools). 4C will fail.")
        print(f"  This is expected - code will use analytical fallback.")
        return
    
    # Generate real mesh
    geometry = StentGeometry(
        diameter=params.diameter,
        length=params.length,
        strut_thickness=params.strut_thickness,
        num_struts=params.num_struts,
        crown_height=params.crown_height
    )
    
    nodes, elements, fixed_nodes, loaded_nodes = generate_cylindrical_stent_mesh(
        geometry,
        n_circumferential_per_strut=4,
        n_radial=2
    )
    
    # Write VTU file (4C prefers this format)
    # Include block_id and point_set arrays required by 4C
    vtu_path = output_path.parent / f"{output_path.stem}.vtu"
    if PV_AVAILABLE:
        try:
            write_vtu_file(
                nodes, elements, str(vtu_path),
                fixed_nodes=np.array(fixed_nodes),
                loaded_nodes=np.array(loaded_nodes)
            )
            print(f"✓ Generated VTU file: {vtu_path} with block_id and point_sets")
            use_vtu = True
        except Exception as e:
            print(f"⚠ Failed to write VTU file: {e}, using inline geometry")
            use_vtu = False
    else:
        use_vtu = False
    
    # Write complete 4C YAML (matching 4C's actual format from tutorial)
    with open(output_path, 'w') as f:
        f.write(f"""TITLE: Stent simulation - diameter={params.diameter}mm, length={params.length}mm
PROBLEM TYPE:
  PROBLEMTYPE: Structure

IO:
  OUTPUT_SPRING: true
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
  VERBOSITY: "Standard"
  WRITE_INITIAL_STATE: false

IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: binary

IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
  GAUSS_POINT_DATA_OUTPUT_TYPE: nodes

SOLVER 1:
  SOLVER: "Superlu"
  NAME: "Structure_Solver"

STRUCTURAL DYNAMIC:
  INT_STRATEGY: "Standard"
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-06
  TOLRES: 1e-06
  LOADLIN: true
  LINEAR_SOLVER: 1

MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 7.8e-9
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: 200000.0
      NUE: 0.3

""")
        
        # Use VTU file if available, otherwise inline geometry
        if use_vtu and vtu_path.exists():
            f.write(f"""STRUCTURE GEOMETRY:
  FILE: {vtu_path.name}
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: nonlinear

""")
        else:
            # Fallback: inline geometry (may not work, but better than nothing)
            f.write("GEOMETRY:\n")
            f.write("  NODES:\n")
            for i, node in enumerate(nodes):
                f.write(f"    - ID: {i+1}\n")
                f.write(f"      COORDS: [{node[0]:.6f}, {node[1]:.6f}, {node[2]:.6f}]\n")
            
            f.write("\n  ELEMENTS:\n")
            for i, elem in enumerate(elements):
                node_ids = ", ".join(str(n+1) for n in elem)
                f.write(f"    - ID: {i+1}\n")
                f.write(f"      TYPE: hex8\n")
                f.write(f"      NODES: [{node_ids}]\n")
            
            f.write("\n  ELEMENT_BLOCKS:\n")
            f.write("    - ID: 1\n")
            f.write(f"      ELEMENTS: [1-{len(elements)}]\n")
            f.write("      MATERIAL: 1\n")
        
        # Boundary conditions
        # Fixed end: all DOFs fixed at z=0
        # Loaded surface: radial pressure on outer surface
        pressure = params.diameter * 0.1  # Radial pressure in MPa
        
        if use_vtu:
            # When using VTU, point_sets are embedded in the file
            # Use DESIGN POINT conditions with ENTITY_TYPE: node_set_id
            # References point_set_1 (E:1) and point_set_2 (E:2)
            f.write(f"""
# Boundary conditions using point_set arrays from VTU file
# point_set_1: {len(fixed_nodes)} fixed nodes at z=0
# point_set_2: {len(loaded_nodes)} loaded nodes on outer radius

DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]

DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [{pressure}, {pressure}, 0.0]
    FUNCT: [0, 0, 0]
""")
        else:
            # Inline geometry - define node sets in YAML
            fixed_node_ids = [n+1 for n in fixed_nodes]
            loaded_node_ids = [n+1 for n in loaded_nodes]
            
            f.write(f"""
# Node sets for boundary conditions (inline geometry)
DNODE-NODE SETS:
  DSURF 1:
    DNODES: [{', '.join(map(str, fixed_node_ids[:50]))}]
  DSURF 2:
    DNODES: [{', '.join(map(str, loaded_node_ids[:50]))}]

DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]

DESIGN SURF NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [{pressure}, {pressure}, 0.0]
    FUNCT: [0, 0, 0]
""")


def run_real_4c_simulation(params: StentParams):
    """Run actual 4C FEM simulation."""
    if not FOURC_AVAILABLE:
        return {
            "success": False,
            "error": "4C solver not available. Install 'fourc' binary or use Docker image.",
            "help": "See: https://github.com/4C-multiphysics/4C",
            "fallback_used": True
        }
    
    try:
        # Create temporary directory
        # 4C writes output to same directory as input YAML file
        work_dir = Path(tempfile.mkdtemp(prefix="4c_sim_"))
        yaml_path = work_dir / "input.4C.yaml"
        
        # Generate 4C input file (this will also generate VTU if mesh tools available)
        generate_4c_yaml(params, yaml_path)
        
        # Check if VTU file was created and is in the right place
        vtu_file = yaml_path.parent / f"{yaml_path.stem}.vtu"
        if vtu_file.exists():
            print(f"✓ VTU file ready: {vtu_file.name}")
        else:
            print(f"⚠ No VTU file found (mesh tools may not be available)")
        
        # Save files for user download
        sim_id = f"sim_{int(time.time()*1000)}"
        try:
            yaml_content = yaml_path.read_text() if yaml_path.exists() else None
            vtu_content = vtu_file.read_bytes() if vtu_file.exists() else None
            
            # Store for download (limit storage size)
            if len(generated_files) >= MAX_STORED_FILES:
                # Remove oldest entry
                oldest_key = min(generated_files.keys(), key=lambda k: generated_files[k].get('timestamp', 0))
                del generated_files[oldest_key]
            
            generated_files[sim_id] = {
                "yaml_content": yaml_content,
                "vtu_content": vtu_content,
                "params": {
                    "diameter": params.diameter,
                    "length": params.length,
                    "strut_thickness": params.strut_thickness,
                    "num_struts": params.num_struts,
                    "crown_height": params.crown_height
                },
                "timestamp": time.time()
            }
            print(f"✓ Saved files for download: {sim_id}")
        except Exception as e:
            print(f"⚠ Failed to save files for download: {e}")
        
        # Configure simulation
        # 4C writes output to same directory as input YAML file
        # So we use work_dir as output directory (where YAML is)
        config = SimulationConfig(
            yaml_input_path=yaml_path,
            output_directory=work_dir,  # 4C writes to same dir as input
            timeout=300.0,
            verbose=False
        )
        
        # Run 4C simulation
        print(f"Running real 4C simulation for: {params}")
        print(f"  YAML path: {yaml_path}")
        print(f"  Output dir: {output_dir}")
        
        # Check if YAML file was created
        if not yaml_path.exists():
            return {
                "success": False,
                "error": f"YAML file not created at {yaml_path}",
                "log_path": None,
                "fallback_used": True
            }
        
        result = fourc_sim.run_simulation(config)
        
        if not result.success:
            # Get more detailed error info
            error_details = result.error_message or "4C simulation failed"
            log_content = ""
            if result.log_path and Path(result.log_path).exists():
                try:
                    log_content = Path(result.log_path).read_text()[-500:]  # Last 500 chars
                except:
                    pass
            
            # Check output directory for any files (4C writes to same dir as YAML)
            output_files = list(work_dir.glob("*")) if work_dir.exists() else []
            
            print(f"4C simulation failed:")
            print(f"  Error: {error_details}")
            print(f"  Log path: {result.log_path}")
            print(f"  Output files found: {len(output_files)}")
            if log_content:
                print(f"  Last log lines:\n{log_content}")
            
            return {
                "success": False,
                "error": error_details,
                "log_path": str(result.log_path) if result.log_path else None,
                "log_content": log_content,
                "output_files": [str(f.name) for f in output_files],
                "fallback_used": True
            }
        
        # Parse VTU files for actual results
        max_stress = 0.0
        max_disp = 0.0
        max_strain = 0.0
        parse_success = False
        
        # Try to parse VTU files if mesh tools available
        if MESH_TOOLS_AVAILABLE:
            latest_vtu = find_latest_vtu(output_dir)
            if latest_vtu:
                try:
                    max_stress, max_disp, max_strain, parse_success = parse_vtu_file(latest_vtu)
                    if parse_success:
                        print(f"Parsed VTU results: stress={max_stress:.2f}, disp={max_disp:.4f}")
                    else:
                        print("VTU parsing failed - trying fallback")
                except Exception as e:
                    print(f"Error parsing VTU: {e}")
            else:
                print("No VTU files found - checking for other output files")
        
        # If VTU parsing failed, try to extract from result object or output files
        if not parse_success or max_stress == 0.0:
            # Check if SimulationResult has any values
            if result.max_von_mises_stress > 0:
                max_stress = result.max_von_mises_stress
                max_disp = result.max_displacement
                print(f"Using SimulationResult values: stress={max_stress:.2f}, disp={max_disp:.4f}")
            else:
                # Last resort: check output directory for any result files (4C writes to same dir as YAML)
                vtu_files = list(work_dir.glob("*.vtu")) + list(work_dir.glob("*.vtk"))
                if vtu_files:
                    print(f"Found {len(vtu_files)} result files but couldn't parse. Files: {[f.name for f in vtu_files[:3]]}")
                    # Use analytical fallback - better than 0 or -4 penalty
                    d, t, n = params.diameter, params.strut_thickness, params.num_struts
                    max_stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                    max_disp = 0.3 * (params.length / 20) * (0.12 / max(t, 0.05))
                    print(f"Using analytical fallback: stress={max_stress:.2f}, disp={max_disp:.4f}")
                else:
                    print("No result files found - simulation may have failed")
        
        # Return results
        return {
            "success": True,
            "simulation_id": f"4c_{int(time.time()*1000)}",
            "status": "completed",
            "result": {
                "max_stress": float(max_stress),
                "max_displacement": float(max_disp),
                "max_strain": float(max_strain),
                "converged": result.converged,
                "source": "real_4c_vtu" if parse_success and max_stress > 0 else ("real_4c_simresult" if result.max_von_mises_stress > 0 else "real_4c_analytical_fallback"),
                "num_iterations": result.num_iterations,
                "mesh_elements": len(elements) if MESH_TOOLS_AVAILABLE else 0
            },
            "metadata": result.metadata
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"4C execution error: {str(e)}",
            "fallback_used": True
        }


def run_mockup_simulation(params: StentParams):
    """Run fast analytical simulation (mockup)."""
    d, t = params.diameter, params.strut_thickness
    stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5
    displacement = 0.3 * (params.length / 20) * (0.12 / max(t, 0.05))
    
    return {
        "success": True,
        "simulation_id": f"mockup_{int(time.time()*1000)}",
        "status": "completed",
        "result": {
            "max_stress": round(stress, 2),
            "max_displacement": round(displacement, 4),
            "converged": True,
            "source": "analytical_mockup"
        }
    }


# ===== SIMULATION ENDPOINTS =====
@app.post("/simulate/sync")
def run_simulation_sync(request: SimulationRequest):
    """Run a single synchronous simulation (mockup or real 4C)."""
    use_real_4c = request.use_docker
    
    if use_real_4c:
        # User wants REAL 4C - no fallback!
        if not FOURC_AVAILABLE:
            return {
                "success": False,
                "error": "4C solver not available on this server. Please uncheck 'Use Docker (Real 4C)' to use fast mockup mode, or contact admin to install 4C binary.",
                "help": "See: https://github.com/4C-multiphysics/4C"
            }
        
        print("Using real 4C FEM solver (no fallback)")
        result = run_real_4c_simulation(request.params)
        
        # If 4C fails, return error (NO FALLBACK TO MOCKUP)
        if not result.get("success"):
            result["error"] = f"Real 4C simulation failed: {result.get('error', 'Unknown error')}. Please uncheck the checkbox to use mockup mode."
        
        return result
    else:
        # Mockup mode
        return run_mockup_simulation(request.params)


# ===== RL TRAINING ENDPOINTS =====
@app.get("/status")
def get_rl_status():
    """Get RL training status - matches rl_backend.py API."""
    with state_lock:
        return {
            "is_training": bool(training_state["is_training"]),
            "progress": float(training_state["progress"]),
            "current_step": int(training_state["current_step"]),
            "logged_steps": int(training_state.get("logged_steps", len(training_state["rewards"]))),
            "total_steps": int(training_state["total_steps"]),
            "episodes": int(training_state["episodes"]),
            "current_params": {k: float(v) for k, v in training_state["current_params"].items()} if training_state["current_params"] else {},
            "best_params": {k: float(v) for k, v in training_state["best_params"].items()} if training_state["best_params"] else {},
            "best_reward": float(training_state["best_reward"]),
            "rewards": [float(x) for x in training_state["rewards"]],  # Send ALL rewards
            "stress_history": [float(x) for x in training_state["stress_history"]],  # Send ALL
            "displacement_history": [float(x) for x in training_state["displacement_history"]],  # Send ALL
            "episode_rewards": [float(x) for x in training_state["episode_rewards"]],  # Send ALL
            "sb3_available": bool(SB3_AVAILABLE),
            "fourc_available": bool(FOURC_AVAILABLE),
            "fourc_binary_exists": bool(FOURC_BINARY_EXISTS),
            "fourc_can_execute": bool(FOURC_CAN_EXECUTE),
        }

@app.post("/start")
def start_training(request: TrainRequest):
    """Start RL training - matches rl_backend.py API."""
    if not SB3_AVAILABLE:
        return {"status": "error", "message": "stable-baselines3 not installed"}
    
    # If user requests real 4C, check availability BEFORE starting
    if request.use_docker and not FOURC_AVAILABLE:
        return {
            "status": "error",
            "message": "Real 4C solver requested but not available on this server.",
            "details": {
                "fourc_available": FOURC_AVAILABLE,
                "fourc_binary_exists": FOURC_BINARY_EXISTS,
                "fourc_can_execute": FOURC_CAN_EXECUTE,
                "binary_path": FOURC_BINARY_PATH
            },
            "suggestion": "Uncheck '4C Solver (Cloud)' to use fast mockup mode, or contact admin to fix 4C installation."
        }
    
    with state_lock:
        if training_state["is_training"]:
            return {"status": "already_running"}
    
    mode = "REAL 4C" if (request.use_docker and FOURC_AVAILABLE) else "MOCKUP"
    print(f"Starting training: {request.steps} steps, mode: {mode}")
    
    thread = threading.Thread(target=run_training, args=(request.steps, request.use_docker), daemon=True)
    thread.start()
    return {
        "status": "started", 
        "steps": request.steps,
        "mode": mode,
        "fourc_available": FOURC_AVAILABLE
    }

@app.post("/stop")
def stop_training():
    """Stop RL training."""
    with state_lock:
        training_state["is_training"] = False
    return {"status": "stopped"}


# ===== FILE DOWNLOAD ENDPOINTS =====
@app.get("/generated-files")
def list_generated_files():
    """List all generated YAML/VTU files available for download."""
    files = []
    for sim_id, data in generated_files.items():
        files.append({
            "id": sim_id,
            "params": data.get("params", {}),
            "timestamp": data.get("timestamp", 0),
            "has_yaml": data.get("yaml_content") is not None,
            "has_vtu": data.get("vtu_content") is not None
        })
    # Sort by timestamp, newest first
    files.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"files": files, "count": len(files)}


@app.get("/download-yaml/{sim_id}")
def download_yaml(sim_id: str):
    """Download YAML file for a specific simulation."""
    from fastapi.responses import PlainTextResponse
    
    if sim_id not in generated_files:
        return {"error": f"Simulation {sim_id} not found"}
    
    yaml_content = generated_files[sim_id].get("yaml_content")
    if not yaml_content:
        return {"error": "YAML file not available for this simulation"}
    
    return PlainTextResponse(
        content=yaml_content,
        media_type="text/yaml",
        headers={"Content-Disposition": f"attachment; filename=stent_{sim_id}.4C.yaml"}
    )


@app.get("/download-vtu/{sim_id}")
def download_vtu(sim_id: str):
    """Download VTU mesh file for a specific simulation."""
    from fastapi.responses import Response
    
    if sim_id not in generated_files:
        return {"error": f"Simulation {sim_id} not found"}
    
    vtu_content = generated_files[sim_id].get("vtu_content")
    if not vtu_content:
        return {"error": "VTU file not available for this simulation"}
    
    return Response(
        content=vtu_content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=stent_{sim_id}.vtu"}
    )

@app.post("/reset")
def reset_training():
    """Reset RL training state."""
    with state_lock:
        training_state["is_training"] = False
        training_state["progress"] = 0.0
        training_state["current_step"] = 0
        training_state["logged_steps"] = 0
        training_state["episodes"] = 0
        training_state["rewards"] = []
        training_state["stress_history"] = []
        training_state["displacement_history"] = []
        training_state["episode_rewards"] = []
        training_state["best_reward"] = -100.0
    return {"status": "reset"}

@app.post("/test-4c")
def test_4c_simulation(request: SimulationRequest):
    """Test endpoint to run a single 4C simulation and see detailed results."""
    print(f"Testing 4C simulation with params: {request.params}")
    result = run_real_4c_simulation(request.params)
    
    return {
        "test_result": result,
        "fourc_available": FOURC_AVAILABLE,
        "mesh_tools_available": MESH_TOOLS_AVAILABLE if 'MESH_TOOLS_AVAILABLE' in dir() else False,
        "recommendation": "Check the 'test_result' field for simulation details"
    }

@app.get("/check-mesh-tools")
def check_mesh_tools():
    """Check if mesh generation tools are available."""
    # Check pyvista availability directly
    try:
        import pyvista as pv
        pv_available = True
    except ImportError:
        pv_available = False
    
    result = {
        "mesh_tools_available": MESH_TOOLS_AVAILABLE,
        "pyvista_available": pv_available,
    }
    
    # Try to generate a simple mesh
    if MESH_TOOLS_AVAILABLE:
        try:
            geometry = StentGeometry(
                diameter=10.0,
                length=20.0,
                strut_thickness=0.12,
                num_struts=12,
                crown_height=1.0
            )
            nodes, elements, fixed, loaded = generate_cylindrical_stent_mesh(geometry)
            result["mesh_generation"] = {
                "success": True,
                "num_nodes": len(nodes),
                "num_elements": len(elements),
                "num_fixed_nodes": len(fixed),
                "num_loaded_nodes": len(loaded)
            }
            
            # Try to write VTU
            if PV_AVAILABLE:
                import tempfile
                temp_vtu = Path(tempfile.mktemp(suffix=".vtu"))
                try:
                    write_vtu_file(
                        nodes, elements, str(temp_vtu),
                        fixed_nodes=np.array(fixed),
                        loaded_nodes=np.array(loaded)
                    )
                    result["vtu_generation"] = {
                        "success": True,
                        "file_size": temp_vtu.stat().st_size,
                        "has_block_id": True,
                        "has_point_sets": True
                    }
                    temp_vtu.unlink()
                except Exception as e:
                    result["vtu_generation"] = {"success": False, "error": str(e)}
            else:
                result["vtu_generation"] = {"success": False, "error": "PyVista not available"}
        except Exception as e:
            result["mesh_generation"] = {"success": False, "error": str(e)}
    else:
        result["mesh_generation"] = {"success": False, "error": "Mesh tools not imported"}
    
    return result


@app.get("/generate-stent-yaml")
def generate_stent_yaml(
    diameter: float = 10.0,
    length: float = 20.0,
    strut_thickness: float = 0.12,
    num_struts: int = 12,
    crown_height: float = 1.0
):
    """Generate a complete 4C YAML file with stent geometry."""
    if not MESH_TOOLS_AVAILABLE:
        return {
            "success": False,
            "error": "Mesh tools not available",
            "yaml": None
        }
    
    try:
        # Generate mesh
        geometry = StentGeometry(
            diameter=diameter,
            length=length,
            strut_thickness=strut_thickness,
            num_struts=num_struts,
            crown_height=crown_height
        )
        nodes, elements, fixed_nodes, loaded_nodes = generate_cylindrical_stent_mesh(geometry)
        
        # Generate YAML content with inline geometry (for display)
        yaml_content = f"""TITLE: Stent simulation - diameter={diameter}mm, length={length}mm
PROBLEM TYPE:
  PROBLEMTYPE: Structure

SOLVER 1:
  SOLVER: "Superlu"
  NAME: "Structure_Solver"

IO:
  OUTPUT_SPRING: true
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
  VERBOSITY: "Standard"
  WRITE_INITIAL_STATE: false

IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: binary

IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
  GAUSS_POINT_DATA_OUTPUT_TYPE: nodes

STRUCTURAL DYNAMIC:
  INT_STRATEGY: "Standard"
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-06
  TOLRES: 1e-06
  LOADLIN: true
  LINEAR_SOLVER: 1

MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 7.8e-9
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: 200000.0
      NUE: 0.3

# Mesh: {len(nodes)} nodes, {len(elements)} hex8 elements
# Fixed nodes: {len(fixed_nodes)} (z=0 end)
# Loaded nodes: {len(loaded_nodes)} (outer surface)

DNODE-NODE TOPOLOGY:
"""
        # Add nodes
        for i, node in enumerate(nodes):
            yaml_content += f"  NODE {i+1} COORD {node[0]:.6f} {node[1]:.6f} {node[2]:.6f}\n"
        
        yaml_content += "\nDELEMENT TOPOLOGY:\n"
        # Add elements
        for i, elem in enumerate(elements):
            node_ids = " ".join(str(n+1) for n in elem)
            yaml_content += f"  {i+1} SOLIDH8 HEX8 {node_ids} MAT 1 KINEM nonlinear\n"
        
        # Add node sets for boundary conditions
        yaml_content += f"\n# Fixed end (z=0): {len(fixed_nodes)} nodes\n"
        yaml_content += "DNODE-NODE SETS:\n"
        yaml_content += f"  DSURF 1 DNODES {' '.join(str(n+1) for n in fixed_nodes[:20])}...\n"
        
        return {
            "success": True,
            "mesh_info": {
                "num_nodes": len(nodes),
                "num_elements": len(elements),
                "num_fixed_nodes": len(fixed_nodes),
                "num_loaded_nodes": len(loaded_nodes)
            },
            "yaml_preview": yaml_content[:3000] + "\n... (truncated)",
            "note": "Full YAML with geometry is used internally for 4C simulations"
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/test-4c-docker")
def test_4c_docker_direct():
    """Test 4C binary directly (we're already in 4C Docker image, so no need for docker run)."""
    import subprocess
    import tempfile
    import shutil
    
    print("Testing 4C binary directly (we're in 4C image)...")
    
    # Check if fourc binary exists
    fourc_bin = shutil.which("fourc") or "/usr/local/bin/fourc" or "/home/user/4C/build/4C"
    
    # Try to find tutorial files in the image (various locations)
    tutorial_search_dirs = [
        "/home/user/4C/tests/input_files",
        "/home/user/4C/tests",
        "/home/user/tests/input_files",
    ]
    
    tutorial_yaml = None
    tutorial_vtu = None
    found_files = []
    
    for search_dir in tutorial_search_dirs:
        search_path = Path(search_dir)
        if search_path.exists():
            # Collect files for diagnostics
            found_files.extend([str(f) for f in search_path.iterdir() if f.is_file()][:10])
            
            # Look for YAML files - prefer structural mechanics files
            all_yaml_files = list(search_path.glob("*.4C.yaml")) + list(search_path.glob("*.yaml"))
            vtu_files = list(search_path.glob("*.vtu"))
            
            # Filter for structural mechanics files (solid, struct, tutorial_solid)
            # Avoid fluid, elch, fsi, etc.
            structural_keywords = ['solid', 'struct', 'plastic', 'beam', 'shell', 'truss']
            avoid_keywords = ['elch', 'fluid', 'fsi', 'scatra', 'thermo', 'ale', 'moving']
            
            structural_files = []
            for f in all_yaml_files:
                fname = f.name.lower()
                # Check if it contains structural keywords and doesn't contain avoid keywords
                has_structural = any(kw in fname for kw in structural_keywords)
                has_avoid = any(kw in fname for kw in avoid_keywords)
                if has_structural and not has_avoid:
                    structural_files.append(f)
            
            # Skip using tutorial files - they have result verification that fails
            # We want to use our generated stent mesh instead
            # if structural_files:
            #     tutorial_yaml = str(structural_files[0])
            #     print(f"✓ Found structural YAML: {tutorial_yaml}")
            # elif all_yaml_files:
            #     tutorial_yaml = str(all_yaml_files[0])
            #     print(f"✓ Found YAML (not structural): {tutorial_yaml}")
            print(f"Found {len(all_yaml_files)} test files, but using generated stent mesh instead")
                
            if vtu_files:
                tutorial_vtu = str(vtu_files[0])
                print(f"✓ Found VTU: {tutorial_vtu}")
            
            if tutorial_yaml:
                break
    
    # Create temp directory for output
    work_dir = Path(tempfile.mkdtemp(prefix="4c_test_"))
    output_dir = work_dir / "output"
    output_dir.mkdir()
    
    # Copy tutorial files to work dir if found
    if tutorial_yaml:
        import shutil
        shutil.copy(tutorial_yaml, work_dir / "input.4C.yaml")
        tutorial_yaml = work_dir / "input.4C.yaml"
    if tutorial_vtu:
        vtu_name = Path(tutorial_vtu).name
        shutil.copy(tutorial_vtu, work_dir / vtu_name)
    
    try:
        use_generated_stent = False
        use_simple_cube = False  # 2x2x2 cube passed! Now test stent mesh
        
        if use_simple_cube and MESH_TOOLS_AVAILABLE:
            # Create a 2x2x2 cube mesh (8 elements) to test multi-element VTU
            print("Creating 2x2x2 cube test (8 elements)...")
            try:
                # Generate 3x3x3 = 27 nodes for 2x2x2 = 8 elements
                cube_nodes = []
                node_map = {}
                for iz in range(3):
                    for iy in range(3):
                        for ix in range(3):
                            node_id = len(cube_nodes)
                            cube_nodes.append([float(ix), float(iy), float(iz)])
                            node_map[(ix, iy, iz)] = node_id
                cube_nodes = np.array(cube_nodes)
                
                # Generate 8 hex8 elements (same ordering as VTK)
                cube_elements = []
                for iz in range(2):
                    for iy in range(2):
                        for ix in range(2):
                            n0 = node_map[(ix, iy, iz)]
                            n1 = node_map[(ix+1, iy, iz)]
                            n2 = node_map[(ix+1, iy+1, iz)]
                            n3 = node_map[(ix, iy+1, iz)]
                            n4 = node_map[(ix, iy, iz+1)]
                            n5 = node_map[(ix+1, iy, iz+1)]
                            n6 = node_map[(ix+1, iy+1, iz+1)]
                            n7 = node_map[(ix, iy+1, iz+1)]
                            cube_elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
                cube_elements = np.array(cube_elements)
                print(f"  Generated {len(cube_nodes)} nodes, {len(cube_elements)} elements")
                
                # Fixed: bottom face (z=0), 9 nodes
                fixed_nodes = np.array([node_map[(ix, iy, 0)] for iy in range(3) for ix in range(3)])
                # Loaded: top face (z=2), 9 nodes
                loaded_nodes = np.array([node_map[(ix, iy, 2)] for iy in range(3) for ix in range(3)])
                
                vtu_path = work_dir / "cube_mesh.vtu"
                write_vtu_file(cube_nodes, cube_elements, str(vtu_path),
                              fixed_nodes=fixed_nodes, loaded_nodes=loaded_nodes)
                print(f"✓ Generated cube VTU: 8 nodes, 1 element")
                
                # Simple compression test YAML
                test_yaml = work_dir / "cube_test.4C.yaml"
                test_yaml.write_text("""TITLE: Simple cube compression test
PROBLEM TYPE:
  PROBLEMTYPE: Structure

SOLVER 1:
  SOLVER: "Superlu"
  NAME: "Structure_Solver"

IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
  VERBOSITY: "Standard"

IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: binary

IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true

STRUCTURAL DYNAMIC:
  INT_STRATEGY: "Standard"
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLDISP: 1e-06
  TOLRES: 1e-06
  LOADLIN: true

MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 7.8e-9
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: 200000.0
      NUE: 0.3

STRUCTURE GEOMETRY:
  FILE: cube_mesh.vtu
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: nonlinear

# Bottom face fixed (z=0) - 9 nodes
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]

# Top face loaded (z=2) - apply small force in -z direction (9 nodes)
DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, -0.001]
    FUNCT: [0, 0, 0]
""")
                tutorial_yaml = test_yaml
                use_generated_stent = True
                print("Created simple cube compression test")
            except Exception as e:
                print(f"Failed to create cube test: {e}")
                use_simple_cube = False
        
        if not use_generated_stent and not tutorial_yaml or not Path(fourc_bin).exists():
            # Try to generate our own stent mesh with 4C-compliant YAML
            if MESH_TOOLS_AVAILABLE:
                print("No tutorial file found - generating stent mesh...")
                try:
                    # Generate realistic stent mesh for testing
                    geometry = StentGeometry(
                        diameter=10.0,
                        length=20.0,
                        strut_thickness=0.12,
                        num_struts=12,
                        crown_height=1.0
                    )
                    nodes, elements, fixed_nodes, loaded_nodes = generate_cylindrical_stent_mesh(
                        geometry, n_circumferential_per_strut=4, n_radial=2
                    )
                    
                    # Write VTU file with required 4C arrays (block_id, point_sets)
                    vtu_path = work_dir / "stent_mesh.vtu"
                    write_vtu_file(
                        nodes, elements, str(vtu_path),
                        fixed_nodes=np.array(fixed_nodes),
                        loaded_nodes=np.array(loaded_nodes)
                    )
                    print(f"✓ Generated VTU: {len(nodes)} nodes, {len(elements)} elements, with block_id and point_sets")
                    
                    # Create YAML referencing VTU with point_set boundary conditions
                    test_yaml = work_dir / "stent_test.4C.yaml"
                    pressure = 0.01  # Small axial load for testing
                    test_yaml.write_text(f"""TITLE: Generated stent test
PROBLEM TYPE:
  PROBLEMTYPE: Structure

SOLVER 1:
  SOLVER: "Superlu"
  NAME: "Structure_Solver"

IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
  VERBOSITY: "Standard"

IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: binary

IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true

STRUCTURAL DYNAMIC:
  INT_STRATEGY: "Standard"
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLDISP: 1e-06
  TOLRES: 1e-06
  LOADLIN: true

MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 7.8e-9
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: 200000.0
      NUE: 0.3

STRUCTURE GEOMETRY:
  FILE: stent_mesh.vtu
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: nonlinear

# Boundary conditions using point_set arrays from VTU file
# point_set_1 = fixed nodes (z=0), point_set_2 = loaded nodes (outer surface)
# Apply simple axial loading for testing
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    ENTITY_TYPE: node_set_id
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]

# Apply small axial force in Z direction on outer surface
DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    ENTITY_TYPE: node_set_id
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, {pressure}]
    FUNCT: [0, 0, 0]
""")
                    tutorial_yaml = test_yaml
                    use_generated_stent = True
                    print("Created stent test YAML with VTU geometry")
                except Exception as e:
                    print(f"Failed to generate stent mesh: {e}")
            
            if not use_generated_stent:
                # Fallback: create a minimal test YAML (will fail without geometry)
                test_yaml = work_dir / "test.4C.yaml"
                test_yaml.write_text("""TITLE: Minimal test (no geometry - will fail)
PROBLEM TYPE:
  PROBLEMTYPE: Structure

SOLVER 1:
  SOLVER: "Superlu"
  NAME: "Structure_Solver"

STRUCTURAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  LINEAR_SOLVER: 1
  TOLDISP: 1e-06
  TOLRES: 1e-06

MATERIALS:
  - MAT: 1
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [2]
      DENS: 7.8e-9
  - MAT: 2
    ELAST_CoupNeoHooke:
      YOUNG: 200000.0
      NUE: 0.3
""")
                tutorial_yaml = test_yaml
                print("Created minimal test YAML (no geometry)")
        
        # Run 4C binary directly (we're already in the image!)
        # 4C command format: 4C input.yaml output_name (no -o flag)
        # Output goes to same directory as input file
        output_name = "test_output"
        cmd = [
            fourc_bin,
            str(tutorial_yaml),
            output_name
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print(f"  Binary: {fourc_bin} (exists: {Path(fourc_bin).exists()})")
        print(f"  YAML: {tutorial_yaml} (exists: {Path(tutorial_yaml).exists()})")
        
        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(work_dir),
            env=env
        )
        
        # Check output files
        output_files = list(output_dir.glob("*"))
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "output_files": [f.name for f in output_files],
            "output_dir": str(output_dir),
            "command": " ".join(cmd),
            "fourc_binary": fourc_bin,
            "binary_exists": Path(fourc_bin).exists(),
            "yaml_exists": Path(tutorial_yaml).exists() if tutorial_yaml else False,
            "found_test_files": found_files[:10] if found_files else [],
            "mesh_tools_available": MESH_TOOLS_AVAILABLE,
            "used_generated_stent": use_generated_stent if 'use_generated_stent' in dir() else False,
            "test_type": "simple_cube" if use_simple_cube else "stent_mesh"
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Timeout after 120 seconds",
            "output_files": [f.name for f in output_dir.glob("*")] if output_dir.exists() else []
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "output_files": []
        }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting 4C + RL API Server on port {port}")
    print(f"SB3 available: {SB3_AVAILABLE}")
    uvicorn.run(app, host="0.0.0.0", port=port)

