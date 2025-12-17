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
        test_result = subprocess.run(
            [FOURC_BINARY_PATH, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        FOURC_VERSION_OUTPUT = test_result.stdout.strip() or test_result.stderr.strip()
        
        # Check if it actually ran successfully (not just exists)
        # Return code 127 = "command not found" or shared library errors
        # Return code 0 = success
        if test_result.returncode == 0:
            FOURC_CAN_EXECUTE = True
            print(f"✓ 4C binary verified at: {FOURC_BINARY_PATH}")
            print(f"  Version: {FOURC_VERSION_OUTPUT}")
        elif "error while loading shared libraries" in FOURC_VERSION_OUTPUT:
            FOURC_CAN_EXECUTE = False
            print(f"✗ 4C binary has MISSING LIBRARIES!")
            print(f"  Error: {FOURC_VERSION_OUTPUT}")
        else:
            # Other non-zero return might still be okay (e.g., --version not supported)
            FOURC_CAN_EXECUTE = True
            print(f"⚠ 4C binary returned code {test_result.returncode}")
            print(f"  Output: {FOURC_VERSION_OUTPUT}")
    except Exception as e:
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
            use_docker=False
        )
        FOURC_AVAILABLE = True
        print("✓ 4C simulator initialized successfully")
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
                        stress = result.get("max_von_mises_stress", 100.0)
                        displacement = result.get("max_displacement", 0.0)
                        print(f"Step {self.step_count}: ✓ 4C REAL simulation in {time.time()-start_time:.2f}s. Stress: {stress:.2f} MPa")
                    else:
                        # Simulation failed - use high penalty
                        print(f"Step {self.step_count}: ✗ 4C Failed: {result.get('error')}")
                        stress = 1000.0  # High penalty
                        displacement = 10.0
                except Exception as e:
                    print(f"Env Exception: {e}")
                    stress = 1000.0
            else:
                # Fast Mockup Calculation
                d, t, n = self.params['diameter'], self.params['strut_thickness'], self.params['num_struts']
                stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                displacement = 0.3 * (self.params['length'] / 20) * (0.12 / max(t, 0.05))

            # Reward calculation
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
            
            if self.n_calls % 2 == 0:  # Log every 2 steps for smoother curves
                with state_lock:
                    training_state["current_step"] = self.n_calls
                    training_state["progress"] = self.n_calls / self.total_steps
                    training_state["rewards"].append(float(reward))
                    try:
                        # Access unwrapped environment to get params
                        env = self.model.env.envs[0].unwrapped
                        if hasattr(env, 'params'):
                            training_state["current_params"] = dict(env.params)
                            # Get real stress if available, else formula
                            if hasattr(env, 'last_stress'): 
                                stress = env.last_stress
                            else:
                                # Use same stress formula as in step()
                                d, t, n = env.params['diameter'], env.params['strut_thickness'], env.params['num_struts']
                                stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                            training_state["stress_history"].append(float(stress))
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
            training_state["progress"] = 0.0
            training_state["episodes"] = 0
            training_state["rewards"] = []
            training_state["stress_history"] = []
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
    if not FOURC_BINARY_EXISTS:
        result["recommendation"] = "4C binary not found. Check Dockerfile COPY commands."
    elif not FOURC_CAN_EXECUTE:
        result["recommendation"] = "4C binary exists but cannot execute (missing shared libraries). Use 4C Docker image as base instead of multi-stage copy."
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
    
    return result

# ===== HELPER FUNCTIONS =====
# Import mesh generation and VTU parsing
try:
    from mesh_generator import (
        StentGeometry,
        generate_cylindrical_stent_mesh,
        write_4c_geometry
    )
    from vtu_parser import parse_vtu_file, find_latest_vtu
    MESH_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Mesh tools not available: {e}")
    MESH_TOOLS_AVAILABLE = False


def generate_4c_yaml(params: StentParams, output_path: Path):
    """Generate complete 4C YAML input file with real mesh."""
    
    if not MESH_TOOLS_AVAILABLE:
        # Fallback to simple YAML without mesh
        yaml_content = f"""# Simplified 4C input (mesh tools not available)
PROBLEM_SIZE: 3
PROBLEM_TYPE: Structure

STRUCTURAL:
  NEWMARK:
    TIMESTEP: 0.001
    NUMSTEP: 50
    
MATERIALS:
  MAT 1:
    TYPE: ElastHyper
    YOUNG: 200000.0
    NUE: 0.3
    DENS: 6.45e-9

# Mesh would be here
"""
        output_path.write_text(yaml_content)
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
    
    # Write complete 4C YAML
    with open(output_path, 'w') as f:
        f.write("""# 4C Stent Simulation with Real Mesh
PROBLEM_SIZE: 3
PROBLEM_TYPE: Structure

STRUCTURAL:
  NEWMARK:
    TIMESTEP: 0.001
    NUMSTEP: 50
    LOADSTEP_SIZE: 0.02

MATERIALS:
  MAT 1:
    TYPE: ElastHyper
    YOUNG: 200000.0  # MPa (NiTi)
    NUE: 0.3
    DENS: 6.45e-9  # kg/mm^3

""")
        
        # Write geometry
        f.write("GEOMETRY:\n")
        f.write("  NODES:\n")
        for i, node in enumerate(nodes):
            f.write(f"    - ID: {i+1}, COORDS: [{node[0]:.6f}, {node[1]:.6f}, {node[2]:.6f}]\n")
        
        f.write("\n  ELEMENTS:\n")
        for i, elem in enumerate(elements):
            node_ids = ", ".join(str(n+1) for n in elem)
            f.write(f"    - ID: {i+1}, TYPE: hex8, NODES: [{node_ids}]\n")
        
        f.write("\n  ELEMENT_BLOCKS:\n")
        f.write("    - ID: 1\n")
        f.write(f"      ELEMENTS: [1-{len(elements)}]\n")
        f.write("      MATERIAL: 1\n")
        
        f.write("\n  NODE_SETS:\n")
        f.write("    - ID: 1, NAME: fixed_end\n")
        fixed_ids = ", ".join(str(n+1) for n in fixed_nodes)
        f.write(f"      NODES: [{fixed_ids}]\n")
        f.write("    - ID: 2, NAME: loaded_surface\n")
        loaded_ids = ", ".join(str(n+1) for n in loaded_nodes)
        f.write(f"      NODES: [{loaded_ids}]\n")
        
        # Boundary conditions
        pressure = params.diameter * 0.1  # Radial pressure
        f.write(f"""
BOUNDARY_CONDITIONS:
  DIRICHLET:
    - NODE_SETS: [1]
      DOF: [1, 2, 3]
      VALUE: 0.0
  
  NEUMANN:
    - NODE_SETS: [2]
      DOF: [2]
      VALUE: {pressure}

OUTPUT:
  VTK:
    INTERVAL: 10
    FIELDS: ['stress', 'displacement', 'strain']
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
        work_dir = Path(tempfile.mkdtemp(prefix="4c_sim_"))
        yaml_path = work_dir / "input.4C.yaml"
        output_dir = work_dir / "output"
        
        # Generate 4C input file
        generate_4c_yaml(params, yaml_path)
        
        # Configure simulation
        config = SimulationConfig(
            yaml_input_path=yaml_path,
            output_directory=output_dir,
            timeout=300.0,
            verbose=False
        )
        
        # Run 4C simulation
        print(f"Running real 4C simulation for: {params}")
        result = fourc_sim.run_simulation(config)
        
        if not result.success:
            return {
                "success": False,
                "error": result.error_message or "4C simulation failed",
                "log_path": str(result.log_path) if result.log_path else None,
                "fallback_used": True
            }
        
        # Parse VTU files for actual results
        max_stress = 0.0
        max_disp = 0.0
        max_strain = 0.0
        
        if MESH_TOOLS_AVAILABLE:
            latest_vtu = find_latest_vtu(output_dir)
            if latest_vtu:
                try:
                    max_stress, max_disp, max_strain, parse_success = parse_vtu_file(latest_vtu)
                    if parse_success:
                        print(f"Parsed VTU results: stress={max_stress:.2f}, disp={max_disp:.4f}")
                    else:
                        print("VTU parsing failed - using placeholder values")
                except Exception as e:
                    print(f"Error parsing VTU: {e}")
            else:
                print("No VTU files found - 4C may not have output results")
        
        # Return results
        return {
            "success": True,
            "simulation_id": f"4c_{int(time.time()*1000)}",
            "status": "completed",
            "result": {
                "max_stress": float(max_stress) if max_stress > 0 else float(result.max_von_mises_stress),
                "max_displacement": float(max_disp) if max_disp > 0 else float(result.max_displacement),
                "max_strain": float(max_strain),
                "converged": result.converged,
                "source": "real_4c_fem" if max_stress > 0 else "real_4c_no_results",
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
            "total_steps": int(training_state["total_steps"]),
            "episodes": int(training_state["episodes"]),
            "current_params": {k: float(v) for k, v in training_state["current_params"].items()} if training_state["current_params"] else {},
            "best_params": {k: float(v) for k, v in training_state["best_params"].items()} if training_state["best_params"] else {},
            "best_reward": float(training_state["best_reward"]),
            "rewards": [float(x) for x in training_state["rewards"]],  # Send ALL rewards
            "stress_history": [float(x) for x in training_state["stress_history"]],  # Send ALL
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

