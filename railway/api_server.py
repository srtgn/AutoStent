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
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import uvicorn
import numpy as np

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium import spaces
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("WARNING: stable-baselines3 not installed, RL training disabled")

# Try to import Queens-Py for UQ
try:
    import queens
    import queens.uq
    QUEENS_AVAILABLE = True
    print("✓ Queens-Py initialized successfully")
except ImportError:
    QUEENS_AVAILABLE = False
    print("WARNING: queens-py not installed, UQ analysis will run in mock mode")

# Try to import 4C interface
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from autostent.simulation.fourc_interface import (
        FourCSimulator,
        SimulationConfig,
        SimulationResult
    )
    # Initialize simulator
    fourc_sim = FourCSimulator(use_docker=False)
    FOURC_AVAILABLE = True
    print("✓ 4C simulator initialized successfully")
except Exception as e:
    FOURC_AVAILABLE = False
    fourc_sim = None
    print(f"WARNING: 4C solver not available: {e}")
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
    "displacement_history": [],
    "episode_rewards": [],
    "yaml_files": {},  # Store YAML and VTU content by step number
}

# ===== STENT ENVIRONMENT FOR RL =====
if SB3_AVAILABLE:
    class SimpleStentEnv(gym.Env):
        """Stent environment for RL training (supports Mockup and Real 4C)"""
        def __init__(self, use_real_4c=False, mesh_coarseness="medium"):
            super().__init__()
            self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
            self.max_episode_steps = 50
            self.step_count = 0
            self.use_real_4c = use_real_4c and FOURC_AVAILABLE
            self.mesh_coarseness = mesh_coarseness
            self.params = {'diameter': 10.0, 'strut_thickness': 0.12, 'num_struts': 12, 'crown_height': 1.0, 'length': 20.0}
            
            if self.use_real_4c:
                print(f"Environment initialized with REAL 4C solver (mesh={mesh_coarseness})")
            else:
                print(f"Environment initialized with MOCKUP solver (requested={use_real_4c}, available={FOURC_AVAILABLE})")
            
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
                    result = run_real_4c_simulation(p, mesh_coarseness=self.mesh_coarseness)
                    
                    if result.get("success"):
                        stress = result.get("max_von_mises_stress", 100.0)
                        displacement = result.get("max_displacement", 0.0)
                        self.last_yaml_content = result.get("yaml_content")
                        self.last_vtu_content = result.get("vtu_content")
                        print(f"Step {self.step_count}: 4C SUCCESS in {time.time()-start_time:.2f}s. Stress={stress:.2f} MPa, Disp={displacement:.3f} mm")
                    else:
                        # Simulation failed
                        error_msg = result.get('error', 'unknown error')
                        print(f"Step {self.step_count}: 4C FAILED: {error_msg}")
                        stress = 1000.0 # High penalty
                        displacement = 10.0
                except Exception as e:
                    print(f"Step {self.step_count}: Exception during 4C: {e}")
                    import traceback
                    traceback.print_exc()
                    stress = 1000.0
                    displacement = 10.0
                    self.last_yaml_content = None
                    self.last_vtu_content = None
            else:
                # Fast Mockup Calculation
                d, t, n = self.params['diameter'], self.params['strut_thickness'], self.params['num_struts']
                stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                displacement = 0.3 * (self.params['length'] / 20) * (0.12 / max(t, 0.05))

            # Store for callback access
            self.last_stress = stress
            self.last_displacement = displacement
            
            # Reward calculation
            reward = -0.4 * (stress / 200) - 0.2 * displacement
            
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
                    training_state["current_step"] = min(self.n_calls, self.total_steps)
                    training_state["progress"] = self.n_calls / self.total_steps
                    training_state["rewards"].append(float(reward))
                    try:
                        # Access unwrapped environment to get params
                        env = self.model.env.envs[0].unwrapped
                        if hasattr(env, 'params'):
                            training_state["current_params"] = dict(env.params)
                            # Get real stress and displacement if available
                            if hasattr(env, 'last_stress'): 
                                stress = env.last_stress
                                displacement = getattr(env, 'last_displacement', 0.0)
                            else:
                                # Use same formula as in step()
                                d, t, n = env.params['diameter'], env.params['strut_thickness'], env.params['num_struts']
                                stress = 100 * (0.12 / max(t, 0.05))**1.5 * (10.0 / max(d, 5))**0.5 * (12 / max(n, 6))**0.8
                                displacement = 0.3 * (env.params['length'] / 20) * (0.12 / max(t, 0.05))
                            training_state["stress_history"].append(float(stress))
                            training_state["displacement_history"].append(float(displacement))
                            
                            # Save generated files for download
                            if hasattr(env, 'last_yaml_content') and env.last_yaml_content:
                                training_state["yaml_files"][training_state["current_step"]] = {
                                    "yaml": env.last_yaml_content,
                                    "vtu": env.last_vtu_content,
                                    "params": dict(env.params),
                                    "reward": float(reward)
                                }
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

    def run_training(total_steps, use_real_4c=False, mesh_coarseness="medium"):
        global training_state
        with state_lock:
            training_state["is_training"] = True
            training_state["total_steps"] = total_steps
            training_state["current_step"] = 0
            training_state["progress"] = 0.0
            training_state["episodes"] = 0
            training_state["rewards"] = []
            training_state["stress_history"] = []
            training_state["displacement_history"] = []
            training_state["episode_rewards"] = []
            training_state["best_reward"] = -100.0
            training_state["yaml_files"] = {}
            training_state["mesh_coarseness"] = mesh_coarseness
        
        try:
            env = SimpleStentEnv(use_real_4c=use_real_4c, mesh_coarseness=mesh_coarseness)
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, 
                        n_steps=128, batch_size=64, n_epochs=5, gamma=0.99, device='cpu')
            callback = TrainingCallback(total_steps)
            model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
            with state_lock:
                training_state["progress"] = 1.0
                training_state["is_training"] = False
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            with state_lock:
                training_state["is_training"] = False


# ===== MODELS =====
class StentParams(BaseModel):
    diameter: float = 10.0
    length: float = 20.0
    strut_thickness: float = 0.12
    num_struts: int = 12
    num_struts: int = 12
    crown_height: float = 1.0
    # Extended physics parameters for UQ (optional)
    youngs_modulus: float = 200000.0  # MPa
    pressure_load: float = 0.0  # MPa (if 0, calculated from diameter)

class SimulationRequest(BaseModel):
    params: StentParams
    timeout: int = 300
    use_docker: bool = False  # Toggle for real 4C vs mockup

class TrainRequest(BaseModel):
    steps: int = 2000
    use_docker: bool = False
    mesh_coarseness: str = "medium"  # fine, medium, coarse


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
    import subprocess
    import shutil
    
    # Check if fourc binary exists and can execute
    fourc_path = shutil.which("fourc")
    binary_exists = False
    can_execute = False
    binary_version = None
    error_msg = None
    
    if not fourc_path:
        # Check common paths from Docker
        for path in ["/usr/local/bin/fourc", "/home/user/4C/build/4C", "/usr/bin/fourc"]:
            if Path(path).exists():
                fourc_path = path
                binary_exists = True
                break
    else:
        binary_exists = True
    
    # Try to execute fourc to verify it works
    if binary_exists and fourc_path:
        try:
            # Set LD_LIBRARY_PATH for 4C execution
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = '/home/user/4C/build:/usr/local/lib:/usr/lib/x86_64-linux-gnu:' + env.get('LD_LIBRARY_PATH', '')
            
            version_output = subprocess.run(
                [fourc_path, "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5,
                env=env
            )
            binary_version = version_output.stdout.strip() or version_output.stderr.strip()
            can_execute = True
        except Exception as e:
            error_msg = f"Could not execute fourc: {str(e)}"
            can_execute = False
    else:
        error_msg = "fourc binary not found"
    
    result = {
        "fourc_available": binary_exists and can_execute,
        "fourc_can_execute": can_execute,
        "mesh_tools_available": MESH_TOOLS_AVAILABLE if 'MESH_TOOLS_AVAILABLE' in dir() else False,
        "sb3_available": SB3_AVAILABLE,
        "binary_path": fourc_path,
        "binary_exists": binary_exists,
        "binary_version": binary_version,
        "error": error_msg,
        "debug_info": {
            "ld_library_path": os.environ.get('LD_LIBRARY_PATH', 'not set'),
            "missing_libs": "none"
        }
    }
    
    return result

# ===== HELPER FUNCTIONS =====
# Import mesh generation and VTU parsing
MESH_IMPORT_ERROR = None
try:
    from mesh_generator import (
        StentGeometry,
        generate_cylindrical_stent_mesh,
        write_4c_geometry,
        write_vtu_file
    )
    from vtu_parser import parse_vtu_file, find_latest_vtu
    MESH_TOOLS_AVAILABLE = True
    print("✓ Mesh tools imported successfully")
except ImportError as e:
    MESH_IMPORT_ERROR = str(e)
    print(f"WARNING: Mesh tools not available: {e}")
    MESH_TOOLS_AVAILABLE = False

@app.get("/check-mesh-tools")
def check_mesh_tools():
    """Diagnostic endpoint to check mesh generation capabilities."""
    result = {
        "mesh_tools_available": MESH_TOOLS_AVAILABLE,
        "import_error": MESH_IMPORT_ERROR
    }
    
    # Try importing individually to identify specific failure
    errors = []
    try:
        import numpy as np
        result["numpy_available"] = True
    except ImportError as e:
        result["numpy_available"] = False
        errors.append(f"numpy: {e}")
    
    try:
        import pyvista as pv
        result["pyvista_available"] = True
        result["pyvista_version"] = pv.__version__
    except ImportError as e:
        result["pyvista_available"] = False
        errors.append(f"pyvista: {e}")
    
    # Check if files exist
    import os
    result["mesh_generator_exists"] = os.path.exists("mesh_generator.py")
    result["vtu_parser_exists"] = os.path.exists("vtu_parser.py")
    result["working_dir"] = os.getcwd()
    result["files_in_dir"] = os.listdir(".")[:20]  # First 20 files
    result["errors"] = errors
    
    return result


def generate_4c_yaml(params: StentParams, output_path: Path, mesh_coarseness: str = "medium"):
    """Generate complete 4C YAML input file with VTU mesh file."""
    
    # Use provided physics params or defaults
    E = params.youngs_modulus or 200000.0
    pressure = params.pressure_load
    if pressure <= 0:
        pressure = params.diameter * 0.01  # Default based on diameter
    
    # Set mesh resolution based on coarseness
    if mesh_coarseness == "fine":
        n_circ_per_strut = 4
        n_radial = 2
    elif mesh_coarseness == "coarse":
        n_circ_per_strut = 1
        n_radial = 1
    else:  # medium (default)
        n_circ_per_strut = 2
        n_radial = 1
    
    if not MESH_TOOLS_AVAILABLE:
        # Fallback to simple YAML without mesh
        yaml_content = f"""TITLE: Stent simulation - diameter={params.diameter}mm, length={params.length}mm
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
      YOUNG: {E}
      NUE: 0.3

# Mesh tools not available - using placeholder
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
        n_circumferential_per_strut=n_circ_per_strut,
        n_radial=n_radial
    )
    
    # Generate VTU file
    vtu_path = output_path.parent / f"{output_path.stem}_mesh.vtu"
    import numpy as np
    write_vtu_file(
        nodes,
        elements,
        str(vtu_path),
        fixed_nodes=np.array(fixed_nodes),
        loaded_nodes=np.array(loaded_nodes)
    )
    
    # Write complete 4C YAML with VTU reference
    vtu_filename = vtu_path.name
    # Pressure is already calculated/set above
    
    yaml_content = f"""TITLE: Stent simulation - diameter={params.diameter}mm, length={params.length}mm
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
      YOUNG: {E}
      NUE: 0.3

STRUCTURE GEOMETRY:
  FILE: {vtu_filename}
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: nonlinear

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
"""
    output_path.write_text(yaml_content)


def run_real_4c_simulation(params: StentParams, mesh_coarseness: str = "medium"):
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
        generate_4c_yaml(params, yaml_path, mesh_coarseness=mesh_coarseness)
        
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
            error_msg = result.error_message or "4C simulation failed"
            # Log full error for debugging
            print(f"4C simulation failed for params {params}:")
            print(error_msg)
            # Truncate very long error messages for response (keep last 2000 chars)
            if len(error_msg) > 2000:
                error_msg = "... (truncated) ...\n" + error_msg[-2000:]
            return {
                "success": False,
                "error": error_msg,
                "log_path": str(result.log_path) if result.log_path else None,
                "fallback_used": True
            }
        
        # Parse VTU/VTK files for actual results (if available)
        max_stress = 0.0
        max_disp = 0.0
        max_strain = 0.0
        debug_info = {"vtu_search": []}
        
        if MESH_TOOLS_AVAILABLE:
            # Search multiple possible output locations
            # IMPORTANT: 4C writes results to output-vtk-files/, not the work_dir root
            search_dirs = []
            
            # Check for output-vtk-files directory first (this is where 4C writes results)
            vtk_output_dir = work_dir / "output-vtk-files"
            if vtk_output_dir.exists():
                search_dirs.append(vtk_output_dir)
            
            if hasattr(result, "output_directory") and result.output_directory:
                search_dirs.append(Path(result.output_directory))
            search_dirs.append(output_dir)
            search_dirs.append(work_dir)  # Last resort
            
            latest_vtu = None
            for search_dir in search_dirs:
                if search_dir.exists():
                    # List files for debugging
                    files_in_dir = [f.name for f in search_dir.glob("*")][:20]
                    debug_info["vtu_search"].append({
                        "dir": str(search_dir),
                        "files": files_in_dir
                    })
                    
                    # Find VTU files, but skip input mesh files (ending in _mesh.vtu)
                    found = find_latest_vtu(search_dir)
                    if found:
                        # Skip input mesh files - we want OUTPUT files
                        if str(found).endswith("_mesh.vtu"):
                            debug_info["skipped_input_mesh"] = str(found)
                            continue
                        latest_vtu = found
                        debug_info["vtu_found"] = str(found)
                        break
            
            if latest_vtu:
                try:
                    max_stress, max_disp, max_strain, parse_success, point_arrays, cell_arrays = parse_vtu_file(latest_vtu)
                    debug_info["parse_success"] = parse_success
                    debug_info["parsed_stress"] = float(max_stress)
                    debug_info["parsed_disp"] = float(max_disp)
                    debug_info["vtu_point_arrays"] = point_arrays
                    debug_info["vtu_cell_arrays"] = cell_arrays
                    if parse_success:
                        print(f"Parsed VTU results: stress={max_stress:.2f}, disp={max_disp:.4f}")
                    else:
                        print("VTU parsing failed - using placeholder values")
                except Exception as e:
                    print(f"Error parsing VTU: {e}")
                    debug_info["parse_error"] = str(e)
            else:
                print("No VTU/VTK files found - 4C may have written no runtime VTK output")
                debug_info["vtu_found"] = None
        
        # Read YAML and VTU content for storage
        yaml_content = None
        vtu_content = None
        try:
            if yaml_path.exists():
                yaml_content = yaml_path.read_bytes()
            vtu_path = work_dir / f"{yaml_path.stem}_mesh.vtu"
            if vtu_path.exists():
                vtu_content = vtu_path.read_bytes()
        except Exception as e:
            print(f"Warning: Could not read YAML/VTU for storage: {e}")
        
        # Choose best available quantities (parsed VTU preferred, otherwise simulator placeholders)
        best_stress = float(max_stress) if max_stress > 0 else float(result.max_von_mises_stress)
        best_disp = float(max_disp) if max_disp > 0 else float(result.max_displacement)
        best_strain = float(max_strain)

        # Return results (IMPORTANT: include top-level keys that the RL env expects)
        payload = {
            "success": True,
            "simulation_id": f"4c_{int(time.time()*1000)}",
            "status": "completed",

            # Top-level (consumed by SimpleStentEnv.step)
            "max_von_mises_stress": best_stress,
            "max_displacement": best_disp,
            "max_principal_strain": best_strain,

            # Structured detail (nice for UI / debugging)
            "result": {
                "max_stress": best_stress,
                "max_displacement": best_disp,
                "max_strain": best_strain,
                "converged": bool(result.converged),
                "source": "real_4c_fem" if max_stress > 0 else "real_4c_no_vtu_parse",
                "num_iterations": int(result.num_iterations),
                "mesh_elements": 0,  # not tracked here
                "output_directory": str(getattr(result, "output_directory", "")),
            },
            "metadata": result.metadata or {},
            "yaml_content": yaml_content,
            "vtu_content": vtu_content,
            "debug_info": debug_info,
        }
        return payload
        
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


@app.get("/diagnose")
def diagnose_simulation():
    """Run a single test simulation and return detailed diagnostic info."""
    params = StentParams(diameter=10.0, length=20.0, strut_thickness=0.12, num_struts=12, crown_height=1.0)
    
    result = run_real_4c_simulation(params)
    
    # Build diagnostic response (excluding binary file content for readability, but noting sizes)
    diag = {
        "success": result.get("success"),
        "max_von_mises_stress": result.get("max_von_mises_stress"),
        "max_displacement": result.get("max_displacement"),
        "result_source": result.get("result", {}).get("source"),
        "converged": result.get("result", {}).get("converged"),
        "num_iterations": result.get("result", {}).get("num_iterations"),
        "debug_info": result.get("debug_info", {}),
        "yaml_size": len(result.get("yaml_content") or b""),
        "vtu_size": len(result.get("vtu_content") or b""),
        "error": result.get("error"),
        "fourc_available": FOURC_AVAILABLE,
        "mesh_tools_available": MESH_TOOLS_AVAILABLE,
    }
    return diag


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
            "displacement_history": [float(x) for x in training_state.get("displacement_history", [])],  # Send ALL
            "episode_rewards": [float(x) for x in training_state["episode_rewards"]],  # Send ALL
            "sb3_available": bool(SB3_AVAILABLE),
        }

@app.post("/start")
def start_training(request: TrainRequest):
    """Start RL training - matches rl_backend.py API."""
    if not SB3_AVAILABLE:
        return {"status": "error", "message": "stable-baselines3 not installed"}
    
    with state_lock:
        if training_state["is_training"]:
            return {"status": "already_running"}
        training_state["mesh_coarseness"] = request.mesh_coarseness
    
    thread = threading.Thread(target=run_training, args=(request.steps, request.use_docker, request.mesh_coarseness), daemon=True)
    thread.start()
    return {"status": "started", "steps": request.steps, "mesh_coarseness": request.mesh_coarseness}

@app.post("/stop")
def stop_training():
    """Stop RL training."""
    with state_lock:
        training_state["is_training"] = False
    return {"status": "stopped"}

@app.get("/training-files")
def list_training_files():
    """List available YAML/VTU files from training."""
    with state_lock:
        files = []
        for step in sorted(training_state.get("yaml_files", {}).keys()):
            files.append({
                "step": step,
                "yaml_url": f"/training-yaml/{step}",
                "vtu_url": f"/training-vtu/{step}"
            })
        return {"files": files, "count": len(files)}

@app.get("/training-yaml/{step}")
def download_training_yaml(step: int):
    """Download YAML file for a specific training step."""
    with state_lock:
        if step not in training_state.get("yaml_files", {}):
            raise HTTPException(status_code=404, detail=f"Step {step} not found")
        yaml_content = training_state["yaml_files"][step].get("yaml")
        if not yaml_content:
            raise HTTPException(status_code=404, detail=f"YAML not available for step {step}")
        return Response(content=yaml_content, media_type="application/x-yaml",
                       headers={"Content-Disposition": f"attachment; filename=step_{step}.4C.yaml"})

@app.get("/training-vtu/{step}")
def download_training_vtu(step: int):
    """Download VTU file for a specific training step."""
    with state_lock:
        if step not in training_state.get("yaml_files", {}):
            raise HTTPException(status_code=404, detail=f"Step {step} not found")
        vtu_content = training_state["yaml_files"][step].get("vtu")
        if not vtu_content:
            raise HTTPException(status_code=404, detail=f"VTU not available for step {step}")
        return Response(content=vtu_content, media_type="application/octet-stream",
                       headers={"Content-Disposition": f"attachment; filename=step_{step}.vtu"})

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
        training_state["displacement_history"] = []
        training_state["episode_rewards"] = []
        training_state["best_reward"] = -100.0
        training_state["yaml_files"] = {}
    return {"status": "reset"}


@app.get("/test-4c-docker")
def test_4c_docker_direct():
    """Test 4C binary directly (we're already in 4C Docker image, so no need for docker run)."""
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
    
    # Create temp directory for output
    work_dir = Path(tempfile.mkdtemp(prefix="4c_test_"))
    output_dir = work_dir / "output"
    output_dir.mkdir()
    
    try:
        use_generated_stent = False
        
        # Generate stent mesh for testing
        if MESH_TOOLS_AVAILABLE:
            print("Generating stent mesh for testing...")
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
        
        # Check output files.
        # NOTE: 4C often writes outputs next to the YAML with an output-name prefix,
        # or into work_dir/<output_name>/..., not into work_dir/output.
        def _list_files(p: Path):
            try:
                return sorted([f.name for f in p.glob("*")])
            except Exception:
                return []

        work_files = _list_files(work_dir)
        output_files = _list_files(output_dir)

        # Also check common 4C output locations for this invocation
        output_name = output_name if "output_name" in locals() else "test_output"
        out_dir_style = work_dir / output_name
        out_dir_files = _list_files(out_dir_style) if out_dir_style.exists() else []

        # 4C runtime VTK output commonly goes into "<output_name>-vtk-files/"
        vtk_dir_style = work_dir / f"{output_name}-vtk-files"
        vtk_dir_files = _list_files(vtk_dir_style) if vtk_dir_style.exists() else []
        vtk_dir_sample = []
        try:
            if vtk_dir_style.exists():
                # Include a small recursive sample so we can see whether VTU/VTK exist
                for p in sorted(vtk_dir_style.rglob("*"))[:50]:
                    if p.is_file():
                        vtk_dir_sample.append(str(p.relative_to(work_dir)))
        except Exception:
            vtk_dir_sample = []

        # Prefix-style outputs in work_dir (e.g., test_output*.vtu/.vtk/.pvtu/.pvd/.log)
        prefix_hits = []
        try:
            for ext in ("*.vtu", "*.vtk", "*.pvtu", "*.pvd", "*.log", "*.txt", "*.csv"):
                for f in work_dir.glob(ext):
                    if f.name.startswith(output_name):
                        prefix_hits.append(f.name)
            prefix_hits = sorted(set(prefix_hits))
        except Exception:
            prefix_hits = []
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            # Back-compat fields
            "output_files": output_files,
            "output_dir": str(output_dir),

            # New diagnostics (this will explain the “output_files=[] but returncode=0” situation)
            "work_dir": str(work_dir),
            "work_dir_files": work_files,
            "output_name": output_name,
            "output_name_dir": str(out_dir_style),
            "output_name_dir_files": out_dir_files,
            "vtk_files_dir": str(vtk_dir_style),
            "vtk_files_dir_files": vtk_dir_files,
            "vtk_files_dir_sample": vtk_dir_sample,
            "output_prefix_hits": prefix_hits,
            "command": " ".join(cmd),
            "fourc_binary": fourc_bin,
            "binary_exists": Path(fourc_bin).exists(),
            "yaml_exists": Path(tutorial_yaml).exists() if tutorial_yaml else False,
            "found_test_files": found_files[:10] if found_files else [],
            "mesh_tools_available": MESH_TOOLS_AVAILABLE,
            "used_generated_stent": use_generated_stent,
            "test_type": "stent_mesh"
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


# ===== UQ ENDPOINTS =====
class UQRequest(BaseModel):
    params: StentParams
    num_samples: int = 20  # Keep low for demo speed
    use_docker: bool = False
    uncertainty_level: str = "medium"  # low, medium, high

@app.post("/analyze/uq")
def run_uq_analysis(request: UQRequest):
    """Run Uncertainty Quantification analysis (Queens-Py or Manual MC)."""
    
    # 1. Define Uncertainty distributions (std dev as % of mean)
    # levels: low=5%, medium=10%, high=20%
    sigma_factor = {"low": 0.05, "medium": 0.10, "high": 0.20}.get(request.uncertainty_level, 0.10)
    
    nominal_E = 200000.0
    nominal_P = request.params.diameter * 0.01
    if request.params.pressure_load > 0:
        nominal_P = request.params.pressure_load
        
    nominal_T = request.params.strut_thickness
    
    samples = []
    
    print(f"Starting UQ Analysis ({request.num_samples} samples, level={request.uncertainty_level})...")
    
    # Simple Monte Carlo Loop (works with or without Queens-Py for now)
    # Robust implementation that doesn't rely on external libraries if missing
    import random
    import numpy as np
    
    for i in range(request.num_samples):
        # Sample parameters
        E_sample = random.gauss(nominal_E, nominal_E * sigma_factor)
        P_sample = random.gauss(nominal_P, nominal_P * sigma_factor)
        T_sample = random.gauss(nominal_T, nominal_T * (sigma_factor * 0.5)) # Manufacturing usually tighter execution
        
        # Clamp to physical limits
        T_sample = max(0.05, T_sample)
        P_sample = max(0.001, P_sample)
        
        # Prepare params
        p = request.params.copy()
        p.youngs_modulus = E_sample
        p.pressure_load = P_sample
        p.strut_thickness = T_sample
        
        # Run Simulation
        if request.use_docker and FOURC_AVAILABLE:
             sim_result = run_real_4c_simulation(p, mesh_coarseness="coarse") # Use coarse for speed in UQ
        else:
             sim_result = run_mockup_simulation(p)
             
        # Collect results
        if sim_result.get("success"):
            res = sim_result.get("result", {})
            stress = res.get("max_stress", 0)
            disp = res.get("max_displacement", 0)
            
            # Compute Safety Factor (Yield Stress approx 400 MPa for Nitinol)
            yield_stress = 400.0
            safety_factor = yield_stress / stress if stress > 0 else 100.0
            
            samples.append({
                "iteration": i,
                "inputs": {"E": E_sample, "Pressure": P_sample, "Thickness": T_sample},
                "outputs": {"Stress": stress, "Displacement": disp, "SafetyFactor": safety_factor}
            })
    
    # Compute Statistics
    if not samples:
        return {"success": False, "error": "No samples generated"}
        
    stresses = [s["outputs"]["Stress"] for s in samples]
    safety_factors = [s["outputs"]["SafetyFactor"] for s in samples]
    
    mean_stress = np.mean(stresses)
    std_stress = np.std(stresses)
    fail_prob = sum(1 for s in stresses if s > 400.0) / len(stresses)
    mean_sf = np.mean(safety_factors)
    
    # Clinical Decision
    decision = "SAFE" if fail_prob < 0.05 and mean_sf > 1.2 else "RISKY"
    if fail_prob > 0.2: decision = "UNSAFE"
    
    return {
        "success": True,
        "summary": {
            "samples": len(samples),
            "mean_stress": float(mean_stress),
            "std_stress": float(std_stress),
            "failure_probability": float(fail_prob),
            "mean_safety_factor": float(mean_sf),
            "decision": decision,
            "uncertainty_level": request.uncertainty_level
        },
        "samples": samples # Return detailed samples for frontend plotting
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting 4C + RL API Server on port {port}")
    print(f"SB3 available: {SB3_AVAILABLE}")
    uvicorn.run(app, host="0.0.0.0", port=port)

