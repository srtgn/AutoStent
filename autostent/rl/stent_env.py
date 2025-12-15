"""
Stent Design RL Environment

Gymnasium-compatible environment for optimizing stent designs using 4C FEM.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import shutil

from ..geometry import SplineStentGeometry, StentGeometryParameters, export_to_4c_mesh
from ..simulation import FourCSimulator, SimulationConfig, SimulationResult
from ..automation import FourCYAMLGenerator
from ..evaluation import compute_biomechanical_metrics
from .observation_space import ObservationSpace
from .action_space import ActionSpace


class StentDesignEnv(gym.Env):
    """
    Gymnasium environment for stent design optimization.
    
    Each step:
    1. Modifies geometry parameters based on action
    2. Generates spline geometry
    3. Exports mesh
    4. Generates 4C YAML input
    5. Runs 4C FEM simulation
    6. Parses results
    7. Computes reward from biomechanical metrics
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        fourc_executable: Optional[str] = None,
        working_dir: Optional[Path] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 50,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize environment.
        
        Args:
            fourc_executable: Path to 4C executable
            working_dir: Working directory for simulations
            render_mode: Rendering mode
            max_episode_steps: Maximum steps per episode
            reward_weights: Weights for reward components
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Working directory
        if working_dir:
            self.working_dir = Path(working_dir)
        else:
            self.working_dir = Path(tempfile.mkdtemp(prefix="autostent_rl_"))
        
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.simulator = FourCSimulator(fourc_executable=fourc_executable)
        self.yaml_generator = FourCYAMLGenerator()
        
        # Observation and action spaces
        self.observation_space_obj = ObservationSpace()
        self.action_space_obj = ActionSpace()
        
        self.observation_space = self.observation_space_obj.space
        self.action_space = self.action_space_obj.space
        
        # Reward weights
        self.reward_weights = reward_weights or {
            "stress": -0.4,  # Minimize stress
            "displacement": -0.2,  # Minimize displacement
            "strain": -0.2,  # Minimize strain
            "convergence": 0.2,  # Reward convergence
        }
        
        # Normalize weights
        total = sum(abs(w) for w in self.reward_weights.values())
        self.reward_weights = {k: v / total for k, v in self.reward_weights.items()}
        
        # State
        self.current_params: Optional[StentGeometryParameters] = None
        self.last_result: Optional[SimulationResult] = None
        self.episode_history: list = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Optional reset options (e.g., initial parameters)
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_history = []
        
        # Initialize parameters
        if options and "initial_parameters" in options:
            self.current_params = options["initial_parameters"]
        else:
            # Random initialization within valid ranges
            self.current_params = StentGeometryParameters(
                diameter=self.np_random.uniform(8.0, 12.0),
                length=self.np_random.uniform(15.0, 25.0),
                strut_thickness=self.np_random.uniform(0.08, 0.15),
                strut_width=self.np_random.uniform(0.12, 0.25),
                num_struts=self.np_random.integers(10, 16),
                crown_height=self.np_random.uniform(0.8, 1.5),
                crown_radius=self.np_random.uniform(0.3, 0.8),
            )
        
        # Generate initial observation
        observation = self._get_observation()
        
        info = {
            "parameters": self._params_to_dict(self.current_params),
            "step": self.current_step,
        }
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action vector
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Apply action to modify parameters
        current_dict = self._params_to_dict(self.current_params)
        modified_dict = self.action_space_obj.apply_action(current_dict, action)
        
        # Update parameters
        self.current_params = StentGeometryParameters(**modified_dict)
        
        # Validate parameters
        errors = self.current_params.validate()
        if errors:
            reward = -100.0  # Large penalty for invalid design
            terminated = True
            truncated = False
            observation = self._get_observation()
            info = {
                "parameters": self._params_to_dict(self.current_params),
                "errors": errors,
                "step": self.current_step,
            }
            return observation, reward, terminated, truncated, info
        
        # Generate geometry
        geometry = SplineStentGeometry(self.current_params)
        
        # Export mesh
        mesh_path = self.working_dir / f"stent_{self.current_step:04d}.vtk"
        export_to_4c_mesh(geometry, mesh_path)
        
        # Generate YAML
        yaml_path = self.working_dir / f"stent_{self.current_step:04d}.4C.yaml"
        self.yaml_generator.generate(
            geometry_params=self.current_params,
            mesh_path=mesh_path,
            output_path=yaml_path,
        )
        
        # Run simulation
        output_dir = self.working_dir / f"output_{self.current_step:04d}"
        config = SimulationConfig(
            yaml_input_path=yaml_path,
            output_directory=output_dir,
            timeout=600.0,
        )
        
        self.last_result = self.simulator.run_simulation(config)
        
        # Compute reward
        if not self.last_result.success:
            reward = -100.0
            terminated = True
        else:
            metrics = compute_biomechanical_metrics(self.last_result)
            reward = self._compute_reward(metrics)
            terminated = False
        
        # Check truncation
        truncated = self.current_step >= self.max_episode_steps
        
        # Update history
        self.episode_history.append({
            "step": self.current_step,
            "parameters": self._params_to_dict(self.current_params),
            "result": self.last_result.to_dict() if self.last_result else None,
            "reward": reward,
        })
        
        # Observation
        observation = self._get_observation()
        
        # Info
        info = {
            "parameters": self._params_to_dict(self.current_params),
            "result": self.last_result.to_dict() if self.last_result else None,
            "step": self.current_step,
            "metrics": metrics if self.last_result and self.last_result.success else {},
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        design_params = self._params_to_dict(self.current_params)
        
        if self.last_result and self.last_result.success:
            simulation_results = {
                "max_stress": self.last_result.max_von_mises_stress,
                "max_displacement": self.last_result.max_displacement,
                "max_strain": self.last_result.max_principal_strain,
            }
            converged = self.last_result.converged
            residual_norm = self.last_result.residual_norm
        else:
            simulation_results = {
                "max_stress": 0.0,
                "max_displacement": 0.0,
                "max_strain": 0.0,
            }
            converged = False
            residual_norm = 1.0
        
        return self.observation_space_obj.encode(
            design_params,
            simulation_results,
            converged,
            residual_norm,
        )
    
    def _compute_reward(self, metrics: Dict[str, float]) -> float:
        """Compute reward from biomechanical metrics."""
        reward = 0.0
        
        # Stress component (minimize)
        stress_norm = metrics.get("max_stress", 0.0) / 500.0
        reward += self.reward_weights["stress"] * (1.0 - stress_norm)
        
        # Displacement component (minimize)
        disp_norm = metrics.get("max_displacement", 0.0) / 10.0
        reward += self.reward_weights["displacement"] * (1.0 - disp_norm)
        
        # Strain component (minimize)
        strain_norm = metrics.get("max_strain", 0.0) / 0.1
        reward += self.reward_weights["strain"] * (1.0 - strain_norm)
        
        # Convergence component (maximize)
        if metrics.get("converged", False):
            reward += self.reward_weights["convergence"]
        
        return float(reward)
    
    def _params_to_dict(self, params: StentGeometryParameters) -> Dict[str, float]:
        """Convert parameters to dictionary."""
        return {
            "diameter": params.diameter,
            "length": params.length,
            "strut_thickness": params.strut_thickness,
            "strut_width": params.strut_width,
            "num_struts": params.num_struts,
            "crown_height": params.crown_height,
            "crown_radius": params.crown_radius,
        }
    
    def close(self):
        """Clean up resources."""
        # Optionally clean up working directory
        # if self.working_dir.exists():
        #     shutil.rmtree(self.working_dir)
        pass

