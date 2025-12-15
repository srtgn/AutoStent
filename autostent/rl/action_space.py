"""
Action Space Definition

Defines the action space for modifying stent design parameters.
"""

import numpy as np
from gymnasium import spaces
from typing import Dict, Any


class ActionSpace:
    """
    Action space for stent design RL environment.
    
    Actions modify design parameters within valid ranges.
    """
    
    def __init__(self, modification_scale: float = 0.1):
        """
        Initialize action space.
        
        Args:
            modification_scale: Maximum relative change per action (0-1)
        """
        self.modification_scale = modification_scale
        
        # 7 design parameters that can be modified
        # Actions are in [-1, 1], representing relative modifications
        self.space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )
    
    def apply_action(
        self,
        current_params: Dict[str, float],
        action: np.ndarray,
    ) -> Dict[str, float]:
        """
        Apply action to modify design parameters.
        
        Args:
            current_params: Current design parameters
            action: Action vector (normalized [-1, 1])
            
        Returns:
            Modified design parameters
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Parameter ranges
        ranges = {
            "diameter": (5.0, 15.0),
            "length": (10.0, 30.0),
            "strut_thickness": (0.05, 0.2),
            "strut_width": (0.1, 0.3),
            "num_struts": (8, 16),
            "crown_height": (0.5, 2.0),
            "crown_radius": (0.2, 1.0),
        }
        
        param_order = [
            "diameter", "length", "strut_thickness", "strut_width",
            "num_struts", "crown_height", "crown_radius"
        ]
        
        modified = {}
        
        for i, param in enumerate(param_order):
            current_value = current_params[param]
            low, high = ranges[param]
            range_size = high - low
            
            # Apply modification
            modification = action[i] * self.modification_scale * range_size
            new_value = current_value + modification
            
            # Clip to valid range
            new_value = np.clip(new_value, low, high)
            
            # Round integer parameters
            if param == "num_struts":
                new_value = int(np.round(new_value))
            
            modified[param] = new_value
        
        return modified

