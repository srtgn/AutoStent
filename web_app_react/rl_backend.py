"""
Real RL Backend Server for AutoStent Web App
Uses actual Stable-Baselines3 PPO with threading
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("WARNING: stable-baselines3 not installed")

import gymnasium as gym
from gymnasium import spaces

# Global state
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
        
        # Update every 20 steps
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
    
    if not SB3_AVAILABLE:
        with state_lock:
            training_state["is_training"] = False
        return
    
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
        import traceback
        traceback.print_exc()
        with state_lock:
            training_state["is_training"] = False


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads"""
    daemon_threads = True


class RLHandler(BaseHTTPRequestHandler):
    def _headers(self, ctype='application/json'):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        self._headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/status':
            self._headers()
            with state_lock:
                resp = {
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
            self.wfile.write(json.dumps(resp).encode())
        elif path == '/':
            self._headers('text/plain')
            self.wfile.write(b"RL Backend Running")
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode() if length > 0 else '{}'
        data = json.loads(body) if body else {}
        
        if path == '/start':
            with state_lock:
                is_training = training_state["is_training"]
            
            if not is_training:
                steps = data.get('steps', 2000)
                thread = threading.Thread(target=run_training, args=(steps,), daemon=True)
                thread.start()
                self._headers()
                self.wfile.write(json.dumps({"status": "started", "steps": steps}).encode())
            else:
                self._headers()
                self.wfile.write(json.dumps({"status": "already_running"}).encode())
                
        elif path == '/stop':
            with state_lock:
                training_state["is_training"] = False
            self._headers()
            self.wfile.write(json.dumps({"status": "stopped"}).encode())
            
        elif path == '/reset':
            with state_lock:
                training_state["is_training"] = False
                training_state["progress"] = 0.0
                training_state["current_step"] = 0
                training_state["episodes"] = 0
                training_state["rewards"] = []
                training_state["stress_history"] = []
                training_state["episode_rewards"] = []
                training_state["best_reward"] = -100.0
            self._headers()
            self.wfile.write(json.dumps({"status": "reset"}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    port = 8765
    print(f"RL Backend at http://localhost:{port}")
    print(f"SB3 available: {SB3_AVAILABLE}")
    server = ThreadedHTTPServer(('localhost', port), RLHandler)
    server.serve_forever()
