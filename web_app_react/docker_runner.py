"""
Docker 4C Runner - Backend endpoint for running 4C simulations via Docker.
Provides endpoints to run the 4C solver in Docker and export results for the webviewer.
"""

import json
import subprocess
import tempfile
import threading
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

# Docker image for 4C
DOCKER_IMAGE = "ghcr.io/4c-multiphysics/4c:main"
DOCKER_FOURC_PATH = "/home/user/4C/build/4C"

# Global state
state_lock = threading.Lock()
simulation_state = {
    "running": False,
    "progress": 0.0,
    "result": None,
    "error": None,
    "output_files": [],
}


def run_docker_simulation(yaml_content: str, output_dir: Path, timeout: float = 300):
    """
    Run a 4C simulation inside Docker container.
    
    Args:
        yaml_content: 4C YAML input file content
        output_dir: Directory for output files
        timeout: Timeout in seconds
    
    Returns:
        dict with success, files, and error info
    """
    global simulation_state
    
    with state_lock:
        simulation_state["running"] = True
        simulation_state["progress"] = 0.0
        simulation_state["error"] = None
        simulation_state["result"] = None
    
    try:
        # Create temp directory for input/output
        work_dir = Path(tempfile.mkdtemp(prefix="4c_sim_"))
        yaml_path = work_dir / "input.4C.yaml"
        results_dir = work_dir / "results"
        results_dir.mkdir()
        
        # Write YAML input
        yaml_path.write_text(yaml_content)
        
        # Build Docker command
        cmd = [
            "docker", "run", "--rm",
            "--platform", "linux/amd64",  # Required for Apple Silicon
            "-v", f"{work_dir}:/workspace",
            "-w", "/workspace",
            DOCKER_IMAGE,
            DOCKER_FOURC_PATH,
            "input.4C.yaml",
            "-o", "results"
        ]
        
        with state_lock:
            simulation_state["progress"] = 0.1
        
        # Run simulation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir)
        )
        
        with state_lock:
            simulation_state["progress"] = 0.9
        
        if result.returncode != 0:
            with state_lock:
                simulation_state["running"] = False
                simulation_state["error"] = result.stderr or "Simulation failed"
            return {"success": False, "error": result.stderr}
        
        # Collect output files
        output_files = list(results_dir.glob("*.vtk")) + list(results_dir.glob("*.vtu"))
        
        # Copy to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        copied_files = []
        for f in output_files:
            dest = output_dir / f.name
            dest.write_bytes(f.read_bytes())
            copied_files.append(str(dest))
        
        with state_lock:
            simulation_state["running"] = False
            simulation_state["progress"] = 1.0
            simulation_state["output_files"] = copied_files
            simulation_state["result"] = {
                "success": True,
                "files": copied_files,
                "log": result.stdout[:1000] if result.stdout else ""
            }
        
        return simulation_state["result"]
        
    except subprocess.TimeoutExpired:
        with state_lock:
            simulation_state["running"] = False
            simulation_state["error"] = f"Timeout after {timeout}s"
        return {"success": False, "error": f"Timeout after {timeout}s"}
        
    except Exception as e:
        with state_lock:
            simulation_state["running"] = False
            simulation_state["error"] = str(e)
        return {"success": False, "error": str(e)}


def generate_stent_yaml(params: dict) -> str:
    """
    Generate a minimal 4C YAML input for stent simulation.
    In production, this would be more complete.
    """
    yaml_content = f"""
# 4C Stent Simulation Input
# Auto-generated from AutoStent Web App

PROBLEM TYP:
  PROBLEMTYP: Structure
  RESTART: 0

DISCRETISATION:
  NUMSTRUCDIS: 1

STRUCTURAL DYNAMIC:
  LINEAR_SOLVER: 1
  INT_STRATEGY: Standard
  DYNAMICTYP: Statics
  RESULTSEVRY: 1
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0

MATERIALS:
  MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 210000.0 NUE 0.3 DENS 7.85e-9

# Stent geometry parameters (for reference):
# Diameter: {params.get('diameter', 10.0)} mm
# Length: {params.get('length', 20.0)} mm
# Strut Thickness: {params.get('strutThickness', 0.12)} mm
# Num Struts: {params.get('numStruts', 12)}
# Crown Height: {params.get('crownHeight', 1.0)} mm
"""
    return yaml_content


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads"""
    daemon_threads = True


class DockerHandler(BaseHTTPRequestHandler):
    def _headers(self, ctype='application/json', status=200):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        self._headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/docker/status':
            self._headers()
            with state_lock:
                resp = {
                    "running": simulation_state["running"],
                    "progress": simulation_state["progress"],
                    "result": simulation_state["result"],
                    "error": simulation_state["error"],
                    "output_files": simulation_state["output_files"],
                }
            self.wfile.write(json.dumps(resp).encode())
            
        elif path == '/docker/check':
            # Check if Docker is available
            self._headers()
            try:
                result = subprocess.run(
                    ["docker", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                available = result.returncode == 0
                self.wfile.write(json.dumps({
                    "available": available,
                    "version": result.stdout.strip() if available else None
                }).encode())
            except Exception as e:
                self.wfile.write(json.dumps({
                    "available": False,
                    "error": str(e)
                }).encode())
                
        elif path == '/':
            self._headers('text/plain')
            self.wfile.write(b"Docker 4C Runner - OK")
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode() if length > 0 else '{}'
        data = json.loads(body) if body else {}
        
        if path == '/docker/run':
            # Start simulation in background thread
            params = data.get('params', {})
            yaml_content = data.get('yaml') or generate_stent_yaml(params)
            output_dir = Path(data.get('output_dir', '/tmp/4c_results'))
            timeout = data.get('timeout', 300)
            
            with state_lock:
                if simulation_state["running"]:
                    self._headers()
                    self.wfile.write(json.dumps({
                        "status": "already_running"
                    }).encode())
                    return
            
            # Run in background
            thread = threading.Thread(
                target=run_docker_simulation,
                args=(yaml_content, output_dir, timeout),
                daemon=True
            )
            thread.start()
            
            self._headers()
            self.wfile.write(json.dumps({"status": "started"}).encode())
            
        elif path == '/docker/export_vtk':
            # Generate VTK file from parameters (for webviewer)
            params = data.get('params', {})
            # For now, return a placeholder - in production would generate actual VTK
            self._headers()
            self.wfile.write(json.dumps({
                "status": "generated",
                "vtk_path": f"/tmp/stent_{params.get('diameter', 10)}.vtk"
            }).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging


if __name__ == '__main__':
    port = 8766
    print(f"Docker 4C Runner at http://localhost:{port}")
    print(f"Endpoints:")
    print(f"  GET  /docker/status - Check simulation status")
    print(f"  GET  /docker/check  - Check Docker availability")
    print(f"  POST /docker/run    - Start a simulation")
    print(f"  POST /docker/export_vtk - Export VTK for webviewer")
    
    # Check Docker availability on startup
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"Docker: {result.stdout.strip()}")
        else:
            print("WARNING: Docker not available")
    except Exception as e:
        print(f"WARNING: Docker check failed: {e}")
    
    server = ThreadedHTTPServer(('localhost', port), DockerHandler)
    server.serve_forever()
