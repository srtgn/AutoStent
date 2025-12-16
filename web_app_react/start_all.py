#!/usr/bin/env python3
"""
Unified startup script for AutoStent Web Application.
Launches all required services:
1. RL Backend (port 8765) - Real RL training with PPO
2. Docker Runner (port 8766) - 4C Docker simulation
3. (Optional) 4C-Webviewer (port 5000) - Official 4C visualization

To use, run: python start_all.py
Then open index.html in your browser.
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

# Service ports
RL_BACKEND_PORT = 8765
DOCKER_RUNNER_PORT = 8766
WEBVIEWER_PORT = 5000

# Process handles
processes = []


def start_service(name, cmd, cwd=None):
    """Start a background service."""
    print(f"Starting {name}...")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append((name, proc))
        time.sleep(1)  # Give it a moment to start
        if proc.poll() is None:
            print(f"  ✓ {name} started (PID: {proc.pid})")
            return True
        else:
            print(f"  ✗ {name} failed to start")
            return False
    except Exception as e:
        print(f"  ✗ {name} error: {e}")
        return False


def stop_all():
    """Stop all started services."""
    print("\nStopping services...")
    for name, proc in processes:
        if proc.poll() is None:
            print(f"  Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except:
                proc.kill()
    print("All services stopped.")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    stop_all()
    sys.exit(0)


def main():
    """Start all services."""
    script_dir = Path(__file__).parent
    
    print("="*60)
    print("AutoStent Web Application - Service Launcher")
    print("="*60)
    print()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start RL Backend
    rl_backend = script_dir / "rl_backend.py"
    if rl_backend.exists():
        start_service(
            f"RL Backend (:{RL_BACKEND_PORT})",
            [sys.executable, str(rl_backend)],
            cwd=script_dir
        )
    else:
        print(f"  ! RL Backend not found at {rl_backend}")
    
    # Start Docker Runner
    docker_runner = script_dir / "docker_runner.py"
    if docker_runner.exists():
        start_service(
            f"Docker 4C Runner (:{DOCKER_RUNNER_PORT})",
            [sys.executable, str(docker_runner)],
            cwd=script_dir
        )
    else:
        print(f"  ! Docker Runner not found at {docker_runner}")
    
    print()
    print("="*60)
    print("Services ready!")
    print("="*60)
    print(f"  RL Backend:     http://localhost:{RL_BACKEND_PORT}")
    print(f"  Docker Runner:  http://localhost:{DOCKER_RUNNER_PORT}")
    print()
    print("Open index.html in your browser to use the app.")
    print("Press Ctrl+C to stop all services.")
    print()
    
    # Keep running
    try:
        while True:
            # Check if services are still running
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"  ! {name} has stopped (exit code: {proc.returncode})")
            time.sleep(5)
    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()
