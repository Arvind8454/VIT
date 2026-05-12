from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import socket
from pathlib import Path


def resolve_nport_command() -> list[str]:
    found = shutil.which("nport")
    if found:
        return [found, "5000"]
    npm_global = Path.home() / "AppData" / "Roaming" / "npm" / "nport.cmd"
    if npm_global.exists():
        return [str(npm_global), "5000"]
    raise FileNotFoundError(
        "NPort command not found. Install with `npm install -g nport` "
        "or add npm global bin to PATH."
    )


def wait_for_port(host: str, port: int, timeout_sec: int = 180) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.8)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(1.0)
    return False


def main() -> int:
    print("Starting Flask app...")
    flask_process = subprocess.Popen([sys.executable, "app.py"])
    print("Waiting for Flask on port 5000...")
    if not wait_for_port("127.0.0.1", 5000, timeout_sec=240):
        print("Flask did not open port 5000 in time. Tunnel will not start.", file=sys.stderr)
        flask_process.terminate()
        return 1

    print("Starting NPort tunnel on port 5000...")
    nport_cmd = resolve_nport_command()
    tunnel_process = subprocess.Popen(nport_cmd, env=os.environ.copy())

    try:
        flask_code = flask_process.wait()
        return flask_code
    except KeyboardInterrupt:
        print("\nStopping Flask and NPort...")
        flask_process.terminate()
        tunnel_process.terminate()
        return 0
    finally:
        if tunnel_process.poll() is None:
            tunnel_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
