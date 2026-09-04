"""
Launch script for the Dead Reckoning Web Application server.
Runs on 0.0.0.0:8000 to enable access from both desktop browsers
and mobile phones connected to the same local network.
"""
import os
import sys
import uvicorn

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.server.app import app

if __name__ == "__main__":
    print("[*] Starting Intelligent Dead Reckoning Navigation Server...")
    print("[*] Local Access:  http://localhost:8000")
    print("[*] Mobile Access: Open http://<YOUR_IP>:8000 on your smartphone in vehicle mount.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
