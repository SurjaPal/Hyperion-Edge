#!/usr/bin/env python3
"""
Convenience launcher: `python scripts/run_demo.py`
Starts the swarm simulation + dashboard server and opens it in a browser.
"""
import os
import sys
import webbrowser
from threading import Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.dirname(__file__))

URL = "http://localhost:8787"


def _open_browser():
    try:
        webbrowser.open(URL)
    except Exception:
        pass


if __name__ == "__main__":
    import fetch_assets  # best-effort: model + sample frames
    fetch_assets.main()

    import server  # backend/server.py

    print(f"Starting Sentinel-Edge -- dashboard will open at {URL}")
    Timer(1.5, _open_browser).start()
    server.main()
