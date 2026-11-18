"""Serve the PathSnap UI on Laptop 2.

Run:  python ui/serve.py --port 5500
Then open http://localhost:5500/?api=http://<laptop-1-ip>:8000
"""
import argparse
import functools
import http.server
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5500)
args = parser.parse_args()

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).parent))
with http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler) as httpd:
    print(f"PathSnap UI on http://localhost:{args.port}/  (add ?api=http://<laptop-1-ip>:8000)")
    httpd.serve_forever()
