"""
Minimal HTTP server for the PuzzleFusion++ viewer.

Serves:
- assembly_viz.html
- viz_data.json
- Breaking Bad fractured meshes under a configurable prefix (default: /bb/)

Example:
    python serve_viz.py --port 8080 \\
        --bb-root /data/gpfs/projects/punim2657/Breaking-Bad-Dataset.github.io/data/breaking_bad

Then open:
    http://localhost:8080/assembly_viz.html?mesh_base=/bb/
"""

import argparse
import http.server
import os
import posixpath
import socketserver
from pathlib import Path
from urllib.parse import unquote


class VizRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, bb_root: Path, bb_prefix: str, **kwargs):
        self.bb_root = bb_root
        # ensure leading and trailing slash, no double slashes
        self.bb_prefix = "/" + bb_prefix.strip("/") + "/"
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        # Normalise path
        path = unquote(path.split("?", 1)[0].split("#", 1)[0])

        # /bb/... -> map into Breaking Bad root
        if path.startswith(self.bb_prefix):
            rel = path[len(self.bb_prefix) :]
            return str(self.bb_root.joinpath(rel))

        # everything else is served from the current working directory (project root)
        # (use the parent implementation but pinned to cwd)
        project_root = Path(os.getcwd())
        # Strip leading slash to avoid absolute-path behaviour
        rel = path.lstrip("/")
        return str(project_root.joinpath(rel))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve PuzzleFusion++ viewer + meshes.")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind (default: 8080).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host/interface to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--bb-root",
        type=str,
        required=True,
        help="Filesystem path to Breaking Bad mesh root (data/breaking_bad).",
    )
    parser.add_argument(
        "--bb-prefix",
        type=str,
        default="/bb/",
        help="URL prefix for meshes (default: /bb/).",
    )
    args = parser.parse_args()

    bb_root = Path(args.bb_root).expanduser().resolve()
    if not bb_root.exists():
        raise SystemExit(f"bb-root does not exist: {bb_root}")

    HandlerClass = lambda *h_args, **h_kwargs: VizRequestHandler(
        *h_args, bb_root=bb_root, bb_prefix=args.bb_prefix, **h_kwargs
    )

    with socketserver.TCPServer((args.host, args.port), HandlerClass) as httpd:
        print(
            f"Serving PuzzleFusion++ viz on http://{args.host}:{args.port}/ "
            f"(meshes under {args.bb_prefix} → {bb_root})"
        )
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()

