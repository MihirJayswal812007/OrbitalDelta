"""
API server launcher.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8001 --host 0.0.0.0 --reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the OrbitalDelta API server")
    parser.add_argument("--host",    default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port",    type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    parser.add_argument("--reload",  action="store_true", help="Enable hot-reload (dev only)")
    args = parser.parse_args()

    print(f"\n  OrbitalDelta API  →  http://{args.host}:{args.port}")
    print(f"  Swagger docs       →  http://{args.host}:{args.port}/docs")
    print(f"  Map viewer         →  http://{args.host}:{args.port}/\n")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
