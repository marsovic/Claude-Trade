"""Real-time web dashboard (read-only) for the survival trading bot.

It reads the files the bot writes in ``STATE_DIR`` and never talks to the
exchange, so it can run on its own and cannot place or cancel any order.

Usage::

    python dashboard/dashboard.py                    # http://127.0.0.1:8050
    python dashboard/dashboard.py --state-dir state --port 8050
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.state import StateStore  # noqa: E402

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

MAX_CURVE_POINTS = 1500


def downsample(points: list, limit: int = MAX_CURVE_POINTS) -> list:
    if len(points) <= limit:
        return points
    step = len(points) / limit
    out = [points[int(i * step)] for i in range(limit)]
    out[-1] = points[-1]
    return out


def read_state(store: StateStore) -> dict | None:
    for _ in range(3):  # the bot replaces the file atomically; retry on a race
        try:
            if not store.state_path.exists():
                return None
            return json.loads(store.state_path.read_text())
        except (json.JSONDecodeError, OSError):
            time.sleep(0.05)
    return None


def create_app(state_dir: str | os.PathLike) -> Flask:
    app = Flask(__name__, static_folder=None)
    store = StateStore(state_dir)
    static_dir = Path(__file__).parent / "static"

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/state")
    def api_state():
        state = read_state(store)
        if state is not None:
            state["equity_history"] = downsample(state.get("equity_history", []))
            state.pop("income_seen", None)
        return jsonify({
            "now": int(time.time() * 1000),
            "state": state,
            "decisions": store.recent_decisions(200)[::-1],
            "trades": store.recent_trades(200)[::-1],
            "tombstone": store.tombstone(),
        })

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.after_request
    def no_cache(resp):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:")
        return resp

    return app


def main(argv=None) -> int:
    if load_dotenv is not None and Path(".env").exists():
        load_dotenv(".env", override=False)
    p = argparse.ArgumentParser(description="Survival trading bot dashboard")
    p.add_argument("--state-dir", default=os.environ.get("STATE_DIR", "state"))
    p.add_argument("--host", default=os.environ.get("DASHBOARD_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8050")))
    args = p.parse_args(argv)
    app = create_app(args.state_dir)
    print(f"Dashboard on http://{args.host}:{args.port} (state: {args.state_dir})")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
