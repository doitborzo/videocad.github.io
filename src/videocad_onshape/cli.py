from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .errors import VideoCADOnshapeError
from .runtime import Application


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videocad-onshape")
    parser.add_argument("--config", type=Path, default=None, help="Path to videocad_onshape.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a prompt autonomously in Onshape")
    run_parser.add_argument("--prompt", required=True)

    dry_run_parser = subparsers.add_parser("dry-run", help="Predict actions without clicking")
    dry_run_parser.add_argument("--prompt", required=True)

    subparsers.add_parser("calibrate", help="Detect and validate the active Onshape canvas")

    healthcheck_parser = subparsers.add_parser("healthcheck", help="Run planner/runtime health checks")
    healthcheck_parser.add_argument("--json", action="store_true", help="Print healthcheck results as JSON")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    app = Application(config)

    try:
        if args.command == "run":
            summary = app.execute_prompt(args.prompt, mode="run")
            print(json.dumps(summary.to_dict(), indent=2))
            return 0
        if args.command == "dry-run":
            summary = app.execute_prompt(args.prompt, mode="dry-run")
            print(json.dumps(summary.to_dict(), indent=2))
            return 0
        if args.command == "calibrate":
            calibration = app.calibrate()
            print(json.dumps(calibration, indent=2))
            return 0
        if args.command == "healthcheck":
            result = app.healthcheck()
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                for item in result.items:
                    status = "OK" if item.ok else "FAIL"
                    print(f"[{status}] {item.name}: {item.detail}")
            return 0 if result.ok else 1
    except VideoCADOnshapeError as exc:
        print(f"error: {exc}")
        return 2

    parser.error(f"Unhandled command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
