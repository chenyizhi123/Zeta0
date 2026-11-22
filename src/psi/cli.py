"""Public Zeta operations CLI; inspection/help do not import model runtimes."""
from __future__ import annotations

import argparse
import json
import sys

from psi.framework.common import checkout_root, save_report


def parser() -> argparse.ArgumentParser:
    from psi.framework.doctor import PROFILES
    from psi.framework.workflow import WORKFLOWS
    p = argparse.ArgumentParser(prog="zeta", description=__doc__)
    p.add_argument("--root", help="Source checkout, required for workflow commands outside the checkout")
    p.add_argument("--json", action="store_true", help="Machine-readable report")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect existing environments/assets without installing anything")
    doctor.add_argument("--profile", choices=PROFILES, default="core")
    doctor.add_argument("--python", action="append", default=[], help="Existing interpreter; repeat to compare environments")
    doctor.add_argument("--asset-path-map", help="Existing asset relocation map")
    doctor.add_argument("--output", help="Write a new JSON report")
    data = sub.add_parser("data", help="Inspect captured datasets").add_subparsers(dest="operation", required=True)
    inspect = data.add_parser("inspect")
    inspect.add_argument("dataset")
    inspect.add_argument("--full", action="store_true", help="Run the existing numerical/provenance training adapter (needs training dependencies)")
    inspect.add_argument("--task", default="inspection")
    inspect.add_argument("--validation-fraction", type=float, default=0.0)
    inspect.add_argument("--seed", type=int, default=292285)
    inspect.add_argument("--output")
    bundle = sub.add_parser("bundle", help="Inspect model packages").add_subparsers(dest="operation", required=True)
    inspect = bundle.add_parser("inspect")
    inspect.add_argument("directory")
    inspect.add_argument("--full", action="store_true", help="Verify file hashes and the existing source contract")
    inspect.add_argument("--output")
    logs = sub.add_parser("logs", help="Summarize existing runtime traces offline").add_subparsers(dest="operation", required=True)
    summary = logs.add_parser("summarize")
    summary.add_argument("filename")
    summary.add_argument("--max-events", type=int, default=100000)
    summary.add_argument("--output")
    workflow = sub.add_parser("workflow", help="Plan or execute a maintained workflow; flags after -- belong to its entrypoint")
    workflow.add_argument("--python", help="Reuse this interpreter; nothing will be installed")
    workflow.add_argument("--execute", action="store_true", help="Execute the plan; without this flag only print it")
    workflow.add_argument("name", choices=WORKFLOWS)
    workflow.add_argument("arguments", nargs=argparse.REMAINDER)
    return p


def emit(report: dict, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return
    print(report.get("format", "zeta"), "OK" if report.get("ok") else "NEEDS ATTENTION")
    for key, value in report.items():
        if key not in {"format", "ok"}:
            print(f"{key}: " + (json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from psi.framework.doctor import doctor
            try:
                root = checkout_root(args.root)
            except ValueError:
                if args.root:
                    raise
                root = None
            report = doctor(root, selected=args.python, profile=args.profile, asset_path_map=args.asset_path_map)
        elif args.command == "data":
            from psi.framework.data import inspect_data
            report = inspect_data(args.dataset, full=args.full, task=args.task,
                                  validation_fraction=args.validation_fraction, seed=args.seed)
        elif args.command == "bundle":
            from psi.framework.bundle import inspect_bundle
            report = inspect_bundle(args.directory, full=args.full)
        elif args.command == "logs":
            from psi.framework.logs import summarize_log
            report = summarize_log(args.filename, max_events=args.max_events)
        else:
            from psi.framework.workflow import execute_plan, workflow_plan
            root = checkout_root(args.root)
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            report = workflow_plan(root, args.name, arguments, args.python)
            if args.execute:
                # Keep stdout available for the target command's own output.
                print(report["command"], file=sys.stderr, flush=True)
                return execute_plan(report, root)
        save_report(report, getattr(args, "output", None))
        emit(report, args.json)
        return 0 if report.get("ok") else 1
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        if isinstance(error, ImportError):
            message = f"Missing optional dependency: {error.name}. Reuse a prepared interpreter; run doctor --profile train --python /path/to/python."
        else:
            message = str(error)
        emit({"format": "zeta0.cli_error.v1", "ok": False, "error": message}, args.json)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
