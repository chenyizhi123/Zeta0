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
    p.add_argument("--root", help="Source checkout, required for workflow/init commands outside the checkout")
    p.add_argument("--json", action="store_true", help="Machine-readable report")
    sub = p.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Inspect existing environments/assets without installing anything")
    doctor.add_argument("--profile", choices=PROFILES, default="core")
    doctor.add_argument("--python", action="append", default=[], help="Existing interpreter; repeat to compare environments")
    doctor.add_argument("--asset-path-map", help="Existing asset relocation map")
    doctor.add_argument("--output", help="Write a new JSON report")

    hosts = sub.add_parser("hosts", help="List supported and planned layered VLA hosts")
    hosts.add_argument("--output", help="Write a new JSON report")

    init = sub.add_parser("init", help="Scaffold a resolved v6/raw training configuration from a host template")
    init.add_argument("--host", choices=("psi0", "zeta_native", "gr00t_n1d7"), required=True)
    init.add_argument("--output", required=True, help="Destination path for the new JSON config")
    init.add_argument("--qwen-model", help="Local Qwen/exported backbone directory")
    init.add_argument("--gr00t-model", help="Local GR00T N1.7 base directory (auto-computes config SHA-256)")
    init.add_argument("--prepared-dir", help="Prepared dataset directory containing registries/index/feature_cache")
    init.add_argument("--output-dir", help="Training run output directory")
    init.add_argument("--max-updates", type=int)
    init.add_argument("--gpus", type=int)
    init.add_argument("--micro-batch", type=int)

    data = sub.add_parser("data", help="Inspect or pull official datasets").add_subparsers(dest="operation", required=True)
    inspect = data.add_parser("inspect")
    inspect.add_argument("dataset")
    inspect.add_argument("--full", action="store_true", help="Run the existing numerical/provenance training adapter (needs training dependencies)")
    inspect.add_argument("--task", default="inspection")
    inspect.add_argument("--validation-fraction", type=float, default=0.0)
    inspect.add_argument("--seed", type=int, default=292285)
    inspect.add_argument("--output")

    pull = data.add_parser("pull", help="Plan or download official Zeta0 datasets from HuggingFace")
    pull.add_argument("dataset", help="Official alias ('real') or HuggingFace 'owner/repo'")
    pull.add_argument("--output", required=True, help="Target local directory")
    pull.add_argument("--tasks", nargs="*", default=None, help="Optional subset of tasks (e.g., place_chair)")
    pull.add_argument("--revision", default="main")
    pull.add_argument("--execute", action="store_true", help="Actually download files via huggingface_hub")
    pull.add_argument("--report-output", help="Write a new JSON report")

    bundle = sub.add_parser("bundle", help="Inspect model packages").add_subparsers(dest="operation", required=True)
    bundle_inspect = bundle.add_parser("inspect")
    bundle_inspect.add_argument("directory")
    bundle_inspect.add_argument("--full", action="store_true", help="Verify file hashes and the existing source contract")
    bundle_inspect.add_argument("--output")

    logs = sub.add_parser("logs", help="Summarize existing runtime traces offline").add_subparsers(dest="operation", required=True)
    summary = logs.add_parser("summarize")
    summary.add_argument("filename")
    summary.add_argument("--max-events", type=int, default=100000)
    summary.add_argument("--output")

    viz = sub.add_parser("visualize", help="Render offline foresight rollouts or RuntimeAudit telemetry dashboards").add_subparsers(dest="operation", required=True)
    viz_log = viz.add_parser("log", help="Plot inference latency and RTC queue depth from a RuntimeAudit JSONL")
    viz_log.add_argument("filename")
    viz_log.add_argument("--output", required=True, help="Destination PNG path")
    viz_log.add_argument("--max-events", type=int, default=100000)
    viz_rollout = viz.add_parser("rollout", help="Plot predicted vs measured future body trajectories")
    viz_rollout.add_argument("summary", help="Path to quickstart summary.json or rollout JSON")
    viz_rollout.add_argument("--output", required=True, help="Destination PNG path")
    viz_rollout.add_argument("--urdf", help="Optional robot URDF for forward-kinematics verification")
    viz_rollout.add_argument("--rerun", action="store_true", help="Also stream trajectories to rerun-sdk if installed")

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
        report_output = getattr(args, "output", None)
        if args.command == "doctor":
            from psi.framework.doctor import doctor
            try:
                root = checkout_root(args.root)
            except ValueError:
                if args.root:
                    raise
                root = None
            report = doctor(root, selected=args.python, profile=args.profile, asset_path_map=args.asset_path_map)
        elif args.command == "hosts":
            # Keep metadata lightweight without importing torch.
            report = {
                "format": "zeta0.hosts.v1",
                "ok": True,
                "hosts": {
                    "zeta_native": {
                        "status": "supported",
                        "backbone_family": "qwen3_vl",
                        "supports_kv_cache": False,
                        "requires_image_mask": False,
                        "default_action_horizon": 50,
                        "default_solver_steps": 10,
                    },
                    "gr00t_n1d7": {
                        "status": "supported",
                        "backbone_family": "gr00t_n1d7_vl",
                        "supports_kv_cache": False,
                        "requires_image_mask": True,
                        "default_action_horizon": 40,
                        "default_solver_steps": 4,
                    },
                    "pi05": {
                        "status": "planned",
                        "backbone_family": "paligemma_siglip",
                        "supports_kv_cache": True,
                        "requires_image_mask": False,
                        "default_action_horizon": 50,
                        "default_solver_steps": 10,
                    },
                },
            }
        elif args.command == "init":
            from psi.framework.init_config import init_training_config
            root = checkout_root(args.root)
            report = init_training_config(
                root,
                host=args.host,
                output=args.output,
                qwen_model=args.qwen_model,
                gr00t_model=args.gr00t_model,
                prepared_dir=args.prepared_dir,
                output_dir=args.output_dir,
                max_updates=args.max_updates,
                gpus_per_node=args.gpus,
                micro_batch=args.micro_batch,
            )
            report_output = None
        elif args.command == "data":
            if args.operation == "inspect":
                from psi.framework.data import inspect_data
                report = inspect_data(args.dataset, full=args.full, task=args.task,
                                      validation_fraction=args.validation_fraction, seed=args.seed)
            else:
                from psi.framework.hub import build_pull_plan, execute_pull
                report = build_pull_plan(
                    args.dataset, output=args.output, tasks=args.tasks, revision=args.revision
                )
                if args.execute:
                    report = execute_pull(report)
                report_output = args.report_output
        elif args.command == "bundle":
            from psi.framework.bundle import inspect_bundle
            report = inspect_bundle(args.directory, full=args.full)
        elif args.command == "logs":
            from psi.framework.logs import summarize_log
            report = summarize_log(args.filename, max_events=args.max_events)
        elif args.command == "visualize":
            from psi.framework.visualize import visualize_log, visualize_rollout
            if args.operation == "log":
                report = visualize_log(args.filename, output=args.output, max_events=args.max_events)
            else:
                report = visualize_rollout(args.summary, output=args.output, urdf=args.urdf, use_rerun=args.rerun)
            report_output = None
        else:
            from psi.framework.workflow import execute_plan, workflow_plan
            root = checkout_root(args.root)
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            report = workflow_plan(root, args.name, arguments, args.python)
            if args.execute:
                # Keep stdout available for the target command's own output.
                print(report["command"], file=sys.stderr, flush=True)
                return execute_plan(report, root)
        save_report(report, report_output)
        emit(report, args.json)
        return 0 if report.get("ok") else 1
    except (OSError, ValueError, KeyError, TypeError, ImportError, NotImplementedError) as error:
        if isinstance(error, ImportError):
            message = f"Missing optional dependency: {error.name}. Reuse a prepared interpreter; run doctor --profile train --python /path/to/python."
        else:
            message = str(error)
        emit({"format": "zeta0.cli_error.v1", "ok": False, "error": message}, args.json)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
