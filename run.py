"""Portable command line entry point for the final R2 PCR analysis package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))


def analysis(arguments):
    spec = importlib.util.spec_from_file_location("analysis_cli", ROOT / "code/analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(arguments)


def aggregate_seeds(paths, output):
    """Summarize new seed reruns only; never read frozen manuscript summaries."""
    import numpy as np
    import pandas as pd

    for task in ["classification", "regression"]:
        records = []
        keys = ["target", "model", "algorithm"] + (["target_scale"] if task == "regression" else [])
        for path in paths:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf8"))
            if manifest["status"] != "complete":
                raise ValueError(f"Incomplete seed run: {path.name}")
            frame = pd.read_csv(path / f"{task}_summary.csv")
            frame["training_seed"] = manifest["config"]["neural"]["seed"]
            records.append(frame)
        combined = pd.concat(records, ignore_index=True)
        if combined.duplicated(keys + ["training_seed"]).any():
            raise ValueError("Duplicate model/seed results.")
        sizes = combined.groupby(keys, dropna=False).training_seed.nunique()
        if not (sizes == len(paths)).all():
            raise ValueError("Seed runs have different model inventories.")
        metrics = (
            [
                "auc",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "average_precision",
                "brier",
            ]
            if task == "classification"
            else ["r2", "mae", "rmse", "r2_raw_ppb", "skill"]
        )
        metrics = [c for c in metrics if c in combined]
        summary = combined.groupby(keys, as_index=False, dropna=False).agg(
            **{f"{c}_{stat}": (c, stat) for c in metrics for stat in ["mean", "std"]}
        )
        summary["n_training_seeds"] = len(paths)
        summary["uncertainty"] = (
            "sample SD across rerun training seeds (ddof=1); not a participant-bootstrap CI"
        )
        combined.to_csv(output / f"{task}_seed_results.csv", index=False)
        summary.to_csv(output / f"{task}_seed_summary.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "check",
            "statistics",
            "figures",
            "linear",
            "transformer",
            "sensitivity",
            "permutation",
            "smoke",
            "test",
        ],
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit profile override; figures and smoke have their own defaults.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Override the public workbook, retaining the cohort/label checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New result directory. Default: results/<command>_<UTC timestamp>.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume permutation checkpoint with the same --output and config.",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Number of new permutations; target B remains unchanged."
    )
    parser.add_argument(
        "--neural",
        action="store_true",
        help="Include short Transformer fits in smoke checks; requires PyTorch.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="PyTorch CPU threads for smoke or Transformer training.",
    )
    args = parser.parse_args(argv)
    if args.resume and (args.command != "permutation" or args.output is None):
        parser.error("--resume requires permutation and the original --output.")
    if args.batch_size is not None and (args.command != "permutation" or args.batch_size < 1):
        parser.error("--batch-size requires permutation and a positive integer.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.command == "test":
        return subprocess.call(
            [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
            cwd=ROOT,
        )
    output = (
        args.output
        or ROOT
        / "results"
        / (args.command + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    ).resolve()
    if args.command == "figures":
        from final_figures import main as figures

        command = ["--output", str(output)]
        if args.input:
            command += ["--input", str(args.input.resolve())]
        if args.config:
            command += ["--config", str(args.config.resolve())]
        return figures(command)
    if args.command in {"transformer", "smoke"}:
        if output.exists() and any(output.iterdir()):
            parser.error("Output directory is not empty. Choose a new --output.")
        output.mkdir(parents=True, exist_ok=True)
    if args.command == "transformer" or (args.command == "smoke" and args.neural):
        import torch

        torch.set_num_threads(args.threads)
    if args.command == "transformer":
        profiles = (
            [args.config]
            if args.config
            else [ROOT / f"config/revision_transformer_{seed}.json" for seed in [11, 42, 2026]]
        )
        paths = []
        for profile in profiles:
            seed = json.loads(profile.read_text(encoding="utf8"))["neural"]["seed"]
            path = output / f"seed_{seed}"
            command = ["--config", str(profile), "--results", str(path)]
            if args.input:
                command += ["--input", str(args.input.resolve())]
            status = analysis(command)
            if status:
                return status
            paths.append(path)
        aggregate_seeds(paths, output)
        return 0
    if args.command == "smoke":
        if args.config:
            parser.error("Use the supplied smoke profiles without --config overrides.")
        for profile in ["smoke_linear"] + (["smoke_transformer"] if args.neural else []):
            command = [
                "--config",
                str(ROOT / f"config/{profile}.json"),
                "--results",
                str(output / profile),
            ]
            if args.input:
                command += ["--input", str(args.input.resolve())]
            status = analysis(command)
            if status:
                return status
        print("Development checks complete. These reduced runs are not manuscript results.")
        return 0
    profile = args.config or ROOT / (
        "config/revision_sensitivity.json"
        if args.command == "sensitivity"
        else "config/revision_main.json"
    )
    stage = {
        "check": "preflight",
        "statistics": "descriptive",
        "linear": "analysis",
        "sensitivity": "sensitivity",
        "permutation": "permutation",
    }[args.command]
    command = ["--config", str(profile.resolve()), "--stage", stage, "--results", str(output)]
    if args.input:
        command += ["--input", str(args.input.resolve())]
    if args.resume:
        command += ["--resume"]
    if args.batch_size is not None:
        command += ["--permutation-batch-size", str(args.batch_size)]
    if args.command == "sensitivity":
        command += ["--sensitivity-models"]
    if args.command == "permutation":
        config = json.loads(profile.read_text(encoding="utf8"))
        algorithms = config["modeling"]["algorithms"]
        if any(
            a not in {"logistic_sklearn", "ridge"} for values in algorithms.values() for a in values
        ):
            parser.error("The reported permutation family is restricted to linear estimators.")
    return analysis(command)


if __name__ == "__main__":
    raise SystemExit(main())
