#!/usr/bin/env python3
"""Convert a saved bilevel experiment and run its lower-level estimator."""

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BLOCKS = {"P0": (0, 18), "Q1": (18, 24), "Q2": (24, 30),
          "Q3": (30, 36), "Q4": (36, 42)}


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def newest_experiment():
    for path in sorted((ROOT / "assets/logs").glob("shared_physical_*"), reverse=True):
        if (path / "experiment.json").is_file() and (path / "parameters.json").is_file():
            if not read_json(path / "parameters.json").get("evaluation_only", False):
                return path
    raise ValueError("No completed shared physical experiment was found.")


def positive_variances(values, count, name):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{name} must contain {count} variances.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name} variances must be finite, numeric, and positive.")
    return values


def converted_robot(manifest, parameters):
    robot = copy.deepcopy(manifest["robot_config"])
    theta = parameters.get("theta")
    if not isinstance(theta, list) or len(theta) not in (36, 42):
        raise ValueError("Saved theta must have 42 entries, or 36 legacy entries with recorded Q4.")
    positive_variances(theta, len(theta), "theta")
    version = 2 if len(theta) == 42 else 1
    if parameters.get("parameter_layout_version", version) != version:
        raise ValueError("Parameter layout version disagrees with theta length.")
    if len(theta) == 36:
        weights = robot["weights"]
        if "fixed_Q4" in parameters:
            Q4 = positive_variances(parameters["fixed_Q4"], 6, "fixed_Q4")
        else:
            Q4 = positive_variances(weights.get("Q4"), 6, "Recorded Q4")
            factor = weights.get("prior_factor", 1)
            positive_variances([factor], 1, "prior_factor")
            Q4 = [factor * value for value in Q4]
        theta = theta + positive_variances(Q4, 6, "Effective Q4")
    for name, (start, end) in BLOCKS.items():
        robot["weights"][name] = theta[start:end]
    robot["weights"]["prior_factor"] = 1
    robot.setdefault("options", {})["estimator"] = "batch"
    return robot


def trial_path(requested, manifest, parameters):
    if requested is None:
        config = parameters.get("config", manifest.get("bilevel_config", {}))
        trials = config.get("physical_testing_trials", [])
        if len(trials) != 1:
            raise ValueError("Specify --trial: the experiment must have exactly one test trial to select a default.")
        requested = trials[0]
    path = Path(requested)
    if not path.is_absolute() and len(path.parts) == 1:
        path = Path("assets/config/trial") / (path.name if path.suffix else path.name + ".json")
    return resolve_path(path)


def export_metrics(output, robot_length):
    positive_variances([robot_length], 1, "Robot length")
    source = output / "logs/pose_metrics.csv"
    if not source.is_file():
        raise ValueError(f"Missing {source}; rebuild estimator and rerun the trial to export marginal NEES.")
    samples = {"base": [], "body": [], "tip": [], "all": []}
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            arclength = float(row["arclength"])
            if not math.isfinite(arclength):
                raise ValueError("Pose metric arclength must be finite.")
            if arclength < -1e-6 or arclength > robot_length + 1e-6:
                continue
            translation = float(row["translation_error_m"])
            rotation = float(row["rotation_error_rad"])
            nees_values = [float(row[name]) if row.get(name) else None
                           for name in ("marginal_nees", "translation_nees", "rotation_nees")]
            if (not math.isfinite(translation) or translation < 0 or
                    not math.isfinite(rotation) or rotation < 0 or
                    any(nees is not None and (not math.isfinite(nees) or nees < 0)
                        for nees in nees_values)):
                raise ValueError("Exported pose metrics must be finite and nonnegative.")
            region = "base" if arclength <= 1e-6 else (
                "tip" if abs(arclength - robot_length) <= 1e-6 else "body")
            denominator = robot_length if region == "base" else arclength
            sample = (translation, rotation, nees_values[0], 100 * translation / denominator,
                      nees_values[1], nees_values[2])
            samples[region].append(sample)
            samples["all"].append(sample)
    if not samples["all"]:
        raise ValueError("No ground-truth pose samples are available for metrics.")
    rows = []
    for region, values in samples.items():
        count = len(values)
        valid_nees = [sample[2] for sample in values if sample[2] is not None]
        translation_nees = [sample[4] for sample in values if sample[4] is not None]
        rotation_nees = [sample[5] for sample in values if sample[5] is not None]
        percentages = [100 * sample[0] / robot_length if region == "all" else sample[3]
                       for sample in values]
        rows.append({
            "region": region,
            "sample_count": count,
            "nees_count": len(valid_nees),
            "translation_mae_m": math.fsum(sample[0] for sample in values) / count if count else None,
            "translation_rmse_m": math.sqrt(math.fsum(sample[0] ** 2 for sample in values) / count) if count else None,
            "translation_mae_pct": math.fsum(percentages) / count if count else None,
            "translation_rmse_pct": math.sqrt(math.fsum(value ** 2 for value in percentages) / count) if count else None,
            "rotation_mae_rad": math.fsum(sample[1] for sample in values) / count if count else None,
            "rotation_rmse_rad": math.sqrt(math.fsum(sample[1] ** 2 for sample in values) / count) if count else None,
            "marginal_nees_mean": math.fsum(valid_nees) / len(valid_nees) if valid_nees else None,
            "translation_nees_count": len(translation_nees),
            "translation_nees_mean": math.fsum(translation_nees) / len(translation_nees) if translation_nees else None,
            "rotation_nees_count": len(rotation_nees),
            "rotation_nees_mean": math.fsum(rotation_nees) / len(rotation_nees) if rotation_nees else None,
        })
    summary_path = output / "logs/metrics_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "logs/metrics_summary.json", {
        "translation_units": "metres; percentages use marker arclength for body/tip and robot length for base/all",
        "rotation_units": "rad (orientation error, not angular velocity)",
        "nees_convention": "mean marginal 6D pose NEES, not joint NEES; nominal reference 6",
        "split_nees_convention": "translation/rotation use the first/last 3 SE(3) log-error components and their 3x3 marginal covariance blocks; nominal reference 3 each; their sum need not equal 6D NEES",
        "body_definition": "0 < arclength < robot_length",
        "tip_definition": "arclength == robot_length within 1e-6 m",
        "base_definition": "abs(arclength) <= 1e-6 m",
        "all_definition": "all base, body and tip samples, pooled by sample count",
        "base_excluded": False,
        "robot_length_m": robot_length,
        "metrics": rows,
    })
    print(f"Metrics: {summary_path}", flush=True)
    for row in rows:
        if not row["sample_count"]:
            print(f"  {row['region']}: no samples", flush=True)
            continue
        nees = row["marginal_nees_mean"]
        nees_text = f"{nees:.6g}" if nees is not None else "unavailable"
        translation_nees = row["translation_nees_mean"]
        translation_nees_text = f"{translation_nees:.6g}" if translation_nees is not None else "unavailable"
        rotation_nees = row["rotation_nees_mean"]
        rotation_nees_text = f"{rotation_nees:.6g}" if rotation_nees is not None else "unavailable"
        print(f"  {row['region']}: translation MAE/RMSE "
              f"{row['translation_mae_m']:.6g}/{row['translation_rmse_m']:.6g} m "
              f"({row['translation_mae_pct']:.6g}/{row['translation_rmse_pct']:.6g} %, "
              f"{'robot length' if row['region'] in ('base', 'all') else 'marker arclength'}), "
              f"rotation MAE/RMSE {row['rotation_mae_rad']:.6g}/{row['rotation_rmse_rad']:.6g} rad, "
              f"marginal NEES {nees_text} ({row['nees_count']}/{row['sample_count']} valid), "
              f"translation NEES {translation_nees_text} ({row['translation_nees_count']}/{row['sample_count']} valid), "
              f"rotation NEES {rotation_nees_text} ({row['rotation_nees_count']}/{row['sample_count']} valid)", flush=True)
    return rows


def export_all_trials_summary(output, trial_roles):
    rows = []
    for trial, role in trial_roles.items():
        source = output / trial.stem / "logs/metrics_summary.csv"
        with source.open(newline="") as stream:
            regional = [row for row in csv.DictReader(stream) if row["region"] in ("body", "tip")]
        for region in ("body", "tip"):
            matches = [row for row in regional if row["region"] == region]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one '{region}' metrics row in {source}.")
            rows.append({"trial": trial.stem, "split": role, **matches[0]})
    summary_path = output / "all_trials_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def formatted(row, name, precision=2, unit=""):
        value = row.get(name)
        return f"{float(value):.{precision}f}{unit}" if value else "unavailable"

    headers = ["Trial", "Split", "Region", "Samples", "Translation MAE/RMSE (m)",
               "Translation MAE/RMSE (%)", "Rotation MAE/RMSE (rad)",
               "Pose NEES (valid)", "Translation NEES (valid)", "Rotation NEES (valid)"]
    table = [headers]
    for row in rows:
        cells = [row["trial"], row["split"], row["region"], row["sample_count"]]
        for quantity, unit, precision in (("translation", "m", 6),
                                          ("translation", "pct", 2), ("rotation", "rad", 2)):
            suffix = "%" if unit == "pct" else unit
            cells.append(" / ".join(formatted(row, f"{quantity}_{metric}_{unit}", precision, suffix)
                                    for metric in ("mae", "rmse")))
        for name, count in (("marginal_nees_mean", "nees_count"),
                            ("translation_nees_mean", "translation_nees_count"),
                            ("rotation_nees_mean", "rotation_nees_count")):
            cells.append(f"{formatted(row, name)} ({row.get(count, '0')}/{row['sample_count']})")
        table.append(cells)
    widths = [max(len(row[column]) for row in table) for column in range(len(headers))]
    print(f"All-trial summary: {summary_path}", flush=True)
    for index, cells in enumerate(table):
        print(" | ".join(cell.ljust(width) for cell, width in zip(cells, widths)), flush=True)
        if index == 0:
            print("-+-".join("-" * width for width in widths), flush=True)
    return rows


def export_optimizer_history(experiment, output):
    timestamp = experiment.name.removeprefix("shared_physical_")
    filename = f"bilevel_optimizer_outer_loop_{timestamp}.log"
    candidates = [experiment / filename, experiment.parent / filename, ROOT / "assets/logs" / filename]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        print(f"Warning: optimizer history plot skipped; no matching {filename} found.", file=sys.stderr)
        return None
    history = []
    columns = ("iteration", "validation_nll", "mean_mahalanobis")
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or []):
            print(f"Warning: optimizer history plot skipped; {source} lacks required columns {columns}.",
                  file=sys.stderr)
            return None
        for row in reader:
            if row["iteration"].strip() == "run_summary":
                break
            values = tuple(float(row[name]) for name in columns)
            if all(math.isfinite(value) for value in values):
                history.append(values)
    if not history:
        print(f"Warning: optimizer history plot skipped; no finite history in {source}.", file=sys.stderr)
        return None
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        print("Warning: optimizer history plot requires Matplotlib; install with python3 -m pip install matplotlib.",
              file=sys.stderr)
        return None
    figure = Figure(figsize=(10, 4.2), layout="constrained")
    FigureCanvasAgg(figure)
    loss_axis, nees_axis = figure.subplots(1, 2)
    iterations, losses, nees = zip(*history)
    loss_axis.plot(iterations, losses, color="#168477", linewidth=1.6)
    loss_axis.set(title="Loss", ylabel="Validation NLL")
    nees_axis.plot(iterations, nees, color="#bd493a", linewidth=1.6, label="Mean pose NEES")
    nees_axis.axhline(6, color="#666666", linestyle="--", linewidth=1, label="Nominal: 6")
    nees_axis.set(title="Mean NEES", ylabel="Pose NEES (6D)")
    nees_axis.legend()
    for axis in (loss_axis, nees_axis):
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.2)
    figure.suptitle(experiment.name)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "optimizer_history.png"
    figure.savefig(destination, dpi=180)
    print(f"Optimizer history: {destination} (source: {source})", flush=True)
    return destination


def run_lower(command, output, robot_length):
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    with (output / "console.log").open("w") as log:
        log.write(shlex.join(command) + "\n")
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            summarized = False
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    if "Launching visualizer..." in line and not summarized:
                        export_metrics(output, robot_length)
                        summarized = True
                returncode = process.wait()
            except (KeyboardInterrupt, OSError, ValueError) as error:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                if isinstance(error, KeyboardInterrupt):
                    return 130
                raise
    if returncode:
        return returncode if returncode > 0 else 128 - returncode
    results = output / "logs/results.txt"
    if results.exists():
        for line in results.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "Success" and value.strip() != "1":
                raise ValueError(f"Lower-level solve failed; see {results} and {output / 'console.log'}.")
    if not summarized:
        export_metrics(output, robot_length)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", help="Experiment directory; default: newest completed training experiment.")
    trial_selection = parser.add_mutually_exclusive_group()
    trial_selection.add_argument("--trial", help="Trial name or JSON path; default: experiment's single test trial.")
    trial_selection.add_argument("--all-trials", action="store_true",
                                 help="Replay all recorded experiment trials sequentially, headless, with per-trial metrics.")
    parser.add_argument("--logs-only", "--log-only", action="store_true",
                        help="Run headless and export logs without opening the visualizer.")
    parser.add_argument("--lower-only", action="store_true",
                        help="Reuse the converted robot config; do not regenerate its weights.")
    parser.add_argument("--robot-config", help="Converted robot path; default: experiment/robot_optimized.json.")
    parser.add_argument("--output", help="Output directory, or parent directory with --all-trials; default: experiment/lower_level/TRIAL.")
    parser.add_argument("--estimator", default="build/release/estimator", help="Lower-level executable path.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare configs and print the command without running it.")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Summarize existing pose_metrics.csv without rerunning the estimator.")
    args = parser.parse_args(argv)
    try:
        experiment = resolve_path(args.experiment) if args.experiment else newest_experiment()
        manifest = read_json(experiment / "experiment.json")
        parameters = read_json(experiment / "parameters.json")
        recorded_trials = dict(manifest.get("trial_configs", {}))
        recorded_trials.update(parameters.get("trial_configs", {}))
        if args.all_trials:
            if not recorded_trials:
                raise ValueError("The experiment has no recorded trial configs to replay.")
            selected_trials = list(dict.fromkeys(resolve_path(name) for name in recorded_trials))
            if len({trial.stem for trial in selected_trials}) != len(selected_trials):
                raise ValueError("Recorded trial names must be unique for --all-trials output directories.")
        else:
            selected_trials = [trial_path(args.trial, manifest, parameters)]
        output_parent = resolve_path(args.output) if args.output else experiment / "lower_level"
        trial_roles = {}
        if args.all_trials:
            config = {**manifest.get("bilevel_config", {}), **parameters.get("config", {})}
            training_trials = {resolve_path(name) for name in config.get("physical_training_trials", [])}
            validation_trials = {resolve_path(name) for name in config.get("physical_testing_trials", [])}
            for selected_trial in selected_trials:
                if (selected_trial in training_trials) == (selected_trial in validation_trials):
                    raise ValueError(f"Trial must belong to exactly one saved training/validation split: {selected_trial}")
                trial_roles[selected_trial] = "training" if selected_trial in training_trials else "validation"
        robot_path = resolve_path(args.robot_config) if args.robot_config else experiment / "robot_optimized.json"
        if args.metrics_only:
            robot = read_json(robot_path) if robot_path.is_file() else manifest["robot_config"]
            for selected_trial in selected_trials:
                output = output_parent
                if args.all_trials or not args.output:
                    output /= selected_trial.stem
                export_metrics(output, robot["topology"]["length"])
            if args.all_trials:
                export_all_trials_summary(output_parent, trial_roles)
                export_optimizer_history(experiment, output_parent)
            return 0
        executable = resolve_path(args.estimator)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"Estimator executable is missing or not executable: {executable}")
        if args.lower_only:
            if not robot_path.is_file():
                raise ValueError(f"Robot config does not exist: {robot_path}; run without --lower-only first.")
        else:
            write_json(robot_path, converted_robot(manifest, parameters))
        robot_length = read_json(robot_path)["topology"]["length"]
        for selected_trial in selected_trials:
            output = output_parent
            if args.all_trials or not args.output:
                output /= selected_trial.stem
            output.mkdir(parents=True, exist_ok=True)
            for name, snapshot in recorded_trials.items():
                if resolve_path(name) == selected_trial:
                    snapshot = copy.deepcopy(snapshot)
                    snapshot.setdefault("options", {})["estimator"] = "batch"
                    selected_trial = output / "trial_config.json"
                    write_json(selected_trial, snapshot)
                    break
            if not selected_trial.is_file():
                raise ValueError(f"Trial config does not exist: {selected_trial}")
            estimator_config = copy.deepcopy(manifest["estimator_config"])
            estimator_config.setdefault("options", {})["estimator"] = "batch"
            estimator_config["options"]["compute_covariances"] = True
            estimator_path = output / "estimator_config.json"
            write_json(estimator_path, estimator_config)
            command = [str(executable), str(robot_path), str(selected_trial), str(estimator_path),
                       "--output_path", str(output) + os.sep]
            if args.logs_only or args.all_trials:
                command.append("--headless")
            print(f"Experiment: {experiment}", flush=True)
            print(f"Robot config: {robot_path}", flush=True)
            print(f"Output: {output}", flush=True)
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                returncode = run_lower(command, output, robot_length)
                if returncode:
                    return returncode
        if args.all_trials and not args.dry_run:
            export_all_trials_summary(output_parent, trial_roles)
            export_optimizer_history(experiment, output_parent)
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())