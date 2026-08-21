from pathlib import Path
import csv
import math
import json
import re


MAX_PLOT_ITERATIONS = 10000
PARAM_DIMS_TO_PLOT = 18


def _to_float(value):
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _to_int(value):
	try:
		return int(float(value))
	except (TypeError, ValueError):
		return None


def _first_present_float(row, keys):
	for key in keys:
		value = _to_float(row.get(key))
		if value is not None:
			return value
	return None


def summarize_log(log_path):
	summary = {
		"path": log_path,
		"rows": 0,
		"first_iteration": None,
		"last_iteration": None,
		"first_consistency": None,
		"last_consistency": None,
		"first_objective": None,
		"last_objective": None,
		"best_objective": None,
		"consistency_name": None,
		"objective_name": None,
		"run_summary": None,
	}

	try:
		with log_path.open("r", newline="") as f:
			reader = csv.DictReader(f)
			fieldnames = reader.fieldnames or []
			if "nees" in fieldnames:
				summary["consistency_name"] = "nees"
			elif "mean_mahalanobis" in fieldnames:
				summary["consistency_name"] = "mean_mahalanobis"

			if "validation_loss" in fieldnames:
				summary["objective_name"] = "validation_loss"
			elif "validation_nll" in fieldnames:
				summary["objective_name"] = "validation_nll"

			for row in reader:
				iteration = _to_int(row.get("iteration"))
				if iteration is None:
					continue
				if iteration >= MAX_PLOT_ITERATIONS:
					continue

				consistency = _first_present_float(row, ["nees", "mean_mahalanobis"])
				objective = _first_present_float(row, ["validation_loss", "validation_nll"])
				if consistency is None and objective is None:
					continue

				if summary["rows"] == 0:
					summary["first_iteration"] = iteration
					summary["first_consistency"] = consistency
					summary["first_objective"] = objective

				summary["rows"] += 1
				summary["last_iteration"] = iteration
				summary["last_consistency"] = consistency
				summary["last_objective"] = objective

				if objective is not None:
					if summary["best_objective"] is None:
						summary["best_objective"] = objective
					else:
						summary["best_objective"] = min(
							summary["best_objective"], objective
						)

		raw_lines = log_path.read_text().splitlines()
		if "run_summary" in raw_lines:
			start = raw_lines.index("run_summary")
			if start + 2 < len(raw_lines):
				summary_reader = csv.reader([raw_lines[start + 1], raw_lines[start + 2]])
				fields = next(summary_reader, [])
				values = next(summary_reader, [])
				summary["run_summary"] = {
					key: value for key, value in zip(fields, values)
				}
	except Exception as exc:
		summary["error"] = str(exc)

	return summary


def _fmt_num(value):
	if value is None:
		return "n/a"
	return f"{value:.6g}"


def _log10_or_nan(values):
	logged = []
	for value in values:
		if value is None or value <= 0.0:
			logged.append(float("nan"))
		else:
			logged.append(math.log10(value))
	return logged


def _parse_numeric_vector(raw_value):
	if not raw_value:
		return []

	numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_value)
	values = []
	for token in numbers:
		value = _to_float(token)
		if value is not None:
			values.append(value)
	return values


def _load_gt_from_log_run_summary(log_path, parameter_name):
	try:
		raw_lines = log_path.read_text().splitlines()
		if "run_summary" not in raw_lines:
			return None, None

		start = raw_lines.index("run_summary")
		if start + 2 >= len(raw_lines):
			return None, None

		summary_reader = csv.reader([raw_lines[start + 1], raw_lines[start + 2]])
		fields = next(summary_reader, [])
		values = next(summary_reader, [])
		run_summary = {key: value for key, value in zip(fields, values)}

		if parameter_name == "lambda":
			validation_precision_raw = run_summary.get("validation_precision")
			if validation_precision_raw is not None:
				validation_precision = _parse_numeric_vector(validation_precision_raw)
				if len(validation_precision) >= PARAM_DIMS_TO_PLOT:
					gt_lambda = validation_precision[:PARAM_DIMS_TO_PLOT]
					if all(value > 0.0 for value in gt_lambda):
						return gt_lambda, f"{log_path.name}:run_summary.validation_precision"

			validation_variance_raw = run_summary.get("validation_variance")
			if validation_variance_raw is None:
				return None, None

			validation_variance = _parse_numeric_vector(validation_variance_raw)
			if len(validation_variance) < PARAM_DIMS_TO_PLOT:
				return None, None

			gt_lambda = []
			for value in validation_variance[:PARAM_DIMS_TO_PLOT]:
				if value <= 0.0:
					return None, None
				gt_lambda.append(1.0 / value)
			return gt_lambda, f"{log_path.name}:run_summary.validation_variance^-1"

		validation_variance_raw = run_summary.get("validation_variance")
		if validation_variance_raw is not None:
			validation_variance = _parse_numeric_vector(validation_variance_raw)
			if len(validation_variance) >= PARAM_DIMS_TO_PLOT:
				gt_theta = validation_variance[:PARAM_DIMS_TO_PLOT]
				if all(value > 0.0 for value in gt_theta):
					return gt_theta, f"{log_path.name}:run_summary.validation_variance"

		validation_precision_raw = run_summary.get("validation_precision")
		if validation_precision_raw is None:
			return None, None

		validation_precision = _parse_numeric_vector(validation_precision_raw)
		if len(validation_precision) < PARAM_DIMS_TO_PLOT:
			return None, None

		gt_theta = []
		for value in validation_precision[:PARAM_DIMS_TO_PLOT]:
			if value <= 0.0:
				return None, None
			gt_theta.append(1.0 / value)
		return gt_theta, f"{log_path.name}:run_summary.validation_precision^-1"
	except Exception:
		return None, None


def _load_ground_truth_parameter(log_path, repo_root, parameter_name):
	# Prefer values embedded in the selected log's run summary when available.
	gt_from_log, log_source = _load_gt_from_log_run_summary(log_path, parameter_name)
	if gt_from_log is not None:
		return gt_from_log, log_source

	# Best-effort: infer GT theta from configured robot weights when available.
	candidate_paths = [
		repo_root / "assets" / "config" / "robot" / "minimal.json",
		repo_root / "assets" / "config" / "robot" / "base.json",
	]

	for config_path in candidate_paths:
		if not config_path.exists():
			continue

		try:
			with config_path.open("r") as f:
				config = json.load(f)
		except Exception:
			continue

		weights = config.get("weights", {})
		p0 = weights.get("P0")
		if not isinstance(p0, list) or len(p0) < PARAM_DIMS_TO_PLOT:
			continue

		gt = []
		for i in range(PARAM_DIMS_TO_PLOT):
			val = _to_float(p0[i])
			if val is None or val <= 0.0:
				break
			gt.append(1.0 / val if parameter_name == "lambda" else val)

		if len(gt) == PARAM_DIMS_TO_PLOT:
			return gt, config_path

	return None, None


def list_logs_with_summary(log_dir):
	log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
	return [summarize_log(log_path) for log_path in log_files]


def prompt_log_selection(log_summaries):
	if not log_summaries:
		raise RuntimeError("No log files were found.")

	print("Available logs:")
	for idx, summary in enumerate(log_summaries, start=1):
		if "error" in summary:
			print(f"{idx:2d}) {summary['path'].name}  [unreadable: {summary['error']}]")
			continue

		rows = summary["rows"]
		iter_range = f"{summary['first_iteration']} -> {summary['last_iteration']}"
		objective_name = summary.get("objective_name") or "objective"
		consistency_name = summary.get("consistency_name") or "consistency"
		objective_range = (
			f"{_fmt_num(summary['first_objective'])} -> "
			f"{_fmt_num(summary['last_objective'])}"
		)
		consistency_range = f"{_fmt_num(summary['first_consistency'])} -> {_fmt_num(summary['last_consistency'])}"
		best_objective = _fmt_num(summary["best_objective"])
		run_summary = summary.get("run_summary") or {}
		converged = run_summary.get("converged", "n/a")
		N = run_summary.get("N", "n/a")
		K = run_summary.get("K", "n/a")

		print(
			f"{idx:2d}) {summary['path'].name}"
			f" | rows={rows}"
			f" | iter={iter_range}"
			f" | {objective_name}={objective_range} (best={best_objective})"
			f" | {consistency_name}={consistency_range}"
			f" | N={N},K={K},conv={converged}"
		)

	while True:
		user_input = input("Choose a log number (or 'q' to quit): ").strip().lower()
		if user_input in {"q", "quit", "exit"}:
			return None

		if user_input.isdigit():
			selected = int(user_input)
			if 1 <= selected <= len(log_summaries):
				return log_summaries[selected - 1]["path"]

		print("Invalid selection. Please enter a valid number from the list.")


def select_log_file():
	repo_root = Path(__file__).resolve().parents[1]
	log_dir = repo_root / "assets" / "logs"

	if not log_dir.exists():
		raise RuntimeError(f"Log directory does not exist: {log_dir}")

	summaries = list_logs_with_summary(log_dir)
	return prompt_log_selection(summaries)


def _detect_parameter_name(fieldnames):
	if any(name.startswith("lambda[") and name.endswith("]") for name in fieldnames):
		return "lambda"
	if any(name.startswith("theta[") and name.endswith("]") for name in fieldnames):
		return "theta"
	return "theta"


def _sorted_gradient_keys(row, parameter_name):
	prefixes = [f"dL_d{parameter_name}["]
	if parameter_name != "theta":
		prefixes.append("dL_dtheta[")
	if parameter_name != "lambda":
		prefixes.append("dL_dlambda[")

	keys = []
	for key in row.keys():
		for prefix in prefixes:
			if key.startswith(prefix) and key.endswith("]"):
				idx = _to_int(key[len(prefix) : -1])
				if idx is not None:
					keys.append((idx, key))
				break
	keys.sort(key=lambda item: item[0])
	return [key for _, key in keys]


def load_plot_data(log_path):
	iterations = []
	parameter_series = [[] for _ in range(PARAM_DIMS_TO_PLOT)]
	gradient_norms = []
	consistency_values = []
	objective_values = []
	consistency_name = None
	objective_name = None

	with log_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames or []
		parameter_name = _detect_parameter_name(fieldnames)
		if "nees" in fieldnames:
			consistency_name = "nees"
		elif "mean_mahalanobis" in fieldnames:
			consistency_name = "mean_mahalanobis"

		if "validation_loss" in fieldnames:
			objective_name = "validation_loss"
		elif "validation_nll" in fieldnames:
			objective_name = "validation_nll"
		gradient_keys = None

		for row in reader:
			iteration = _to_int(row.get("iteration"))
			consistency = _first_present_float(row, ["nees", "mean_mahalanobis"])
			objective = _first_present_float(row, ["validation_loss", "validation_nll"])

			# Skip non-iteration sections (for example trailing run_summary block).
			if iteration is None or consistency is None or objective is None:
				continue
			if iteration >= MAX_PLOT_ITERATIONS:
				continue

			parameter_values = []
			for i in range(PARAM_DIMS_TO_PLOT):
				parameter_value = _to_float(row.get(f"{parameter_name}[{i}]"))
				if parameter_value is None:
					break
				parameter_values.append(parameter_value)
			if len(parameter_values) != PARAM_DIMS_TO_PLOT:
				continue

			if gradient_keys is None:
				gradient_keys = _sorted_gradient_keys(row, parameter_name)

			grad_components = []
			for key in gradient_keys:
				grad_val = _to_float(row.get(key))
				if grad_val is not None:
					grad_components.append(grad_val)

			if not grad_components:
				continue

			grad_norm = math.sqrt(sum(component * component for component in grad_components))

			iterations.append(iteration)
			for i in range(PARAM_DIMS_TO_PLOT):
				parameter_series[i].append(parameter_values[i])
			gradient_norms.append(grad_norm)
			consistency_values.append(consistency)
			objective_values.append(objective)

	if not iterations:
		raise RuntimeError(f"No valid optimization rows found in log: {log_path}")

	return {
		"iterations": iterations,
		"parameter_series": parameter_series,
		"parameter_name": parameter_name,
		"gradient_norms": gradient_norms,
		"consistency_values": consistency_values,
		"objective_values": objective_values,
		"consistency_name": consistency_name or "consistency",
		"objective_name": objective_name or "objective",
	}


def plot_selected_log(log_path):
	try:
		import matplotlib.pyplot as plt
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"matplotlib is required for plotting. Install it with: pip install matplotlib"
		) from exc

	plot_data = load_plot_data(log_path)
	repo_root = Path(__file__).resolve().parents[1]
	parameter_name = plot_data["parameter_name"]
	gt_parameter, gt_parameter_source = _load_ground_truth_parameter(log_path, repo_root, parameter_name)
	iterations = plot_data["iterations"]
	parameter_series = plot_data["parameter_series"]
	gradient_norms = plot_data["gradient_norms"]
	consistency_values = plot_data["consistency_values"]
	objective_values = plot_data["objective_values"]
	consistency_name = plot_data["consistency_name"]
	objective_name = plot_data["objective_name"]

	fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
	fig.suptitle(f"Optimization Trace: {log_path.name}", fontsize=13)

	ax_theta = axes[0][0]
	for i in range(PARAM_DIMS_TO_PLOT):
		parameter_line, = ax_theta.plot(iterations, _log10_or_nan(parameter_series[i]), label=f"{parameter_name}[{i}]")
		if gt_parameter is not None:
			gt_value = gt_parameter[i]
			if gt_value is not None and gt_value > 0.0:
				ax_theta.plot(
					iterations,
					[math.log10(gt_value)] * len(iterations),
					linestyle="--",
					linewidth=1.0,
					alpha=0.7,
					color=parameter_line.get_color(),
					label=f"gt_{parameter_name}[{i}]",
				)
	ax_theta.set_title(f"log10({parameter_name}[0-17]) vs iteration")
	ax_theta.set_ylabel(f"log10({parameter_name})")
	ax_theta.grid(True, alpha=0.3)
	ax_theta.legend(loc="best", fontsize=6, ncol=3)

	ax_grad = axes[0][1]
	ax_grad.plot(iterations, _log10_or_nan(gradient_norms), color="tab:orange")
	ax_grad.set_title("log10(gradient norm) vs iteration")
	ax_grad.set_ylabel(f"log10(||dL/d{parameter_name}||)")
	ax_grad.grid(True, alpha=0.3)

	ax_nees = axes[1][0]
	ax_nees.plot(iterations, consistency_values, color="tab:green")
	ax_nees.set_title(f"{consistency_name} vs iteration")
	ax_nees.set_xlabel("iteration")
	ax_nees.set_ylabel(consistency_name)
	ax_nees.grid(True, alpha=0.3)

	ax_loss = axes[1][1]
	ax_loss.plot(iterations, objective_values, color="tab:red")
	ax_loss.set_title(f"{objective_name} vs iteration")
	ax_loss.set_xlabel("iteration")
	ax_loss.set_ylabel(objective_name)
	ax_loss.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()


def main():
	selected_log = select_log_file()
	if selected_log is None:
		print("No log selected. Exiting.")
		return

	print(f"Selected log: {selected_log}")
	try:
		plot_selected_log(selected_log)
	except RuntimeError as exc:
		print(str(exc))


if __name__ == "__main__":
	main()
