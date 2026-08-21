import csv
import math

log_path = 'assets/logs/bilevel_optimizer_outer_loop_20260820_132613.log'

with open(log_path, 'r') as f:
    # Read until we hit "run_summary" or end of file
    lines = []
    for line in f:
        if 'run_summary' in line:
            break
        lines.append(line)

# Let's parse CSV from lines
reader = csv.DictReader(lines)
data = list(reader)

print(f"Total iterations parsed: {len(data)}")
print("First row keys:", data[0].keys() if data else "None")

# 1. converged status
print("\n--- converged check ---")
# Let's search the file again to look for converged value
converged = False
with open(log_path, 'r') as f:
    text = f.read()
    if 'converged' in text:
        # Find lines below converged to see if it is true or false
        idx = text.find('converged')
        sub = text[idx:idx+300]
        print("Converged block:", sub)
        if 'true' in sub.lower():
            converged = True
        elif 'false' in sub.lower():
            converged = False

# 2. First and last iteration rows metrics: iteration, validation_nll, mean_mahalanobis, mean_logdet
first_row = data[0]
last_row = data[-1]

print("\n--- First and Last Rows ---")
print(f"First row (iter {first_row['iteration']}):")
print(f"  validation_nll: {first_row['validation_nll']}")
print(f"  mean_mahalanobis: {first_row['mean_mahalanobis']}")
print(f"  mean_logdet: {first_row['mean_logdet']}")

print(f"Last row (iter {last_row['iteration']}):")
print(f"  validation_nll: {last_row['validation_nll']}")
print(f"  mean_mahalanobis: {last_row['mean_mahalanobis']}")
print(f"  mean_logdet: {last_row['mean_logdet']}")


# 3. Trend over last 50 iterations: validation_nll and gradient proxy from dL_dlambda norm (compute approximate L2 from 36 entries)
print("\n--- Last 50 Iterations Statistics ---")
last_50 = data[-50:]
last_50_nll = [float(row['validation_nll']) for row in last_50]

# Compute dL_dlambda L2 norm for each of the last 50 rows
dl_dl_norms = []
for row in last_50:
    vals = [float(row[f'dL_dlambda[{i}]']) for i in range(36)]
    l2 = math.sqrt(sum(v**2 for v in vals))
    dl_dl_norms.append(l2)

print(f"validation_nll over last 50 iterations: min={min(last_50_nll)}, max={max(last_50_nll)}, first={last_50_nll[0]}, last={last_50_nll[-1]}")
print(f"dL_dlambda L2 norm over last 50 iterations: min={min(dl_dl_norms):.4e}, max={max(dl_dl_norms):.4e}, first={dl_dl_norms[0]:.4e}, last={dl_dl_norms[-1]:.4e}")

# Let's inspect some values of dL_dlambda to verify why they are extremely small or large
print("\nLast iteration dL_dlambda entries (all 36):")
last_dl = [float(last_row[f'dL_dlambda[{i}]']) for i in range(36)]
print(last_dl)

# 4. Signs of instability (oscillation, divergence, exploding lambda)
# Let's look for maximum lambda value over all iterations to check for explosion
max_lambda_per_iter = []
lambdas_all = []
for row in data:
    l_vals = [float(row[f'lambda[{i}]']) for i in range(36)]
    max_lambda_per_iter.append(max(l_vals))

print("\n--- Instability Checks ---")
print(f"Max lambda value over all iterations: {max(max_lambda_per_iter)}")
print(f"Last lambda values (first 10):", [float(last_row[f'lambda[{i}]']) for i in range(10)])

# Let's look at the validation_nll trend overall
nll_all = [float(row['validation_nll']) for row in data]
print(f"Overall validation_nll: initial={nll_all[0]}, min={min(nll_all)}, final={nll_all[-1]}")

