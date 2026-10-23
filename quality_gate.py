import json
import math
import sys

# Maximum acceptable mean absolute error.
# Review the actual MAE from Practical 2 before finalizing.
MAX_MAE = 3.0

with open("metrics.json", encoding="utf-8") as file:
    metrics = json.load(file)

mae = float(metrics["mae"])
rmse = float(metrics["rmse"])
r2 = float(metrics["r2"])

print(f"Model MAE: {mae:.4f}")
print(f"Model RMSE: {rmse:.4f}")
print(f"Model R2: {r2:.4f}")
print(f"Maximum allowed MAE: {MAX_MAE:.4f}")

if not all(math.isfinite(value) for value in (mae, rmse, r2)):
    print("QUALITY GATE FAILED: Invalid model metrics.")
    sys.exit(1)

if mae > MAX_MAE:
    print("QUALITY GATE FAILED: Prediction error is too high.")
    sys.exit(1)

print("QUALITY GATE PASSED")
