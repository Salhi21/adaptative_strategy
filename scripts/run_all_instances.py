import os
import subprocess
import re
import statistics
import time

# === CONFIGURATION ===
INSTANCE_DIR = "instances"     # path to your .evrp files
RUNS_PER_INSTANCE = 12
OUTPUT_FILE = "results_summary.txt"
PYTHON_CMD = "python"          # or "python3" depending on your setup
SCRIPT = "scripts/run_instance.py"
MAX_GENS = 200                 # or adjust as needed
POP = 50

# === REGEX to extract data from output ===
COST_PATTERN = re.compile(r"best\s*=\s*([0-9]+\.[0-9]+)")
TIME_PATTERN = re.compile(r"CPU\s*time\s*[:=]\s*([0-9]+\.[0-9]+)")

# === MAIN ===
def run_instance(instance_path):
    cmd = [
        PYTHON_CMD, SCRIPT,
        "--instance", instance_path,
        "--max-gens", str(MAX_GENS),
        "--pop", str(POP)
    ]
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    # Try to extract best cost from stdout
    out = result.stdout
    match_cost = COST_PATTERN.search(out)
    best_cost = float(match_cost.group(1)) if match_cost else None

    # Optional CPU time if your script prints it
    match_time = TIME_PATTERN.search(out)
    cpu_time = float(match_time.group(1)) if match_time else elapsed

    return best_cost, cpu_time


def main():
    instances = [f for f in os.listdir(INSTANCE_DIR) if f.endswith(".evrp")]
    if not instances:
        print("No .evrp instances found in", INSTANCE_DIR)
        return

    with open(OUTPUT_FILE, "w") as f_out:
        f_out.write("=== EVRP Benchmark Summary ===\n\n")
        for instance in instances:
            print(f"Running {instance} ...")
            costs, times = [], []
            for r in range(RUNS_PER_INSTANCE):
                cost, t = run_instance(os.path.join(INSTANCE_DIR, instance))
                if cost is not None:
                    costs.append(cost)
                    times.append(t)
                    print(f"  Run {r+1}/{RUNS_PER_INSTANCE}: cost={cost:.2f}, time={t:.2f}s")
                else:
                    print(f"  Run {r+1}: ❌ no cost detected")

            if costs:
                best = min(costs)
                worst = max(costs)
                mean = statistics.mean(costs)
                avg_time = statistics.mean(times)
                summary = (
                    f"Instance: {instance}\n"
                    f"  Best cost   : {best:.2f}\n"
                    f"  Mean cost   : {mean:.2f}\n"
                    f"  Worst cost  : {worst:.2f}\n"
                    f"  Avg CPU time: {avg_time:.2f} s\n\n"
                )
                print(summary)
                f_out.write(summary)
            else:
                f_out.write(f"Instance: {instance} — No valid results.\n\n")

    print("\n✅ Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
