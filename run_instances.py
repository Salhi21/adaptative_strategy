import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# === Configuration ===
instance_files = [
    "E-n29-k4-s7.evrp",
    "E-n30-k3-s7.evrp",
    "E-n35-k3-s5.evrp",
    "E-n37-k4-s4.evrp",
    "E-n60-k5-s9.evrp",
    "E-n89-k7-s13.evrp",
    "E-n112-k8-s11.evrp",
    "F-n49-k4-s4.evrp",
    "F-n80-k4-s8.evrp",
    "F-n140-k7-s5.evrp",
    "M-n110-k10-s9.evrp",
    "M-n126-k7-s5.evrp",
    "M-n163-k12-s12.evrp",
    "M-n212-k16-s12.evrp",
]

runs_per_instance = 1
max_parallel_jobs = 10  # Safe value for your 10-core CPU

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


def run_single_job(instance_file, run_id):
    """Runs one instance once and saves output."""
    # Create subfolder for this instance file
    instance_name = instance_file[:-5]
    instance_dir = os.path.join(output_dir, instance_name)
    os.makedirs(instance_dir, exist_ok=True)

    output_filename = os.path.join(instance_dir, f"run_{run_id}_output.txt")
    cmd = [
        "python", "-m", "scripts.run_instance",
        "--instance", f"instance/{instance_file}",
        "--max-gens", "20",
        "--pop", "20"
    ]

    with open(output_filename, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"✅ Finished {instance_file}, run {run_id} → {output_filename}")


if __name__ == "__main__":
    # Create all jobs: (instance_file, run_id)
    jobs = [(inst, i + 1) for inst in instance_files for i in range(runs_per_instance)]

    print(f"🚀 Starting {len(jobs)} runs across {len(instance_files)} instances...")
    print(f"⚙️  Using up to {max_parallel_jobs} parallel processes.\n")

    with ProcessPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = [executor.submit(run_single_job, inst, run_id) for inst, run_id in jobs]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error in a run: {e}")

    print("\n🎉 All runs completed successfully!")
