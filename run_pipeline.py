import subprocess

scripts = ["data.py", "datapreproc.py", "baseline_model.py"]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python3", script], check=True)
    if result.returncode == 0:
        print(f"{script} completed successfully.")
    else:
        print(f"Error running {script}.")
        break
print("Pipeline completed.")
