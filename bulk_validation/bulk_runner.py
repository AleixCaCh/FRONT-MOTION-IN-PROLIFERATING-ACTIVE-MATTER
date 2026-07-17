from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time


def read_run_ids(param_file):
    run_ids = []

    with open(param_file, "r") as file:
        for line in file:
            line = line.split("#", 1)[0].strip()

            if line.startswith("run_id") and "=" in line:
                run_id = int(line.split("=", 1)[1])
                run_ids.append(run_id)

    return run_ids


def compile_code(c_file, exe_file):
    command = [
        "gcc",
        "-O3",
        "-std=c99",
        "-Wall",
        "-Wextra",
        str(c_file),
        "-lm",
        "-o",
        str(exe_file),
    ]

    print(" ".join(command))
    subprocess.run(command, check=True)
    print(f"Compiled: {exe_file}")


def output_files(data_folder, param_base, run_id):
    name = f"{param_base}_run_{run_id:03d}.dat"

    return [
        data_folder / name,
        data_folder / f"g_{name}",
        data_folder / f"S_{name}",
    ]


def run_one(exe_file, param_file, run_id, data_folder, skip_existing):
    param_base = Path(param_file).stem
    outputs = output_files(data_folder, param_base, run_id)

    if skip_existing and all(filename.exists() for filename in outputs):
        return {
            "run_id": run_id,
            "status": "skipped",
            "seconds": 0,
        }

    log_file = data_folder / f"{param_base}_run_{run_id:03d}.log"
    start = time.time()

    with open(log_file, "w") as log:
        result = subprocess.run(
            [str(exe_file), str(param_file), str(run_id)],
            cwd=data_folder,
            stdout=log,
            stderr=log,
        )

    status = "finished" if result.returncode == 0 else "error"

    return {
        "run_id": run_id,
        "status": status,
        "seconds": time.time() - start,
        "log_file": str(log_file),
    }


def run_parallel(
    exe_file,
    param_file,
    run_ids,
    data_folder,
    max_parallel=4,
    skip_existing=False,
):
    exe_file = Path(exe_file).resolve()
    param_file = Path(param_file).resolve()
    data_folder = Path(data_folder).resolve()
    data_folder.mkdir(parents=True, exist_ok=True)

    print(f"Executable: {exe_file}")
    print(f"Parameters: {param_file}")
    print(f"Data folder: {data_folder}")
    print(f"Run IDs: {run_ids}")

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = []

        for run_id in run_ids:
            future = pool.submit(
                run_one,
                exe_file,
                param_file,
                run_id,
                data_folder,
                skip_existing,
            )
            futures.append(future)

        results = [future.result() for future in futures]

    results.sort(key=lambda row: row["run_id"])

    for row in results:
        print(
            f"run {row['run_id']:03d}: "
            f"{row['status']} "
            f"({row['seconds']:.2f} s)"
        )

    return results
