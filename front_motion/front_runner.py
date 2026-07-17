"""Compilation and parallel-run helpers for the front-propagation C code.

Main functions
--------------
read_run_blocks: read [run] blocks from a parameter file as dictionaries.
select_runs: keep all runs or only selected run_id values.
compile_c_code: compile the C source with gcc.
expected_output_files: build snapshot/front/rho/log filenames for one run.
run_front_simulations: launch selected C runs in parallel and print timing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable, Sequence


# ==========================================================
# PARAMETER-FILE HELPERS
# ==========================================================

def strip_comment(line: str) -> str:
    """Remove anything after # and strip surrounding whitespace."""
    return line.split("#", 1)[0].strip()


def parse_value(value: str):
    """Convert a parameter value to int/float when possible; otherwise keep text."""
    value = value.strip()
    try:
        if any(ch in value.lower() for ch in [".", "e"]):
            return float(value)
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return value


def read_named_blocks(filename: str | Path, block_name: str = "run") -> list[dict]:
    """Read named [block_name] blocks from an INI-like parameter file."""
    filename = Path(filename)
    blocks: list[dict] = []
    current: dict | None = None
    target_header = f"[{block_name}]"

    with filename.open("r") as f:
        for raw_line in f:
            line = strip_comment(raw_line)
            if line == "":
                continue

            is_header = line.startswith("[") and line.endswith("]")
            if line == target_header:
                if current is not None:
                    blocks.append(current)
                current = {}
                continue

            if is_header and current is not None:
                blocks.append(current)
                current = None
                continue

            if current is not None and "=" in line:
                key, value = line.split("=", 1)
                current[key.strip()] = parse_value(value)

    if current is not None:
        blocks.append(current)

    return blocks


def read_run_blocks(filename: str | Path) -> list[dict]:
    """Read all [run] blocks and require each block to contain run_id."""
    runs = read_named_blocks(filename, block_name="run")
    for i, run in enumerate(runs):
        if "run_id" not in run:
            raise ValueError(f"Missing run_id in [run] block number {i}")
        run["run_id"] = int(run["run_id"])
    return runs


def select_runs(runs: Sequence[dict], selected_run_ids: Iterable[int] | None = None) -> list[dict]:
    """Return all runs or only those whose run_id is in selected_run_ids."""
    if selected_run_ids is None:
        return list(runs)
    selected = {int(rid) for rid in selected_run_ids}
    return [run for run in runs if int(run["run_id"]) in selected]


# ==========================================================
# COMPILATION
# ==========================================================

def compile_c_code(c_file: str | Path,
                   exe_file: str | Path,
                   compiler: str = "gcc",
                   flags: Sequence[str] | None = None,
                   verbose: bool = True) -> subprocess.CompletedProcess:
    """Compile the C simulation code and raise RuntimeError on failure."""
    c_file = Path(c_file)
    exe_file = Path(exe_file)

    if flags is None:
        flags = ["-O3", "-std=c99", "-Wall", "-Wextra"]

    command = [compiler, *flags, str(c_file), "-lm", "-o", str(exe_file)]

    if verbose:
        print("Compiling C code:")
        print(" ".join(command))

    result = subprocess.run(command, capture_output=True, text=True)

    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed with return code {result.returncode}")

    if verbose:
        print(f"Compilation finished: {exe_file}")

    return result


# ==========================================================
# PARALLEL C RUNS
# ==========================================================

def expected_output_files(data_folder: str | Path, param_base: str, run_id: int) -> dict[str, Path]:
    """Return expected snapshot/front/rho/log paths for one run_id."""
    data_folder = Path(data_folder)
    run_id = int(run_id)
    return {
        "snapshot": data_folder / f"snapshot_{param_base}_run_{run_id:03d}.dat",
        "front": data_folder / f"front_{param_base}_run_{run_id:03d}.dat",
        "rho": data_folder / f"rho_{param_base}_run_{run_id:03d}.dat",
        "log": data_folder / f"{param_base}_run_{run_id:03d}.log",
    }


def _should_skip_run(data_folder: Path, param_base: str, run_id: int,
                     skip_existing: bool, required_output: str) -> bool:
    if not skip_existing:
        return False
    files = expected_output_files(data_folder, param_base, run_id)
    if required_output == "all":
        return files["snapshot"].exists() and files["front"].exists() and files["rho"].exists()
    if required_output not in files:
        raise ValueError("required_output must be 'front', 'snapshot', 'rho', or 'all'")
    return files[required_output].exists()


def run_front_simulations(exe_file: str | Path,
                          param_file: str | Path,
                          max_parallel: int = 8,
                          selected_run_ids: Iterable[int] | None = None,
                          data_folder: str | Path | None = None,
                          skip_existing: bool = False,
                          required_output: str = "front",
                          poll_seconds: float = 1.0,
                          verbose: bool = True) -> list[dict]:
    """Run selected C simulations in parallel and print start/finish timings."""
    exe_file = Path(exe_file).resolve()
    param_file = Path(param_file).resolve()
    param_base = param_file.stem

    if data_folder is None:
        data_folder = Path("data") / param_base
    data_folder = Path(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

    runs = read_run_blocks(param_file)
    selected_runs = select_runs(runs, selected_run_ids)

    if verbose:
        print(f"Executable    : {exe_file}")
        print(f"Parameter file: {param_file}")
        print(f"Output folder : {data_folder.resolve()}")
        print("Selected runs :", [int(r["run_id"]) for r in selected_runs])

    results: list[dict] = []
    running: list[tuple[int, subprocess.Popen, object, float, Path]] = []
    next_run = 0
    global_start = time.perf_counter()

    while next_run < len(selected_runs) or running:
        while next_run < len(selected_runs) and len(running) < int(max_parallel):
            run = selected_runs[next_run]
            run_id = int(run["run_id"])
            files = expected_output_files(data_folder, param_base, run_id)

            if _should_skip_run(data_folder, param_base, run_id, skip_existing, required_output):
                if verbose:
                    print(f"Skipped run_id {run_id:03d}: existing {required_output} output")
                results.append({
                    "run_id": run_id,
                    "status": "skipped",
                    "return_code": 0,
                    "elapsed_seconds": 0.0,
                    "log_file": str(files["log"]),
                })
                next_run += 1
                continue

            log_file = files["log"].open("w")
            start_time = time.perf_counter()
            process = subprocess.Popen(
                [str(exe_file), str(param_file), str(run_id)],
                stdout=log_file,
                stderr=log_file,
                cwd=data_folder,
            )
            running.append((run_id, process, log_file, start_time, files["log"]))

            if verbose:
                print(f"Started run_id {run_id:03d} -> {files['log']}")
            next_run += 1

        still_running = []
        for run_id, process, log_file, start_time, log_path in running:
            code = process.poll()
            if code is None:
                still_running.append((run_id, process, log_file, start_time, log_path))
                continue

            log_file.close()
            elapsed = time.perf_counter() - start_time
            status = "finished" if code == 0 else "error"
            results.append({
                "run_id": run_id,
                "status": status,
                "return_code": code,
                "elapsed_seconds": elapsed,
                "log_file": str(log_path),
            })

            if verbose:
                if code == 0:
                    print(f"Finished run_id {run_id:03d} in {elapsed:.2f} s")
                else:
                    print(f"ERROR run_id {run_id:03d}; return code = {code}; elapsed = {elapsed:.2f} s")

        running = still_running
        if running:
            time.sleep(poll_seconds)

    total_elapsed = time.perf_counter() - global_start
    if verbose:
        print(f"All selected runs finished in {total_elapsed:.2f} s.")

    return sorted(results, key=lambda r: int(r["run_id"]))


# ==========================================================
# COMMAND-LINE ENTRY POINT
# ==========================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and/or run front-propagation simulations.")
    parser.add_argument("param_file", help="Parameter .txt file containing [run] blocks.")
    parser.add_argument("--c-file", default="abm_front_propagation_glued.c")
    parser.add_argument("--exe-file", default="abm_front_propagation_glued.exe")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--selected-run-ids", type=int, nargs="*", default=None)
    parser.add_argument("--data-folder", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--required-output", choices=["front", "snapshot", "rho", "all"], default="front")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.no_compile:
        compile_c_code(args.c_file, args.exe_file)
    run_front_simulations(
        exe_file=args.exe_file,
        param_file=args.param_file,
        max_parallel=args.max_parallel,
        selected_run_ids=args.selected_run_ids,
        data_folder=args.data_folder,
        skip_existing=args.skip_existing,
        required_output=args.required_output,
    )


if __name__ == "__main__":
    main()
