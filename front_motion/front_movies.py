"""Parallel movie-launch helpers for the front-propagation workflow.

Main functions
--------------
trajectory_files_for_runs: find snapshot files for selected run_id values.
build_output_mp4: construct the movie filename for one movie kind.
build_movie_jobs: make the full list of movie jobs.
run_movie_jobs_parallel: launch front_movie_worker.py jobs in parallel with timing.
make_front_movies: high-level function used directly from notebooks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from front_analysis import discover_run_ids, get_run_files


def trajectory_files_for_runs(data_folder: str | Path,
                              param_base: str,
                              selected_run_ids: Iterable[int] | None = None,
                              require_front: bool = True) -> list[Path]:
    """Return existing snapshot files for selected runs.

    If selected_run_ids is None, runs are discovered from front_ files by
    default, because front observables are required for zoom movies and most
    analysis.
    """
    data_folder = Path(data_folder)
    if selected_run_ids is None:
        source = "front" if require_front else "snapshot"
        selected_run_ids = discover_run_ids(data_folder, param_base, source=source)

    dat_files = []
    missing = []
    for rid in selected_run_ids:
        snapshot_file, front_file, _ = get_run_files(data_folder, param_base, int(rid))
        snapshot_file = Path(snapshot_file)
        front_file = Path(front_file)
        if not snapshot_file.exists():
            missing.append(snapshot_file)
            continue
        if require_front and not front_file.exists():
            missing.append(front_file)
            continue
        dat_files.append(snapshot_file)

    if missing:
        print("Missing files:")
        for path in missing:
            print("  ", path)
        print("Existing complete runs will still be processed.")

    return sorted(dat_files)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "auto"
    return f"{value:g}"


def build_output_mp4(dat_file: str | Path,
                     movie_folder: str | Path,
                     movie_kind: str,
                     zoom_width: float = 1.0,
                     zoom_height: float | None = 0.5,
                     center_threshold: str = "th_2") -> Path:
    """Build the output .mp4 path for one input trajectory and movie kind."""
    dat_file = Path(dat_file)
    movie_folder = Path(movie_folder)
    base = dat_file.stem

    if movie_kind == "full":
        suffix = "_full"
    elif movie_kind == "left_zoom":
        suffix = f"_left_zoom_{center_threshold}_w{zoom_width:g}_h{_format_optional_float(zoom_height)}"
    elif movie_kind == "right_zoom":
        suffix = f"_right_zoom_{center_threshold}_w{zoom_width:g}_h{_format_optional_float(zoom_height)}"
    else:
        raise ValueError("movie_kind must be 'full', 'left_zoom', or 'right_zoom'")

    return movie_folder / f"{base}{suffix}.mp4"


def build_movie_jobs(data_folder: str | Path,
                     param_base: str,
                     movie_folder: str | Path,
                     selected_run_ids: Iterable[int] | None = None,
                     movie_kinds: Sequence[str] | None = None,
                     zoom_width: float = 1.0,
                     zoom_height: float | None = 0.5,
                     center_threshold: str = "th_2") -> list[tuple[Path, Path, str]]:
    """Build a list of (trajectory, output_mp4, movie_kind) jobs."""
    if movie_kinds is None:
        movie_kinds = ["full", "left_zoom", "right_zoom"]

    dat_files = trajectory_files_for_runs(
        data_folder=data_folder,
        param_base=param_base,
        selected_run_ids=selected_run_ids,
        require_front=any(k in ["left_zoom", "right_zoom"] for k in movie_kinds),
    )

    jobs = []
    for dat_file in dat_files:
        for movie_kind in movie_kinds:
            output_mp4 = build_output_mp4(
                dat_file,
                movie_folder,
                movie_kind,
                zoom_width=zoom_width,
                zoom_height=zoom_height,
                center_threshold=center_threshold,
            )
            jobs.append((dat_file, output_mp4, movie_kind))
    return jobs


def run_movie_jobs_parallel(jobs: Sequence[tuple[Path, Path, str]],
                            max_parallel: int = 4,
                            worker_script: str | Path = "front_movie_worker.py",
                            python_exe: str | None = None,
                            mode: str = "preview",
                            skip_existing: bool = False,
                            zoom_width: float = 1.0,
                            zoom_height: float | None = 0.5,
                            center_threshold: str = "th_2",
                            log_folder: str | Path | None = None,
                            poll_seconds: float = 1.0,
                            verbose: bool = True) -> list[dict]:
    """Launch movie jobs in parallel and print start/finish timings."""
    if python_exe is None:
        python_exe = sys.executable
    worker_script = Path(worker_script).resolve()

    if log_folder is None:
        if len(jobs) > 0:
            log_folder = Path(jobs[0][1]).parent / "movie_logs"
        else:
            log_folder = Path("movie_logs")
    log_folder = Path(log_folder)
    log_folder.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Worker script: {worker_script}")
        print(f"Log folder   : {log_folder.resolve()}")
        print(f"Movie jobs   : {len(jobs)}")

    results = []
    running: list[tuple[Path, Path, str, subprocess.Popen, object, float, Path]] = []
    next_job = 0
    global_start = time.perf_counter()

    while next_job < len(jobs) or running:
        while next_job < len(jobs) and len(running) < int(max_parallel):
            dat_file, output_mp4, movie_kind = jobs[next_job]
            output_mp4.parent.mkdir(parents=True, exist_ok=True)
            log_path = log_folder / f"{output_mp4.stem}.log"
            log_file = log_path.open("w")

            cmd = [
                python_exe,
                str(worker_script),
                str(dat_file),
                str(output_mp4),
                "--mode", mode,
                "--movie-kind", movie_kind,
                "--zoom-width", str(zoom_width),
                "--center-threshold", center_threshold,
            ]
            if zoom_height is not None:
                cmd.extend(["--zoom-height", str(zoom_height)])
            if skip_existing:
                cmd.append("--skip-existing")

            start_time = time.perf_counter()
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            running.append((dat_file, output_mp4, movie_kind, process, log_file, start_time, log_path))

            if verbose:
                print(f"Started {movie_kind}: {dat_file} -> {output_mp4}")
            next_job += 1

        still_running = []
        for dat_file, output_mp4, movie_kind, process, log_file, start_time, log_path in running:
            code = process.poll()
            if code is None:
                still_running.append((dat_file, output_mp4, movie_kind, process, log_file, start_time, log_path))
                continue

            log_file.close()
            elapsed = time.perf_counter() - start_time
            status = "finished" if code == 0 else "error"
            results.append({
                "input_dat": str(dat_file),
                "output_mp4": str(output_mp4),
                "movie_kind": movie_kind,
                "status": status,
                "return_code": code,
                "elapsed_seconds": elapsed,
                "log_file": str(log_path),
            })

            if verbose:
                if code == 0:
                    print(f"Finished {movie_kind}: {output_mp4} in {elapsed:.2f} s")
                else:
                    print(f"ERROR {movie_kind}: {dat_file}; return code = {code}; elapsed = {elapsed:.2f} s")

        running = still_running
        if running:
            time.sleep(poll_seconds)

    total_elapsed = time.perf_counter() - global_start
    if verbose:
        print(f"All requested movies finished in {total_elapsed:.2f} s.")

    return results


def make_front_movies(param_base: str | None = None,
                      param_file: str | Path | None = None,
                      data_folder: str | Path | None = None,
                      movie_folder: str | Path | None = None,
                      selected_run_ids: Iterable[int] | None = None,
                      movie_kinds: Sequence[str] | None = None,
                      max_parallel: int = 4,
                      mode: str = "preview",
                      skip_existing: bool = False,
                      zoom_width: float = 1.0,
                      zoom_height: float | None = 0.5,
                      center_threshold: str = "th_2",
                      worker_script: str | Path = "front_movie_worker.py") -> list[dict]:
    """High-level notebook helper: build jobs and run them in parallel."""
    if param_base is None:
        if param_file is None:
            raise ValueError("Provide either param_base or param_file")
        param_base = Path(param_file).stem

    if data_folder is None:
        data_folder = Path("data") / param_base
    if movie_folder is None:
        movie_folder = Path("movies") / param_base

    data_folder = Path(data_folder)
    movie_folder = Path(movie_folder)
    log_folder = movie_folder / "movie_logs"
    movie_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)

    print(f"Data folder : {data_folder.resolve()}")
    print(f"Movie folder: {movie_folder.resolve()}")

    jobs = build_movie_jobs(
        data_folder=data_folder,
        param_base=param_base,
        movie_folder=movie_folder,
        selected_run_ids=selected_run_ids,
        movie_kinds=movie_kinds,
        zoom_width=zoom_width,
        zoom_height=zoom_height,
        center_threshold=center_threshold,
    )

    return run_movie_jobs_parallel(
        jobs,
        max_parallel=max_parallel,
        worker_script=worker_script,
        mode=mode,
        skip_existing=skip_existing,
        zoom_width=zoom_width,
        zoom_height=zoom_height,
        center_threshold=center_threshold,
        log_folder=log_folder,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch front movie workers in parallel.")
    parser.add_argument("--param-file", default=None)
    parser.add_argument("--param-base", default=None)
    parser.add_argument("--data-folder", default=None)
    parser.add_argument("--movie-folder", default=None)
    parser.add_argument("--selected-run-ids", type=int, nargs="*", default=None)
    parser.add_argument("--movie-kinds", nargs="*", default=["full", "left_zoom", "right_zoom"])
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--mode", choices=["preview", "hq"], default="preview")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--zoom-width", type=float, default=1.0)
    parser.add_argument("--zoom-height", type=float, default=0.5)
    parser.add_argument("--center-threshold", choices=["th_1", "th_2", "th_3"], default="th_2")
    parser.add_argument("--worker-script", default="front_movie_worker.py")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    make_front_movies(
        param_base=args.param_base,
        param_file=args.param_file,
        data_folder=args.data_folder,
        movie_folder=args.movie_folder,
        selected_run_ids=args.selected_run_ids,
        movie_kinds=args.movie_kinds,
        max_parallel=args.max_parallel,
        mode=args.mode,
        skip_existing=args.skip_existing,
        zoom_width=args.zoom_width,
        zoom_height=args.zoom_height,
        center_threshold=args.center_threshold,
        worker_script=args.worker_script,
    )


if __name__ == "__main__":
    main()
