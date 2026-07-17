from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

from bulk_analysis import read_trajectory, theta_colors

mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def save_movie(input_file, output_file, mode="preview", particle_size=15):
    params, frames = read_trajectory(input_file)

    if mode == "preview":
        fps = 5
        frames = frames[::5]
    elif mode == "hq":
        fps = 10
    else:
        raise ValueError("mode must be 'preview' or 'hq'")

    L = float(params["L"])
    arrow_length = 0.03 * L

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    artists = {"scatter": None, "quiver": None}

    def update(frame_number):
        frame = frames[frame_number]
        data = frame["data"]

        x = data[:, 1]
        y = data[:, 2]
        theta = data[:, 3]
        colors = theta_colors(theta)

        if artists["scatter"] is not None:
            artists["scatter"].remove()
            artists["quiver"].remove()

        artists["scatter"] = ax.scatter(
            x,
            y,
            s=particle_size,
            c=colors,
            linewidths=0,
        )

        artists["quiver"] = ax.quiver(
            x,
            y,
            arrow_length * np.cos(theta),
            arrow_length * np.sin(theta),
            color=colors,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            pivot="tail",
        )

        ax.set_title(
            f"step = {frame['step']}, "
            f"t = {frame['time']:.2f}, "
            f"N = {frame['N']}"
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer = FFMpegWriter(fps=fps, bitrate=8000)
    animation.save(output_file, writer=writer, dpi=200)
    plt.close(fig)


def movie_worker(script_file, input_file, output_file, mode):
    command = [
        sys.executable,
        str(script_file),
        str(input_file),
        str(output_file),
        mode,
    ]

    result = subprocess.run(command)
    return input_file, output_file, result.returncode


def make_movies(
    data_folder,
    param_base,
    run_ids,
    movie_folder,
    mode="preview",
    max_parallel=4,
    skip_existing=True,
):
    data_folder = Path(data_folder)
    movie_folder = Path(movie_folder)
    movie_folder.mkdir(parents=True, exist_ok=True)

    script_file = Path(__file__).resolve()
    jobs = []

    for run_id in run_ids:
        name = f"{param_base}_run_{run_id:03d}"
        input_file = data_folder / f"{name}.dat"
        output_file = movie_folder / f"{name}.mp4"

        if not input_file.exists():
            print(f"Missing: {input_file}")
            continue

        if skip_existing and output_file.exists():
            print(f"Skipped: {output_file}")
            continue

        jobs.append((input_file, output_file))

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = []

        for input_file, output_file in jobs:
            print(f"Started: {input_file.name}")
            future = pool.submit(
                movie_worker,
                script_file,
                input_file,
                output_file,
                mode,
            )
            futures.append(future)

        results = [future.result() for future in futures]

    for input_file, output_file, return_code in results:
        if return_code == 0:
            print(f"Finished: {output_file}")
        else:
            print(f"Error while creating: {output_file}")

    return results


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python bulk_movies.py input.dat output.mp4 preview"
        )

    save_movie(
        input_file=sys.argv[1],
        output_file=sys.argv[2],
        mode=sys.argv[3],
    )
