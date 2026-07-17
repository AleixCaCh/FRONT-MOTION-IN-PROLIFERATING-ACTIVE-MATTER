"""Build one MP4 movie from one C trajectory file.

Main functions
--------------
downsample_frames: keep every Nth frame, preserving the final frame.
front_column_for_movie_kind: choose the front column used for zoom movies.
build_animation: create the matplotlib animation object.
save_mp4: save one full/left_zoom/right_zoom movie.
main: command-line worker used by front_movies.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from front_analysis import (
    get_associated_front_file,
    nearest_front_index,
    read_front_file,
    read_trajectory,
    theta_to_rgba,
    threshold_label,
)


# Matplotlib needs to know where ffmpeg is to write MP4 files.
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def downsample_frames(frames: list[dict], stride: int) -> list[dict]:
    """Return every stride-th frame and always include the final frame."""
    if stride <= 1:
        return frames
    sampled = frames[::stride]
    if len(frames) > 0 and sampled[-1] is not frames[-1]:
        sampled.append(frames[-1])
    return sampled


def front_column_for_movie_kind(movie_kind: str, center_threshold: str) -> str | None:
    """Return the front column used to center a zoom movie."""
    if center_threshold not in ["th_1", "th_2", "th_3"]:
        raise ValueError("center_threshold must be one of: th_1, th_2, th_3")
    if movie_kind == "left_zoom":
        return f"x_left_{center_threshold}"
    if movie_kind == "right_zoom":
        return f"x_right_{center_threshold}"
    return None


def build_animation(params: dict,
                    frames: list[dict],
                    front_data: dict | None = None,
                    front_metadata: dict | None = None,
                    movie_kind: str = "full",
                    fps: int = 10,
                    arrow_length: float | None = None,
                    particle_size: float = 12,
                    zoom_width: float = 1.0,
                    zoom_height: float | None = None,
                    center_threshold: str = "th_2"):
    """Create a full-domain or front-following matplotlib animation."""
    Lx = float(params.get("Lx", params.get("L", 1.0)))
    Ly = float(params.get("Ly", params.get("L", 1.0)))

    is_zoom = movie_kind in ["left_zoom", "right_zoom"]
    if zoom_height is None:
        zoom_height = Ly
    if arrow_length is None:
        arrow_length = 0.08 * (zoom_height if is_zoom else Ly)

    if is_zoom:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
    else:
        fig, ax = plt.subplots(figsize=(12, 2.8))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    if not is_zoom:
        ax.set_xlim(0.0, Lx)
        ax.set_ylim(0.0, Ly)

    artists = {"scatter": None, "quiver": None, "front_line": None}
    center_column = front_column_for_movie_kind(movie_kind, center_threshold)

    def update(frame_index: int):
        frame = frames[frame_index]
        data = frame["data"]

        x = data[:, 1]
        y = data[:, 2]
        theta = data[:, 3]
        u = arrow_length * np.cos(theta)
        v = arrow_length * np.sin(theta)
        rgba = theta_to_rgba(theta)

        for key in ["scatter", "quiver", "front_line"]:
            if artists[key] is not None:
                artists[key].remove()
                artists[key] = None

        artists["scatter"] = ax.scatter(x, y, s=particle_size, c=rgba)
        artists["quiver"] = ax.quiver(
            x, y, u, v,
            color=rgba,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0025,
            pivot="tail",
        )

        title_extra = "full tube"
        if is_zoom and front_data is not None and center_column in front_data:
            j = nearest_front_index(front_data, frame["time"])
            xc = front_data[center_column][j]
            if np.isfinite(xc):
                xmin = max(0.0, xc - 0.5 * zoom_width)
                xmax = min(Lx, xc + 0.5 * zoom_width)
                ymin = max(0.0, 0.5 * Ly - 0.5 * zoom_height)
                ymax = min(Ly, 0.5 * Ly + 0.5 * zoom_height)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                artists["front_line"] = ax.axvline(xc, linestyle="-", linewidth=1.5)
                side = "left" if movie_kind == "left_zoom" else "right"
                title_extra = f"{side} zoom, {threshold_label(center_threshold, front_metadata)}"
        elif is_zoom:
            ax.set_xlim(0.0, min(Lx, zoom_width))
            ax.set_ylim(max(0.0, 0.5 * Ly - 0.5 * zoom_height),
                        min(Ly, 0.5 * Ly + 0.5 * zoom_height))
            title_extra = f"{movie_kind}, front file missing"

        ax.set_title(
            f"{title_extra}: step = {frame['step']}, t = {frame['time']:.3f}, N = {frame['N']}",
            fontsize=11,
        )

        out = [artists["scatter"], artists["quiver"]]
        if artists["front_line"] is not None:
            out.append(artists["front_line"])
        return tuple(out)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    return fig, anim


def save_mp4(params: dict,
             frames: list[dict],
             output_name: str | Path,
             front_data: dict | None = None,
             front_metadata: dict | None = None,
             movie_kind: str = "full",
             fps: int = 10,
             arrow_length: float | None = None,
             particle_size: float = 12,
             dpi: int = 160,
             zoom_width: float = 1.0,
             zoom_height: float | None = None,
             center_threshold: str = "th_2") -> None:
    """Build and save one MP4 movie."""
    fig, anim = build_animation(
        params=params,
        frames=frames,
        front_data=front_data,
        front_metadata=front_metadata,
        movie_kind=movie_kind,
        fps=fps,
        arrow_length=arrow_length,
        particle_size=particle_size,
        zoom_width=zoom_width,
        zoom_height=zoom_height,
        center_threshold=center_threshold,
    )
    writer = FFMpegWriter(fps=fps, bitrate=8000)
    anim.save(str(output_name), writer=writer, dpi=dpi)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one full-tube or zoomed front movie.")
    parser.add_argument("input_dat")
    parser.add_argument("output_mp4")
    parser.add_argument("--mode", choices=["preview", "hq"], default="preview")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--movie-kind", choices=["full", "left_zoom", "right_zoom"], default="full")
    parser.add_argument("--zoom-width", type=float, default=1.0)
    parser.add_argument("--zoom-height", type=float, default=None)
    parser.add_argument("--center-threshold", choices=["th_1", "th_2", "th_3"], default="th_2")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    input_dat = Path(args.input_dat)
    output_mp4 = Path(args.output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and output_mp4.exists():
        print(f"[SKIP] Movie already exists: {output_mp4}", flush=True)
        return

    print(f"[INFO] Reading trajectory: {input_dat}", flush=True)
    params, frames = read_trajectory(input_dat)
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {input_dat}")

    front_data = None
    front_metadata = None
    if args.movie_kind in ["left_zoom", "right_zoom"]:
        front_file = Path(get_associated_front_file(input_dat))
        if front_file.exists():
            print(f"[INFO] Reading front data: {front_file}", flush=True)
            front_metadata, front_data = read_front_file(front_file)
        else:
            print(f"[WARN] Front file not found; zoom will not follow the front: {front_file}", flush=True)

    if args.mode == "preview":
        fps = 5
        frame_stride = 5
    else:
        fps = 10
        frame_stride = 1

    frames_to_use = downsample_frames(frames, frame_stride)

    print(f"[INFO] Mode: {args.mode}", flush=True)
    print(f"[INFO] Movie kind: {args.movie_kind}", flush=True)
    print(f"[INFO] Frames: original={len(frames)}, used={len(frames_to_use)}", flush=True)
    print(f"[INFO] Output: {output_mp4}", flush=True)

    save_mp4(
        params=params,
        frames=frames_to_use,
        output_name=output_mp4,
        front_data=front_data,
        front_metadata=front_metadata,
        movie_kind=args.movie_kind,
        fps=fps,
        particle_size=12,
        dpi=160,
        zoom_width=args.zoom_width,
        zoom_height=args.zoom_height,
        center_threshold=args.center_threshold,
    )

    print(f"[DONE] Saved movie: {output_mp4}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
