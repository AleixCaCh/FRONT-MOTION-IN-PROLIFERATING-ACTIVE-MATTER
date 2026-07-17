from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def set_plot_style(font_size=11):
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def set_few_ticks(ax, number=4):
    ax.xaxis.set_major_locator(MaxNLocator(number))
    ax.yaxis.set_major_locator(MaxNLocator(number))


def get_run_files(data_folder, param_base, run_id):
    data_folder = Path(data_folder)
    name = f"{param_base}_run_{run_id:03d}.dat"

    return {
        "trajectory": data_folder / name,
        "gr": data_folder / f"g_{name}",
        "S": data_folder / f"S_{name}",
    }


def read_header(line, params):
    words = line[1:].split()

    if not words or words[0] == "source_param_file":
        return

    for i in range(0, len(words) - 1, 2):
        key = words[i]
        value = words[i + 1]

        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value


def read_trajectory(filename):
    params = {}
    frames = []

    with open(filename, "r") as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if line.startswith("#"):
            read_header(line, params)
            i += 1
            continue

        if line.startswith("FRAME"):
            words = line.split()
            step = int(words[1])
            time = float(words[2])
            N = int(words[3])

            data = np.zeros((N, 4))

            for j in range(N):
                row = lines[i + 1 + j].split()
                data[j] = [
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                ]

            frames.append({
                "step": step,
                "time": time,
                "N": N,
                "data": data,
            })

            i += N + 1
        else:
            i += 1

    return params, frames


def read_gr(filename):
    frames = []

    with open(filename, "r") as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line.startswith("FRAME"):
            i += 1
            continue

        words = line.split()
        step = int(words[1])
        time = float(words[2])
        N = int(words[3])

        r = []
        g = []
        i += 1

        while i < len(lines) and not lines[i].startswith("FRAME"):
            row = lines[i].strip()

            if row and not row.startswith("#"):
                values = row.split()
                r.append(float(values[0]))
                g.append(float(values[1]))

            i += 1

        frames.append({
            "step": step,
            "time": time,
            "N": N,
            "r": np.array(r),
            "g": np.array(g),
        })

    return frames


def read_S(filename):
    step = []
    time = []
    N = []
    S = []

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            values = line.split()
            step.append(int(values[0]))
            time.append(float(values[1]))
            N.append(int(values[2]))
            S.append(float(values[3]))

    return {
        "step": np.array(step),
        "time": np.array(time),
        "N": np.array(N),
        "S": np.array(S),
    }


def load_run(data_folder, param_base, run_id):
    files = get_run_files(data_folder, param_base, run_id)

    for filename in files.values():
        if not filename.exists():
            raise FileNotFoundError(filename)

    params, frames = read_trajectory(files["trajectory"])
    gr_frames = read_gr(files["gr"])
    S_data = read_S(files["S"])

    return {
        "params": params,
        "frames": frames,
        "gr_frames": gr_frames,
        "S_data": S_data,
        "files": files,
    }


def average_gr(gr_frames, t_min=1000):
    selected = []

    for frame in gr_frames:
        if frame["time"] >= t_min:
            selected.append(frame["g"])

    if len(selected) == 0:
        raise ValueError(f"No g(r) frames found for t >= {t_min}")

    r = gr_frames[-1]["r"]
    g_mean = np.mean(np.array(selected), axis=0)

    return r, g_mean


def second_peak(r, g):
    below_one = False
    peak_index = None
    peak_value = 1.0

    for i, value in enumerate(g):
        if not below_one and value < 1.0:
            below_one = True
        elif below_one and value >= peak_value:
            peak_value = value
            peak_index = i

    if peak_index is None:
        return None, None

    return r[peak_index], g[peak_index]


def theta_colors(theta):
    values = np.mod(theta, 2.0 * np.pi) / (2.0 * np.pi)
    return plt.get_cmap("hsv")(values)


def draw_snapshot(ax, frame, L, particle_size=5, arrow_stride=1):
    data = frame["data"]
    x = data[:, 1]
    y = data[:, 2]
    theta = data[:, 3]
    colors = theta_colors(theta)

    arrow_length = 0.035 * L
    indices = np.arange(0, len(x), arrow_stride)

    ax.scatter(x, y, s=particle_size, c=colors, linewidths=0)

    ax.quiver(
        x[indices],
        y[indices],
        arrow_length * np.cos(theta[indices]),
        arrow_length * np.sin(theta[indices]),
        color=colors[indices],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0028,
        pivot="tail",
    )

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([0, 0.5 * L, L])
    ax.set_yticks([0, 0.5 * L, L])


def plot_run_summary(
    data_folder,
    param_base,
    run_id,
    title,
    t_min=1000,
    save_path=None,
):
    run = load_run(data_folder, param_base, run_id)

    frame = run["frames"][-1]
    S_data = run["S_data"]
    L = float(run["params"]["L"])

    r, g_mean = average_gr(run["gr_frames"], t_min)
    peak_r, peak_g = second_peak(r, g_mean)

    fig = plt.figure(figsize=(9.5, 4.4))
    grid = fig.add_gridspec(
        2, 4,
        width_ratios=[1.6, 0.06, 0.15, 1.0],
        hspace=0.42,
        wspace=0.22,
    )

    ax_snapshot = fig.add_subplot(grid[:, 0])
    cax = fig.add_subplot(grid[:, 1])
    ax_gr = fig.add_subplot(grid[0, 3])
    ax_S = fig.add_subplot(grid[1, 3])

    draw_snapshot(ax_snapshot, frame, L)
    ax_snapshot.set_title(title)

    colorbar_map = plt.cm.ScalarMappable(cmap="hsv")
    colorbar_map.set_clim(0, 1)
    colorbar = fig.colorbar(colorbar_map, cax=cax)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    colorbar.set_ticklabels([
        r"$0$",
        r"$\pi/2$",
        r"$\pi$",
        r"$3\pi/2$",
        r"$2\pi$",
    ])
    colorbar.ax.set_title(r"$\theta$")

    ax_gr.plot(r, g_mean)
    ax_gr.axhline(1, linestyle=":", label=r"$g=1$")

    if peak_r is not None:
        ax_gr.plot(peak_r, peak_g, "o", markersize=4)
        ax_gr.axhline(
            peak_g,
            linestyle="--",
            label=f"2nd peak = {peak_g:.3f}",
        )

    ax_gr.set_xlabel(r"$r$")
    ax_gr.set_ylabel(r"$\langle g(r)\rangle_t$")
    ax_gr.legend(frameon=False)
    set_few_ticks(ax_gr, 3)

    ax_S.plot(S_data["time"], S_data["S"])
    ax_S.set_xlabel(r"$t$")
    ax_S.set_ylabel(r"$S$")
    ax_S.set_ylim(0, 1)
    set_few_ticks(ax_S, 3)
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_population_density_order(S_data, L, save_path=None):
    time = S_data["time"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    axes[0].plot(time, S_data["N"])
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$N(t)$")

    axes[1].plot(time, S_data["N"] / L**2)
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$\rho(t)$")

    axes[2].plot(time, S_data["S"])
    axes[2].set_xlabel(r"$t$")
    axes[2].set_ylabel(r"$S(t)$")
    axes[2].set_ylim(0, 1)

    for ax in axes:
        set_few_ticks(ax)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()


def add_phase_row(fig, outer_cell, case, panel_label, t_min=1000):
    run = load_run(
        case["data_folder"],
        case["param_base"],
        case["run_id"],
    )

    frame = run["frames"][-1]
    S_data = run["S_data"]
    L = float(run["params"]["L"])

    r, g_mean = average_gr(run["gr_frames"], t_min)
    peak_r, peak_g = second_peak(r, g_mean)

    grid = outer_cell.subgridspec(
        2, 4,
        width_ratios=[1.65, 0.055, 0.12, 1.0],
        wspace=0.22,
        hspace=0.42,
    )

    ax_snapshot = fig.add_subplot(grid[:, 0])
    cax = fig.add_subplot(grid[:, 1])
    ax_gr = fig.add_subplot(grid[0, 3])
    ax_S = fig.add_subplot(grid[1, 3])

    draw_snapshot(ax_snapshot, frame, L)
    ax_snapshot.set_title(f"({panel_label}) {case['name']}")

    colorbar_map = plt.cm.ScalarMappable(cmap="hsv")
    colorbar_map.set_clim(0, 1)
    colorbar = fig.colorbar(colorbar_map, cax=cax)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    colorbar.set_ticklabels([
        r"$0$",
        r"$\pi/2$",
        r"$\pi$",
        r"$3\pi/2$",
        r"$2\pi$",
    ])
    colorbar.ax.set_title(r"$\theta$")

    ax_gr.plot(r, g_mean)
    ax_gr.axhline(1, linestyle=":", label=r"$g=1$")

    if peak_r is not None:
        ax_gr.plot(peak_r, peak_g, "o", markersize=4)
        ax_gr.axhline(
            peak_g,
            linestyle="--",
            label=f"2nd peak = {peak_g:.3f}",
        )

    ax_gr.set_xlabel(r"$r$")
    ax_gr.set_ylabel(r"$\langle g(r)\rangle_t$")
    ax_gr.legend(frameon=False)
    set_few_ticks(ax_gr, 3)

    ax_S.plot(S_data["time"], S_data["S"])
    ax_S.set_xlabel(r"$t$")
    ax_S.set_ylabel(r"$S$")
    ax_S.set_ylim(0, 1)
    set_few_ticks(ax_S, 3)


def make_two_phase_figure(cases, output_stem, t_min=1000):
    if len(cases) != 2:
        raise ValueError("Give exactly two cases")

    fig = plt.figure(figsize=(11.5, 10.2))
    outer = fig.add_gridspec(2, 1, hspace=0.30)

    add_phase_row(fig, outer[0], cases[0], "a", t_min)
    add_phase_row(fig, outer[1], cases[1], "b", t_min)

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")

    plt.show()
