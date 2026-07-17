"""Shared analysis utilities for the active-matter front-propagation workflow.

Main groups of functions
------------------------
Style helpers: apply_front_plot_style, style_axis, set_max_ticks.
File helpers: get_run_files, discover_run_ids, get_associated_front_file,
    get_associated_density_file, get_associated_warmup_file.
Readers: read_trajectory, read_front_file, read_density_profile_file,
    read_warmup_file.
Front analysis: plot_front_timeseries_lr, fit_front_speed_side,
    velocity_summary_table, plot_front_fit_side, plot_velocity_comparison,
    fit_front_sweep.
Density validation: plot_density_profiles, plot_density_profile_with_fronts,
    plot_density_heatmap_lr, estimate_bulk_density_from_thresholds.
Warmup validation: plot_warmup_population.
Snapshots: plot_snapshot_with_front, make_front_screenshots_from_folder.
Theory diagnostics: fkpp_estimates.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ==========================================================
# PLOT STYLE HELPERS
# ==========================================================

def apply_front_plot_style():
    """Use this style."""
    plt.rcParams.update({
        "figure.dpi"       : 150,
        "font.family"      : "serif",
        "font.size"        : 18,
        "axes.labelsize"   : 25,
        "xtick.labelsize"  : 20,
        "ytick.labelsize"  : 20,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.grid"        : False,
    })


def set_max_ticks(ax, n=4, x=True, y=True):
    """Limit the number of major ticks on an axis."""
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n))


def style_axis(ax, n_ticks=4, x_ticks=True, y_ticks=True, grid=False):
    """Apply the default axis style used in the thesis plots."""
    set_max_ticks(ax, n=n_ticks, x=x_ticks, y=y_ticks)
    ax.grid(grid)


apply_front_plot_style()


# ==========================================================
# ANGLE / COLOR HELPERS
# ==========================================================

def theta_in_2pi_array(theta):
    return np.mod(theta, 2.0 * np.pi)


def theta_to_rgba(theta):
    cmap = plt.get_cmap("hsv")
    values = theta_in_2pi_array(theta) / (2.0 * np.pi)
    return cmap(values)


# ==========================================================
# SMALL HELPERS
# ==========================================================

def _try_number(value):
    try:
        if any(ch in value.lower() for ch in [".", "e"]):
            return float(value)
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return value


def _parse_metadata_pairs(parts):
    metadata = {}
    j = 0
    while j + 1 < len(parts):
        key = parts[j]
        value = parts[j + 1]
        metadata[key] = _try_number(value)
        j += 2
    return metadata


def _ensure_folder_for_file(save_path):
    if save_path is None:
        return
    folder = os.path.dirname(str(save_path))
    if folder != "":
        os.makedirs(folder, exist_ok=True)


def resolve_frame_index(nframes, frame_index):
    if nframes == 0:
        return None
    if frame_index is None:
        return nframes - 1
    if frame_index < 0:
        frame_index = nframes + frame_index
    if frame_index < 0 or frame_index >= nframes:
        return None
    return frame_index


def nearest_index(values, value):
    values = np.asarray(values)
    return int(np.nanargmin(np.abs(values - value)))


def nearest_front_index(front_data, time):
    return nearest_index(front_data["time"], time)


def nearest_density_index(rho_data, time):
    return nearest_index(rho_data["time"], time)


# ==========================================================
# FILE NAME HELPERS
# ==========================================================

def strip_snapshot_prefix(filename):
    """Return the physical run basename, removing the snapshot_ trajectory prefix."""
    base = os.path.basename(str(filename))
    if base.startswith("snapshot_"):
        return base[len("snapshot_"):]
    return base


def get_associated_front_file(dat_file):
    folder = os.path.dirname(str(dat_file))
    base = strip_snapshot_prefix(dat_file)
    return os.path.join(folder, "front_" + base)


def get_associated_density_file(dat_file):
    folder = os.path.dirname(str(dat_file))
    base = strip_snapshot_prefix(dat_file)
    return os.path.join(folder, "rho_" + base)


def get_associated_warmup_file(dat_file):
    """Return the warmup diagnostic file associated with a snapshot file."""
    folder = os.path.dirname(str(dat_file))
    base = strip_snapshot_prefix(dat_file)
    return os.path.join(folder, "warmup_" + base)


def get_run_files(data_folder, param_base, run_id):
    snapshot_file = os.path.join(data_folder, f"snapshot_{param_base}_run_{run_id:03d}.dat")
    front_file = os.path.join(data_folder, f"front_{param_base}_run_{run_id:03d}.dat")
    rho_file = os.path.join(data_folder, f"rho_{param_base}_run_{run_id:03d}.dat")
    return snapshot_file, front_file, rho_file


def discover_run_ids(data_folder, param_base, source="front"):
    """Discover available run_id values from output files.

    By default this scans front files, because the front observable is required
    for most front-propagation analysis. Set source="snapshot" or source="rho"
    only when you intentionally want to scan those file types instead.
    """
    run_ids = []
    if not os.path.isdir(data_folder):
        return run_ids

    valid_sources = {"snapshot", "front", "rho"}
    if source not in valid_sources:
        raise ValueError(f"source must be one of {sorted(valid_sources)}")

    prefix = f"{source}_{param_base}_run_"
    for name in os.listdir(data_folder):
        if not name.startswith(prefix) or not name.endswith(".dat"):
            continue
        middle = name[len(prefix):-4]
        try:
            run_ids.append(int(middle))
        except ValueError:
            pass

    return sorted(run_ids)


# ==========================================================
# READ TRAJECTORY FILE
# ==========================================================

def read_trajectory(filename):
    params = {}
    frames = []

    with open(filename, "r") as f:
        lines = f.readlines()

    params["trajectory_file"] = filename

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "":
            i += 1
            continue

        if line.startswith("#"):
            parts = line[1:].split()
            params.update(_parse_metadata_pairs(parts))
            i += 1
            continue

        if line.startswith("FRAME"):
            parts = line.split()
            step = int(parts[1])
            time = float(parts[2])
            N = int(parts[3])

            data = np.zeros((N, 4))
            for k in range(N):
                vals = lines[i + 1 + k].split()
                data[k, 0] = int(vals[0])
                data[k, 1] = float(vals[1])
                data[k, 2] = float(vals[2])
                data[k, 3] = float(vals[3])

            frames.append({"step": step, "time": time, "N": N, "data": data})
            i += N + 1
        else:
            i += 1

    return params, frames


# ==========================================================
# READ FRONT OBSERVABLE FILE
# ==========================================================

def read_front_file(filename):
    metadata = {}
    column_names = None
    rows = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            if line.startswith("#"):
                parts = line[1:].split()
                if len(parts) > 0 and parts[0] == "step":
                    column_names = parts
                else:
                    metadata.update(_parse_metadata_pairs(parts))
                continue

            rows.append([float(x) for x in line.split()])

    if column_names is None:
        raise ValueError("Could not find a '# step time ...' header in the front file.")

    arr = np.array(rows, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    data = {name: arr[:, j] for j, name in enumerate(column_names)}
    data["column_names"] = column_names
    return metadata, data


# ==========================================================
# READ DENSITY PROFILE FILE
# ==========================================================

def read_density_profile_file(filename):
    metadata = {}
    column_names = None
    bin_centers = None
    rows = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            if line.startswith("#"):
                parts = line[1:].split()
                if len(parts) == 0:
                    continue
                if parts[0] == "bin_centers":
                    bin_centers = np.array([float(x) for x in parts[1:]], dtype=float)
                elif parts[0] == "step":
                    column_names = parts
                else:
                    metadata.update(_parse_metadata_pairs(parts))
                continue

            rows.append([float(x) for x in line.split()])

    if column_names is None:
        raise ValueError("Could not find a '# step time N rho_0 ...' header in the density file.")

    arr = np.array(rows, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if bin_centers is None:
        nbins_x = int(metadata.get("nbins_x", arr.shape[1] - 3))
        Lx = float(metadata.get("Lx", nbins_x))
        dx = float(metadata.get("dx", Lx / nbins_x))
        bin_centers = (np.arange(nbins_x) + 0.5) * dx

    data = {
        "step": arr[:, 0].astype(int),
        "time": arr[:, 1],
        "N": arr[:, 2].astype(int),
        "x": bin_centers,
        "rho": arr[:, 3:],
        "column_names": column_names,
    }
    return metadata, data


# ==========================================================
# READ WARMUP POPULATION FILE
# ==========================================================

def read_warmup_file(filename):
    """Read warmup_N diagnostics written by the C code.

    Expected format:
        # metadata_key metadata_value ...
        # step time N rho_seed
        100 1.00000000 187 748.0

    Returns
    -------
    metadata : dict
        Metadata parsed from comment lines.
    data : dict
        Arrays with columns such as "step", "time", "N", and "rho_seed".
    """
    metadata = {}
    column_names = None
    rows = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            if line.startswith("#"):
                parts = line[1:].split()
                if len(parts) == 0:
                    continue
                if parts[0] == "step":
                    column_names = parts
                else:
                    metadata.update(_parse_metadata_pairs(parts))
                continue

            rows.append([float(x) for x in line.split()])

    if column_names is None:
        raise ValueError("Could not find a '# step time N ...' header in the warmup file.")

    if len(rows) == 0:
        arr = np.empty((0, len(column_names)), dtype=float)
    else:
        arr = np.array(rows, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

    data = {name: arr[:, j] for j, name in enumerate(column_names)}
    if "step" in data:
        data["step"] = data["step"].astype(int)
    if "N" in data:
        data["N"] = data["N"].astype(int)
    data["column_names"] = column_names
    return metadata, data


def plot_warmup_population(warmup_data, use_density=False,
                           rho_sat=None, ax=None,
                           save_path=None, show=True):
    """Plot the warmup population or seed-region density versus warmup time.

    Parameters
    ----------
    warmup_data : dict
        Output of read_warmup_file().
    use_density : bool
        If False, plot N(t). If True, plot rho_seed(t).
    rho_sat : float or None
        Optional reference density. Only used when use_density=True.
    ax : matplotlib axis or None
        Existing axis to plot on. If None, a new figure is created.
    save_path : str or None
        Optional path where the figure is saved.
    show : bool
        If True, display the figure. If False, close it after saving.
    """
    created_ax = ax is None

    if created_ax:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
    else:
        fig = ax.figure

    time = warmup_data["time"]

    if use_density:
        if "rho_seed" not in warmup_data:
            raise ValueError("warmup_data does not contain 'rho_seed'.")
        y = warmup_data["rho_seed"]
        ylabel = r"$\rho_{seed}(t)$"
        label = r"$\rho_{seed}$ during warmup"
    else:
        y = warmup_data["N"]
        ylabel = r"$N(t)$"
        label = r"$N$ during warmup"

    ax.plot(time, y, linewidth=2.0, label=label)

    if use_density and rho_sat is not None and np.isfinite(rho_sat) and rho_sat > 0:
        ax.axhline(rho_sat, linestyle="--", linewidth=1.5,
                   label=r"$\rho_{sat}$ reference")

    ax.set_xlabel("warmup time")
    ax.set_ylabel(ylabel)
    ax.set_title("Warmup quasi-equilibrium diagnostic")
    style_axis(ax, n_ticks=4, grid=False)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        _ensure_folder_for_file(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if created_ax:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig, ax
    
# ==========================================================
# FRONT COLUMN CONVENTIONS
# ==========================================================

METHOD_COLUMNS = {
    "tip": ("x_left_tip", "x_right_tip", "tip"),
    "quantile": ("x_q01", "x_q99", "quantile"),
    "th_1": ("x_left_th_1", "x_right_th_1", r"$\alpha_1\rho_{sat}$"),
    "th_2": ("x_left_th_2", "x_right_th_2", r"$\alpha_2\rho_{sat}$"),
    "th_3": ("x_left_th_3", "x_right_th_3", r"$\alpha_3\rho_{sat}$"),
}

DEFAULT_METHODS = ["tip", "quantile", "th_1", "th_2", "th_3"]
THRESHOLD_METHODS = ["th_1", "th_2", "th_3"]


def threshold_index(method):
    if method in ["th_1", "th_2", "th_3"]:
        return int(method.split("_")[1])
    return None


def threshold_fraction_from_metadata(metadata, method):
    idx = threshold_index(method)
    if idx is None or metadata is None:
        return None
    key = f"threshold_frac{idx}"
    if key not in metadata:
        return None
    try:
        return float(metadata[key])
    except Exception:
        return None


def threshold_label(method, metadata=None):
    frac = threshold_fraction_from_metadata(metadata, method)
    if frac is None:
        return get_front_label(method)
    return rf"${frac:g}\rho_{{sat}}$"



def get_front_column(side, method):
    if method not in METHOD_COLUMNS:
        raise ValueError(f"Unknown method '{method}'. Valid methods: {list(METHOD_COLUMNS)}")
    left_col, right_col, _ = METHOD_COLUMNS[method]
    if side == "left":
        return left_col
    if side == "right":
        return right_col
    raise ValueError("side must be 'left' or 'right'")


def get_front_label(method):
    if method not in METHOD_COLUMNS:
        return method
    return METHOD_COLUMNS[method][2]


def side_sign(side):
    if side == "left":
        return -1.0
    if side == "right":
        return 1.0
    raise ValueError("side must be 'left' or 'right'")


def side_title(side):
    return "left front" if side == "left" else "right front"


def front_side_color(side):
    """Consistent colors for left/right front positions."""
    if side == "left":
        return "tab:blue"
    if side == "right":
        return "tab:red"
    return "black"


def front_method_linestyle(method):
    """Consistent linestyles for the different front definitions."""
    styles = {
        "th_1": "--",
        "th_2": "-",
        "th_3": ":",
        "tip": "-.",
        "quantile": "-.",
    }
    return styles.get(method, "-")


# ==========================================================
# FRONT TIME SERIES AND VELOCITY FITS
# ==========================================================

def _time_mask(time, t_min=None, t_max=None):
    mask = np.isfinite(time)
    if t_min is not None:
        mask &= time >= t_min
    if t_max is not None:
        mask &= time <= t_max
    return mask


def plot_front_timeseries_side(front_data, side="right", methods=None,
                               t_min=None, t_max=None, ax=None):
    if methods is None:
        methods = DEFAULT_METHODS

    time = front_data["time"]
    mask = _time_mask(time, t_min, t_max)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    for method in methods:
        col = get_front_column(side, method)
        if col not in front_data:
            continue
        y = front_data[col]
        m = mask & np.isfinite(y)
        ax.plot(time[m], y[m], label=get_front_label(method))

    ax.set_xlabel("time")
    ax.set_ylabel("front position x")
    ax.set_title(side_title(side))
    style_axis(ax, n_ticks=4, grid=False)
    ax.legend(fontsize=11)
    return fig, ax


def plot_front_timeseries_lr(front_data, methods=None, t_min=None, t_max=None,
                             save_path=None, show=True):
    if methods is None:
        methods = DEFAULT_METHODS

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plot_front_timeseries_side(front_data, side="left", methods=methods,
                               t_min=t_min, t_max=t_max, ax=axes[0])
    plot_front_timeseries_side(front_data, side="right", methods=methods,
                               t_min=t_min, t_max=t_max, ax=axes[1])
    axes[0].set_xlabel("")
    fig.tight_layout()

    if save_path is not None:
        _ensure_folder_for_file(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def fit_front_speed(front_data, column, t_min, t_max):
    time = front_data["time"]
    x = front_data[column]
    mask = _time_mask(time, t_min, t_max) & np.isfinite(x)
    if np.sum(mask) < 2:
        raise ValueError(f"Not enough valid points for {column} in the selected fitting interval.")
    slope, intercept = np.polyfit(time[mask], x[mask], 1)
    return slope, intercept


def fit_front_speed_side(front_data, side, method, t_min, t_max):
    column = get_front_column(side, method)
    slope, intercept = fit_front_speed(front_data, column, t_min, t_max)
    outward_speed = side_sign(side) * slope
    return {
        "side": side,
        "method": method,
        "column": column,
        "slope_dxdt": slope,
        "intercept": intercept,
        "outward_speed": outward_speed,
    }


def velocity_summary_table(front_data, t_min, t_max, methods=None):
    if methods is None:
        methods = DEFAULT_METHODS

    rows = []
    for method in methods:
        row = {"method": method, "label": get_front_label(method)}
        try:
            left = fit_front_speed_side(front_data, "left", method, t_min, t_max)
            row["left_dxdt"] = left["slope_dxdt"]
            row["left_outward_speed"] = left["outward_speed"]
        except Exception:
            row["left_dxdt"] = np.nan
            row["left_outward_speed"] = np.nan

        try:
            right = fit_front_speed_side(front_data, "right", method, t_min, t_max)
            row["right_dxdt"] = right["slope_dxdt"]
            row["right_outward_speed"] = right["outward_speed"]
        except Exception:
            row["right_dxdt"] = np.nan
            row["right_outward_speed"] = np.nan

        left_v = row["left_outward_speed"]
        right_v = row["right_outward_speed"]
        row["mean_outward_speed"] = np.nanmean([left_v, right_v])
        row["left_minus_right"] = left_v - right_v
        if np.isfinite(row["mean_outward_speed"]) and row["mean_outward_speed"] != 0:
            row["relative_asymmetry"] = (left_v - right_v) / row["mean_outward_speed"]
        else:
            row["relative_asymmetry"] = np.nan
        rows.append(row)

    return rows


def plot_front_fit_side(front_data, side, method, t_min, t_max,
                        ax=None, save_path=None, show=True):
    column = get_front_column(side, method)
    slope, intercept = fit_front_speed(front_data, column, t_min, t_max)
    outward_speed = side_sign(side) * slope

    time = front_data["time"]
    x = front_data[column]
    mask = _time_mask(time, t_min, t_max) & np.isfinite(x)

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
    else:
        fig = ax.figure

    side_color = front_side_color(side)

    # Main trajectory
    ax.plot(
        time, x,
        color=side_color,
        linewidth=1.8,
        label="front position"
    )

    # Linear fit over selected window
    side_symbol = "L" if side == "left" else "R"
    ax.plot(
        time[mask],
        slope * time[mask] + intercept,
        "--",
        color="tab:orange",
        linewidth=1.8,
        label=fr"linear fit ($v_{side_symbol}={outward_speed:.4f}$)"
    )

    # Fit window limits
    ax.axvline(t_min, linestyle=":", color="0.45", linewidth=1.0)
    ax.axvline(t_max, linestyle=":", color="0.45", linewidth=1.0)

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"front position $x$")

    style_axis(ax, n_ticks=4, grid=False)
    ax.legend(frameon=False, loc="best")

    if save_path is not None:
        _ensure_folder_for_file(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if created_ax:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return slope, intercept, outward_speed

def plot_velocity_comparison(rows, save_path=None, show=True):
    labels = [r["method"] for r in rows]
    y = np.arange(len(labels))
    left_v = np.array([r["left_outward_speed"] for r in rows], dtype=float)
    right_v = np.array([r["right_outward_speed"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.38
    ax.barh(y - width / 2, left_v, height=width, label="left outward speed")
    ax.barh(y + width / 2, right_v, height=width, label="right outward speed")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("outward speed")
    style_axis(ax, n_ticks=4, y_ticks=False, grid=False)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        _ensure_folder_for_file(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ==========================================================
# FKPP ESTIMATES
# ==========================================================

def fkpp_estimates(params_or_metadata):
    """Return FKPP-like speed estimates from metadata.

    The dilute-edge linear growth rate in this ABM is approximated as
    r = p0 - q0, because neighbor inhibition is negligible at the leading edge.

    Brownian FKPP: v = 2 sqrt(Dr * r)
    Active-effective FKPP diagnostic: D_eff = Dr + v0^2/(2 Dtheta)
    """
    p0 = float(params_or_metadata.get("p0", np.nan))
    q0 = float(params_or_metadata.get("q0", np.nan))
    Dr = float(params_or_metadata.get("Dr", np.nan))
    Dtheta = float(params_or_metadata.get("Dtheta", np.nan))
    v0 = float(params_or_metadata.get("v0", np.nan))

    r = p0 - q0
    result = {
        "p0": p0,
        "q0": q0,
        "r_edge": r,
        "Dr": Dr,
        "Dtheta": Dtheta,
        "v0": v0,
        "v_fkpp_brownian": np.nan,
        "D_eff_active": np.nan,
        "v_fkpp_active_eff": np.nan,
    }

    if np.isfinite(r) and r > 0 and np.isfinite(Dr) and Dr >= 0:
        result["v_fkpp_brownian"] = 2.0 * np.sqrt(Dr * r)

    if (np.isfinite(r) and r > 0 and np.isfinite(Dr) and Dr >= 0 and
            np.isfinite(v0) and np.isfinite(Dtheta) and Dtheta > 0):
        D_eff = Dr + v0 * v0 / (2.0 * Dtheta)
        result["D_eff_active"] = D_eff
        result["v_fkpp_active_eff"] = 2.0 * np.sqrt(D_eff * r)

    return result


# ==========================================================
# SNAPSHOTS WITH FRONT LINES AND OPTIONAL ZOOM
# ==========================================================

def _front_line_specs(side, include_tip=True):
    """Return front-line specifications.

    The linestyle encodes the front definition/threshold, while the color is
    chosen separately from the side: left = blue, right = red.
    """
    specs = [
        ("th_1", get_front_column(side, "th_1"), front_method_linestyle("th_1"),
         r"$\alpha_1\rho_{sat}$"),
        ("th_2", get_front_column(side, "th_2"), front_method_linestyle("th_2"),
         r"$\alpha_2\rho_{sat}$"),
        ("th_3", get_front_column(side, "th_3"), front_method_linestyle("th_3"),
         r"$\alpha_3\rho_{sat}$"),
    ]
    if include_tip:
        specs.append(("tip", get_front_column(side, "tip"), front_method_linestyle("tip"), "tip"))
    return specs


def plot_snapshot_with_front(params, frames, front_data=None, frame_index=None,
                             side="right", zoom=False, center_method="th_2",
                             zoom_width=1.0, zoom_x_range=None,
                             zoom_height=None, zoom_y_center=None,
                             arrow_length=None, particle_size=12,
                             aspect="auto",
                             show_theta_histogram=True, histogram_bins=24,
                             save_path=None, show=True):
    if len(frames) == 0:
        print("No trajectory frames available.")
        return None, None

    idx = resolve_frame_index(len(frames), frame_index)
    if idx is None:
        print(f"Invalid frame_index = {frame_index}")
        return None, None

    frame = frames[idx]
    data = frame["data"]
    Lx = float(params.get("Lx", params.get("L", 1.0)))
    Ly = float(params.get("Ly", params.get("L", 1.0)))

    x = data[:, 1]
    y = data[:, 2]
    theta = data[:, 3]

    if zoom_height is None:
        zoom_height = Ly
    if zoom_y_center is None:
        zoom_y_center = 0.5 * Ly

    if arrow_length is None:
        arrow_length = 0.08 * (zoom_height if zoom else Ly)

    u = arrow_length * np.cos(theta)
    v = arrow_length * np.sin(theta)
    rgba = theta_to_rgba(theta)

    # Better figure dimensions for a long, thin domain
    if zoom:
        fig_width = 8
        fig_height = 4
    else:
        fig_width = 13
        fig_height = 3

    if show_theta_histogram:
        fig, (ax, ax_hist) = plt.subplots(
            1, 2,
            figsize=(fig_width + 2.2, fig_height),
            gridspec_kw={"width_ratios": [4.8, 1.15], "wspace": 0.28},
        )
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax_hist = None

    ax.scatter(x, y, s=particle_size, c=rgba)

    ax.quiver(x, y, u, v, color=rgba, angles="xy",
              scale_units="xy", scale=1.0, width=0.0025, pivot="tail")

    front_center = None

    if front_data is not None:
        j = nearest_front_index(front_data, frame["time"])
        sides_to_plot = ["left", "right"] if side == "both" else [side]

        for s in sides_to_plot:
            color = front_side_color(s)
            for method, column, linestyle, label in _front_line_specs(s):
                if column in front_data:
                    value = front_data[column][j]
                    if np.isfinite(value):
                        ax.axvline(value, linestyle=linestyle, linewidth=1.8,
                                   color=color,
                                   label=f"{s} {label}" if side == "both" else label)

        if zoom and side in ["left", "right"]:
            center_column = get_front_column(side, center_method)
            if center_column in front_data:
                front_center = front_data[center_column][j]

        if len(ax.lines) > 0:
            ax.legend(loc="upper right", fontsize=10)

    # -----------------------------
    # Axis limits
    # -----------------------------
    if zoom:
        # Case 1: user explicitly chooses the x-range
        if zoom_x_range is not None:
            xmin, xmax = zoom_x_range
            xmin = max(0.0, float(xmin))
            xmax = min(Lx, float(xmax))

            if xmin >= xmax:
                print(f"Invalid zoom_x_range = {zoom_x_range}")
                plt.close(fig)
                return None, None

            zoom_label = f", zoom x ∈ [{xmin:.2f}, {xmax:.2f}]"

        # Case 2: automatic zoom around the detected front
        elif side in ["left", "right"] and front_center is not None and np.isfinite(front_center):
            xmin = max(0.0, front_center - 0.5 * zoom_width)
            xmax = min(Lx, front_center + 0.5 * zoom_width)
            zoom_label = f", {side} zoom around {center_method}"

        # Case 3: fallback
        else:
            xmin, xmax = 0.0, Lx
            zoom_label = ", zoom"

        ymin = max(0.0, zoom_y_center - 0.5 * zoom_height)
        ymax = min(Ly, zoom_y_center + 0.5 * zoom_height)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    else:
        ax.set_xlim(0.0, Lx)
        ax.set_ylim(0.0, Ly)
        zoom_label = ""

    # Important: do NOT force equal aspect for this geometry
    ax.set_aspect(aspect, adjustable="box")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"step = {frame['step']}, t = {frame['time']:.3f}, "
        f"N = {frame['N']}{zoom_label}"
    )

    style_axis(ax, n_ticks=4, grid=False)

    if ax_hist is not None:
        theta_mod = theta_in_2pi_array(theta)
        bins = np.linspace(0.0, 2.0 * np.pi, histogram_bins + 1)
        ax_hist.hist(theta_mod, bins=bins, orientation="horizontal")
        ax_hist.set_ylim(0.0, 2.0 * np.pi)
        ax_hist.set_yticks([0.0, np.pi, 2.0 * np.pi])
        ax_hist.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        ax_hist.set_xlabel("count")
        ax_hist.set_ylabel(r"$\theta$")
        ax_hist.set_title("orientation")
        style_axis(ax_hist, n_ticks=3, grid=False)

    if save_path is not None:
        _ensure_folder_for_file(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

def make_front_screenshots_from_folder(data_folder, output_folder=None,
                                       frame_index=None, file_list=None,
                                       side="right", zoom=False,
                                       center_method="th_2",
                                       zoom_width=1.0, zoom_height=None,
                                       show_theta_histogram=True,
                                       show=True):
    if file_list is None:
        dat_files = sorted([
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.endswith(".dat") and f.startswith("snapshot_")
        ])
    else:
        dat_files = [os.path.join(data_folder, f) for f in file_list]

    if len(dat_files) == 0:
        print("No trajectory .dat files found.")
        return

    for dat_file in dat_files:
        print(f"Processing: {dat_file}")
        if not os.path.exists(dat_file):
            print("  Skipped: trajectory file missing")
            continue

        front_file = get_associated_front_file(dat_file)
        if not os.path.exists(front_file):
            print(f"  Skipped: front file missing -> {front_file}")
            continue

        params, frames = read_trajectory(dat_file)
        metadata, front_data = read_front_file(front_file)

        save_path = None
        if output_folder is not None:
            base = os.path.splitext(os.path.basename(dat_file))[0]
            suffix = f"_{side}"
            if zoom:
                suffix += f"_zoom_{center_method}_w{zoom_width:g}"
            save_path = os.path.join(output_folder, base + suffix + ".png")

        plot_snapshot_with_front(params=params, frames=frames, front_data=front_data,
                                 frame_index=frame_index, side=side, zoom=zoom,
                                 center_method=center_method,
                                 zoom_width=zoom_width, zoom_height=zoom_height,
                                 show_theta_histogram=show_theta_histogram,
                                 save_path=save_path, show=show)


# ==========================================================
# DENSITY PROFILE VALIDATION PLOTS
# ==========================================================

def plot_density_profiles(rho_data, rho_sat=np.nan, times=None, normalize=True,
                          show_reference_lines=True,
                          ax=None, save_path=None, show=True):
    x = rho_data["x"]
    rho = rho_data["rho"]
    profile_times = rho_data["time"]

    if times is None:
        if len(profile_times) <= 5:
            indices = np.arange(len(profile_times))
        else:
            indices = np.linspace(0, len(profile_times) - 1, 5).astype(int)
    else:
        indices = [nearest_density_index(rho_data, t) for t in times]

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(9, 4.8))
    else:
        fig = ax.figure

    scale = rho_sat if normalize and np.isfinite(rho_sat) and rho_sat > 0 else 1.0
    ylabel = r"$\rho/\rho_{sat}$" if scale != 1.0 else r"$\rho$"

    for idx in indices:
        ax.plot(x, rho[idx] / scale, linewidth=2.0, label=f"t={profile_times[idx]:.3g}")

    if show_reference_lines and scale != 1.0:
        reference_specs = [
            (1.0, "-",  r"$\rho_{sat}$"),
            (0.8, ":",  r"$\alpha_3\rho_{sat}$"),
            (0.5, "--", r"$\alpha_2\rho_{sat}$"),
            (0.2, "-.", r"$\alpha_1\rho_{sat}$"),
        ]
        for level, linestyle, label in reference_specs:
            ax.axhline(level, linestyle=linestyle, linewidth=1.0,
                       alpha=0.55, label=label)

    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    style_axis(ax, n_ticks=4, grid=False)
    ax.legend(fontsize=11, ncol=2)
    fig.tight_layout()

    if save_path is not None:
        _ensure_folder_for_file(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if created_ax:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig, ax


def plot_density_profile_with_fronts(rho_data, front_data, rho_sat=np.nan, time=None,
                                     side="both", normalize=True,
                                     show_reference_lines=True,
                                     show_front_lines=True,
                                     front_methods=("th_1", "th_2", "th_3"),
                                     x_min=None, x_max=None,
                                     save_path=None, show=True):
    """Plot one density profile and optionally overlay selected front observables.

    Horizontal reference lines are density levels:
        rho/rho_sat = 1, 0.8, 0.5, 0.2.

    Vertical front lines are selected using front_methods.

    Available front_methods:
        "th_1"   -> threshold_frac1
        "th_2"   -> threshold_frac2
        "th_3"   -> threshold_frac3
        "tip"      -> particle tip
        "quantile" -> q01/q99 front estimate

    You can also use:
        front_methods="all"

    Optional x-axis limits:
        x_min, x_max -> restrict the plotted x range.
    """
    if time is None:
        time = rho_data["time"][-1]

    i = nearest_density_index(rho_data, time)
    j = nearest_front_index(front_data, rho_data["time"][i])

    x = rho_data["x"]
    profile = rho_data["rho"][i]

    scale = rho_sat if normalize and np.isfinite(rho_sat) and rho_sat > 0 else 1.0
    y = profile / scale

    if front_methods == "all":
        front_methods = DEFAULT_METHODS
    elif front_methods is None:
        front_methods = []
    else:
        front_methods = list(front_methods)

    fig, ax = plt.subplots(figsize=(12, 4.8))

    ax.plot(
        x, y,
        linewidth=2.0,
        color="black",
        label=fr"$\rho(x,t)$"
    )

    if show_reference_lines and scale != 1.0:
        reference_specs = [
            (1.0, "-",  r"$\rho_{sat}$"),
            (0.8, ":",  r"$\alpha_3\rho_{sat}$"),
            (0.5, "--", r"$\alpha_2\rho_{sat}$"),
            (0.2, "-.", r"$\alpha_1\rho_{sat}$"),
        ]

        for level, linestyle, label in reference_specs:
            ax.axhline(
                level,
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.55,
                color="gray",
                label=label
            )

    if show_front_lines:
        sides_to_plot = ["left", "right"] if side == "both" else [side]

        for s in sides_to_plot:
            color = front_side_color(s)

            for method in front_methods:
                col = get_front_column(s, method)

                if col in front_data and np.isfinite(front_data[col][j]):
                    linestyle = front_method_linestyle(method)
                    linewidth = 1.8 if method == "th_2" else 1.4
                    label = f"{s} {get_front_label(method)}"

                    ax.axvline(
                        front_data[col][j],
                        linewidth=linewidth,
                        linestyle=linestyle,
                        color=color,
                        label=label
                    )

    # Optional x-axis zoom
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho/\rho_{sat}$" if scale != 1.0 else r"$\rho$")
    # ax.set_title("Density profile with measured front positions")

    style_axis(ax, n_ticks=4, grid=False)

    ax.legend(
        fontsize=11,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    if save_path is not None:
        _ensure_folder_for_file(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_density_heatmap_lr(rho_data, front_data=None, rho_sat=np.nan,
                            methods=None, t_min=None, t_max=None,
                            save_path=None, show=True):
    if methods is None:
        methods = THRESHOLD_METHODS

    x = rho_data["x"]
    time = rho_data["time"]
    rho = rho_data["rho"]
    mask = _time_mask(time, t_min, t_max)

    scale = rho_sat if np.isfinite(rho_sat) and rho_sat > 0 else 1.0
    z = rho[mask] / scale
    tt = time[mask]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    if len(tt) >= 2 and len(x) >= 2:
        im = ax.pcolormesh(x, tt, z, shading="auto")
    else:
        im = ax.imshow(
            z,
            aspect="auto",
            origin="lower",
            extent=[x.min(), x.max(), tt.min(), tt.max()]
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\rho/\rho_{sat}$" if scale != 1.0 else r"$\rho$")
    cbar.ax.tick_params(labelsize=13)

    if front_data is not None:
        ft = front_data["time"]
        fm = _time_mask(ft, t_min, t_max)

        for side in ["left", "right"]:
            color = front_side_color(side)
            for method in methods:
                col = get_front_column(side, method)
                if col in front_data:
                    label = f"{side} {get_front_label(method)}"
                    ax.plot(
                        front_data[col][fm], ft[fm],
                        linewidth=1.5,
                        linestyle=front_method_linestyle(method),
                        color=color,
                        label=label
                    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"time $t$")
    style_axis(ax, n_ticks=4, grid=False)

    if front_data is not None:
        ax.legend(
            fontsize=11,
            ncol=1,
            loc="upper right",
            frameon=False
        )

    fig.tight_layout()

    if save_path is not None:
        _ensure_folder_for_file(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def estimate_bulk_density_from_thresholds(rho_data, front_data, method="th_3",
                                          margin_bins=2):
    """Estimate saturated bulk density from the invaded interior.

    For each saved density profile, this chooses the region between the measured
    left and right fronts of the chosen method, then removes margin_bins from
    each edge. The remaining bins are treated as the bulk behind the fronts.
    """
    x = rho_data["x"]
    rho = rho_data["rho"]
    times = rho_data["time"]
    left_col = get_front_column("left", method)
    right_col = get_front_column("right", method)

    rows = []
    for i, t in enumerate(times):
        j = nearest_front_index(front_data, t)
        xl = front_data[left_col][j]
        xr = front_data[right_col][j]
        if not (np.isfinite(xl) and np.isfinite(xr) and xl < xr):
            rows.append((t, np.nan, np.nan, 0))
            continue

        mask = (x > xl) & (x < xr)
        idx = np.where(mask)[0]
        if len(idx) > 2 * margin_bins:
            idx = idx[margin_bins:-margin_bins]
        if len(idx) == 0:
            rows.append((t, np.nan, np.nan, 0))
            continue

        vals = rho[i, idx]
        rows.append((t, float(np.nanmean(vals)), float(np.nanstd(vals)), int(len(idx))))

    arr = np.array(rows, dtype=float)
    return {
        "time": arr[:, 0],
        "rho_bulk_mean": arr[:, 1],
        "rho_bulk_std": arr[:, 2],
        "n_bins": arr[:, 3].astype(int),
    }


def plot_bulk_density_validation(bulk, rho_sat=np.nan, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(bulk["time"], bulk["rho_bulk_mean"], linewidth=2.0,
            label="bulk density estimate")
    if np.isfinite(rho_sat) and rho_sat > 0:
        ax.axhline(rho_sat, linestyle="--", linewidth=1.5,
                   label=r"$\rho_{sat}$ used")
    ax.set_xlabel("time")
    ax.set_ylabel(r"bulk $\rho$")
    style_axis(ax, n_ticks=4, grid=False)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        _ensure_folder_for_file(save_path)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax




# ==========================================================
# SWEEP FITTING HELPERS
# ==========================================================

def choose_fit_window(front_data, fit_t_min=None, fit_t_max=None,
                      fit_start_fraction=0.7, fit_end_fraction=0.95):
    """Choose a linear-fit time window from explicit times or time fractions."""
    time = np.asarray(front_data["time"], dtype=float)
    time = time[np.isfinite(time)]
    if len(time) < 2:
        raise ValueError("Not enough valid time points to choose a fit window.")

    t_min = fit_t_min
    t_max = fit_t_max
    if t_min is None:
        t_min = time[0] + fit_start_fraction * (time[-1] - time[0])
    if t_max is None:
        t_max = time[0] + fit_end_fraction * (time[-1] - time[0])
    return float(t_min), float(t_max)


def fit_front_speeds_for_front_data(front_data, methods=None, fit_t_min=None,
                                    fit_t_max=None, fit_start_fraction=0.70,
                                    fit_end_fraction=0.95):
    """Fit left/right outward speeds for one already-read front_data object."""
    if methods is None:
        methods = THRESHOLD_METHODS

    t_min, t_max = choose_fit_window(
        front_data,
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
        fit_start_fraction=fit_start_fraction,
        fit_end_fraction=fit_end_fraction,
    )

    row = {"t_min": t_min, "t_max": t_max}
    for method in methods:
        for side in ["left", "right"]:
            try:
                fit = fit_front_speed_side(front_data, side, method, t_min, t_max)
                row[f"{side}_{method}_speed"] = fit["outward_speed"]
                row[f"{side}_{method}_dxdt"] = fit["slope_dxdt"]
            except Exception:
                row[f"{side}_{method}_speed"] = np.nan
                row[f"{side}_{method}_dxdt"] = np.nan

        lv = row[f"left_{method}_speed"]
        rv = row[f"right_{method}_speed"]
        row[f"mean_{method}_speed"] = np.nanmean([lv, rv])
        row[f"asym_{method}"] = lv - rv

    return row


def fit_front_sweep(run_table, data_folder, param_base, methods=None,
                    fit_t_min=None, fit_t_max=None,
                    fit_start_fraction=0.7, fit_end_fraction=0.95,
                    return_dataframe=True):
    """Fit front speeds for every run listed in a run table.

    run_table may be a pandas DataFrame or a list of dictionaries. The output
    preserves the run metadata columns and adds fitted speed columns.
    """
    if methods is None:
        methods = THRESHOLD_METHODS

    rows = []
    missing = []

    if hasattr(run_table, "iterrows"):
        iterator = (row.to_dict() for _, row in run_table.iterrows())
    else:
        iterator = (dict(row) for row in run_table)

    for run in iterator:
        rid = int(run["run_id"])
        _, front_file, _ = get_run_files(data_folder, param_base, rid)
        if not os.path.exists(front_file):
            missing.append(rid)
            continue

        _, front_data = read_front_file(front_file)
        fit_row = dict(run)
        fit_row.update(fit_front_speeds_for_front_data(
            front_data,
            methods=methods,
            fit_t_min=fit_t_min,
            fit_t_max=fit_t_max,
            fit_start_fraction=fit_start_fraction,
            fit_end_fraction=fit_end_fraction,
        ))
        rows.append(fit_row)

    if return_dataframe:
        try:
            import pandas as pd
            return pd.DataFrame(rows), missing
        except Exception:
            pass

    return rows, missing

def side_speed_long_table(fit_df, method, keep_columns=None):
    """
    This function only converts the left and right speeds into a long-table
    format. For statistical summaries, the two sides should first be
    averaged within each simulation run.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("side_speed_long_table requires pandas.") from exc

    if keep_columns is None:
        keep_columns = []

    keep_columns = [c for c in keep_columns if c in fit_df.columns]

    rows = []
    for side in ["left", "right"]:
        col = f"{side}_{method}_speed"
        if col not in fit_df.columns:
            continue

        tmp = fit_df[keep_columns].copy()
        tmp["method"] = method
        tmp["side"] = side
        tmp["speed"] = fit_df[col].astype(float)
        rows.append(tmp)

    if len(rows) == 0:
        return pd.DataFrame(columns=keep_columns + ["method", "side", "speed"])

    out = pd.concat(rows, ignore_index=True)
    out = out[np.isfinite(out["speed"])].copy()
    return out


def pooled_speed_summary(fit_df, group_col, method, metadata_cols=None):
    """Summarize front speeds using each simulation run as one measurement.
        1. For each run, average the left and right outward speeds:

               run_speed = (left_speed + right_speed) / 2

        2. For each value of `group_col`, calculate the mean, standard
           deviation, and SEM across the independent run-level speeds.

    The left and right fronts from the same simulation are therefore not
    treated as independent measurements.

    Runs for which either the left or right speed is missing or non-finite
    are excluded from the summary.

    Parameters
    ----------
    fit_df : pandas.DataFrame
        Table containing the fitted front speeds.

    group_col : str
        Parameter used to group simulations, such as "Dr", "r_edge",
        "v0", or "Dtheta".

    method : str
        Front-position method passed to `side_speed_long_table`.

    metadata_cols : list of str, optional
        Additional parameter columns to preserve in the output.

    Returns
    -------
    pandas.DataFrame
        One row per value of `group_col`.

        The main statistical columns are:

        - n_runs: number of independent realizations
        - mean_speed: mean of the run-level front speeds
        - speed_std: sample standard deviation across runs
        - speed_sem: standard error across runs

        Left-right quantities are retained as diagnostics.
    """
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise ImportError(
            "pooled_speed_summary requires NumPy and pandas."
        ) from exc

    if metadata_cols is None:
        metadata_cols = []

    required_fit_columns = {"run_id", group_col}
    missing_fit_columns = required_fit_columns.difference(fit_df.columns)

    if missing_fit_columns:
        missing_text = ", ".join(sorted(missing_fit_columns))
        raise ValueError(
            f"fit_df is missing required columns: {missing_text}"
        )

    keep_columns = []
    for column in ["run_id", group_col] + list(metadata_cols):
        if column in fit_df.columns and column not in keep_columns:
            keep_columns.append(column)

    speed_long = side_speed_long_table(
        fit_df,
        method=method,
        keep_columns=keep_columns,
    )

    if len(speed_long) == 0:
        return pd.DataFrame()

    required_long_columns = {"run_id", group_col, "side", "speed"}
    missing_long_columns = required_long_columns.difference(
        speed_long.columns
    )

    if missing_long_columns:
        missing_text = ", ".join(sorted(missing_long_columns))
        raise ValueError(
            "side_speed_long_table did not return the required columns: "
            f"{missing_text}"
        )

    run_level = (
        speed_long
        .pivot_table(
            index=["run_id", group_col],
            columns="side",
            values="speed",
            aggfunc="mean",
        )
        .reset_index()
    )

    run_level.columns.name = None

    if "left" not in run_level.columns or "right" not in run_level.columns:
        return pd.DataFrame()

    valid_run = (
        np.isfinite(run_level["left"])
        & np.isfinite(run_level["right"])
    )

    run_level = run_level.loc[valid_run].copy()

    if len(run_level) == 0:
        return pd.DataFrame()

    run_level["run_speed"] = (
        0.5 * (run_level["left"] + run_level["right"])
    )

    run_level["run_asymmetry"] = (
        run_level["left"] - run_level["right"]
    )

    # ----------------------------------------------------------
    # Statistics across independent simulations
    # ----------------------------------------------------------
    summary = (
        run_level
        .groupby(group_col)
        .agg(
            n_runs=("run_speed", "count"),
            mean_speed=("run_speed", "mean"),
            speed_std=("run_speed", "std"),
            left_mean=("left", "mean"),
            right_mean=("right", "mean"),
            left_std=("left", "std"),
            right_std=("right", "std"),
            asymmetry=("run_asymmetry", "mean"),
            asymmetry_std=("run_asymmetry", "std"),
        )
        .reset_index()
    )

    summary["speed_sem"] = (
        summary["speed_std"] / np.sqrt(summary["n_runs"])
    )

    summary["asymmetry_sem"] = (
        summary["asymmetry_std"] / np.sqrt(summary["n_runs"])
    )

    summary["n_speeds"] = 2 * summary["n_runs"]

    summary["relative_asymmetry"] = np.nan

    valid_mean = (
        np.isfinite(summary["mean_speed"])
        & (summary["mean_speed"] != 0)
    )

    summary.loc[valid_mean, "relative_asymmetry"] = (
        summary.loc[valid_mean, "asymmetry"]
        / summary.loc[valid_mean, "mean_speed"]
    )

    meta_cols = [
        column
        for column in metadata_cols
        if (
            column in fit_df.columns
            and column not in {"run_id", group_col}
        )
    ]

    if meta_cols:
        # Restrict metadata to runs that were actually included.
        included_runs = run_level[["run_id", group_col]].drop_duplicates()

        meta_source = fit_df.merge(
            included_runs,
            on=["run_id", group_col],
            how="inner",
        )

        meta = (
            meta_source
            .groupby(group_col)
            .agg({column: "first" for column in meta_cols})
            .reset_index()
        )

        summary = summary.merge(
            meta,
            on=group_col,
            how="left",
        )

    summary["method"] = method

    first_cols = [
        group_col,
        "method",
        "n_runs",
        "n_speeds",
        "mean_speed",
        "speed_std",
        "speed_sem",
        "left_mean",
        "right_mean",
        "asymmetry",
        "relative_asymmetry",
        "left_std",
        "right_std",
        "asymmetry_std",
        "asymmetry_sem",
    ]

    other_cols = [
        column
        for column in summary.columns
        if column not in first_cols
    ]

    summary = summary[
        [column for column in first_cols if column in summary.columns]
        + other_cols
    ]

    return summary

def pooled_speed_summary_by_method(fit_df, group_col, methods, metadata_cols=None):
    """Compute run-level front-speed summaries for several front methods."""
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pooled_speed_summary_by_method requires pandas.") from exc

    rows = []
    for method in methods:
        tmp = pooled_speed_summary(
            fit_df,
            group_col=group_col,
            method=method,
            metadata_cols=metadata_cols,
        )
        if len(tmp) > 0:
            rows.append(tmp)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)



if __name__ == "__main__":
    dat_file = "data/params_front_test/snapshot_params_front_test_run_000.dat"
    front_file = get_associated_front_file(dat_file)
    rho_file = get_associated_density_file(dat_file)

    if os.path.exists(dat_file) and os.path.exists(front_file):
        params, frames = read_trajectory(dat_file)
        metadata, front_data = read_front_file(front_file)
        plot_snapshot_with_front(params, frames, front_data, frame_index=-1, side="both")
        plot_front_timeseries_lr(front_data)

    if os.path.exists(rho_file) and os.path.exists(front_file):
        rho_meta, rho_data = read_density_profile_file(rho_file)
        _, front_data = read_front_file(front_file)
        rho_sat = float(rho_meta.get("rho_sat_used", np.nan))
        plot_density_heatmap_lr(rho_data, front_data, rho_sat=rho_sat)
