# Front Motion in Proliferating Active Matter

This repository contains the simulation and analysis code developed for the master's thesis **Front Motion in Proliferating Active Matter**.

The project studies a stochastic population of proliferating active Brownian particles with translational and angular diffusion, self-propulsion, birth and death, and neighbour-dependent inhibition of reproduction.

## Repository structure

```text
.
├── bulk_validation/
│   ├── C simulation codes
│   ├── Python helper files
│   ├── Jupyter notebooks
│   ├── parameter files
│   └── bulk_env.txt
│
├── front_motion/
│   ├── C simulation code
│   ├── Python helper files
│   ├── Jupyter notebooks
│   └── front_motion_env.txt
│
├── README.md
└── .gitignore
```

## Bulk validation

The `bulk_validation` folder contains the periodic-box workflow used to compare the model with representative bulk phases from the literature.

It includes:

- the main code with inherited newborn orientation,
- the alternative code with randomized newborn orientation,
- compilation and parallel-run tools,
- analysis of the particle number, density, polar order, and pair correlation,
- screenshot and movie tools.

Install the Python packages with:

```bash
python -m pip install -r bulk_validation/bulk_env.txt
```

## Front motion

The `front_motion` folder contains the workflow used to simulate and analyse population fronts expanding into initially empty regions.

It includes:

- the front simulation code with the glued-bulk implementation,
- compilation and parallel-run tools,
- density-profile and front-position analysis,
- front-velocity measurements,
- non-active and active parameter sweeps,
- assessment of the glued-bulk method,
- screenshot and movie tools.

Install the Python packages with:

```bash
python -m pip install -r front_motion/front_motion_env.txt
```

## Simulation workflow and parameter files

The C programs read text parameter files formed by one or more `[run]` blocks. Each block contains a complete parameter set and is identified by `run_id`. The selected simulation is executed by passing the parameter file and the corresponding identifier to the compiled program:

```bash
./executable_name parameter_file.txt run_id
```

The supplied notebooks prepare the parameter files used in the different sweeps, compile the corresponding C code, launch independent simulations, and analyse the generated output. In the bulk workflow, particles evolve in a square domain with periodic boundary conditions. In the front workflow, the population is first equilibrated in a periodic central strip and is then allowed to propagate through the full rectangular domain. During the propagation stage, the glued-bulk method can remove part of the dense central population while preserving a retained region behind each front.

At each numerical step, particles are moved, the cell list is rebuilt, neighbours are counted, and the birth or death update is applied. Measurements are saved at the intervals specified in the parameter file.

### Parameters shared by the bulk and front simulations

The code names are kept in the table below. In particular, `R_inter` corresponds to the interaction radius $R$ and `Dtheta` to the angular diffusion coefficient $D_\theta$ used in the thesis.

| Parameter | Meaning |
|---|---|
| `run_id` | Integer identifier of the parameter block. It is used to select the simulation and to label the output files. |
| `seed` | Seed of the random-number generator. Different seeds define independent realizations. |
| `p0` | Birth rate of an isolated particle, before the reduction caused by neighbouring particles. |
| `q0` | Constant death rate of each particle. |
| `Ns` | Neighbour number controlling local saturation. The birth rate becomes zero when $N_i\geq N_s$. |
| `R_inter` | Interaction radius used to count the neighbours of each particle. It corresponds to $R$ in the thesis. |
| `rho0` | Initial particle number density. In the front simulations, it applies to the initial central strip. |
| `v0` | Self-propulsion speed. The non-active case is obtained with `v0 = 0`. |
| `Dr` | Translational diffusion coefficient $D_r$. |
| `Dtheta` | Angular diffusion coefficient $D_\theta$. Smaller values produce longer orientational persistence. |
| `dt` | Fixed numerical time step $\Delta t$. |
| `T` | Duration of the simulation. In the front code, this is the propagation time after the warmup stage. |
| `save_per_step` | Number of integration steps between saved particle configurations. |

### Parameters specific to the bulk validation

| Parameter | Meaning |
|---|---|
| `L` | Side length of the square simulation domain. Periodic boundary conditions are applied in both directions. |

The files `params_four_phases.txt` and `params_four_phases_altered.txt` contain the representative parameter sets used for the four bulk regimes. The first file is used with `abm_bulk_validation.c`, where newborn particles inherit the parent orientation. The second is used with `abm_bulk_validation_non_inheritance.c`, where the newborn orientation is selected uniformly at random.

### Parameters specific to the front simulations

| Parameter | Meaning |
|---|---|
| `Lx` | Length of the full rectangular domain in the propagation direction. |
| `Ly` | Transverse size of the domain. Periodic boundary conditions are applied in this direction. |
| `x_init_min`, `x_init_max` | Left and right limits of the initially populated strip. Their difference is the strip width $L_{\mathrm{band}}$. |
| `warmup_T` | Duration of the periodic warmup performed inside the initial strip. |
| `rho_sat` | Saturated density used to define the front thresholds. Setting `rho_sat = -1` makes the code estimate it from the final 50 time units of the warmup. |
| `nbins_x` | Number of spatial bins used to construct the one-dimensional density profile along $x$. |
| `threshold_frac1`, `threshold_frac2`, `threshold_frac3` | Fractions of $\rho_{\mathrm{sat}}$ used to locate the density-threshold fronts. The thesis uses 0.2, 0.5, and 0.8. |
| `front_per_step` | Number of integration steps between consecutive front-position measurements. |
| `rho_profile_every_front` | Density-profile saving interval, expressed in numbers of front measurements. A value of 0 disables this output. |
| `isolation_buffer_factor` | Dimensionless glued-bulk buffer factor $B$. The retained buffer distance behind each bulk anchor is $B R_{\mathrm{inter}}$. |

The front parameter files for the different sweeps are generated in the corresponding Jupyter notebooks. The notebooks then group the output by the varied parameters and calculate the front observables and velocities used in the thesis.

## Additional requirements

A C compiler such as GCC is required to compile the simulation codes.

The notebooks use relative paths and should be run from their corresponding workflow folder.

## Data and generated files

Simulation data, logs, compiled executables, figures, and movies are not included in this repository. They can be generated by running the supplied C codes and Jupyter notebooks.

## Author

Aleix Casals Cheng  
Master's Degree in Physics of Complex Systems  
University of the Balearic Islands  
Academic year 2025-2026
