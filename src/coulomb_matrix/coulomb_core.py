"""Shared Coulomb matrix computation core for `ijij` and `ijji` modes.

This module provides `load_config` and `CoulombCalculatorBase` which
encapsulates the common MPI/grid/IO setup. Mode-specific behavior is
implemented as two helper methods inside the class.
"""
import glob
import re
from pathlib import Path

import gpaw.mpi as mpi
import numpy as np
import tomllib
from gpaw.poisson import PoissonSolver

from .grid_utils import PoissonGrid
from .mpi_utils import compute_mpi_distribution
from .xsf import Xsf2Np


def load_config(config_path):
    config_file = None
    if config_path:
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
    else:
        default_path = Path.cwd() / "coulomb_config.toml"
        if default_path.exists():
            config_file = default_path.resolve()

    config_dir = config_file.parent if config_file else Path.cwd()

    raw_config = {}
    if config_file is not None:
        with open(config_file, "rb") as f:
            raw_config = tomllib.load(f)

    config = raw_config

    # Lightweight validation / normalization
    interaction = config.get("interaction", {})
    mpi_cfg = config.get("mpi", {})
    paths = config.get("paths", {})
    io_cfg = config.get("io", {})
    output = config.get("output", {})

    for key in ("Rx", "Ry", "Rz"):
        if key in interaction:
            interaction[key] = int(interaction[key])
            if interaction[key] < 0:
                raise ValueError(f"Config field 'interaction.{key}' must be >= 0.")
        else:
            interaction[key] = 0

    # remove resolution handling; grids will be derived directly from XSF data

    mpi_cfg["number_poisson_pools"] = int(mpi_cfg.get("number_poisson_pools", 1))
    if mpi_cfg["number_poisson_pools"] < 1:
        raise ValueError("Config field 'mpi.number_poisson_pools' must be >= 1.")

    output["use_unique_filenames"] = bool(output.get("use_unique_filenames", False))

    # Ensure paths strings exist (may be empty if user relies on defaults)
    for section in (paths, output):
        for k, v in list(section.items()):
            if isinstance(v, str):
                # section[k] = Path(v).expanduser().resolve() if v else config_dir
                section[k] = Path(v) if v else config_dir
                if not section[k].is_absolute():
                    # Join the config folder path with the relative path, then normalize it
                    absolute_input_path = (config_dir / section[k]).resolve()
                    section[k] = absolute_input_path

    config.update(
        {
            "interaction": interaction,
            "mpi": mpi_cfg,
            "paths": paths,
            "io": io_cfg,
            "output": output,
        }
    )

    return config, config_file


class CoulombCalculatorBase:
    def __init__(
        self,
        config_path=None,
        mode="ijij",
    ):
        if mode not in ("ijij", "ijji"):
            raise ValueError("mode must be 'ijij' or 'ijji'")
        self.mode = mode
        self.config, self.config_file = load_config(config_path)
        # Shared initializations used by subclasses
        self.comm = mpi.world
        self.rank = self.comm.rank
        self.mpi_size = self.comm.size

        config = self.config
        self.paths_cfg = config.get("paths", {})
        self.interaction_cfg = config.get("interaction", {})
        self.mpi_cfg = config.get("mpi", {})
        self.io_cfg = config.get("io", {})

        # Derived file paths and patterns
        default_dir = Path.cwd()
        xsf_path = self.paths_cfg.get("xsf_dir", default_dir)
        npy_path = self.paths_cfg.get("npy_dir", default_dir)
        pattern_xsf = self.io_cfg.get("xsf_glob", "wannier90*.xsf")
        pattern_npy = self.io_cfg.get("npy_glob", "wannier90*.npy")

        self.npy_files = glob.glob(str(npy_path) + "/" + str(pattern_npy))
        self.npy_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        self.xsf_files = glob.glob(str(xsf_path) + "/" + str(pattern_xsf))
        self.xsf_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        if self.npy_files == [] or self.xsf_files == []:
            raise ValueError(
                f"No npy or xsf files found in {xsf_path.expanduser()} or {npy_path.expanduser()} with pattern {pattern_xsf} or {pattern_npy}."
            )

        self.Rx = int(self.interaction_cfg.get("Rx", 0))
        self.Ry = int(self.interaction_cfg.get("Ry", 0))
        self.Rz = int(self.interaction_cfg.get("Rz", 0))

        self.num_wann = len(self.npy_files)

        dist = compute_mpi_distribution(
            mpi_size=self.mpi_size,
            rank=self.rank,
            num_wann=self.num_wann,
            n_poisson_pools=int(self.mpi_cfg.get("number_poisson_pools", 1)),
            mode=self.mode,
        )

        if self.mode == "ijij":
            self.pool_wf_indices = dist["pool_wf_indices"]
            self.rank_wf_indices = dist["rank_wf_indices"]
        elif self.mode == "ijji":
            self.pool_pair_indices = dist["pool_pair_indices"]
        self.comm_poisson = self.comm.new_communicator(dist["comm_poisson_ranks"])

        # Build Poisson grid
        # Read one of the XSF files to get the grid descriptor for the Poisson solver
        if self.comm.rank == 0:
            with open(self.xsf_files[0], "r") as xsf:
                wf_file = Xsf2Np(xsf, skip_wf=True)
            lattice_vectors = wf_file.get_lattice_vectors()
            supercell_vectors = wf_file.get_supercell()
            real_space_grid = np.array(wf_file.get_real_space_grid())
        else:
            lattice_vectors = np.zeros((3, 3), dtype=float)
            supercell_vectors = np.zeros((3, 3), dtype=float)
            real_space_grid = np.zeros((3,), dtype=int)
        self.comm.broadcast(lattice_vectors, 0)
        self.comm.broadcast(supercell_vectors, 0)
        self.comm.broadcast(real_space_grid, 0)

        # Very slow initialization, can I make it faster?
        self.pg = PoissonGrid(
            lattice_vectors,
            supercell_vectors,
            real_space_grid,
            self.comm_poisson,
            self.interaction_cfg,
        )

        # commonly used objects
        self.dV = self.pg.dV
        self.n_grid_uc = self.pg.n_grid_uc
        self.GD = self.pg.GD
        self.grid = self.pg.grid
        self.grid_full = self.pg.grid_full
        self.map_wf_to_poisson = self.pg.map_wf_to_poisson
        # This should not matter. WF is just put on a larger grid with the same spacing,
        # so nearest neighbor interpolation should be sufficient. I believe it is faster than linear interpolation,
        # but this should be tested.
        self.interp_method = "nearest"  # or "linear"

        # Build Poisson solver
        # Very slow initialization, can I make it faster?
        self.poisson_solver = PoissonSolver(name="fast", nn=3)
        self.poisson_solver.set_grid_descriptor(self.GD)

    def solve_poisson(self, charge_density, global_potential=True):
        """Solve Poisson equation for a given charge density.

        Parameters
        ----------
            charge_density (np.ndarray): Charge density array on the Poisson grid.
            global_potential (bool): Whether to compute the global potential. Default is True.

        Returns
        -------
            np.ndarray: Potential array on the Poisson grid.
        """
        local_potential = self.GD.zeros()
        self.poisson_solver.solve(local_potential, charge_density)
        if global_potential:
            potential = self.pg.map_local_to_global(local_potential)
            self.comm_poisson.sum(potential)
        else:
            potential = local_potential
        return potential

    def run(self):
        raise NotImplementedError("Subclasses must implement run().")
