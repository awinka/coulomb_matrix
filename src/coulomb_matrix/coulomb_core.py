"""Shared Coulomb matrix computation core for `ijij` and `ijji` modes.

This module provides `CoulombCalculatorBase`, which encapsulates the common
MPI/grid/IO setup. Mode-specific behavior is implemented as two helper methods
inside the class.
"""
import glob
import re
from pathlib import Path

import gpaw.mpi as mpi
import numpy as np
from gpaw.poisson import PoissonSolver

from .config import CoulombConfig
from .grid_utils import PoissonGrid
from .mpi_utils import compute_mpi_distribution
from .xsf import Xsf2Np


class CoulombCalculatorBase:
    def __init__(
        self,
        config_path=None,
        mode="ijij",
    ):
        if mode not in ("ijij", "ijji"):
            raise ValueError("mode must be 'ijij' or 'ijji'")
        self.mode = mode
        self.config, self.config_file = CoulombConfig.from_toml(config_path)
        # Shared initializations used by subclasses
        self.comm = mpi.world
        self.rank = self.comm.rank
        self.mpi_size = self.comm.size

        self.paths_cfg = self.config.paths
        self.interaction_cfg = self.config.interaction
        self.mpi_cfg = self.config.mpi
        self.io_cfg = self.config.io

        # Derived file paths and patterns
        xsf_path = self.paths_cfg.xsf_dir
        npy_path = self.paths_cfg.npy_dir
        pattern_xsf = self.io_cfg.xsf_glob
        pattern_npy = self.io_cfg.npy_glob

        self.npy_files = glob.glob(str(npy_path) + "/" + str(pattern_npy))
        self.npy_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        self.xsf_files = glob.glob(str(xsf_path) + "/" + str(pattern_xsf))
        self.xsf_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        if self.npy_files == [] or self.xsf_files == []:
            raise ValueError(
                f"No npy or xsf files found in {xsf_path.expanduser()} or {npy_path.expanduser()} with pattern {pattern_xsf} or {pattern_npy}."
            )

        self.Rx = self.interaction_cfg.Rx
        self.Ry = self.interaction_cfg.Ry
        self.Rz = self.interaction_cfg.Rz

        self.num_wann = len(self.npy_files)

        dist = compute_mpi_distribution(
            mpi_size=self.mpi_size,
            rank=self.rank,
            num_wann=self.num_wann,
            n_poisson_pools=self.mpi_cfg.number_poisson_pools,
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
