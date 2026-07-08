"""Poisson grid utilities.

Provides `PoissonGrid` which computes Poisson grid sizing and offers an
interpolation helper to map Wannier-function arrays onto the Poisson grid.
"""
from typing import Dict
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from ase.units import Bohr
from scipy.interpolate import RegularGridInterpolator


class PoissonGrid:
    """Encapsulate Poisson-grid construction and interpolation helpers.

    Example:
        pg = PoissonGrid(wf_file, comm_poisson, interaction_cfg, grid_cfg, mode='ijij')
        wf_on_poisson = pg.map_wf_to_poisson(wf_array, pg.grid, method='nearest')
    """

    def __init__(self, wf_file, comm_poisson, interaction_cfg: dict, grid_cfg: dict, mode: str = "ijij"):
        self.wf_file = wf_file
        self.comm_poisson = comm_poisson
        self.interaction_cfg = interaction_cfg
        self.grid_cfg = grid_cfg
        self.mode = mode

        self.unit_cell_vectors = wf_file.get_lattice_vectors() / Bohr
        self.supercell_vectors = wf_file.get_supercell() / Bohr
        self.n_unit_cells = np.round(np.diag(self.supercell_vectors) / np.diag(self.unit_cell_vectors)).reshape((3, 1))

        nx, ny, nz = wf_file.get_real_space_grid()
        self.wf_grid = np.array([nx, ny, nz])
        self.dV = np.linalg.det(self.unit_cell_vectors * self.n_unit_cells) / np.prod(self.wf_grid)
        self.n_grid_uc = np.int_(self.wf_grid / self.n_unit_cells[:, 0])

        Rx = int(interaction_cfg.get("Rx", 1))
        Ry = int(interaction_cfg.get("Ry", 1))
        Rz = int(interaction_cfg.get("Rz", 1))

        default_expand = [2 * Rx, 2 * Ry, 2 * Rz] if mode == "ijij" else [0, 0, 0]
        self.poisson_expand = np.array(grid_cfg.get("poisson_expand", default_expand)).reshape(3, 1)
        self.poisson_size = self.n_unit_cells.copy()
        self.poisson_size += self.poisson_expand
        self.poisson_size[self.poisson_size < 1] = 1

        self.n_grid = np.int_(self.n_grid_uc * self.poisson_size[:, 0])

        self.GD = GridDescriptor(comm=comm_poisson, N_c=self.n_grid, cell_cv=self.unit_cell_vectors * self.poisson_size, pbc_c=True)

        X, Y, Z = self.GD.get_grid_point_coordinates()
        self.grid_spacing = self.GD.h_cv
        p_grid_shift = self.unit_cell_vectors * (self.n_unit_cells - self.poisson_size) / 2
        p_grid_shift = sum(p_grid_shift)
        grid = np.stack((X + p_grid_shift[0], Y + p_grid_shift[1], Z + p_grid_shift[2]), axis=-1)
        self.grid = grid @ np.linalg.inv(self.grid_spacing)

        X_full, Y_full, Z_full = self.GD.get_grid_point_coordinates(global_array=True)
        grid_full = np.stack((X_full + p_grid_shift[0], Y_full + p_grid_shift[1], Z_full + p_grid_shift[2]), axis=-1)
        self.grid_full = grid_full @ np.linalg.inv(self.grid_spacing)

    def map_wf_to_poisson(self, wf_array: np.ndarray, target_grid: np.ndarray, method: str = "linear") -> np.ndarray:
        """Interpolate a WF array (shape `wf_grid`) onto `target_grid`.

        The interpolator uses index-space coordinates defined by the WF grid
        (0..nx-1, 0..ny-1, 0..nz-1) as its domain.
        """
        nx_local, ny_local, nz_local = self.wf_grid
        interp = RegularGridInterpolator(
            (np.arange(nx_local), np.arange(ny_local), np.arange(nz_local)),
            wf_array,
            method=method,
            bounds_error=False,
            fill_value=0.0,
        )
        return interp(target_grid)
