"""Poisson grid utilities.

Provides `PoissonGrid` which computes Poisson grid sizing and offers an
interpolation helper to map Wannier-function arrays onto the Poisson grid.
"""
import numpy as np
from ase.units import Bohr
from gpaw.grid_descriptor import GridDescriptor
from scipy.interpolate import RegularGridInterpolator


class PoissonGrid:
    """Encapsulate Poisson-grid construction and interpolation helpers.

    Example:
        pg = PoissonGrid(lattice_vectors, supercell_vectors, real_space_grid, comm_poisson, interaction_cfg)
        wf_on_poisson = pg.map_wf_to_poisson(wf_array, pg.grid, method='nearest')
    """

    def __init__(
        self,
        lattice_vectors,
        supercell_vectors,
        real_space_grid,
        comm_poisson,
        interaction_cfg: dict,
    ):
        self.lattice_vectors = lattice_vectors
        self.comm_poisson = comm_poisson
        self.interaction_cfg = interaction_cfg

        self.unit_cell_vectors = lattice_vectors / Bohr
        self.supercell_vectors = supercell_vectors / Bohr
        self.n_unit_cells = np.round(
            np.diag(self.supercell_vectors) / np.diag(self.unit_cell_vectors)
        ).reshape((3, 1))

        nx, ny, nz = real_space_grid
        self.wf_grid = np.array([nx, ny, nz])
        self.dV = np.linalg.det(self.unit_cell_vectors * self.n_unit_cells) / np.prod(
            self.wf_grid
        )
        self.n_grid_uc = np.int_(self.wf_grid / self.n_unit_cells[:, 0])

        Rx = int(interaction_cfg.get("Rx", 0))
        Ry = int(interaction_cfg.get("Ry", 0))
        Rz = int(interaction_cfg.get("Rz", 0))

        # Check that the interaction range is in a direction that is periodic. If not, raise an error.
        if (Rx > 0 and self.n_unit_cells[0, 0] < 2) or (Ry > 0 and self.n_unit_cells[1, 0] < 2) or (
            Rz > 0 and self.n_unit_cells[2, 0] < 2
        ):
            raise ValueError(
                f"Interaction range (Rx={Rx}, Ry={Ry}, Rz={Rz}) is not compatible with the number of unit cells in the supercell (n_unit_cells={self.n_unit_cells.flatten()})."
            )

        # NOTE: Size of the Poisson grid is dependent on the interaction range (Rx, Ry, Rz).
        # In the future, we might want to make this more flexible or configurable.
        default_expand = [2 * Rx, 2 * Ry, 2 * Rz]
        self.poisson_expand = np.array(default_expand).reshape(3, 1)
        self.poisson_size = self.n_unit_cells.copy()
        self.poisson_size += self.poisson_expand
        self.poisson_size[self.poisson_size < 1] = 1

        self.n_grid = np.int_(self.n_grid_uc * self.poisson_size[:, 0])

        self.GD = GridDescriptor(
            comm=comm_poisson,
            N_c=self.n_grid,
            cell_cv=self.unit_cell_vectors * self.poisson_size,
            pbc_c=True,
        )

        X, Y, Z = self.GD.get_grid_point_coordinates()
        # Center the Poisson grid in the supercell, so that when interpolating the WF onto the Poisson grid, the WF is centered in the Poisson grid.
        p_grid_shift = (
            self.unit_cell_vectors * (self.n_unit_cells - self.poisson_size) / 2
        )
        p_grid_shift = sum(p_grid_shift)
        grid = np.stack(
            (X + p_grid_shift[0], Y + p_grid_shift[1], Z + p_grid_shift[2]), axis=-1
        )
        # Go from real-space coordinates to index-space coordinates for interpolation.
        self.grid_spacing = self.GD.h_cv
        self.grid = grid @ np.linalg.inv(self.grid_spacing)

        X_full, Y_full, Z_full = self.GD.get_grid_point_coordinates(global_array=True)
        grid_full = np.stack(
            (
                X_full + p_grid_shift[0],
                Y_full + p_grid_shift[1],
                Z_full + p_grid_shift[2],
            ),
            axis=-1,
        )
        self.grid_full = grid_full @ np.linalg.inv(self.grid_spacing)

    def map_wf_to_poisson(
        self, wf_array: np.ndarray, target_grid: np.ndarray, method: str = "linear"
    ) -> np.ndarray:
        """Interpolate a WF array (shape `wf_grid`) onto `target_grid`.

        The interpolator uses index-space coordinates defined by the WF grid
        (0..nx-1, 0..ny-1, 0..nz-1) as its domain.
        """
        # Make sure the input WF array has the expected shape
        if wf_array.shape != tuple(self.wf_grid):
            raise ValueError(
                f"Expected wf_array shape {tuple(self.wf_grid)}, got {wf_array.shape}"
            )
        nx_local, ny_local, nz_local = self.wf_grid
        interp = RegularGridInterpolator(
            (np.arange(nx_local), np.arange(ny_local), np.arange(nz_local)),
            wf_array,
            method=method,
            bounds_error=False,
            fill_value=0.0,
        )
        return interp(target_grid)

    def map_local_to_global(self, local_array: np.ndarray) -> np.ndarray:
        """Map a local array (shape `n_grid`) to a global array (shape `n_grid_full`).

        This is useful for MPI parallelization, where each rank may only have a local portion of the Poisson grid.
        """
        if local_array.shape != self.grid.shape[:3]:
            raise ValueError(
                f"Expected local_array shape {tuple(self.grid.shape[:3])}, got {local_array.shape}"
            )
        global_array = self.GD.zeros(global_array=True)
        global_array[
            self.GD.beg_c[0] : self.GD.end_c[0],
            self.GD.beg_c[1] : self.GD.end_c[1],
            self.GD.beg_c[2] : self.GD.end_c[2],
        ] = local_array
        return global_array
