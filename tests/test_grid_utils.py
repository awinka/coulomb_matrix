"""
Tests for the grid_utils module.

The tests should be reviewed, currently I haven't spent any time on them.

An mpi test should be added, since the GridDescriptor that is used in PoissonGrid is an MPI-aware object. 
The tests here are only for the non-MPI parts of the code.
"""
import numpy as np


class _FakeGD:
    def __init__(self, comm, N_c, cell_cv, pbc_c=True):
        # N_c is an array-like of grid sizes
        self.N_c = np.int_(N_c)
        self.shape = tuple(self.N_c.tolist())
        # set grid spacing to identity so that index-space == real-space
        self.h_cv = np.eye(3)

    def get_grid_point_coordinates(self, global_array=False):
        nx, ny, nz = self.shape
        X, Y, Z = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        return X.astype(float), Y.astype(float), Z.astype(float)


def test_poissongrid_basic_properties(monkeypatch):
    # Import module and monkeypatch GridDescriptor to a lightweight fake
    import coulomb_matrix.grid_utils as gu

    monkeypatch.setattr(gu, "GridDescriptor", _FakeGD)

    lattice_vectors = np.eye(3) * 1.0
    supercell_vectors = np.eye(3) * 2.0
    real_space_grid = (4, 4, 4)
    comm = None
    interaction_cfg = {"Rx": 1, "Ry": 1, "Rz": 1}

    pg = gu.PoissonGrid(
        lattice_vectors, supercell_vectors, real_space_grid, comm, interaction_cfg
    )

    # basic numeric properties
    assert isinstance(pg.dV, float) and pg.dV > 0
    assert pg.grid.ndim == 4 and pg.grid.shape[-1] == 3
    assert pg.grid_full.ndim == 4 and pg.grid_full.shape[-1] == 3
    assert pg.n_grid.shape[0] == 3


def test_map_wf_to_poisson_nearest_preserves_point(monkeypatch):
    import coulomb_matrix.grid_utils as gu

    monkeypatch.setattr(gu, "GridDescriptor", _FakeGD)

    lattice_vectors = np.eye(3) * 1.0
    supercell_vectors = np.eye(3) * 2.0
    real_space_grid = (6, 6, 6)
    comm = None
    interaction_cfg = {"Rx": 1, "Ry": 1, "Rz": 1}
    pg = gu.PoissonGrid(
        lattice_vectors, supercell_vectors, real_space_grid, comm, interaction_cfg
    )

    # create a WF with a single spike at (1,2,3)
    wf_array = np.zeros(tuple(pg.wf_grid))
    wf_array[1, 2, 3] = 7.5

    interp = pg.map_wf_to_poisson(wf_array, pg.grid, method="nearest")
    # find grid point closest to (1,2,3) in index-space
    grid_coords = pg.grid.reshape(-1, 3)
    dists = np.linalg.norm(grid_coords - np.array([1.0, 2.0, 3.0]), axis=1)
    idx = np.argmin(dists)
    interp_flat = interp.reshape(-1)
    assert interp_flat[idx] == 7.5


def test_map_wf_to_poisson_out_of_bounds_fills_zero(monkeypatch):
    import coulomb_matrix.grid_utils as gu

    monkeypatch.setattr(gu, "GridDescriptor", _FakeGD)

    lattice_vectors = np.eye(3) * 1.0
    supercell_vectors = np.eye(3) * 2.0
    real_space_grid = (4, 4, 4)
    comm = None
    interaction_cfg = {"Rx": 1, "Ry": 1, "Rz": 1}
    pg = gu.PoissonGrid(
        lattice_vectors, supercell_vectors, real_space_grid, comm, interaction_cfg
    )

    wf_array = np.ones(tuple(pg.wf_grid))

    # create a target grid with one point far outside the interpolator domain
    target = pg.grid.reshape(-1, 3).copy()
    target = np.vstack([target, np.array([[999.0, 999.0, 999.0]])])
    out = pg.map_wf_to_poisson(wf_array, target, method="linear")
    assert out[-1] == 0.0


def test_map_local_to_global():
    """
    Test that the local grid coordinates are correctly mapped to global coordinates.

    Note: This test should be tested with MPI.
    """
    import gpaw.mpi as mpi

    import coulomb_matrix.grid_utils as gu

    lattice_vectors = np.eye(3) * 1.0
    supercell_vectors = np.eye(3) * 2.0
    real_space_grid = (4, 4, 4)
    comm = mpi.world
    interaction_cfg = {"Rx": 1, "Ry": 1, "Rz": 1}
    pg = gu.PoissonGrid(
        lattice_vectors, supercell_vectors, real_space_grid, comm, interaction_cfg
    )

    # create a local grid with a single point at (1,1,1)
    local_grid = np.ones(pg.GD.n_c)
    # Global reference is ones everywhere.
    global_ref = np.ones(pg.GD.N_c)

    global_grid = pg.map_local_to_global(local_grid)
    # Have to sum over all ranks to get the global grid
    comm.sum(global_grid)

    assert np.allclose(global_grid, global_ref)
