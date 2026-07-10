"""
Integration tests for the V and J matrix calculations.

These tests should be run in an MPI environment, especially the test `test_v_and_j_run_consistent_across_pools`, 
which checks that the results are consistent across different numbers of MPI pools. 
"""
import os

import gpaw.mpi as mpi
import numpy as np
import pytest
import tomli_w


def _write_xsf(path, nx, ny, nz):
    with open(path, "w") as f:
        f.write("PRIMVEC\n")
        f.write("1.0 0.0 0.0\n")
        f.write("0.0 1.0 0.0\n")
        f.write("0.0 0.0 1.0\n")
        f.write("PRIMCOORD\n")
        f.write("0\n")
        f.write("BEGIN_DATAGRID_3D_UNKNOWN\n")
        f.write(f"{nx} {ny} {nz}\n")
        f.write("0.0 0.0 0.0\n")
        for i in range(3):
            f.write(
                "1.0 0.0 0.0\n"
                if i == 0
                else ("0.0 1.0 0.0\n" if i == 1 else "0.0 0.0 1.0\n")
            )
        # write some data values (nx*ny*nz floats)
        vals = [str(float(i % 6)) for i in range(nx * ny * nz)]
        for i in range(0, len(vals), 6):
            f.write(" ".join(vals[i : i + 6]) + "\n")
        f.write("END_DATAGRID_3D\n")


def _make_files(tmpdir, num_wann=3, wf_shape=(4, 4, 4)):
    npy_paths = []
    for i in range(num_wann):
        wf = np.zeros(wf_shape, dtype=float)
        wf[i % wf_shape[0], i % wf_shape[1], i % wf_shape[2]] = float(i + 1)
        path = os.path.join(tmpdir, f"wf_{i}.npy")
        np.save(path, wf)
        npy_paths.append(path)

    xsf_path = os.path.join(tmpdir, "wf_00001.xsf")
    _write_xsf(xsf_path, *wf_shape)

    npy_glob = "wf_*.npy"
    xsf_glob = "wf_*.xsf"
    return npy_glob, xsf_glob


# @pytest.mark.integration
@pytest.mark.parametrize("mode", ["ijij", "ijji"])
def test_v_and_j_run_consistent_across_pools(tmp_path, mode):
    tmpdir = str(tmp_path)
    num_wann = 3
    # wf_shape = (4, 4, 4)
    wf_shape = (8, 8, 8)

    npy_glob, xsf_glob = _make_files(tmpdir, num_wann=num_wann, wf_shape=wf_shape)

    world = mpi.world
    size = getattr(world, "size", 1)
    rank = getattr(world, "rank", 0)

    # try several pool values up to size (or 2)
    pool_choices = [1]
    if size >= 2:
        pool_choices.append(2)

    results = []

    for pools in pool_choices:
        # write a small config file per pool setting
        cfg = {
            "interaction": {"Rx": 1, "Ry": 1, "Rz": 1},
            "mpi": {"number_poisson_pools": pools},
            "paths": {"xsf_dir": tmpdir, "npy_dir": tmpdir},
            "io": {"xsf_glob": xsf_glob, "npy_glob": npy_glob},
        }
        cfg_path = os.path.join(tmpdir, f"cfg_{pools}.toml")
        with open(cfg_path, "wb") as f:
            f.write(tomli_w.dumps(cfg).encode())

        if mode == "ijij":
            from coulomb_matrix.v_matrix import VCoulombCalculator

            calc = VCoulombCalculator(config_path=cfg_path)
        else:
            from coulomb_matrix.j_matrix import JCoulombCalculator

            calc = JCoulombCalculator(config_path=cfg_path)

        V = calc.run()
        mpi.world.barrier()
        # collect results only on rank 0
        if rank == 0:
            results.append(V)

    # only assert on rank 0
    if rank == 0:
        for pool in range(1, len(results)):
            assert np.allclose(
                results[pool], results[0]
            ), f"Results differ for pool setting {pool_choices[pool]} and mode {mode}"


def test_same_diagonal(tmp_path):
    """The diagonal elements of the V_ijij and V_ijji matrices should be the same."""
    tmpdir = str(tmp_path)
    num_wann = 3
    wf_shape = (8, 8, 8)

    npy_glob, xsf_glob = _make_files(tmpdir, num_wann=num_wann, wf_shape=wf_shape)

    cfg = {
        "interaction": {"Rx": 1, "Ry": 1, "Rz": 1},
        "mpi": {"number_poisson_pools": 1},
        "paths": {"xsf_dir": tmpdir, "npy_dir": tmpdir},
        "io": {"xsf_glob": xsf_glob, "npy_glob": npy_glob},
    }
    cfg_path = os.path.join(tmpdir, "cfg.toml")
    with open(cfg_path, "wb") as f:
        f.write(tomli_w.dumps(cfg).encode())

    from coulomb_matrix.j_matrix import JCoulombCalculator
    from coulomb_matrix.v_matrix import VCoulombCalculator

    v_calc = VCoulombCalculator(config_path=cfg_path)
    j_calc = JCoulombCalculator(config_path=cfg_path)

    V_ijij = v_calc.run()
    V_ijji = j_calc.run()

    # Compare diagonals
    diag_ijij = np.diagonal(V_ijij[0, 0, 0], axis1=-2, axis2=-1)
    diag_ijji = np.diagonal(V_ijji[0, 0, 0], axis1=-2, axis2=-1)

    assert np.allclose(
        diag_ijij, diag_ijji
    ), "Diagonal elements of V_ijij and V_ijji do not match."
